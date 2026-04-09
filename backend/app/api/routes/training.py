"""
Training endpoints + WebSocket for real-time log streaming.
"""
import asyncio
import json
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.models.schemas import APIResponse, TrainingConfig
from app.services.trainer import train_model, auto_train_with_retry, get_training_status, TRAINING_REGISTRY
from app.services.evaluator import get_all_results, compare_models, evaluate_model
from app.services.metadata_store import get_session_summary, read_training_log
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/training", tags=["training"])

# In-memory log queues per session
_log_queues: dict[str, asyncio.Queue] = {}
# In-memory task refs for cancellation
_training_tasks: dict[str, asyncio.Task] = {}


@router.post("/start", response_model=APIResponse)
async def start_training(config: TrainingConfig):
    """Start full model training (non-blocking, use WebSocket for logs)."""
    sid = config.session_id
    log_queue: asyncio.Queue[str] = asyncio.Queue()
    _log_queues[sid] = log_queue

    async def _run():
        await train_model(config, log_queue)

    task = asyncio.create_task(_run())
    _training_tasks[sid] = task
    return APIResponse(
        status="success",
        data={"session_id": sid, "model_type": config.model_type.value},
        message="Training started. Connect to WebSocket for live logs.",
    )


@router.post("/stop/{session_id}", response_model=APIResponse)
async def stop_training(session_id: str):
    """Cancel an in-progress training run."""
    task = _training_tasks.pop(session_id, None)
    status = get_training_status(session_id)
    stopped = False
    if task and not task.done():
        task.cancel()
        stopped = True
    if status:
        status.status = "failed"
    queue = _log_queues.get(session_id)
    if queue:
        await queue.put("=== Training stopped by user ===")
    return APIResponse(
        status="success",
        data={"stopped": stopped},
        message="Training stopped" if stopped else "No active training found",
    )


@router.post("/auto-start", response_model=APIResponse)
async def auto_start_training(session_id: str):
    """Get LLM recommendation, then train automatically — retries with fallback on error."""
    from app.core.config import settings
    from app.core.llm_engine import get_recommendation
    from app.services.preprocessor import detect_format, detect_task_type, get_dataset_summary as _get_summary
    from app.models.schemas import ModelType, TaskType, TrainingConfig as TC, DatasetFormat

    session_dir = settings.UPLOAD_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # Locate dataset
    csvs = list(session_dir.rglob("*.csv"))
    dataset_path = csvs[0] if csvs else session_dir

    try:
        fmt = detect_format(dataset_path)
        tt = detect_task_type(fmt, dataset_path)
        summary = _get_summary(dataset_path, session_id)
        recommendation = get_recommendation(tt, summary)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {exc}")

    # Proactive check: image tasks require PyTorch — fail fast with a clear message
    _image_task = tt in (TaskType.CLASSIFICATION, TaskType.SEGMENTATION)
    _image_format = fmt in (DatasetFormat.IMAGE_FOLDER, DatasetFormat.ZIP)
    if _image_task and _image_format:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail=(
                    "PyTorch is not installed but this dataset requires deep learning. "
                    "Install it with: pip install torch torchvision "
                    "--index-url https://download.pytorch.org/whl/cpu"
                ),
            )

    # Map recommended model string to ModelType
    from app.models.schemas import DatasetFormat
    try:
        model_type = ModelType(recommendation.model_type)
    except (ValueError, KeyError):
        if tt == TaskType.SEGMENTATION:
            model_type = ModelType.UNET
        elif tt == TaskType.CLASSIFICATION and fmt in (DatasetFormat.IMAGE_FOLDER, DatasetFormat.ZIP):
            model_type = ModelType.CNN
        else:
            model_type = ModelType.RANDOM_FOREST

    hp = recommendation.hyperparams or {}
    config = TC(
        session_id=session_id,
        model_type=model_type,
        task_type=tt,
        epochs=10,
        batch_size=int(hp.get("batch_size", 32)),
        learning_rate=float(hp.get("learning_rate", 0.001)),
        early_stopping=True,
        patience=5,
        hyperparams=hp,
    )

    log_queue: asyncio.Queue[str] = asyncio.Queue()
    _log_queues[session_id] = log_queue

    async def _auto_run():
        await auto_train_with_retry(config, log_queue)

    task = asyncio.create_task(_auto_run())
    _training_tasks[session_id] = task

    return APIResponse(
        status="success",
        data={
            "session_id": session_id,
            "model_type": model_type.value,
            "recommendation": recommendation.model_dump(),
        },
        message=f"Auto-training started with {model_type.value}",
    )


@router.get("/status/{session_id}", response_model=APIResponse)
async def training_status(session_id: str):
    status = get_training_status(session_id)
    if not status:
        raise HTTPException(status_code=404, detail="No training found for this session")
    return APIResponse(status="success", data=status.model_dump(), message="")


@router.get("/results/{session_id}", response_model=APIResponse)
async def training_results(session_id: str):
    results = get_all_results(session_id)
    return APIResponse(status="success", data=results, message="")


@router.get("/compare/{session_id}", response_model=APIResponse)
async def model_comparison(session_id: str):
    comparison = compare_models(session_id)
    return APIResponse(status="success", data=comparison, message="")


@router.get("/log/{session_id}/{model_type}", response_model=APIResponse)
async def get_log(session_id: str, model_type: str):
    """Return the full persisted training log for a session+model."""
    log = read_training_log(session_id, model_type)
    return APIResponse(status="success", data={"log": log}, message="")


@router.get("/history/{session_id}", response_model=APIResponse)
async def training_history(session_id: str):
    """Return full session summary — dataset + all runs + log tails."""
    summary = get_session_summary(session_id)
    return APIResponse(status="success", data=summary, message="")


@router.post("/evaluate", response_model=APIResponse)
async def evaluate(session_id: str, model_type: str, task_type: str):
    from app.models.schemas import TaskType
    try:
        tt = TaskType(task_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown task type: {task_type}")

    results = evaluate_model(session_id, model_type, tt)
    return APIResponse(status="success", data=results, message="Evaluation complete")


@router.post("/autonomous", response_model=APIResponse)
async def start_autonomous_training(
    session_id: str,
    max_experiments: int = 3,
    task_type: Optional[str] = None,
    dataset_name: str = "Dataset",
):
    """
    Start fully autonomous ML pipeline using AI agents.
    Connect via WebSocket at /training/ws/{session_id} for real-time logs.
    """
    from app.core.config import settings
    from app.services.pilot_runner import run_autonomous_pipeline

    session_dir = settings.UPLOAD_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # Find dataset path
    csvs = list(session_dir.rglob("*.csv"))
    dataset_path = csvs[0] if csvs else session_dir

    log_queue: asyncio.Queue[str] = asyncio.Queue()
    _log_queues[session_id] = log_queue

    async def _autonomous_run():
        try:
            result = await run_autonomous_pipeline(
                session_id=session_id,
                dataset_path=str(dataset_path),
                log_queue=log_queue,
                max_experiments=max_experiments,
                task_type_override=task_type,
                dataset_name=dataset_name,
            )
            if result and result.get("status") == "completed":
                await log_queue.put("✅ AUTONOMOUS PIPELINE COMPLETE")
            else:
                await log_queue.put("❌ AUTONOMOUS PIPELINE FAILED")
        except Exception as exc:
            logger.error(f"Autonomous run uncaught error: {exc}", exc_info=True)
            await log_queue.put(f"❌ AUTONOMOUS PIPELINE FAILED (error: {exc})")

    task = asyncio.create_task(_autonomous_run())
    _training_tasks[session_id] = task

    return APIResponse(
        status="success",
        data={
            "session_id": session_id,
            "mode": "autonomous",
            "max_experiments": max_experiments,
        },
        message="Autonomous training started. Connect to WebSocket for logs.",
    )


@router.websocket("/ws/{session_id}")
async def training_ws(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming training logs."""
    await websocket.accept()
    logger.info(f"WS connected for session {session_id}")

    # Create queue if not exists
    if session_id not in _log_queues:
        _log_queues[session_id] = asyncio.Queue()

    queue = _log_queues[session_id]

    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
                status = get_training_status(session_id)
                ws_status = status.status if status else "running"

                # Autonomous pipeline may not populate TRAINING_REGISTRY status.
                if "AUTONOMOUS PIPELINE COMPLETE" in msg:
                    ws_status = "completed"
                elif "AUTONOMOUS PIPELINE FAILED" in msg:
                    ws_status = "failed"

                # Derive rough progress % from log message step tags [1/3], [2/3] etc.
                pct = 0
                import re
                m = re.search(r'\[(\d+)/(\d+)\]', msg)
                if m:
                    pct = int(int(m.group(1)) / int(m.group(2)) * 100)
                elif status and status.status == "completed":
                    pct = 100

                payload = {
                    "log": msg,
                    "status": ws_status,
                    "metrics": status.metrics if status else {},
                    "progress": pct,
                }
                await websocket.send_json(payload)

                # When done, drain remaining queued messages before closing
                if ws_status in ("completed", "failed"):
                    while not queue.empty():
                        try:
                            extra = queue.get_nowait()
                            m2 = re.search(r'\[(\d+)/(\d+)\]', extra)
                            p2 = int(int(m2.group(1)) / int(m2.group(2)) * 100) if m2 else pct
                            await websocket.send_json({
                                "log": extra,
                                "status": ws_status,
                                "metrics": status.metrics or {},
                                "progress": p2,
                            })
                        except Exception:
                            break
                    await asyncio.sleep(0.2)
                    try:
                        await websocket.close()
                    except Exception:
                        pass
                    break
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"heartbeat": True})
    except WebSocketDisconnect:
        logger.info(f"WS disconnected for session {session_id}")
    except Exception as exc:
        logger.error(f"WS error for {session_id}: {exc}")
    finally:
        _log_queues.pop(session_id, None)
