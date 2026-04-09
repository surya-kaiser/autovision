"""
Pilot run endpoints — quick 3-epoch test with time estimate.
"""
import asyncio
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.models.schemas import APIResponse, TrainingConfig
from app.services.pilot_runner import run_pilot
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/pilot", tags=["pilot"])

_pilot_queues: dict[str, asyncio.Queue] = {}


@router.post("/run", response_model=APIResponse)
async def pilot_run(config: TrainingConfig):
    """Start a pilot run (non-blocking). Connect WS for live output."""
    sid = config.session_id
    q: asyncio.Queue[str] = asyncio.Queue()
    _pilot_queues[sid] = q

    async def _run():
        result = await run_pilot(config, q)
        _pilot_queues[sid] = q  # keep queue alive for WS

    asyncio.create_task(_run())
    return APIResponse(
        status="success",
        data={"session_id": sid},
        message="Pilot run started. Connect to WebSocket for output.",
    )


@router.websocket("/ws/{session_id}")
async def pilot_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in _pilot_queues:
        _pilot_queues[session_id] = asyncio.Queue()
    q = _pilot_queues[session_id]
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=60.0)
                await websocket.send_json({"log": msg})
                if "Confirm to proceed" in msg or "ERROR" in msg:
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({"heartbeat": True})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error(f"Pilot WS error: {exc}")
    finally:
        _pilot_queues.pop(session_id, None)
