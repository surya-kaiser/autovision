"""
AutoVision FastAPI application entry point.
"""
import platform
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import dataset, training, inference, pilot
from app.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"AutoVision {settings.VERSION} starting up...")
    from app.core.llm_engine import check_ollama_available
    if check_ollama_available():
        logger.info(f"Ollama online — model: {settings.OLLAMA_MODEL}")
    else:
        logger.warning("Ollama not available — using rule-based recommendations")
    yield
    logger.info("AutoVision shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AutoVision MLOps Platform — automated ML training and inference",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(dataset.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")
app.include_router(inference.router, prefix="/api/v1")
app.include_router(pilot.router, prefix="/api/v1")


# ── Health & Info ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.VERSION}


@app.get("/api/v1/system/info")
async def system_info():
    import psutil
    try:
        cpu_pct = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        info = {
            "cpu_percent": cpu_pct,
            "memory_total_gb": round(mem.total / 1e9, 2),
            "memory_used_gb": round(mem.used / 1e9, 2),
            "platform": platform.system(),
            "python": platform.python_version(),
        }
        try:
            import torch
            info["gpu_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["gpu_available"] = False
    except Exception as e:
        info = {"error": str(e)}
    return {"status": "success", "data": info}


@app.get("/api/v1/llm/status")
async def llm_status():
    from app.core.llm_engine import check_ollama_available
    online = check_ollama_available()
    return {
        "status": "success",
        "data": {
            "ollama_online": online,
            "model": settings.OLLAMA_MODEL,
            "base_url": settings.OLLAMA_BASE_URL,
        },
    }


