"""Model and GPU health diagnostics."""
from fastapi import APIRouter, Request
from backend.release import public_release

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health")
def health_check(request: Request):
    state = request.app.state
    gpu = state.chat_service.model_service.get_gpu_status()
    return {"status": "online", "mode": "model-only", "gpu": gpu, "release": public_release()}
