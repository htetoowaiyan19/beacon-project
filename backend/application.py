"""Create the model-only chat application."""
from __future__ import annotations
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.config import FRONTEND_DIR
from backend.routes.chat import router as chat_router
from backend.routes.health import router as health_router
from backend.services.chat_service import ChatService


def create_app(*, chat_service=None) -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="BEACON Burmese AI", version="3.0.0")
    app.state.chat_service = chat_service or ChatService()
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
        expose_headers=["x-vercel-ai-ui-message-stream"],
    )
    app.include_router(health_router)
    app.include_router(chat_router)

    @app.get("/api/config")
    def capabilities():
        return {"mode": "model-only", "stream_protocol": "ui-message-stream-v1"}

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    return app
