"""Create the model-only chat application."""
from __future__ import annotations
import logging
import os
from contextlib import asynccontextmanager
import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.config import FRONTEND_DIR
from backend.routes.chat import router as chat_router
from backend.routes.health import router as health_router
from backend.services.chat_service import ChatService
from backend.release import public_release
from backend.routes.show_day import router as show_day_router
from backend.services.show_day_monitor import ShowDayMonitor


def create_app(*, chat_service=None, monitor=None) -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            service = getattr(app.state.chat_service, '_model_service', None)
            close = getattr(service, 'close', None)
            if close:
                await anyio.to_thread.run_sync(close)
    app = FastAPI(title="BEACON Burmese AI", version="3.0.0", lifespan=lifespan)
    app.state.chat_service = chat_service or ChatService()
    token = os.getenv('BEACON_MONITOR_TOKEN') if os.getenv('BEACON_SHOW_DAY') == '1' else None
    app.state.show_day_monitor = monitor or (ShowDayMonitor(token) if token else None)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
        expose_headers=["x-vercel-ai-ui-message-stream"],
    )
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(show_day_router)

    @app.get("/api/config")
    def capabilities():
        return {"mode": "model-only", "stream_protocol": "ui-message-stream-v1", "release": public_release(),
                "show_day": app.state.show_day_monitor is not None}

    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    return app
