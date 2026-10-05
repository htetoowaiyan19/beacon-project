"""Chat API routes with real-time Server-Sent Events (SSE) token streaming."""

from __future__ import annotations

import logging
from typing import Literal, Optional
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)

logger = logging.getLogger("beacon.routes.chat")

router = APIRouter(prefix="/api/chat", tags=["Chat"])


class MessageItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000, description="User query in Burmese or English")
    history: list[MessageItem] = Field(default_factory=list, description="Prior conversation history")
    use_base_model: bool = Field(default=False, description="Use Base model instead of LoRA fine-tuned model")
    think: bool = Field(default=False, description="Enable thinking mode")
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(default=DEFAULT_TOP_P, ge=0.0, le=1.0)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, ge=16, le=2048)
    system_prompt: Optional[str] = None


@router.post("/stream")
def chat_stream(request: ChatRequest, http_request: Request):
    """AI SDK UI Message Stream: text, metrics, errors and completion."""
    history_dicts = [{"role": m.role, "content": m.content} for m in request.history]

    event_generator = http_request.app.state.chat_service.stream_response(
        user_message=request.message,
        conversation_history=history_dicts,
        use_base_model=request.use_base_model,
        think=request.think,
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_new_tokens,
        system_prompt=request.system_prompt,
    )

    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1",
        },
    )
