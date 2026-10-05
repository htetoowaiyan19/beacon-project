"""Shared chat orchestration using Vercel's AI SDK UI Message Stream v1."""
from __future__ import annotations
import json
import logging
from uuid import uuid4

logger = logging.getLogger("beacon.chat")


def event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class ChatService:
    """Plain model chat with no retrieval or OCR dependencies."""

    def __init__(self, model_service=None):
        self._model_service = model_service

    @property
    def model_service(self):
        if self._model_service is None:
            from backend.services.model_service import ModelService
            self._model_service = ModelService.get_instance()
        return self._model_service

    def stream_response(self, user_message: str, conversation_history: list[dict], **kwargs):
        message_id, text_id = uuid4().hex, uuid4().hex
        yield event({"type": "start", "messageId": message_id})
        generator = None
        text_started = False
        try:
            yield event({"type": "start-step"})
            yield event({"type": "text-start", "id": text_id})
            text_started = True
            options = {k: v for k, v in kwargs.items() if k in {
                "use_base_model", "think", "temperature", "top_p", "max_new_tokens", "system_prompt"
            }}
            generator = self.model_service.stream_chat(
                messages=[*conversation_history, {"role": "user", "content": user_message}], **options
            )
            while True:
                try:
                    delta = next(generator)
                except StopIteration as finished:
                    metrics = finished.value or {}
                    break
                if delta:
                    yield event({"type": "text-delta", "id": text_id, "delta": delta})
            yield event({"type": "text-end", "id": text_id})
            text_started = False
            yield event({"type": "data-metrics", "data": metrics})
            yield event({"type": "finish-step"})
            yield event({"type": "finish", "finishReason": metrics.get("finish_reason", "stop")})
        except Exception:
            logger.exception("Chat stream failed")
            if text_started:
                yield event({"type": "text-end", "id": text_id})
            yield event({"type": "error", "errorText": "Chat generation failed. Check the server log."})
            yield event({"type": "finish", "finishReason": "error"})
        finally:
            if generator is not None:
                generator.close()
        yield "data: [DONE]\n\n"
