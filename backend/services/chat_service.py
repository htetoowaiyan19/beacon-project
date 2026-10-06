"""Shared chat orchestration using Vercel's AI SDK UI Message Stream v1."""
from __future__ import annotations
import json
import logging
import time
from uuid import uuid4
from backend.services.chat_context import ChatContextError

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
            import os
            if os.getenv('BEACON_RUNTIME') == 'gguf':
                from backend.services.gguf_model_service import GGUFModelService
                self._model_service = GGUFModelService()
            else:
                from backend.services.model_service import ModelService
                self._model_service = ModelService.get_instance()
        return self._model_service

    def stream_response(self, user_message: str, conversation_history: list[dict], **kwargs):
        started_at = time.perf_counter()
        first_text_seconds = None
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
                    if first_text_seconds is None:
                        first_text_seconds = time.perf_counter() - started_at
                    yield event({"type": "text-delta", "id": text_id, "delta": delta})
            metrics['time_to_first_text_seconds'] = round(first_text_seconds, 3) if first_text_seconds is not None else None
            metrics['request_elapsed_seconds'] = round(time.perf_counter() - started_at, 3)
            logger.info('Chat timing: input=%s first_text=%ss queue=%ss total=%ss history_dropped=%s',
                        metrics.get('prompt_tokens', 'unknown'), metrics['time_to_first_text_seconds'],
                        metrics.get('queue_seconds', 'unknown'), metrics['request_elapsed_seconds'],
                        metrics.get('history_messages_dropped', 0))
            yield event({"type": "text-end", "id": text_id})
            text_started = False
            yield event({"type": "data-metrics", "data": metrics})
            yield event({"type": "finish-step"})
            yield event({"type": "finish", "finishReason": metrics.get("finish_reason", "stop")})
        except ChatContextError as exc:
            if text_started:
                yield event({"type": "text-end", "id": text_id})
            yield event({"type": "error", "errorText": str(exc)})
            yield event({"type": "finish", "finishReason": "error"})
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
