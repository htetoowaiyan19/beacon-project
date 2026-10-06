"""Model management and real-time token streaming using Qwen3-4B and PEFT LoRA."""

from __future__ import annotations

import logging
from pathlib import Path
from contextlib import nullcontext
from threading import Event, Lock, Thread
import time
from typing import Any, AsyncGenerator, Generator, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from backend.services.chat_context import prepare_chat
from backend.services.text_streamer import UnicodeTextIteratorStreamer

from backend.config import (
    DEFAULT_LORA_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MAX_HISTORY_TURNS,
)

logger = logging.getLogger("beacon.model_service")


class CancelGeneration(StoppingCriteria):
    def __init__(self, cancelled: Event):
        self.cancelled = cancelled

    def __call__(self, input_ids, scores, **kwargs):
        return self.cancelled.is_set()


def get_inference_dtype() -> torch.dtype:
    """Return optimal precision for inference."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class ModelService:
    """Singleton-style service holding tokenizer and models for streaming generation."""

    _instance: Optional[ModelService] = None

    def __init__(
        self,
        base_model_path: Optional[Path | str] = None,
        lora_checkpoint_path: Optional[Path | str] = None,
        lazy_load: bool = True,
    ) -> None:
        self.base_model_path = Path(base_model_path or DEFAULT_MODEL_PATH)
        self.lora_checkpoint_path = Path(lora_checkpoint_path or DEFAULT_LORA_PATH)

        self.tokenizer = None
        self.base_model = None
        self.lora_model = None
        self._is_loaded = False
        self._generation_lock = Lock()

        if not lazy_load:
            self.load()

    @classmethod
    def get_instance(cls) -> ModelService:
        """Access or initialize global ModelService instance."""
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def load(self) -> None:
        """Load tokenizer, base model, and LoRA adapter into memory."""
        if self._is_loaded:
            return

        logger.info(f"Loading tokenizer from: {self.base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.base_model_path))

        logger.info(f"Loading base model from: {self.base_model_path}...")
        device_map = "auto" if torch.cuda.is_available() else None
        dtype = get_inference_dtype()

        self.base_model = AutoModelForCausalLM.from_pretrained(
            str(self.base_model_path),
            dtype=dtype,
            attn_implementation="sdpa",
            device_map=device_map,
        )
        self.base_model.eval()
        placement = getattr(self.base_model, 'hf_device_map', {})
        if torch.cuda.is_available() and any(str(device) in {'cpu', 'disk'} for device in placement.values()):
            logger.warning('Model layers are offloaded to CPU/disk. Free GPU memory and restart for faster chat.')

        if (self.lora_checkpoint_path / "adapter_config.json").exists():
            logger.info(f"Loading LoRA adapter from: {self.lora_checkpoint_path}...")
            self.lora_model = PeftModel.from_pretrained(self.base_model, str(self.lora_checkpoint_path))
            self.lora_model.eval()
        else:
            logger.warning(f"LoRA path not found: {self.lora_checkpoint_path}. Using base model.")
            self.lora_model = None

        self._is_loaded = True
        logger.info("Model and tokenizer loaded successfully.")

    def get_model(self, use_base_model: bool = False):
        """Get the active model (LoRA or Base)."""
        if not self._is_loaded:
            self.load()
        if use_base_model or self.lora_model is None:
            return self.base_model
        return self.lora_model

    def stream_chat(
        self, *args, **kwargs,
    ) -> Generator[str, None, dict[str, Any]]:
        # PEFT attaches adapters to the base in place. Serialize switching and generation.
        queued_at = time.perf_counter()
        with self._generation_lock:
            queue_seconds = time.perf_counter() - queued_at
            metrics = yield from self._stream_chat(*args, **kwargs)
            metrics['queue_seconds'] = round(queue_seconds, 3)
            return metrics

    def _stream_chat(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
        use_base_model: bool = False,
        think: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> Generator[str, None, dict[str, Any]]:
        """Stream generated tokens one-by-one from the model.

        Yields:
            Text chunks (tokens) as generated.
        Returns:
            Dictionary with generation metrics (total_tokens, elapsed_seconds, tokens_per_second).
        """
        if not self._is_loaded:
            self.load()

        active_model = self.get_model(use_base_model=use_base_model)
        if not use_base_model and self.lora_model is None:
            raise RuntimeError('The trained adapter is missing. Restore the release adapter or explicitly select the base model.')
        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        # The Burmese-heavy adapter can ignore the general language instruction.
        # Reinforce it for English input without changing the user's message.
        latest_user = next((m.get("content", "") for m in reversed(messages)
                            if m.get("role") == "user"), "")
        if not system_prompt and latest_user.isascii() and any(c.isalpha() for c in latest_user):
            sys_prompt += (
                " The current user message is in English. Write your entire answer in English, "
                "unless the user explicitly requests another language."
            )

        context_limit = getattr(active_model.config, "max_position_embeddings", None)
        encoded, _, context_metrics = prepare_chat(
            self.tokenizer, messages, sys_prompt, think=think,
            max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
            max_history_turns=DEFAULT_MAX_HISTORY_TURNS,
            context_limit=context_limit, max_new_tokens=max_new_tokens,
        )

        # Get actual model device safely
        model_device = next(active_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in encoded.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        streamer = UnicodeTextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,
        )

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        cancelled = Event()
        outcome = {}

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "top_p": top_p if temperature > 0 else None,
            "repetition_penalty": 1.15,
            "pad_token_id": pad_token_id,
            "use_cache": True,
            "stopping_criteria": StoppingCriteriaList([CancelGeneration(cancelled)]),
        }

        # Run generation in a background thread with inference_mode & error handling
        def _run_generation():
            try:
                adapter_context = (
                    self.lora_model.disable_adapter()
                    if use_base_model and self.lora_model is not None else nullcontext()
                )
                with adapter_context, torch.inference_mode():
                    output = active_model.generate(**gen_kwargs)
                outcome["total_tokens"] = int(output.shape[-1] - prompt_tokens)
                outcome["finish_reason"] = "length" if outcome["total_tokens"] >= max_new_tokens else "stop"
            except Exception as exc:
                logger.error(f"Generation error in worker thread: {exc}", exc_info=True)
                outcome["error"] = exc
                streamer.end()

        thread = Thread(target=_run_generation)
        start_time = time.perf_counter()
        thread.start()

        try:
            for token_chunk in streamer:
                yield token_chunk
        finally:
            cancelled.set()
            thread.join()
        if "error" in outcome:
            raise RuntimeError("Model generation failed") from outcome["error"]
        token_count = outcome.get("total_tokens", 0)
        elapsed = time.perf_counter() - start_time
        tps = token_count / elapsed if elapsed > 0 else 0

        return {
            **context_metrics,
            "total_tokens": token_count,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(tps, 1),
            "prompt_tokens": prompt_tokens,
            "finish_reason": outcome.get("finish_reason", "stop"),
        }

    def get_gpu_status(self) -> dict[str, Any]:
        """Return GPU and memory usage statistics."""
        status = {
            "is_loaded": self._is_loaded,
            "has_lora": self.lora_model is not None,
            "adapter_available": all((self.lora_checkpoint_path / name).is_file()
                                     for name in ("adapter_model.safetensors", "adapter_config.json")),
            "adapter_name": self.lora_checkpoint_path.name,
            "execution_devices": sorted({str(device) for device in getattr(self.base_model, 'hf_device_map', {}).values()}),
            "cpu_offload": any(str(device) in {'cpu', 'disk'} for device in getattr(self.base_model, 'hf_device_map', {}).values()),
        }
        if not torch.cuda.is_available():
            return dict(status, device="CPU", vram_allocated_gb=0.0, vram_reserved_gb=0.0)

        device = torch.cuda.current_device()
        bytes_per_gb = 1024**3
        return {
            "device": torch.cuda.get_device_name(device),
            **status,
            "vram_allocated_gb": round(torch.cuda.memory_allocated(device) / bytes_per_gb, 2),
            "vram_reserved_gb": round(torch.cuda.memory_reserved(device) / bytes_per_gb, 2),
        }
