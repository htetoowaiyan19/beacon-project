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
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

from backend.config import (
    DEFAULT_LORA_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
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
        with self._generation_lock:
            return (yield from self._stream_chat(*args, **kwargs))

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
        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        # Build message history
        conversation: list[dict[str, str]] = []
        if sys_prompt:
            conversation.append({"role": "system", "content": sys_prompt})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # If this is the last user message and think is False, prepend /no_think
            conversation.append({"role": role, "content": content})

        # Format chat template
        chat_text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think,
        )

        # Get actual model device safely
        model_device = next(active_model.parameters()).device
        encoded = self.tokenizer(chat_text, return_tensors="pt")
        inputs = {k: v.to(model_device) for k, v in encoded.items()}
        prompt_tokens = inputs["input_ids"].shape[1]
        context_limit = getattr(active_model.config, "max_position_embeddings", None)
        if context_limit and prompt_tokens + max_new_tokens > context_limit:
            raise ValueError("Conversation exceeds the model context window. Start a new chat.")

        streamer = TextIteratorStreamer(
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
            "total_tokens": token_count,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(tps, 1),
            "prompt_tokens": prompt_tokens,
            "finish_reason": outcome.get("finish_reason", "stop"),
        }

    def get_gpu_status(self) -> dict[str, Any]:
        """Return GPU and memory usage statistics."""
        if not torch.cuda.is_available():
            return {"device": "CPU", "vram_allocated_gb": 0.0, "vram_reserved_gb": 0.0}

        device = torch.cuda.current_device()
        bytes_per_gb = 1024**3
        return {
            "device": torch.cuda.get_device_name(device),
            "vram_allocated_gb": round(torch.cuda.memory_allocated(device) / bytes_per_gb, 2),
            "vram_reserved_gb": round(torch.cuda.memory_reserved(device) / bytes_per_gb, 2),
            "is_loaded": self._is_loaded,
            "has_lora": self.lora_model is not None,
        }
