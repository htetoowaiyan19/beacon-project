"""Download and cache the base model locally."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen/Qwen3-4B"
SAVE_PATH = PROJECT_ROOT / "models" / "qwen3-4b"


def main() -> None:
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map="auto",
    )

    print(f"Saving to {SAVE_PATH}...")
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(SAVE_PATH)
    model.save_pretrained(SAVE_PATH, safe_serialization=True)

    print("Done!")


if __name__ == "__main__":
    main()
