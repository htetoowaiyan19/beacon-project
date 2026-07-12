"""Shared generation defaults for model chat and evaluation scripts."""

from __future__ import annotations

from typing import Final


GENERATION_CONFIG: Final[dict[str, float | int | bool]] = {
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "repetition_penalty": 1.15,
}
