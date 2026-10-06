"""Backend configuration settings."""

from __future__ import annotations

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "qwen3-4b"
from backend.release import RELEASE_ADAPTER
DEFAULT_LORA_PATH = Path(os.getenv("BEACON_LORA_PATH", str(RELEASE_ADAPTER)))

# Runtime directories are created only by the services that use them.

# Shared language-adaptive companion persona
from scripts.utils.persona import DEFAULT_SYSTEM_PROMPT as TRAINING_SYSTEM_PROMPT
DEFAULT_SYSTEM_PROMPT = TRAINING_SYSTEM_PROMPT + (
    ' Use short, natural sentences. Answer technical questions directly. '
    'You are an AI: do not invent physical experiences, personal plans, locations or current facts. '
    'If information is missing, ask a brief clarifying question.'
)

# Generation defaults
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.8

# Server
HOST = os.getenv("BEACON_HOST", "0.0.0.0")
PORT = int(os.getenv("BEACON_PORT", "8000"))
