# BEACON — Burmese AI companion

**v1.0.0, IT Seminar.** Local Qwen3-4B chat with the trained Burmese LoRA adapter.
It follows the user's language and polite, casual or close-friend tone. The active
application is model-only. Burmese fluency and factual accuracy remain uneven;
see the measured results in [RELEASE_V1.md](RELEASE_V1.md).

For a folder map, measured disk usage, review order and backup/cleanup decisions,
read [PROJECT_REVIEW_GUIDE.md](PROJECT_REVIEW_GUIDE.md).

## Run the project

For detailed installation options, verification, start/stop commands and
troubleshooting, read [SETUP.md](SETUP.md).

On the configured seminar machine, double-click **show_day_ui.bat**. Start Backend
and Frontend from the panel, then open the visitor page on the front monitor.
The members' monitor shows logs, GPU statistics and submitted visitor chats.
See [SHOW_DAY.md](SHOW_DAY.md). Closing the panel stops its servers.

For ordinary chat, double-click **run_trained_chat.bat** and visit
**http://127.0.0.1:8000/**. Press Ctrl+C in its window to stop it. The launcher
verifies the frozen adapter in `models/adapters/beacon-v1.0.0` before serving.
The API documentation is at `/docs` and in [API.md](API.md).

## Move to another device

**Windows laptop with RTX 4050 / 6 GB VRAM:** use the trained Q4 GGUF/CUDA
edition in `builds/BEACON-v1.0.0-windows-cuda.zip`. Follow the step-by-step
[WINDOWS_GPU.md](WINDOWS_GPU.md). It includes the latest frontend and native
CUDA/CPU runtimes without the original training stack.

**M2 MacBook Air with 8 GB RAM:** use the smaller
`builds/BEACON-v1.0.0-m2-air.zip` with trained Q4_K_M GGUF and Metal inference.
See [M2_AIR.md](M2_AIR.md). It retains the visitor page and members' panel and needs
no PyTorch for inference. The full package below preserves training data and weights.

Use `builds/BEACON-v1.0.0-portable.zip`, which includes source, the complete base
model, trained adapter, datasets and review provenance. Follow
[TRANSFER.md](TRANSFER.md) to verify it and install dependencies in a fresh
environment. Python packages and GPU drivers require installation; the ZIP does
not copy the original machine's virtual environment.

Rebuild the verified transfer archive after changes:

```powershell
.\.venv\Scripts\python.exe scripts/package_release.py
```

Git contains source and documentation. Weights, datasets, training outputs and
local archives are deliberately excluded. A source-only clone needs those artifacts
to run the real model.

## Frontend development

Start with [frontend/README.md](frontend/README.md). Its explicitly labeled mock
preview needs no GPU or weights. The client is plain HTML, CSS and JavaScript,
with Vercel AI SDK UI Message Stream v1 over SSE. No frontend build is needed.

The real model processes recent conversation turns within a 2,048-token input
budget. Older messages remain visible/exportable in the browser. Replies stream
incrementally, including Burmese text; waiting time and queue time appear in stats.

## Data and optional training

The completed experiment used **14,257 conversations**: 12,759 train, 749
validation and 749 held-out test. Original team submissions, recovered older IT
data, review decisions and release manifests are preserved. Read
[datasets/README.md](datasets/README.md) and
[FINAL_DATASET_REVIEW.md](FINAL_DATASET_REVIEW.md) for quality findings and remaining
content review. Do not change these splits when comparing against the completed run.

Double-click **train_ui.bat** for training controls, progress and safe stop/resume.
See [TRAINING_UI.md](TRAINING_UI.md). Training and chat should run separately on
the seminar GPU. Training creates a new run; it does not replace the serving
adapter. Select another evaluated adapter explicitly with `BEACON_LORA_PATH` and
`scripts/run_server.py`.

## Project layout

| Path | Purpose |
| --- | --- |
| `backend/`, `frontend/` | Model API and visitor chat |
| `scripts/` | Server controls, training, dataset builders, evaluation and packaging |
| `tests/`, `prompts/` | Automated checks and evaluation prompts |
| `datasets/active.json`, `datasets/releases/` | Selected dataset and frozen splits |
| `datasets/freshes/`, `datasets/sources/`, `datasets/reviews/` | Originals and preparation evidence |
| `models/qwen3-4b/`, `models/adapters/beacon-v1.0.0/` | Base weights and serving adapter |
| `models/gguf/`, `runtime/` | Quantized laptop exports and bundled native runtime |
| `outputs/` | Local training checkpoints, logs and evaluation results |
| `builds/` | Model and transfer ZIPs with checksum sidecars |
| `archives/` | Preserved RAG and project/data history |

The removed retrieval project remains in
`archives/rag_5th_year_2026-10-04.zip` for the fifth-year project. Superseded guides,
unused CLI chat tools and old notebooks are preserved in
`archives/project_history_2026-10-06.zip` and Git history. These archives are
separate from the transfer package; the current app does not need them.
