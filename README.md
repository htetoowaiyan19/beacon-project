# BEACON - Burmese AI companion

Model-only Qwen3-4B chat with LoRA training for fluent Burmese and accurate IT explanations. The assistant follows the user's language and tone: polite, casual, or close-friend.

## Train using the Python UI

**Frontend members:** start with [the frontend team guide](frontend/README.md).
Its streaming preview needs no GPU, weights or datasets and can run while training
is active. The API contract is in [API.md](API.md).

Git contains source, tools, tests and documentation. Model weights, datasets,
training outputs and the fifth-year archive are local artifacts shared separately
by the project owner. A fresh clone needs its own environment and dependencies.
Do not move or edit active training files while a run is in progress.

Double-click `train_ui.bat`, or run:

```powershell
.\.venv\Scripts\python.exe scripts/training_ui.py
```

Select the datasets and training settings, then Start training. The panel shows status, progress, loss, validation loss, learning rate, RAM/VRAM, checkpoints, a loss plot, and live logs. Stop safely saves before exiting. Each run gets a separate folder under `outputs/training_runs`; existing adapters are preserved. See [the UI guide](TRAINING_UI.md) for setup, controls, run files, and serving a new adapter.

The active IT seminar release has **14,257 conversations**: 12,759 training, 749 validation, and 749 test. It combines team-collected IT data, recovered older examples, and style conversations. See [the final dataset scan](FINAL_DATASET_REVIEW.md). Content review remains pending; see [the training plan](TRAINING_PLAN.md) and [dataset organization](datasets/README.md). Review before training, and keep the test split untouched during parameter selection.

## Run chat

```powershell
.\.venv\Scripts\python.exe scripts/run_server.py --host 127.0.0.1 --port 8000
```

Or double-click `run_server.bat`. Open `http://localhost:8000`. API docs are available at `/docs`; see [API.md](API.md). Chat output uses Vercel AI SDK UI Message Stream v1. Inference dependencies are in `requirements-inference.txt`; training dependencies are in `requirements-training.txt`. Create a virtual environment and install the appropriate dependencies on a fresh clone.

Base weights are in `models/qwen3-4b`. The server defaults to the existing adapter in `outputs/checkpoints`. To use an evaluated new adapter, set `BEACON_LORA_PATH` to its full adapter folder before starting the server. Training does not automatically replace the serving adapter. Run training and chat separately on the 16 GB GPU.

## Project layout

| Path | Purpose |
| --- | --- |
| `backend/`, `frontend/` | Model chat API and browser interface |
| `scripts/training_ui.py`, `scripts/train_lora.py` | Desktop controls and LoRA trainer |
| `datasets/freshes/` | Original team submissions |
| `datasets/releases/`, `datasets/active.json` | Prepared releases and active selection |
| `datasets/archive/`, `datasets/reviews/` | Preserved originals and review decisions |
| `models/` | Base model weights |
| `outputs/checkpoints/` | Existing trained adapter |
| `outputs/training_runs/` | New UI runs, logs, stats, adapters |
| `outputs/evaluations/` | Audits and benchmark results |
| `archives/` | Preserved fifth-year retrieval project |

## Fifth-year retrieval project

The RAG pipeline has been removed from the active project. Its documents, database/index, OCR weights, source, old frontend, tests and requirements are preserved in [the ZIP](archives/rag_5th_year_2026-10-04.zip). The archive has a SHA-256 sidecar and an internal manifest; all 56 original entries were byte-verified before removal. Extract into a separate project for reuse and read its `RESTORE.md`. Base weights, adapters and training datasets remain here rather than being duplicated into the ZIP.

[PROJECT_REVIEW.md](PROJECT_REVIEW.md) and [TRAINING_REPORT.md](TRAINING_REPORT.md) retain historical findings. Historical losses/token accuracy do not prove factual accuracy or fluent Burmese. Evaluate the reviewed new adapter on held-out IT answers and native-reviewed conversation prompts before the seminar.
