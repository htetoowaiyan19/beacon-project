# BEACON - Burmese AI companion

**BEACON v1.0.0 (IT Seminar)** is ready for local chat. Double-click `run_trained_chat.bat` and open **http://127.0.0.1:8000/**. See [the release guide](RELEASE_V1.md) for the verified adapter, results and sharing instructions.

Model-only Qwen3-4B chat with Burmese LoRA training and IT conversations. The assistant follows the user's language and tone: polite, casual, or close-friend. Burmese fluency and factual accuracy remain uneven; check important answers.

## Train using the Python UI

**Show day:** double-click `show_day_ui.bat` for separate server controls, logs,
model stats and live visitor conversations. Put the panel on the members' monitor
and open the visitor browser on the front monitor. See [SHOW_DAY.md](SHOW_DAY.md).

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

Base weights are in `models/qwen3-4b`. The server defaults to the frozen release adapter in `models/adapters/beacon-v1.0.0`. The release launcher verifies its checksum before starting. To use another evaluated adapter, set `BEACON_LORA_PATH` to its full adapter folder before starting `scripts/run_server.py`. Training does not automatically replace the serving adapter. Run training and chat separately on the 16 GB GPU.

## Project layout

| Path | Purpose |
| --- | --- |
| `backend/`, `frontend/` | Model chat API and browser interface |
| `scripts/training_ui.py`, `scripts/train_lora.py` | Desktop controls and LoRA trainer |
| `datasets/freshes/` | Original team submissions |
| `datasets/releases/`, `datasets/active.json` | Prepared releases and active selection |
| `datasets/archive/`, `datasets/reviews/` | Preserved originals and review decisions |
| `models/` | Base weights and frozen v1 adapter |
| `outputs/checkpoints/` | Existing trained adapter |
| `outputs/training_runs/` | New UI runs, logs, stats, adapters |
| `outputs/evaluations/` | Audits and benchmark results |
| `archives/` | Preserved fifth-year retrieval project |

## Fifth-year retrieval project

The RAG pipeline has been removed from the active project. Its documents, database/index, OCR weights, source, old frontend, tests and requirements are preserved in [the ZIP](archives/rag_5th_year_2026-10-04.zip). The archive has a SHA-256 sidecar and an internal manifest; all 56 original entries were byte-verified before removal. Extract into a separate project for reuse and read its `RESTORE.md`. Base weights, adapters and training datasets remain here rather than being duplicated into the ZIP.

[PROJECT_REVIEW.md](PROJECT_REVIEW.md) and [TRAINING_REPORT.md](TRAINING_REPORT.md) retain historical findings. Historical losses/token accuracy do not prove factual accuracy or fluent Burmese. Evaluate the reviewed new adapter on held-out IT answers and native-reviewed conversation prompts before the seminar.
