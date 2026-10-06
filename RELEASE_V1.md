# BEACON v1.0.0 — IT Seminar

The first seminar release uses Qwen3-4B with the completed Burmese LoRA adapter.
It supports Burmese and English, IT explanations, and conversation with a tone
that follows the user. It is a model-only application with no RAG dependency.

## Start the live chat

Double-click **run_trained_chat.bat**, keep its server window open, then visit
**http://127.0.0.1:8000/**. API documentation is at **http://127.0.0.1:8000/docs**.
The first reply loads the model; later replies stream immediately as generated.
Close the server with Ctrl+C. Chatting does not train or change the adapter.

```powershell
.\.venv\Scripts\python.exe scripts/run_release.py --check
.\.venv\Scripts\python.exe scripts/run_release.py
```

The launcher verifies the adapter SHA-256 and selects the frozen snapshot in
`models/adapters/beacon-v1.0.0`. It defaults to localhost. Other evaluated adapters
can still be served with `BEACON_LORA_PATH` and `scripts/run_server.py`.

## What is included

- Responsive chat page with streaming, Stop, New conversation, Copy and Save chat.
- Visible model status, example prompts, generation settings and token/speed stats.
- Vercel AI SDK UI Message Stream v1 and the existing API 3.0.0 contract.
- Runtime instructions for matching English input, concise answers, uncertainty and avoiding invented
  personal experiences. These instructions do not guarantee factual accuracy.
- Frozen best adapter from checkpoint 3000 of the completed 3190-step run.

Validation loss at the selected checkpoint was **0.373622**. A seeded 32-example
held-out comparison produced mean sentence chrF **35.41** for the trained adapter
and **22.32** for the base model; the trained adapter scored higher on **24/32**.
There were no empty replies; one trained reply hit the output limit, versus 20
base replies. Those numbers measure similarity to references, not correctness
or native fluency. IT inaccuracies and unnatural Burmese remain. These results
used the original evaluation persona; the additional v1 runtime instructions
have only received a live smoke test, not the full held-out comparison.

Detailed local results: `outputs/evaluations/heldout_20261006_144656/REVIEW.md`.
This release is for the seminar; check important technical answers before using
them in real work. Test split contents and training data have not been changed
as part of this release.

Initial v1 release checks passed: **11 Python tests and 7 JavaScript tests**, eight live HTTP
endpoints, and real English/Burmese trained-adapter streams. The English smoke
answer followed English after adding the request-specific language instruction.
Dataset split checksums match the completed training release. Browser visual
review could not be performed because browser automation was unavailable.
Local smoke report: `outputs/release_checks/v1-smoke.json`.

## Share or restore

For the seminar's two-monitor setup, double-click **show_day_ui.bat**.
The desktop panel controls the model backend and visitor frontend separately,
and shows logs, model statistics and live conversations. See **SHOW_DAY.md**.

The chat now uses recent context within a 2,048-token input budget and streams
Burmese without waiting for spaces. Older exchanges remain visible in the page
but are omitted from the model's context when necessary. The panel shows actual
first-text waiting time and queue time; no retraining is required for these fixes.

`archives/BEACON-v1.0.0-portable.zip` contains the application, complete base model,
trained adapter, datasets and review provenance. It supersedes the older
adapter-only seminar ZIP. Its `MANIFEST.json` records every file's SHA-256 and the
adjacent `.zip.sha256` verifies the ZIP itself. It excludes training optimizer
states, logs and virtual environments. The RAG archive remains separate for the
fifth-year project.

Follow [TRANSFER.md](TRANSFER.md) on another device: extract the ZIP, run
`python scripts/verify_package.py`, then `python scripts/setup_device.py` to create
a new environment. Setup requires internet for dependencies; weights are included.
Then double-click `show_day_ui.bat` or `run_trained_chat.bat`.

To rebuild the ZIP after a source change:

```powershell
.\.venv\Scripts\python.exe scripts/package_release.py
```

The source repository intentionally excludes model artifacts. Frontend members
can use the existing mock preview without downloading weights; see
`frontend/README.md` in the repository. The mock is labeled and is not this model.
