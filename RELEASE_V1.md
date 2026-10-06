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

Release checks passed: **11 Python tests and 7 JavaScript tests**, eight live HTTP
endpoints, and real English/Burmese trained-adapter streams. The English smoke
answer followed English after adding the request-specific language instruction.
Dataset split checksums match the completed training release. Browser visual
review could not be performed because browser automation was unavailable.
Local smoke report: `outputs/release_checks/v1-smoke.json`.

## Share or restore

`archives/BEACON-v1.0.0-seminar.zip` contains the application and adapter, plus a
manifest with per-file checksums. Its adjacent `.sha256` file verifies the ZIP.
It excludes datasets, training states, logs, virtual environments and base weights.
The RAG archive remains separate for the fifth-year project.

On another machine:

1. Extract the ZIP into its own folder. Use Python 3.11 or newer (the current
   seminar environment uses Python 3.14). Create `.venv` with `python -m venv .venv`.
2. Install CUDA-enabled PyTorch appropriate for the GPU, then install
   `requirements-inference.txt` in that environment. The listed versions are the
   versions validated on the seminar machine. The model has been run on a 16 GB GPU.
3. Copy the existing **models/qwen3-4b** folder into the extracted project. If
   unavailable, `python scripts/download_model.py` downloads Qwen/Qwen3-4B;
   this requires internet access and sufficient disk space and memory.
4. Run `scripts/run_release.py --check`, then double-click `run_trained_chat.bat`.

To rebuild the ZIP after a source change:

```powershell
.\.venv\Scripts\python.exe scripts/package_release.py
```

The source repository intentionally excludes model artifacts. Frontend members
can use the existing mock preview without downloading weights; see
`frontend/README.md` in the repository. The mock is labeled and is not this model.
