# Frontend team quick start

The current client is plain HTML, CSS and JavaScript. No Node dependencies or build
step are needed. Edit `index.html`, `style.css`, `app.js` and `stream.js` here.

## Run without a GPU, model or training data

From the repository root on Windows (Python 3.11+):

```powershell
python -m venv .venv-frontend
.\.venv-frontend\Scripts\python.exe -m pip install -r requirements-frontend.txt
.\.venv-frontend\Scripts\python.exe -m uvicorn backend.dev_app:app --host 127.0.0.1 --port 8001 --reload
```

On macOS/Linux, use `.venv-frontend/bin/python` instead.
Open **http://127.0.0.1:8001**. This serves the frontend and real chat route with
a clearly labeled synthetic Burmese/English response. It never imports torch or
loads weights, so it can run alongside the trainer. Metrics in this mode are
placeholders. The real model server remains `backend.app:app`; the mock is enabled
only by explicitly launching `backend.dev_app`.

## Connect to the model API

Read [API.md](../API.md) for the request schema and streaming events. The client uses
relative `/api/...` URLs. When using a separate frontend dev server, proxy `/api`
to the backend URL your teammate provides, or configure your client with that URL.
Do not start another real model server on the training machine while training uses
its GPU. The backend has no authentication; keep development access on trusted networks.

Chat uses **Vercel AI SDK UI Message Stream v1 over SSE**. Preserve `stream.js` handling
of partial frames, UTF-8 chunks, errors and completion. The request is this project's
`{message, history, ...}` schema, not the AI SDK's default `messages` request;
adapt the request if adopting an SDK client.

Check Burmese rendering, incremental replies, Stop/cancel, Clear chat, mobile
layout, errors, and history only after a completed response. Display messages as
text or sanitize rendered Markdown; never insert raw message HTML.

## Lightweight checks

```powershell
.\.venv-frontend\Scripts\python.exe -m pytest tests/test_frontend_handoff.py tests/test_variants_streaming.py -q -p no:cacheprovider
node --test tests/test_frontend_stream.cjs tests/test_frontend_app.cjs
```

Node is optional for development and needed only for the JavaScript tests.
The full training test suite also needs training dependencies and local datasets
and weights. Use a feature branch and open a pull request against `main`.
Do not commit secrets, datasets, models, checkpoints, logs or local archives.
`datasets/active.json` describes a local release; it does not download it.
