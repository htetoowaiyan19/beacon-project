# BEACON show-day control panel

Double-click **show_day_ui.bat**. The Python desktop dashboard belongs on the
members' monitor; the browser belongs on the visitors' monitor.

## Start the demonstration

1. Stop any previous chat server first (Ctrl+C in its terminal). The panel
   refuses to take over occupied ports or stop servers it does not own.
2. Click **Start Backend**. Leave Preview mode unchecked to use the frozen
   trained adapter. Startup verifies the adapter checksum and loads the weights
   before accepting visitors. Wait for **Trained · loaded** in the Adapter card.
   See the Backend logs for loading progress or errors.
3. Click **Start Frontend**, then **Open visitor page**. Its URL is
   **http://127.0.0.1:8080/**. Move that browser window to the front monitor;
   use the browser's fullscreen mode if desired.
4. Keep the desktop panel on the members' monitor. Select a visitor's question
   to view that conversation, or keep **Follow latest visitor** checked.
5. Click each server's **Stop** button when finished. Closing the panel also
   stops both servers it started. Starting a server again does not retrain.

The backend uses localhost port 8000, the frontend uses localhost port 8080.
Choose different unused ports in the panel if necessary. Stop the frontend
before changing the backend port. Both monitors use the same computer; this
launcher does not expose the application on a public network.

## What members see

- Separate server controls, process status and UTF-8 logs for backend/frontend.
- Model device, trained adapter state, allocated/reserved model VRAM, and
  optional NVIDIA driver utilization, temperature and whole-GPU memory.
- Total, active, completed, failed and cancelled requests; token count,
  generation speed and duration of the most recent reply.
- Submitted visitor questions and assistant text as it streams, grouped by
  conversation. Partial/cancelled/failed replies remain clearly labeled.
- Visitor activity: page connected, input focused, typing, new conversation,
  Stop, Copy and Save. Draft text, keystrokes and screen recordings are not collected.

The chat feed is retained **in memory only**, with the latest 40 requests and
250 activity events. It resets when the backend restarts. A visitor has an
anonymous browser ID, not a verified identity. Online means a heartbeat was seen
recently, so a closed tab can take about 35 seconds to disappear. Logs are saved
under `outputs/show_day/` and contain server output, not a chat transcript.

Visitors see a monitoring notice. The operator feed requires a random token
held by the desktop panel. It is not placed in frontend assets. The visitor
server proxies only public chat, health, config and activity endpoints; it blocks
operator routes, backend documentation and arbitrary files.

## Rehearse without a GPU

Check **Preview mode** before starting the backend, or run:

```powershell
.\.venv\Scripts\python.exe scripts/show_day_ui.py --preview --start
```

Preview responses and stats are visibly labeled synthetic. The panel, proxy and
mock backend do not import torch or load the model. They use Python's Tkinter and
the existing frontend dependencies; install `requirements-frontend.txt` for a
preview-only environment. Real chat uses `requirements-inference.txt`, the base
model and the frozen adapter described in `RELEASE_V1.md`.

Manual panel launch:

```powershell
.\.venv\Scripts\python.exe scripts/show_day_ui.py
```

The backend's Vercel UI Message Stream contract is unchanged. `client_id` and
`session_id` are optional request fields used to group the seminar feed. Existing
chat clients continue to work; monitoring is disabled in normal server launches.

Verification: **19 Python tests and 10 JavaScript tests passed**, including
incremental proxy streaming, disconnect cancellation, operator authentication,
port ownership, backend restart, preview isolation and desktop feed rendering.
The real Qwen3-4B + frozen LoRA also passed a visitor-proxy/operator-feed smoke
test. The local report is `outputs/show_day/real-model-smoke.json`.
