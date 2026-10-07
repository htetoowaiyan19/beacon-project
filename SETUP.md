# BEACON setup guide

This guide covers the current **BEACON v1.0.0 IT Seminar** project: a local
Qwen3-4B Burmese/English companion with a trained adapter, a streaming web chat,
and a Python desktop panel for members. The active application has no RAG pipeline.

**For the M2 MacBook Air with 8 GB RAM, choose Option A.** Install and rehearse
before show day. The original NVIDIA desktop setup is Option B; frontend members
who only need to develop the interface can use Option C.

Run commands from the **project root**, where `backend/`, `frontend/` and
`scripts/` are located. Replace example folder paths with your actual paths.
Quoted paths work when folder names contain spaces. Commands labeled `sh` run
in macOS/Linux Terminal; commands labeled `powershell` run in Windows PowerShell.

## Contents

- [Choose a setup option](#choose-a-setup-option)
- [Understand the packages and model formats](#understand-the-packages-and-model-formats)
- [Option A: M2 Air, 8 GB RAM](#option-a-m2-air-8-gb-ram)
- [Option B: original NVIDIA desktop](#option-b-original-nvidia-desktop)
- [Option C: frontend development without a model](#option-c-frontend-development-without-a-model)
- [Set up from a Git clone](#set-up-from-a-git-clone)
- [Optional training and evaluation](#optional-training-and-evaluation)
- [Ports, settings and environment variables](#ports-settings-and-environment-variables)
- [Troubleshooting](#troubleshooting)
- [Show-day checklist](#show-day-checklist)
- [Backups and package rebuilding](#backups-and-package-rebuilding)

## Choose a setup option

For the **Windows RTX 4050 laptop with 6 GB VRAM and 16 GB RAM**, use the
separate trained Q4 CUDA build and follow [WINDOWS_GPU.md](WINDOWS_GPU.md).
Its Windows launchers and bundled native DLLs replace the original-model
installation path for that deployment. No training is required.

| Option | Use it for | Model runtime | Files to obtain |
| --- | --- | --- | --- |
| A: M2 Air | The 8 GB seminar laptop | Trained Q4_K_M GGUF through llama.cpp Metal | `BEACON-v1.0.0-m2-air.zip` and its checksum |
| B: NVIDIA desktop | Original trained model, optional training/evaluation | PyTorch/Transformers with base SafeTensors and separate LoRA | `BEACON-v1.0.0-portable.zip` and its checksum |
| C: frontend preview | Interface work, layout and monitoring rehearsals | Synthetic responses; no real model | Git source or full portable ZIP |
| CPU fallback | Diagnostics on a machine without a usable GPU | GGUF CPU on the Mac, or original PyTorch CPU | Appropriate package from A or B |

The CPU fallback is available, but it has no response-time guarantee. The original
unquantized model is unsuitable as the default plan for the 8 GB Air. Use its
quantized laptop package rather than installing the full desktop training stack.

Only choose one serving workflow at a time: the members' panel or the single chat
server. Each loads one model; starting both can use extra memory and conflict on ports.

## Understand the packages and model formats

| Package | Approximate ZIP size | Contents | Exclusions |
| --- | ---: | --- | --- |
| `BEACON-v1.0.0-m2-air.zip` | 2.51 GB | Application, members' panel, merged trained Q4 model, Apple Silicon runtime, setup tools and desktop check reports | Original BF16 weights, datasets, training tools/states, Python environment |
| `BEACON-v1.0.0-portable.zip` | 8.39 GB | Application, full base weights, frozen trained adapter, datasets, provenance, training/evaluation tools | GGUF model, native laptop runtime, optimizer recovery states, Python environment, local logs |
| `BEACON-v1.0.0-model-only.zip` | 8.32 GB | Original base model, frozen adapter and related model files | Complete application and Python environment |

Sizes are decimal GB and can change slightly after documentation updates. Keep the
ZIP and extracted folder in your disk-space calculation. The M2 package needs about
2.51 GB for its files after extraction, plus dependencies and logs; keeping its ZIP
uses approximately another 2.51 GB. The full package needs roughly 9 GB for extracted
files, plus a larger Python environment and space for optional training outputs.

The original base weights are **BF16 SafeTensors**; the trained LoRA is a separate
SafeTensors adapter. Both must be present for original trained-model inference.
An adapter file alone is not the complete model.

The laptop model is **Q4_K_M GGUF**, approximately 2.50 GB for its weights. The
trained adapter is already merged into it; the laptop does not load a separate
LoRA file. Model file size is not total RAM consumption.

The transfer packages do not contain `.venv`. Always install a fresh environment
on the destination machine. Dependency installation requires internet; inference
uses the supplied local weights and can run offline afterward.

## Option A: M2 Air, 8 GB RAM

### A1. Prepare macOS and Python

The bundled native runtime requires **macOS 13.3 or newer** and Apple Silicon.
The setup script requires **Python 3.11 or newer** running as `arm64`, with
Tcl/Tk support if you want the desktop panel. Use a current maintenance release
of your chosen Python version.

In Terminal, check:

```sh
sw_vers -productVersion
uname -m
python3 --version
python3 -c 'import platform, sys; print(sys.executable); print(platform.machine())'
python3 -m tkinter
```

The architecture checks should report `arm64`. The Tkinter command should open
a small test window; close it after checking. A Python process reporting `x86_64`
is using Intel emulation; install/use native Apple Silicon Python and reopen Terminal.

If Python is missing or Tkinter fails, the standard macOS installer from
[Python.org](https://www.python.org/downloads/macos/) provides a universal2 build
and native Tk. Use the normal interpreter for this project. Complete the included
`Install Certificates.command` step if needed for HTTPS package installation.
See the [official Python macOS instructions](https://docs.python.org/3/using/mac.html).
Setup does not install Python, update macOS, or install a compiler.

### A2. Copy, verify and extract the laptop package

Copy these two files from the original project's `builds/` folder:

```text
BEACON-v1.0.0-m2-air.zip
BEACON-v1.0.0-m2-air.zip.sha256
```

Place them together, then open Terminal in that directory. For example:

```sh
cd "$HOME/Downloads"
shasum -a 256 -c BEACON-v1.0.0-m2-air.zip.sha256
```

The result should end with `OK`. If it fails, recopy the archive before proceeding.

Extract into a **fresh folder**, keeping the internal structure. Finder can extract
the ZIP; alternatively, use:

```sh
mkdir -p "$HOME/Desktop/BEACON-M2"
ditto -x -k BEACON-v1.0.0-m2-air.zip "$HOME/Desktop/BEACON-M2"
cd "$HOME/Desktop/BEACON-M2"
python3 scripts/verify_package.py
```

Choose a new destination if that folder already contains another copy. File
verification checks the manifest's sizes and SHA-256 hashes, including the GGUF
and bundled runtime. Run it before changing packaged files.

The laptop model files should be located at:

```text
models/gguf/beacon-v1.0.0-Q4_K_M.gguf
models/gguf/export.json
runtime/macos-arm64.tar.gz
runtime/provenance.json
```

### A3. Install the laptop environment once

From the extracted project root:

```sh
sh setup_laptop.command
```

This verifies the package, extracts llama.cpp into `runtime/llama/`, creates
`.venv`, installs `requirements-laptop.txt`, and verifies the trained model.
The laptop requirements are FastAPI, Uvicorn and optional memory telemetry via
psutil. They do not install PyTorch, Transformers or training libraries.

Wait for the `Ready` message. Then check:

```sh
.venv/bin/python scripts/run_laptop.py --check
.venv/bin/python -m tkinter
```

The first command verifies the GGUF and its frozen adapter identity; it does not
load the model or benchmark Metal. Close the Tk test window. Once installation
and a real chat rehearsal succeed, the laptop no longer needs internet for chat.

### A4. Start the members' panel and visitor page

```sh
sh start_laptop.command
```

In the panel:

1. Keep **Preview mode unchecked** for the real trained AI.
2. Click **Start Backend**. Watch Backend logs and wait for
   `Trained model ready for the visitor frontend.`
3. Click **Start Frontend**, then **Open visitor page**.
4. Open **http://127.0.0.1:8080/** on the visitors' monitor. Keep the desktop
   panel on the laptop screen for members.
5. Send a short Burmese question and a short English question. Check streaming,
   completion and the matching conversation in the members' feed.

The panel shows separate logs, request counts, model state, response timing and
submitted visitor chats as replies stream. It records activity such as typing
status and Stop/Save actions; it does not show unsent draft text or screen captures.
The visitor chat feed is in memory and resets when the backend restarts.

Stop each server with its **Stop** button, or close the panel and wait for shutdown.
This stops the servers and native model process owned by the panel. Run
`sh start_laptop.command` again to restart; retraining is unnecessary.

### A5. Alternative: one chat page without the desktop panel

```sh
sh chat_laptop.command
```

Open **http://127.0.0.1:8000/**. Keep Terminal open and press **Ctrl+C** there
to stop. This mode does not provide the members' live monitoring panel.
The model loads when the first generation starts, so the first reply can take longer.

Direct equivalents, useful for diagnosis:

```sh
.venv/bin/python scripts/run_laptop.py
# Alternatively, one chat server:
.venv/bin/python scripts/run_laptop.py --chat
```

Run either serving command, not both simultaneously.

### A6. Rehearse model loading, language and speed

Stop the live servers before this standalone check so it can own the native model
port and avoid loading a second model:

```sh
.venv/bin/python scripts/smoke_laptop.py
```

It loads the real GGUF, generates English and Burmese answers, checks nonempty
UTF-8 streams and the requested language, and stops its model process afterward.
The report is `outputs/laptop_export/smoke.json`; timings apply to the device
where you ran the command. It is a basic check, not a fluency or factuality grade.

Do not add `--heldout` on the minimal M2 package: that option needs the separate
test dataset, which this package does not include. Existing desktop comparisons
are shipped under `checks/` for review.

### A7. Laptop resource settings and quality limits

| Setting | Current laptop value |
| --- | --- |
| Model | Trained Qwen3-4B, Q4_K_M GGUF |
| Acceleration | Metal, first Apple GPU (`MTL0`) |
| Native context allocation | 2,048 tokens; one generation slot |
| Input budget | 1,536 tokens, including instructions and retained history |
| Retained history | At most two recent complete exchanges |
| Maximum reply | 192 tokens, even if a client asks for more |
| Thinking and base comparison | Disabled |
| CPU / batch threads | 4 / 4 |
| Batch / microbatch | 128 / 128 |
| Additional RAM cache | Disabled; ordinary prompt reuse remains enabled |

Older messages remain visible/exportable in the browser. The model receives only
the retained context. Oversized current messages are rejected with a request to
shorten them; they are not silently cut. If a reply hits its limit, ask a focused
follow-up. These settings are in `scripts/utils/llama_runtime.py` and
`backend/services/gguf_model_service.py`; there is no general performance settings file.

The panel reports native process resident RAM and available system RAM. On Apple
Silicon these are not dedicated VRAM measurements, GPU utilization, or wattage;
resident RAM does not include every Metal allocation. Check Activity Monitor's
Memory Pressure and close unnecessary applications/tabs before your rehearsal.

Desktop streaming and lifecycle checks passed, but **M2 speed, total memory use
and sustained performance have not been measured**. Test multiple successive chats
on the Air. Some Burmese phrasing became weaker after quantization in the small
desktop comparison. Review the answers in `checks/` and test your seminar's IT
questions; successful installation does not establish answer quality.

## Option B: original NVIDIA desktop

### B1. Prepare Windows, Python and the GPU

The verified desktop environment used **Python 3.14**, an **RTX 5060 Ti with
16 GB VRAM**, and **32 GB system RAM**. Setup accepts Python 3.11 or newer;
other versions/devices have not received the same complete checks.

Install Python with pip, venv and Tcl/Tk support. The Python executable must be
available as `python` for the batch launchers. Check in PowerShell:

```powershell
python --version
python -m tkinter
nvidia-smi
```

Close the Tk test window. Install a compatible NVIDIA driver before GPU setup.
The script installs the tested CUDA 12.8 PyTorch wheel; it does not install the
driver. The desktop model may offload to CPU on smaller GPUs, increasing latency.

### B2. Verify and extract the full portable package

Copy `BEACON-v1.0.0-portable.zip` and its `.zip.sha256` file together. From their
directory in PowerShell:

```powershell
Get-FileHash .\BEACON-v1.0.0-portable.zip -Algorithm SHA256
Get-Content .\BEACON-v1.0.0-portable.zip.sha256
```

Compare the two hash values; they must match. Extract the ZIP with a ZIP64-capable
extractor into a fresh folder, such as `C:\BEACON`. Then:

```powershell
Set-Location 'C:\BEACON'
python scripts/verify_package.py
python scripts/setup_device.py
```

Setup creates `.venv`, installs CUDA PyTorch and `requirements-inference.txt`,
checks the original trained release files, and reports CUDA availability. It
needs internet for packages, while the supplied base/adapter need no download.

Confirm the result:

```powershell
.\.venv\Scripts\python.exe scripts/run_release.py --check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

For intended NVIDIA inference, the CUDA check should print `True`. If it prints
`False`, resolve the driver/environment issue before presenting GPU inference.

### B3. Start and stop the desktop application

For the monitored two-screen demonstration, double-click `show_day_ui.bat`, or:

```powershell
.\.venv\Scripts\python.exe scripts/show_day_ui.py
```

Use the same Backend-then-Frontend sequence from A4. The visitor URL is
**http://127.0.0.1:8080/**. Closing the panel stops its owned servers.

For a single chat page, double-click `run_trained_chat.bat`, or:

```powershell
.\.venv\Scripts\python.exe scripts/run_release.py
```

Open **http://127.0.0.1:8000/**. Keep the server window open; **Ctrl+C** stops it.
The release launcher always selects and verifies the frozen trained adapter.
It does not retrain or replace weights.

For an alternate single-server port:

```powershell
.\.venv\Scripts\python.exe scripts/run_release.py --host 127.0.0.1 --port 8002
```

Open `http://127.0.0.1:8002/` in this example. The original runtime retains at
most three recent exchanges within its 2,048-token input budget. Its default
reply budget is 512 tokens; this differs from the laptop cap.

### B4. CPU-only installation and Linux

For a CPU-only original-model environment:

```powershell
python scripts/setup_device.py --cpu
```

It still requires the full base model, separate adapter and sufficient system
RAM. This is a fallback, not the M2 Air's recommended configuration.

On Linux, use an appropriate Python installation with Tkinter for desktop panels,
extract the full portable package and run:

```sh
python3 scripts/verify_package.py
python3 scripts/setup_device.py
.venv/bin/python scripts/run_release.py
```

Use `python3 scripts/setup_device.py --cpu` if you want the CPU wheel.
For the original Transformers runtime on macOS, `--cpu` is required; that path
does not use Metal. For Apple GPU inference, use Option A.

## Option C: frontend development without a model

The frontend is plain HTML, CSS and JavaScript: no npm install or build step is
needed to run it. This mode serves clearly labeled synthetic responses and never
loads model weights. Its statistics are placeholders.

Use a Git clone or the full portable package, which includes the preview tools.
Create a separate environment to keep it independent of training/inference:

**Windows:**

```powershell
python -m venv .venv-frontend
.\.venv-frontend\Scripts\python.exe -m pip install -r requirements-frontend.txt
.\.venv-frontend\Scripts\python.exe -m uvicorn backend.dev_app:app --host 127.0.0.1 --port 8001 --reload
```

**macOS/Linux:**

```sh
python3 -m venv .venv-frontend
.venv-frontend/bin/python -m pip install -r requirements-frontend.txt
.venv-frontend/bin/python -m uvicorn backend.dev_app:app --host 127.0.0.1 --port 8001 --reload
```

Visit **http://127.0.0.1:8001/**. Press **Ctrl+C** in the terminal to stop.
Edit the files in `frontend/`; reload the page to see changes.

To rehearse the members' panel, stop that preview server first, then use:

```powershell
.\.venv-frontend\Scripts\python.exe scripts/show_day_ui.py --preview --start
```

On macOS/Linux, substitute `.venv-frontend/bin/python`. This needs Tkinter.
The panel starts preview servers on 8000/8080 and marks their replies synthetic.
On an installed M2 package, you can instead open the normal laptop panel and
check Preview before starting Backend; no extra preview environment is necessary.

For a real backend, the client uses relative `/api/...` URLs. The show-day visitor
proxy already routes these to the selected backend. If introducing a different
frontend server, provide equivalent routing. Preserve the **Vercel AI SDK UI
Message Stream v1 over SSE** contract and the project's request schema from
[API.md](API.md). More team instructions are in `frontend/README.md` in the full
source checkout.

## Set up from a Git clone

```sh
git clone https://github.com/htetoowaiyan19/beacon-project.git
cd beacon-project
```

Git contains code and documentation. It deliberately excludes model weights,
native binaries, local archives, dataset contents, `.venv`, logs and training
outputs. A clone alone can run Option C, but cannot serve the trained model.

For original-model inference, copy these complete directories from the full
portable package or the model-only package into the checkout:

```text
models/qwen3-4b/
models/adapters/beacon-v1.0.0/
```

Keep their tokenizer/config files alongside the weights. Then follow B1 and B2's
`setup_device.py` command. A Git checkout does not have `MANIFEST.json`, so use
`scripts/run_release.py --check` for its serving-artifact check; run the full
package verifier on the extracted package before copying its files.

For M2 inference from a checkout, copy the complete `models/gguf/` and `runtime/`
directories from the verified M2 package, then run `sh setup_laptop.command`.
Using the complete M2 package directly is the simpler show-day workflow.

For training/evaluation, also copy the complete `datasets/` tree from the full
portable package. The tracked `datasets/active.json` describes a release; it does
not download it. The M2 package contains neither the raw datasets nor the original
training weights.

## Optional training and evaluation

The seminar model is already trained. These steps are for a new experiment or
evaluation on the capable desktop, not prerequisites for chat on the Air.

### Install extra dependencies

From the full project/portable copy on the NVIDIA desktop:

```powershell
python scripts/setup_device.py --training --evaluation
```

In an already configured CUDA environment, you can add the extras directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-evaluation.txt
```

Training uses ordinary BF16/FP16 LoRA through the current training script. There
is no MLX training workflow for the 8 GB Air in this project.

### Start, stop safely and resume

Stop real chat/model servers first to free their memory. Launch `train_ui.bat`, or:

```powershell
.\.venv\Scripts\python.exe scripts/training_ui.py
```

The selected dataset is `it_seminar_v3`: **12,759 training**, **749 validation**
and **749 held-out test** conversations. Check `datasets/active.json` and review
the content before starting a new run. Starting training creates a separate run
and does not replace the frozen serving adapter.

For an initial training smoke check, select the release's `smoke/train.jsonl` and
`smoke/validation.jsonl`, and set Max steps to 10. For full settings, see
[TRAINING_UI.md](TRAINING_UI.md) in the full source/package.

For a planned shutdown, click **Stop safely** and wait for `stopped` and process
exit. Continue with **Resume run**, selecting the original run folder containing
`config.json`. A power cut can recover only from a complete saved checkpoint;
unsaved steps are lost. Default checkpoints save every 25 optimizer steps and
validation runs every 250. Do not alter dataset contents/settings midway through
a resumable run.

The full portable ZIP excludes optimizer, scheduler and RNG recovery states.
To resume on another capable device, separately transfer the original
`outputs/training_runs/<run>/` folder with its complete checkpoint state, and
check saved paths. The inference adapter or GGUF alone cannot resume training.

### Evaluate and serve a candidate

On the NVIDIA desktop with evaluation dependencies and all frozen splits:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_heldout_sample.py --adapter models/adapters/beacon-v1.0.0
```

This runs a stratified sample comparison against the base model and writes results
under `outputs/evaluations/`; it is not a run over every held-out record. Stop the
chat server beforehand. Keep validation for choosing settings and preserve the
held-out split. Review language, tone and factual correctness in addition to loss
and reference-similarity metrics.

To serve a separately reviewed original-format adapter:

```powershell
$env:BEACON_LORA_PATH = 'C:\BEACON\outputs\training_runs\YOUR-RUN\adapter'
.\.venv\Scripts\python.exe scripts/run_server.py --host 127.0.0.1
```

Use a real evaluated adapter directory. This generic server honors that override;
`run_release.py` deliberately selects the frozen seminar adapter instead. Restart
the server after changing the override. Remove it when done:

```powershell
Remove-Item Env:BEACON_LORA_PATH -ErrorAction SilentlyContinue
```

Changing a SafeTensors adapter does not update an existing merged GGUF. A new
laptop model requires a separate merge/export and quality check.

## Ports, settings and environment variables

| Service | Default URL/port | Purpose |
| --- | --- | --- |
| Real backend or single chat | `127.0.0.1:8000` | Model API; single mode also serves chat |
| Show-day visitor frontend | `127.0.0.1:8080` | Visitor assets and public API proxy |
| GGUF native model service | `127.0.0.1:8090` | Internal llama.cpp server; not the visitor page |
| Frontend developer example | `127.0.0.1:8001` | Synthetic development preview |

Backend diagnostics are **`/api/health`**, not `/health`. API docs are at backend
`/docs`; the visitor proxy does not expose documentation or operator endpoints.
An `online` health response means the backend is running; check `gpu.is_loaded`
to see whether weights are loaded. Single-chat mode loads lazily; real panel mode
preloads before declaring readiness.

All examples in this guide use localhost, and the show-day panel is designed for
two screens on the same computer. A browser on another computer cannot reach the
laptop through its own `127.0.0.1`. Network visitor hosting needs a separate routing,
binding and access-control plan; the current public chat endpoints have no login.

| Variable | Meaning |
| --- | --- |
| `BEACON_RUNTIME=gguf` | Select the laptop service; set automatically by the laptop launcher |
| `BEACON_LLAMA_PORT` | Native llama.cpp port, default 8090 |
| `BEACON_GGUF_GPU_LAYERS=0` | CPU diagnostic fallback; default is all supported layers |
| `BEACON_LLAMA_SERVER_PATH` | Explicit native executable path; normal setup finds the bundled binary |
| `BEACON_GGUF_METADATA` | GGUF metadata filename within `models/gguf/`; default `export.json` |
| `BEACON_LORA_PATH` | Separate adapter for the generic original-model server |

Set variables **before launching** the process. The scripts read environment
variables directly; they do not automatically load a `.env` file. The panel
creates its own random operator token; no manual token setup is needed.

To change the native port on the Mac, for example:

```sh
BEACON_LLAMA_PORT=8091 .venv/bin/python scripts/run_laptop.py
```

Choose backend/frontend ports in the panel. Stop Frontend before changing its
backend port. Keep all three ports distinct. For a CPU diagnostic chat on the Mac:

```sh
.venv/bin/python scripts/run_laptop.py --cpu --chat
```

## Troubleshooting

### `python`/`python3` not found, wrong version or wrong architecture

Install the appropriate Python interpreter and reopen the terminal. Check the
version and executable path. On the Mac, confirm Python reports `arm64`. On
Windows, if using `py` to choose a version, use that interpreter to run setup;
later commands can use `.venv\Scripts\python.exe` directly.

### Tkinter is missing or the panel cannot open

Run `python3 -m tkinter` on the Mac or `python -m tkinter` on Windows. Use a
Python installation with Tcl/Tk, then recreate the environment in a fresh copy
if the interpreter changed. Tkinter is an interpreter/OS component, not a package
fixed by `pip install tkinter`. Single chat can run without the desktop panel.

### Missing model, missing adapter or checksum mismatch

Confirm you obtained the correct package, preserved its folder structure and ran
commands from its root. Recopy damaged files from the original verified package.
The original runtime needs both base and adapter; the laptop needs GGUF plus its
matching `export.json`. Do not change expected hashes simply to bypass verification.

Full-package checks report intentional edits as differences. Verify a fresh copy
before editing; use that original copy to distinguish transfer corruption from
your later changes.

### `llama-server` is missing

On the Mac, run `sh setup_laptop.command` from the M2 package and check that it
finishes. The executable should exist under `runtime/llama/`. A source-only clone
or the full original-model ZIP does not supply the bundled native archive.

### macOS blocks the native executable

Read the displayed security message and check `runtime/provenance.json`, the
archive checksum and the supplied llama.cpp license. If you trust this verified
copy, follow Apple's supported approval process in System Settings > Privacy &
Security. Do not disable Gatekeeper globally. See
[Apple's instructions for opening trusted apps](https://support.apple.com/en-us/102445).

### Package installation fails

Check internet access, disk space and the exact error in Terminal. For a Python.org
macOS installation reporting certificate errors, complete its
`Install Certificates.command` step. Retry the setup command after addressing
the error. A preinstalled environment and real rehearsal are needed for an
offline show; having only the ZIP is insufficient.

### Port already in use

Stop the old server in its original terminal with Ctrl+C, or close the panel
that owns it. The panel intentionally refuses to take over another process's
port. Select another unused port if appropriate; change 8090 using the native
port variable when needed. Do not kill unrelated processes to free a port.

### Browser cannot connect or health is online but no reply appears

Use port 8080 for the monitored visitor page or 8000 for default single chat.
Check both server logs. Frontend alone cannot generate answers. In panel mode,
wait for trained-model readiness; in single mode, the first request loads weights.
Preview mode must be unchecked to show the real model.

### Responses become slow or the laptop swaps

Check Activity Monitor Memory Pressure, close unused applications, use one active
visitor chat, and check whether another model server is running. Keep thinking
disabled and the bounded laptop context. Review first-text time and queue time
separately: a request waiting behind another generation is not model compute time.

On the desktop, check CUDA availability and any CPU-offload label. On the Mac,
Metal startup errors are in Backend logs; the normal runtime requests `MTL0`
explicitly. CPU fallback is useful to diagnose loading, but may be slower.

### Reply stops early or forgets an older message

The laptop caps output at 192 tokens and retains at most two recent exchanges.
Ask a concise follow-up or repeat important older context. Starting a new chat
clears browser conversation context, not the trained weights. Neither chat
history nor the members' feed updates the model through training.

### Windows activation is blocked by PowerShell

No activation command is required for this guide. Use the full environment
executable, such as `.\.venv\Scripts\python.exe`, as shown above.

## Show-day checklist

1. Verify and install the chosen package on the actual show device.
2. Rehearse real Burmese/English conversations and your main IT questions.
3. Review quantized-model comparison answers; do not present installation checks
   as proof of fluent or technically correct output.
4. Test several consecutive exchanges, Stop/cancel, New conversation, Copy and
   Save chat. Confirm the members' feed and memory statistics update.
5. Test with internet disconnected after dependency setup.
6. Connect the visitor monitor and arrange windows. Keep the panel visible to
   members and the chat visible to visitors.
7. Stop previous server sessions, start one real Backend, wait for readiness,
   then start Frontend. Keep the laptop awake during the demonstration.
8. At the end, save any chats you want from the visitor UI and close the panel.
   Its in-memory feed is not a persistent chat archive.

## Backups and package rebuilding

Keep the full portable ZIP for original weights, datasets and preparation
provenance. Keep the M2 ZIP for the laptop demonstration. Preserve the original
training run separately if you need resume states. The archived fifth-year RAG
project is separate and unnecessary for the current application.

From the original project machine, rebuild after source or documentation changes:

```powershell
.\.venv\Scripts\python.exe scripts/package_laptop.py
.\.venv\Scripts\python.exe scripts/package_release.py
```

The first command requires the existing trained GGUF and native runtime; it
packages them without retraining or conversion. The second requires the original
base/adapter and complete datasets. Both write ZIPs and checksum sidecars in
`builds/` and verify every archived entry. These packagers are part of the full
source checkout, not the minimal M2 distribution.

If editing source on the Mac, do not expect an old ZIP manifest to match it.
Keep the verified original archive, record your changes, and generate a new
verified package from the complete source/artifacts when distributing updates.

Related guides in the full project: [M2 Air notes](M2_AIR.md),
[original transfer guide](TRANSFER.md), [show-day panel](SHOW_DAY.md),
[API contract](API.md), [release results](RELEASE_V1.md), and
[training controls](TRAINING_UI.md). The minimal M2 package includes only the
guides relevant to its runtime plus this combined setup guide.
