# BEACON Windows CUDA setup: RTX 4050, 6 GB VRAM

Use **`builds/BEACON-v1.0.0-windows-cuda.zip`** for the Windows laptop with an
RTX 4050 and 16 GB system RAM. It includes your trained Burmese adapter merged
into Qwen3-4B, quantized to **Q4_K_M GGUF**, the latest landing/chat pages, the
members' panel, and pinned Windows native CUDA/CPU runtimes.

The GGUF weights are approximately **2.50 GB**. Context, native buffers, the
display and other applications also consume memory. Final speed and memory use
must be checked on the actual laptop. Normal chat does not train or change weights.

## What has been verified

On the source Windows desktop with an **RTX 5060 Ti**, eight consecutive real
CUDA turns passed stream/language/limit checks. The cold run took about 29 seconds
to load and another 25 seconds to produce its first text; subsequent first-text
times were 0.05–0.11 seconds. Allow time for startup and warm-up before visitors
arrive. These are desktop measurements, not a promised RTX 4050 speed.

The real visitor proxy also passed English/Burmese streaming, current frontend
asset delivery, members' chat feed, cancellation, private-route protection and
shutdown checks. Desktop evidence is included in `checks/windows-smoke.json`
and `checks/windows-http-check.json`. Rehearse on the laptop using section 8.
The separate CPU runtime also generated two real test replies; its report is
`checks/windows-smoke-cpu.json`. A fresh extracted installation successfully
created the small Python environment without training libraries.

## 1. Prepare the laptop

1. Use **Windows 10/11 x64**, with your RTX 4050 enabled.
2. Install **64-bit Python 3.11 or newer**, including pip, venv and Tcl/Tk.
   Python 3.14 is the version used for desktop verification. Use a current
   maintenance release and ensure `python` works in a new terminal.
   [Official Python Windows instructions](https://docs.python.org/3/using/windows.html).
3. Install a current compatible NVIDIA laptop driver from NVIDIA or your laptop
   manufacturer's supported driver channel. This package uses the official
   **llama.cpp b11429 CUDA 12.4 build** and its paired DLLs. Setup checks device
   enumeration; it does not install or change the GPU driver.
4. If the native executable reports missing Microsoft runtime DLLs, install the
   [Microsoft Visual C++ v14 x64 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
5. Plug in the laptop for rehearsal and keep it awake during the demonstration.

Open **PowerShell** and run:

```powershell
python --version
python -c "import platform, struct; print(platform.machine()); print(struct.calcsize('P') * 8)"
python -m tkinter
nvidia-smi
```

The architecture should be `AMD64`/`x86_64`, and the pointer size should be `64`.
Tkinter should open a test window; close it. `nvidia-smi` should list the RTX 4050
and about 6 GB total dedicated GPU memory. If `python` is missing, reopen the
terminal after installation or run the setup script with your configured `py`
interpreter. Subsequent launchers use the project's own `.venv`.

Initial setup needs internet for the small Python dependencies. The model,
llama.cpp executables and paired CUDA DLLs are supplied, so no model download or
CUDA Toolkit/compiler installation is required. Inference works offline after
setup and a successful rehearsal. Allow at least **12 GB free disk space** for
the outer ZIP, extracted files, native runtime extraction, environment and logs.

## 2. Copy and verify the build

Copy these two files from the original machine's `builds/` folder:

```text
BEACON-v1.0.0-windows-cuda.zip
BEACON-v1.0.0-windows-cuda.zip.sha256
```

Place them together, for example in Downloads. Open PowerShell in that folder:

```powershell
Get-FileHash .\BEACON-v1.0.0-windows-cuda.zip -Algorithm SHA256
Get-Content .\BEACON-v1.0.0-windows-cuda.zip.sha256
```

The hash values must match; letter case does not matter. Recopy the files if they
do not match. Keep the sidecar with the ZIP when making another backup.

## 3. Extract into a fresh folder

Right-click the ZIP in File Explorer, choose **Extract All**, and extract into a
new folder such as:

```text
C:\Users\YOUR-NAME\Desktop\BEACON-RTX4050
```

Keep the directory structure. Avoid mixing files with an old BEACON installation
or transferring the original desktop's `.venv`. The folder containing
`setup_windows_gpu.bat`, `backend/`, `frontend/`, `models/` and `scripts/` is the
project root.

Open PowerShell there, for example:

```powershell
Set-Location "$env:USERPROFILE\Desktop\BEACON-RTX4050"
python scripts/verify_package.py
```

The verifier checks every packaged file against `MANIFEST.json`, including
weights, native ZIPs, fonts and scripts. Run it before editing packaged files.
It requires no installed model libraries or GPU. The outer ZIP checksum also
protects the manifest itself.

## 4. Install once

Double-click **`setup_windows_gpu.bat`**, or run:

```powershell
python scripts/setup_windows_gpu.py
```

Setup performs these actions:

1. Verifies the package, trained GGUF and native archive checksums.
2. Extracts the CUDA executable and paired CUDA DLLs into `runtime/llama-cuda/`.
3. Extracts a separate CPU runtime into `runtime/llama-cpu/`.
4. Runs the native device check and requires **`CUDA0`** for normal GPU setup.
5. Creates `.venv` and installs `requirements-laptop.txt`.
6. Checks the trained model again, then prints **Ready**.

This inference environment uses FastAPI, Uvicorn and psutil; it needs no PyTorch,
Transformers, PEFT, training data or separate LoRA adapter. Native archive
identity is recorded in `runtime/windows-provenance.json`.

Keep the setup window open until completion. If it fails, fix the reported issue
and retry. Installing the Microsoft or NVIDIA system prerequisites may require
their normal administrator approval; the project setup does not install them.

## 5. Confirm model loading before opening the visitor screen

With all live BEACON servers stopped:

```powershell
.\.venv\Scripts\python.exe scripts/run_laptop.py --check
.\.venv\Scripts\python.exe scripts/run_laptop.py --check-runtime
```

The first verifies model bytes without loading weights. The second loads the
native model once, prints status, then stops its process. Look for:

```text
"device": "NVIDIA GPU (CUDA)"
"is_loaded": true
"adapter_merged": true
"execution_devices": ["CUDA0"]
```

Native loading logs should show GPU layer offloading. The launcher explicitly
requests `CUDA0`, so a missing CUDA backend/device causes an error rather than
quietly presenting normal GPU mode as CPU inference. An explicit `--cpu` launch
uses the separate CPU runtime and is labeled CPU.

## 6. Start the monitored show-day setup

Double-click **`start_windows_gpu.bat`**, or run:

```powershell
.\.venv\Scripts\python.exe scripts/run_laptop.py
```

In the members' panel:

1. Leave **Preview mode unchecked** for the trained model.
2. Click **Start Backend**. Watch the Backend log until it says
   **Trained model ready for the visitor frontend**.
3. Click **Start Frontend**.
4. Click **Open visitor page**, or visit **http://127.0.0.1:8080/**.
5. The landing page opens first; click **Start chatting** to enter the live chat.
   Direct chat URL: **http://127.0.0.1:8080/chat.html**.
6. Move that browser window to the visitors' monitor. Keep the members' panel
   on the laptop screen.
7. Send Burmese and English questions and check the corresponding feed entries.

The panel displays server logs, model state, first-text/queue times, request
counts, native process RAM and available system RAM. Its NVIDIA driver card
shows whole-GPU utilization, memory and temperature when `nvidia-smi` is available.
These are different measurements: process resident RAM is not VRAM; whole-GPU
memory includes the display and other programs. Unsupported measurements remain
unavailable rather than being invented.

Submitted visitor chats and partial responses appear in the members' feed. That
feed is in memory, not a persistent chat database. Draft text and screen recordings
are not collected. Visitors see the monitoring notice.

### Stop and restart

Click **Stop** for each server, or close the members' panel and wait for its
shutdown to finish. It stops the web servers and native model process it owns.
Double-click `start_windows_gpu.bat` again to restart. No retraining is necessary.

## 7. Alternative: a single chat server

Double-click **`chat_windows_gpu.bat`**, or:

```powershell
.\.venv\Scripts\python.exe scripts/run_laptop.py --chat
```

Open **http://127.0.0.1:8000/** for the landing page, or
**http://127.0.0.1:8000/chat.html** for chat. Keep the terminal open. The first
generation loads weights, so its first reply can include startup time. Press
**Ctrl+C** in that terminal to stop the server and native model.

Choose either this mode or the members' panel; running both duplicates model
work and conflicts on their default ports. API docs are on backend `/docs` and
health is `/api/health`. The visitor proxy does not expose protected operator
routes or backend docs. These launchers serve localhost for two screens on one
computer; they do not configure access from another device or a public website.

## 8. Rehearse consecutive conversations

Stop the live servers first, then run:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_windows_gpu.py
```

It loads the real CUDA GGUF, runs eight consecutive English/Burmese turns, checks
UTF-8/language/nonempty streams and generation limits, records timings and driver
memory samples, and stops its model process. Results are saved to:

```text
outputs/windows_gpu_checks/smoke.json
```

For a longer automatic rehearsal:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_windows_gpu.py --turns 16
```

These checks are not a factuality or fluency score. Read the generated answers,
then rehearse your actual seminar prompts, Stop/cancel, new conversation and
monitoring. Test offline after setup. Timing and memory measurements from the
original desktop do not establish performance on the RTX 4050 laptop.

The ZIP includes existing quantization comparisons under `checks/`. Some Burmese
phrasing became weaker in Q4; IT errors also remain in the original model. The
CUDA edition uses the same trained Q4 weights as the M2 edition and needs the
same language/content review.

## 9. Settings for the 6 GB GPU

| Setting | Windows CUDA edition |
| --- | --- |
| Model | Trained Q4_K_M GGUF, approximately 2.50 GB |
| GPU | First CUDA device, `CUDA0`; all supported model layers |
| Native context allocation | 2,048 tokens, one slot |
| Input budget | 1,536 tokens, including instructions/history |
| Retained history | At most two recent complete exchanges |
| Maximum reply | 192 tokens |
| Thinking/base comparison | Disabled; trained adapter is merged |
| CPU/batch threads | 4/4 |
| Batch/microbatch | 128/128 |
| Extra RAM prompt cache | Disabled; ordinary prompt reuse enabled |

Older messages remain visible in the page, while model context is bounded.
Oversized current messages receive a shortening request. If a reply hits the
output cap, ask a focused follow-up. Close games and other GPU-heavy software
before the show. This configuration is intended to fit a 6 GB GPU, but actual
headroom depends on your device and other programs; verify it during rehearsal.

## 10. Troubleshooting and CPU fallback

| Problem | Action |
| --- | --- |
| Python missing/wrong architecture | Install/select 64-bit Python; reopen PowerShell and repeat the version/architecture checks. |
| Tkinter missing | Use Python with Tcl/Tk. Single chat does not require the desktop panel. |
| Missing `VCRUNTIME`/`MSVCP` DLL | Install the official Microsoft x64 redistributable, then retry setup. |
| Missing CUDA DLL or native executable | Rerun setup and check native archive hashes. CUDA executable and paired DLLs must stay together. |
| No `CUDA0`, driver error, or native exit | Check `nvidia-smi`, enabled GPU and compatible current NVIDIA driver. Read the setup/backend output. |
| Port in use | Stop the server in the terminal/panel that owns it. Select other panel ports or change native port as below. |
| Very slow response | Check the Model card says CUDA, close other GPU workloads, examine queue time and whole-GPU memory, and rehearse one chat at a time. |
| Preview answers instead of AI | Uncheck Preview and restart Backend. |
| Checksum mismatch | Restore a fresh verified ZIP; do not edit expected hashes to bypass the check. |

Explicit CPU diagnostic mode:

```powershell
.\.venv\Scripts\python.exe scripts/run_laptop.py --cpu --chat
```

Or run `.\start_windows_gpu.bat --cpu` from PowerShell for the panel. If GPU setup
cannot pass its device check and you intentionally want CPU-only diagnostics,
install with `python scripts/setup_windows_gpu.py --cpu`, then continue to use
`--cpu` in the serving command. CPU mode may be much slower.

Native port is normally 8090. Backend/front ports are 8000/8080; all must be
distinct. To change only the native port:

```powershell
$env:BEACON_LLAMA_PORT = '8091'
.\.venv\Scripts\python.exe scripts/run_laptop.py
```

PowerShell activation is unnecessary: commands use `.venv\Scripts\python.exe`
directly. Log files are under `outputs/show_day/`. Keep any desired transcript
using the frontend's available chat export controls; restarting the backend
clears its in-memory monitoring feed.

## Package contents and rebuilding

This ZIP includes app source, frontend assets/fonts, the Q4 GGUF and metadata,
native CUDA/CPU archives with provenance, launchers, verification tools and guides.
It excludes the original 8 GB base weights, separate adapter, datasets,
optimizer states, `.venv`, secrets and visitor logs.

From the complete original project, rebuild with:

```powershell
.\.venv\Scripts\python.exe scripts/package_laptop.py --windows
```

The packager verifies model/native inputs and every archived entry, then writes
the ZIP and `.zip.sha256` in `builds/`. Rebuilding a package does not retrain the
model. Package verification and real device rehearsal are separate checks.

Native sources: [pinned llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b11429),
[CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
For the API contract and panel behavior, see [API.md](API.md) and [SHOW_DAY.md](SHOW_DAY.md).
