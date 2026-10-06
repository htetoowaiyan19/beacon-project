# Run BEACON on another device

`archives/BEACON-v1.0.0-portable.zip` contains the cleaned source, Qwen3-4B base
weights, frozen trained LoRA adapter, tokenizers, frontend, show-day panel,
training/evaluation tools and preserved datasets/review decisions. No separate
model download is needed. It excludes the original `.venv`, Git metadata, secrets,
visitor logs, training optimizer checkpoints and the separate fifth-year RAG archive.

## Verify and extract

Copy the ZIP and its adjacent `.zip.sha256` file. On Windows, compare
`Get-FileHash .\BEACON-v1.0.0-portable.zip -Algorithm SHA256` with the sidecar.
Extract the entire ZIP into a new folder using a ZIP64-capable extractor (for
example Windows Extract All). Keep the directory structure, including `models/`.
Allow about 9 GB for extracted project files, plus space for Python packages;
keeping the ZIP also needs approximately another 8 GB.

From the extracted project folder, with Python 3.11 or newer installed:

```powershell
python scripts/verify_package.py
python scripts/setup_device.py
```

The verifier checks every packaged file's size and SHA-256, including both model
weights and datasets. It needs no installed packages or GPU. Run it before editing
packaged files. The adjacent ZIP checksum also checks the manifest itself.

Setup creates a fresh `.venv` and installs the pinned dependencies with the CUDA
12.8 PyTorch build tested on the seminar machine. It requires internet access and
a compatible NVIDIA driver/GPU; it does not install a GPU driver. The tested
machine has an RTX 5060 Ti with 16 GB VRAM and 32 GB system RAM. Smaller GPUs may
offload work to CPU and respond much more slowly. Python 3.14 is the tested runtime;
older compatible Python versions have not received the full project checks.
Enable Tcl/Tk in the Windows Python installer for the desktop panels.

For CPU-only setup, use `python scripts/setup_device.py --cpu`. CPU inference needs
enough system RAM and will be slower. On Linux/macOS, use `.venv/bin/python` in
commands instead of `.venv\Scripts\python.exe`; Tkinter may require a separate OS
package. macOS uses the CPU path; Apple GPU inference is not implemented.

## Start and stop

For the two-monitor seminar setup, double-click **show_day_ui.bat**, start Backend
and Frontend, then open the visitor page (default **http://127.0.0.1:8080/**).
Put the panel on the members' monitor. See [SHOW_DAY.md](SHOW_DAY.md).

For a single chat page, double-click **run_trained_chat.bat** and open
**http://127.0.0.1:8000/**. Ctrl+C stops that server. Do not start both workflows
at the same time on the same ports.

CLI alternatives:

```powershell
.\.venv\Scripts\python.exe scripts/run_release.py --check
.\.venv\Scripts\python.exe scripts/show_day_ui.py
# Or, for a single server:
.\.venv\Scripts\python.exe scripts/run_release.py
```

After dependency installation, inference uses the included local weights.
The base model and adapter must both remain present. The release launcher verifies
the adapter matches the selected checkpoint; normal chat does not train it.

## Optional training and evaluation

Use `python scripts/setup_device.py --training --evaluation` to install those
dependencies too. The transfer includes the original datasets and preparation
evidence, with `datasets/active.json` selecting the completed v3 experiment.
See [TRAINING_UI.md](TRAINING_UI.md) and [datasets/README.md](datasets/README.md).
Do not alter the completed experiment's held-out splits.

The ZIP can start a new training run. To resume an existing run, additionally
copy its original `outputs/training_runs/<run>` directory with complete optimizer,
scheduler and RNG checkpoints, and check its saved paths on the new device.
Those large recovery states are preserved on the original machine.

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts/package_release.py
```

Packaging uses ZIP64, includes a per-file `MANIFEST.json`, verifies every archived
entry against source hashes, rejects files that change while packing and writes
a ZIP SHA-256 sidecar. Large weights are stored without compression to keep
packaging fast. The older `BEACON-v1.0.0-seminar.zip` is superseded by this complete
transfer ZIP.
