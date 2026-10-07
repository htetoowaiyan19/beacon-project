# BEACON builds

Built model and application ZIPs are stored here with matching `.zip.sha256`
sidecars. These large local artifacts are excluded from Git.

| ZIP | Contents | Approximate size |
| --- | --- | ---: |
| `BEACON-v1.0.0-m2-air.zip` | Trained Q4_K_M GGUF, Apple Silicon runtime, chat and monitoring panel | 2.51 GB |
| `BEACON-v1.0.0-windows-cuda.zip` | Trained Q4_K_M GGUF, Windows CUDA/CPU runtimes and latest app | About 3.2 GB |
| `BEACON-v1.0.0-model-only.zip` | Original SafeTensors base model and frozen trained LoRA adapter | 8.32 GB |
| `BEACON-v1.0.0-portable.zip` | Full project, original model/adapter, datasets and preparation provenance | 8.39 GB |

For the **M2 Air with 8 GB RAM**, use the M2 ZIP. Follow [SETUP.md](../SETUP.md)
for installation. The model-only ZIP needs a separate application checkout.

For the **Windows RTX 4050 laptop**, use the Windows CUDA ZIP and follow
[WINDOWS_GPU.md](../WINDOWS_GPU.md).

The M2 Air and full portable packages are rebuilt from the current working source
with the frontend team's landing page, separate chat page, logo and local fonts
(frontend commit `62d1e9b`). Current project guides and packagers use `builds/`.
The model-only ZIP was moved from `archives/` without changing its contents;
it has no frontend and does not need rebuilding for this UI update.

Rebuild the laptop or full transfer package from the complete source machine:

```powershell
.\.venv\Scripts\python.exe scripts/package_laptop.py
.\.venv\Scripts\python.exe scripts/package_laptop.py --windows
.\.venv\Scripts\python.exe scripts/package_release.py
```

Both packagers now write here and verify archived file hashes. Preserve the
checksum sidecar with each ZIP when transferring it. Historical RAG and project
archives remain under `archives/`.
