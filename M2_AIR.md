# BEACON on the M2 MacBook Air (8 GB)

For the complete installation walkthrough and other setup options, see
[SETUP.md](SETUP.md).

Use **builds/BEACON-v1.0.0-m2-air.zip** for show day. It contains your trained
Burmese adapter merged into Qwen3-4B and quantized to **Q4_K_M GGUF**, the visitor
chat, members' panel and a bundled Apple Silicon llama.cpp runtime.
Weights are approximately **2.50 GB**. Total RAM use also includes context cache,
model buffers, macOS and the browser; file size is not a RAM measurement.

Inference uses **Metal on the M2 GPU**, with no PyTorch, Transformers or training
libraries. The laptop package excludes datasets, original BF16 weights and resume
states. Those are preserved in the full transfer package and original project.

## Setup before show day

The bundled binary requires **macOS 13.3 or newer**. Install Apple Silicon Python
**3.11 or newer** with Tcl/Tk support for the desktop panel if it is not installed.

1. Copy the ZIP and its `.zip.sha256` file. Compare the sidecar with
   `shasum -a 256 BEACON-v1.0.0-m2-air.zip` in Terminal.
2. Extract into a fresh folder. Open Terminal in that folder.
3. Run `sh setup_laptop.command`. It verifies files, extracts the bundled runtime,
   creates `.venv` and installs `requirements-laptop.txt`. Initial dependency
   installation requires internet. The model and native runtime are already included.
4. Run `sh start_laptop.command`. Start Backend and wait for **trained model ready**,
   then Start Frontend and Open visitor page. Default: **http://127.0.0.1:8080/**.
   Put the browser on the external monitor and the members' panel on the Air.

The `.command` files can also open from Finder when executable. If macOS blocks
the runtime, inspect its security prompt and verify the included GitHub provenance
before allowing it. Setup does not install an OS update or Python itself.

For a single chat page, run `sh chat_laptop.command` and visit
**http://127.0.0.1:8000/**. Ctrl+C stops it. Do not run both launchers on the same
ports. Closing the members' panel stops its servers and native model process.
Inference works offline after setup.

## Resource settings

| Setting | Laptop edition |
| --- | --- |
| GPU | Metal, all supported model layers |
| Context allocation | 2,048 tokens, one slot |
| Input budget | 1,536 tokens |
| Remembered exchanges | Two recent complete turns at most |
| Response limit | 192 tokens |
| Thinking | Disabled |
| CPU / batch threads | 4 / 4 |
| Batch / microbatch | 128 / 128 |
| Extra RAM cache storage | Disabled; ordinary prompt reuse remains enabled |

The full conversation remains visible/exportable. Oversized current messages are
rejected rather than cut. Base comparison and thinking are disabled. If an answer
hits the response limit, ask a focused follow-up.

The panel reports native process resident RAM and available system RAM. Resident
RAM does not include every Metal allocation and is not dedicated VRAM, GPU
utilization or electrical power. Check
Activity Monitor's Memory Pressure on the Air, close unnecessary applications and
tabs, and keep one visitor chat active at a time. The fanless Air may slow during
sustained use; test a longer session before the seminar.

## Verification

`models/gguf/export.json` records the original base and frozen adapter hashes.
The launcher verifies the GGUF checksum. No retraining is required; quantization
can change answers, so review Burmese fluency and IT answers on the laptop.
Desktop CPU smoke results do not establish M2 speed or power consumption.

Ten desktop streaming checks passed. A small eight-example test comparison gave
mean chrF 29.38 for the original trained adapter and 24.58 for Q4 at the same
192-token output budget. Some Burmese phrasing became less natural; IT errors
also remain in the original model. Three separate validation examples did not
show a Q5 improvement. These tiny checks are diagnostic, not reliable estimates
of overall quality. chrF measures agreement with references, not factual accuracy.
Review the included answers in `checks/` before presenting the quantized model.

The default ZIP contains Q4 only. A Q5 diagnostic export is preserved locally;
`scripts/package_laptop.py --q5` can package it separately. It is about 2.89 GB
and has not demonstrated better quality in this small check.

Before editing files, run `python3 scripts/verify_package.py`. After setup, run
`.venv/bin/python scripts/smoke_laptop.py` for real English/Burmese streaming
checks. Results go to `outputs/laptop_export/smoke.json`; the model stops afterward.
The package includes desktop checks in `checks/desktop-gguf-smoke.json` when available.

The API and Vercel streaming contract remain the same. See [API.md](API.md).
Sources: [Qwen llama.cpp guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/run_locally/llama.cpp.md),
[Metal build](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md),
[quantization](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md).
