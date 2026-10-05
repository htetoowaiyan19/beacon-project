# Training control panel

Double-click `train_ui.bat`, or run:

```powershell
.\.venv\Scripts\python.exe scripts/training_ui.py
```

The UI uses Python's built-in Tkinter. It imports no model weights; Start launches
`scripts/train_lora.py` in the project's virtual environment as a separate process.
Install the training dependencies with `pip install -r requirements-training.txt`
if creating a new environment. CUDA-enabled PyTorch must match your GPU; keep the
currently verified CUDA environment for the seminar. Tkinter is included in the
standard Windows Python installer (enable Tcl/Tk if it is missing).

The model path and all three dataset splits are editable. Defaults follow
`datasets/active.json`; the current release contains 12,759 train, 749 validation,
and 749 held-out test conversations. Review the content before starting a full run.
Settings default to two epochs, batch size 1, accumulation 8, learning rate 0.0001,
LoRA rank 32 / alpha 64, and sequence length 2048. This is ordinary BF16/FP16 LoRA,
not quantized training. It uses assistant-only loss and validation early stopping.
For an initial smoke run, set Max steps to 10 and select the release's smoke files
as described in [TRAINING_PLAN.md](TRAINING_PLAN.md). The UI does not start training
automatically. Close the chat server first to release its GPU model memory.

The panel displays phase, optimizer steps, epoch, elapsed time, estimated remaining
time, loss, validation loss, learning rate, training loss plot, GPU device, allocated
and reserved VRAM, process RAM, and best checkpoint. Loss is updated every 10 steps;
checkpoints normally save every 25 optimizer steps and validation runs every 250.
Bounded smoke runs shorten these intervals to fit their step budget.
GPU memory is measured inside the training process. It is not total GPU utilization.
ETA is approximate and includes preparation overhead; it can change during evaluation.
Loss is not a measure of factual correctness or Burmese fluency.

Each run has an isolated folder:

```text
outputs/training_runs/<timestamp>-<id>/
  config.json       # exact parameters / input paths
  process.json      # child PID and start time
  train.log         # complete stdout and stderr
  status.json       # atomically refreshed trainer metrics
  exit.json         # process exit code and final status
  stop.request      # present only after Stop safely
  adapter/          # final adapter, tokenizer, manifest and bounded checkpoints
```

Start refuses a second concurrent run in the same UI. New runs never overwrite
`outputs/checkpoints` or previous UI runs. View previous run loads its saved stats
and log; Open run folder opens Explorer. Keep the UI open while training.
Stop safely requests cancellation at the next optimizer-step boundary, saves a
checkpoint and the final adapter, and labels the run `stopped`. During model loading,
evaluation, or saving, stopping can take time. Closing the window while training
offers the same safe-stop request and leaves the panel open until the process exits.
There is no automatic force-kill. A failed process is labeled `failed`; inspect its
log for the error. A saved partial adapter still needs evaluation before use.

The trainer's final export uses its best validation checkpoint when one exists;
the last saved checkpoint may contain later optimizer state. Use **Resume run** and
select the previous run folder (the folder containing `config.json`). The UI reloads
its settings, verifies the latest completed checkpoint, and resumes optimizer,
scheduler, RNG and step state. It creates a new session log folder while continuing
to save checkpoints in the original adapter directory. Do not use Start training
to continue an old run; that creates a fresh run.

For planned shutdown, click **Stop safely** and wait for `stopped` and process exit
before shutting down. A power cut can only recover up to the last complete save;
unsaved updates are lost. At the observed 5 seconds/step, 25 steps is about two
minutes plus checkpoint-writing time, not a guaranteed wall-clock interval.
Keep the default three checkpoints. Incomplete or hash-mismatched saves are skipped
in favor of an older verified checkpoint. This cannot protect against a failing disk;
an external backup or UPS provides additional protection.

Resume requires unchanged dataset contents and core training settings. Epochs are
the total target, not extra epochs to add. Do not edit the dataset midway through
a resumable run. Only one trainer should use an output folder at a time. If the UI
itself crashes, check that its trainer process has exited before resuming.
Legacy checkpoints without completion markers are not automatically resumed.
The failed `20261005-183800-fb3a42` run had no checkpoint to recover.

CLI continuation uses the same original arguments and output directory, plus
`--resume_from_checkpoint latest`. `--save_steps 25` controls saving frequency.
The final adapter by itself is for inference; it is not a complete resumable checkpoint.
Status-file sharing errors now retry and emit a warning instead of terminating training.

After you review and evaluate an adapter, serve it explicitly:

```powershell
$env:BEACON_LORA_PATH = 'C:\full\path\to\outputs\training_runs\RUN\adapter'
.\.venv\Scripts\python.exe scripts/run_server.py --host 127.0.0.1
```

Restart the server when changing adapters. Without this setting, it uses the existing
`outputs/checkpoints` adapter. Check the first chat's server log for the actual
adapter path; `/api/health` reports whether a LoRA adapter loaded. English requests
should receive English replies unless another language is requested. Evaluate
polite, casual, BFF, IT correctness, and unknown-answer behavior separately.

Implementation references: [Transformers callbacks](https://huggingface.co/docs/transformers/main_classes/callback)
and [Vercel stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol).
