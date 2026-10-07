# BEACON project review guide

Use this guide to understand the current project, inspect its release and data,
and decide what to keep locally. For installation commands, use [SETUP.md](SETUP.md).

The project contained approximately **46.20 GB (43.03 GiB)** of files on
**2026-10-06**. Most space belongs to model copies, ZIP builds, checkpoints and
the Python environment. The source code and documentation account for very little.
**No existing models, datasets, checkpoints or builds were deleted during this review.**

## 1. Start with the purpose of each copy

The original project is the **development and training machine's working copy**.
It contains more than the laptop needs: original weights, quantized exports,
training recovery states, datasets, evaluation evidence and distribution ZIPs.

The **M2 Air deployment** needs the files extracted from
`builds/BEACON-v1.0.0-m2-air.zip` and its destination `.venv`. It does not need
the entire 46 GB working project. That ZIP is about 2.51 GB and contains the
trained Q4_K_M GGUF, native runtime, application and members' panel.

The **full portable ZIP** preserves the application, original base/adapter and
datasets for transfer. It does not contain training recovery checkpoints or the
native GGUF runtime. Git preserves source and documentation; it does not back up
the ignored datasets, weights, ZIPs or training outputs.

### Read these guides in order

| Guide | Question it answers |
| --- | --- |
| [README.md](README.md) | What is BEACON and what are its entry points? |
| [SETUP.md](SETUP.md) | How do I install, start, stop and troubleshoot each option? |
| [M2_AIR.md](M2_AIR.md) | What settings and limitations apply to the 8 GB laptop? |
| [SHOW_DAY.md](SHOW_DAY.md) | How do the visitor screen and members' monitoring work? |
| [API.md](API.md) | What requests, stream events and diagnostics can clients use? |
| [RELEASE_V1.md](RELEASE_V1.md) | Which trained adapter is served and what was measured? |
| [datasets/README.md](datasets/README.md) | Where are originals, final splits and review decisions? |
| [FINAL_DATASET_REVIEW.md](FINAL_DATASET_REVIEW.md) | What was corrected and what still needs content review? |
| [TRAINING_UI.md](TRAINING_UI.md) | How do I start, stop safely and resume training? |
| [TRANSFER.md](TRANSFER.md) | How do I restore the original model on another device? |
| [builds/README.md](builds/README.md) | Which ZIP should I copy? |

## 2. Where the disk space goes

These measurements are logical file sizes, including hidden folders. They do not
measure allocated blocks, RAM, or caches outside this project. Sizes will change
as new logs, training runs and builds are added.

| Folder | Size, decimal GB | What uses the space | Review priority |
| --- | ---: | --- | --- |
| `builds/` | 19.217 | Full portable ZIP, model-only ZIP and M2 ZIP | First: distribution copies overlap |
| `models/` | 13.719 | Original base, serving adapter, Q4 and Q5 exports | Keep active weights; Q5 is optional |
| `outputs/` | 6.726 | Older checkpoints, current training recovery states and conversion tools | Separate recovery states from disposable output |
| `.venv/` | 5.924 | Installed PyTorch/CUDA and other packages | Needed by this configured desktop |
| `.git/` | 0.259 | Source history and Git objects | Keep for development/history |
| `datasets/` | 0.241 | Originals, releases, components and review evidence | Small, valuable provenance; keep |
| `archives/` | 0.101 | Retired RAG, earlier project files and pre-correction data | Keep for fifth-year/history needs |
| `runtime/` | 0.012 | Bundled macOS runtime archive and provenance | Needed to build/install the laptop package |
| Source and documentation | Under 0.01 | Backend, frontend, scripts, tests, prompts and guides | Keep; deleting these saves little |

### Largest individual artifacts

| Artifact | Size | Why it exists |
| --- | ---: | --- |
| `builds/BEACON-v1.0.0-portable.zip` | 8.392 GB | Complete original-model transfer build with datasets |
| `builds/BEACON-v1.0.0-model-only.zip` | 8.315 GB | Base/adapter transfer without the complete application |
| `models/qwen3-4b/model.safetensors` | 8.045 GB | Original base weights used by desktop chat and training |
| `models/gguf/beacon-v1.0.0-Q5_K_M.gguf` | 2.890 GB | Optional five-bit diagnostic export |
| `builds/BEACON-v1.0.0-m2-air.zip` | 2.509 GB | Complete lightweight laptop distribution |
| `models/gguf/beacon-v1.0.0-Q4_K_M.gguf` | 2.497 GB | Merged trained laptop weights used to build the M2 ZIP |
| `models/adapters/beacon-v1.0.0/adapter_model.safetensors` | 0.264 GB | Frozen trained seminar adapter |

The ZIPs and extracted weights contain overlapping data. This is deliberate
packaging, not evidence of duplicate training examples. Deleting the raw weights
would break the corresponding local inference/training/export workflow; moving
distribution ZIPs to a verified backup has a different effect.

### Refresh the measurements

From the project root on Windows:

```powershell
.\.venv\Scripts\python.exe scripts/project_inventory.py
```

On macOS/Linux with a suitable Python interpreter:

```sh
python3 scripts/project_inventory.py
```

The inventory script uses only Python's standard library, loads no model and
writes `outputs/project_review/inventory.json`. The report includes top-level
sizes, subfolder sizes, every file at least 100 MB, read errors and skipped links.
It skips symlinks/junctions to avoid counting linked folders outside the project.
Run it after any future cleanup to measure the actual result. Inventorying files
is not a code, content, checksum or model-quality audit.

## 3. Folder map and dependencies

```text
Project/
  backend/       API, streaming and selected inference service
  frontend/      Visitor chat assets; no build step
  scripts/       Launchers, UI controls, data preparation, checks and packaging
  tests/         Python and JavaScript regression checks
  prompts/       Manual evaluation prompts
  datasets/      Human originals, recovered IT data, frozen splits and decisions
  models/        Original weights/adapter and quantized laptop exports
  runtime/       Bundled Apple Silicon native runtime and source provenance
  outputs/       Training recovery states, results, logs and conversion work
  builds/        Distributable ZIPs and checksum sidecars
  archives/      Retired RAG and historical project/data backups
  .venv/         Device-specific desktop Python dependencies
  .git/          Source history
```

### What each workflow needs

| Workflow | Required artifacts | Not needed for that workflow |
| --- | --- | --- |
| Original trained desktop chat | Application, original base, frozen adapter, inference environment | Dataset contents and optimizer checkpoints |
| M2 trained chat | Laptop application, Q4 GGUF plus export metadata, extracted Apple runtime, laptop environment | Original SafeTensors, dataset contents, training states |
| Frontend preview | Source assets, mock backend and frontend dependencies | Models, datasets, CUDA |
| New original-model training | Base weights, selected frozen splits, training tools/environment | Distribution ZIPs and GGUF runtime |
| Resume a previous training run | Original run/config, complete recovery checkpoint, unchanged inputs and compatible environment | A model-only ZIP is insufficient |
| Rebuild full portable ZIP | Full source, base/serving adapter and datasets | Training optimizer states |
| Rebuild M2 ZIP | Relevant source, Q4/export metadata and bundled native archive/provenance | Original training recovery states |
| Re-export GGUF | Original base/serving adapter, export dependencies, converter/quantizer and temporary space | Existing ZIPs are not a substitute for those inputs |

## 4. Review the application source

### Backend review order

| File or group | What to inspect |
| --- | --- |
| `backend/app.py`, `application.py` | Application creation, route registration, static assets and shutdown cleanup |
| `backend/dev_app.py` | Explicit mock preview; synthetic outputs remain labeled |
| `backend/config.py`, `release.py`, `beacon-release.json` | Defaults, selected frozen adapter and public release metadata |
| `backend/routes/chat.py` | Request validation, chat stream and cancellation integration |
| `backend/routes/health.py` | `/api/health`, model-load and device status |
| `backend/routes/show_day.py` | Visitor events and protected operator endpoints |
| `backend/services/chat_service.py` | Runtime selection, Vercel SSE events, timings and errors |
| `backend/services/model_service.py` | Original Transformers/LoRA loading and generation |
| `backend/services/gguf_model_service.py` | Native runtime requests, laptop limits and merged-adapter behavior |
| `backend/services/chat_context.py`, `text_streamer.py` | Bounded history and Unicode streaming for original inference |
| `backend/services/streaming_response.py` | Generator cleanup on completion/disconnect |
| `backend/services/show_day_monitor.py` | In-memory visitor activity/chat feed and retention |

Check which runtime a launcher selects before interpreting its statistics. The
normal desktop defaults to the original model; laptop launchers set
`BEACON_RUNTIME=gguf`. Device RAM and GPU metrics have different meanings between
the runtimes. The product release is v1.0.0; API contract version 3.0.0 is a
separate version, not another trained model.

### Frontend review order

| File | Purpose |
| --- | --- |
| `frontend/index.html`, `style.css` | Page structure, controls and Burmese rendering |
| `frontend/app.js` | Requests, conversation history, settings, Stop/Copy/Save and health display |
| `frontend/stream.js` | Incremental Vercel SSE frame parsing and completion/errors |
| `frontend/show-day.js` | Visitor/session activity and monitoring notice |
| `frontend/favicon.svg` | Browser icon |
| `frontend/README.md` | Preview development instructions |

Rehearse a Burmese reply, an English reply, a follow-up, cancellation, a new
conversation and an exported chat. Ensure cancelled answers are visibly labeled,
and ensure the visitor page and members' feed show the same submitted exchange.
Draft text is not a live screen capture; monitoring tracks submitted chats and
activity events. Keep streamed text rendering safe when changing presentation.

### Scripts by responsibility

| Group | Scripts and purpose |
| --- | --- |
| Original setup/serving | `setup_device.py`, `run_release.py`, `run_server.py`, `download_model.py` |
| Laptop setup/serving | `setup_laptop.py`, `run_laptop.py`, `utils/llama_runtime.py` |
| Show-day controls | `show_day_ui.py`, `show_day_backend.py`, `show_day_frontend.py`, `utils/show_day_control.py` |
| Training | `train_lora.py`, `training_ui.py`, `utils/training_status.py`, `utils/training_checkpoints.py` |
| Data preparation | `build_dataset.py`, `recover_legacy_dataset.py`, `build_fluency_dataset.py`, `build_seminar_dataset.py` |
| Data checks/review | `audit_training.py`, `audit_final_dataset.py`, `build_human_review_packet.py`, `finalize_seminar_review.py` |
| Shared data/model utilities | `utils/active_dataset.py`, `dataset_checks.py`, `text_normalizer.py`, `persona.py`, `model_utils.py`, `generation_utils.py` |
| Original evaluation | `evaluate_model.py`, `evaluate_heldout_sample.py`, `profile_chat_latency.py` |
| Real serving smoke checks | `smoke_release.py`, `smoke_show_day.py`, `smoke_laptop.py` |
| Quantization diagnostics | `export_laptop_model.py`, `compare_laptop_quality.py`, `validate_laptop_quantization.py` |
| Packaging/inventory | `package_release.py`, `package_laptop.py`, `verify_package.py`, `project_inventory.py` |
| Review templates | `scripts/templates/` HTML used by dataset review tools |

Read data builders and review finalizers before running them: they generate or
change release/review files. A code review does not require executing every script.
Keep this completed experiment's data frozen. Standalone model smoke checks load
weights; stop live model servers first and run on the intended device.

### Launchers and dependency files

| File | Intended use |
| --- | --- |
| `show_day_ui.bat` | Desktop members' panel |
| `run_trained_chat.bat` | Single chat using the verified frozen desktop adapter |
| `run_server.bat` | Generic original-model server |
| `setup_device.bat` | Original-model environment installation |
| `train_ui.bat` | Optional desktop training controls |
| `setup_laptop.command` | Mac laptop installation |
| `start_laptop.command` | Mac members' panel |
| `chat_laptop.command` | Mac single chat server |
| `requirements-inference.txt` | Original model and API libraries |
| `requirements-training.txt` | Additional training libraries |
| `requirements-evaluation.txt` | Additional evaluation libraries |
| `requirements-export.txt` | Model conversion dependencies |
| `requirements-laptop.txt` | Small native-runtime API environment |
| `requirements-frontend.txt` | Model-free development preview and checks |

The Windows `.venv` is specific to this device. Do not copy it to the Mac or
delete individual CUDA DLLs to shrink it. Recreate the appropriate environment
from requirements if you intentionally retire this installation.

## 5. Review models and the selected experiment

### Active original model

The serving snapshot is `models/adapters/beacon-v1.0.0/`, not whichever adapter
directory has the latest modified date. Its identity is pinned in
`beacon-release.json`; the original base is `models/qwen3-4b/`.

The selected adapter is from **checkpoint 3000** of the completed **3190-step**
experiment. Its LoRA rank is 32 and alpha is 64. Use the release check without
loading weights:

```powershell
.\.venv\Scripts\python.exe scripts/run_release.py --check
```

This verifies the frozen adapter checksum and basic base-file availability.
It does not evaluate the model or hash every base file. Package verification
checks all listed files; use the full portable ZIP as the separate manifest-backed
copy when assessing artifact integrity.

### Laptop models

`models/gguf/export.json` identifies the active Q4 model, its checksum and source
adapter/base hashes. `export-Q5_K_M.json` describes the optional Q5 diagnostic copy.
The trained adapter is merged into these models; loading another separate adapter
does not update their weights.

```powershell
.\.venv\Scripts\python.exe scripts/run_laptop.py --check
```

This is a file/identity check and can run on this Windows machine. It does not
prove that Metal works on the Air. Q5 is not used by the default M2 ZIP and has
not shown a quality improvement in the tiny existing validation comparison.

### Existing quality evidence

| Evidence location | What it establishes and what it does not |
| --- | --- |
| `outputs/evaluations/` | Original base/adapter sampled held-out answers, reference scores and review reports |
| `outputs/laptop_export/smoke.json` | Desktop native-runtime streaming checks; not an M2 speed/power benchmark |
| `outputs/laptop_export/quality-comparison.json` | Eight-example original/Q4 diagnostic comparison |
| `outputs/laptop_export/validation-quantization.json` | Three-example BF16/Q4/Q5 diagnostic comparison |
| `outputs/laptop_export/http-check.json` | Real visitor proxy, cancellation, operator feed and shutdown evidence |
| `outputs/laptop_export/template-check.json` | Matching original/native template and token IDs for one tested conversation |
| `outputs/laptop_export/restore-check.json` | Verified extraction and relocated API checks on Windows; not macOS execution |
| `outputs/show_day/`, `release_checks/`, `performance/` | Serving smoke checks, logs and latency samples |

Burmese fluency and IT accuracy remain uneven. Reference similarity and validation
loss do not certify factual correctness or natural Burmese. Read actual answers
and rehearse on the M2, including several successive chats. Retain these small
reports even if large conversion tooling is moved elsewhere.

## 6. Review the datasets without losing their history

The active descriptor is `datasets/active.json`, selecting
`datasets/releases/it_seminar_v3/`. The completed release has **14,257 conversations**:
12,759 train, 749 validation and 749 held-out test.

| Dataset location | Review purpose | Retention advice |
| --- | --- | --- |
| `datasets/freshes/` | Nine original team submissions | Preserve unchanged |
| `datasets/archive/` | Older corpus, including the original 7,220 records | Preserve legacy provenance |
| `datasets/sources/` | Normalized human, recovered and authored components | Preserve reproducibility |
| `datasets/releases/it_seminar_v3/train/` | Actual training split | Freeze the completed experiment |
| `datasets/releases/it_seminar_v3/validation/` | Selection/validation split | Never append to training |
| `datasets/releases/it_seminar_v3/test/` | Held-out evaluation split | Keep held out |
| `datasets/releases/it_seminar_v3/review/` | Audit, alternative answers and human review packet | Review content here |
| `datasets/reviews/` | Decisions, corrections and before/after evidence | Preserve saved review overlays |
| Other `datasets/releases/` components/versions | Previous releases and preparation components | Review as history, not extra unique training rows |

The release retains 9,503 team examples, 4,604 recovered legacy examples and 150
authored conversations. The dataset is only about **241 MB** in total; deleting
human originals or review evidence would save little compared with the ZIPs.

### Review sequence

1. Confirm `active.json` selects v3 and read the release manifest/README.
2. Read `FINAL_DATASET_REVIEW.md`, including its update about the 16 delegated
   technical/contextual corrections. Earlier counts describe the earlier scan.
3. Review `review/human_review/README.md` and `review.html` for the review packet.
4. Review `review/alternative_answers.md`: there are 394 groups sharing an opening
   question, not necessarily 394 defects. Distinguish paraphrases from disagreement.
5. Check technical correctness and Burmese naturalness separately. Saved assistant
   corrections are not independent native-speaker approval.
6. Keep the completed splits and overlay frozen. Further corrections should form
   a separately versioned experiment before new training/evaluation.

The last recorded complete audit reported no mechanical failures or detected
split leakage; that is not a new content audit of every record in this folder review.
Do not append raw files, smoke views, evaluation subsets or component releases to
the training split: they overlap with the selected release.

Pinned hashes are in `datasets/releases/it_seminar_v3/release.sha256`. To compare
current files manually in PowerShell:

```powershell
Get-Content datasets/releases/it_seminar_v3/release.sha256
Get-FileHash datasets/releases/it_seminar_v3/train/train_combined.jsonl -Algorithm SHA256
Get-FileHash datasets/releases/it_seminar_v3/validation/validation_combined.jsonl -Algorithm SHA256
Get-FileHash datasets/releases/it_seminar_v3/test/test_combined.jsonl -Algorithm SHA256
```

These are read-only integrity checks. Do not run the dataset builder just to
inspect the project: rebuilding is a separate data-changing action.

## 7. Review outputs before deciding what to archive

| Output area | Size | Meaning |
| --- | ---: | --- |
| `outputs/checkpoints/` | 3.735 GB | Older rank-64/alpha-128 experiment and checkpoints 3250/4875 |
| `outputs/training_runs/` | 2.691 GB | Current UI run chain, exported adapter and rank-32 recovery states |
| `outputs/laptop_export/` | 0.299 GB | Converter source/tools, downloaded tooling and quantization/check evidence |
| Other output folders | Under 0.001 GB combined at review | Evaluation reports, server logs, latency and release checks |

### Current training recovery states

The current experiment's original run directory is:

```text
outputs/training_runs/20261005-185304-2d2fb2/
```

Its adapter contains checkpoints **3000, 3175 and 3190**. Final trainer state
records step 3190 and best checkpoint 3000. Later stop/resume sessions have their
own status/log folders; the final session is
`outputs/training_runs/20261006-132518-faad6f/`. An earlier session can show
`stopped` while the shared experiment has subsequently completed. Check final
trainer state, session logs and `beacon-release.json` together.

A recoverable checkpoint needs adapter/config, optimizer, scheduler, RNG,
trainer state and the verified `checkpoint_complete.json` marker. The inference
snapshot is not a substitute. Preserve the original run folder, related resume
session configs/logs, inputs and paths if you want to resume or reproduce it.

### Older checkpoints are another experiment

`outputs/checkpoints/` uses rank 64/alpha 128, while the served release uses rank
32/alpha 64. It is not merely a redundant copy of the current serving adapter.
Archive it separately if you want to retain that experiment. It is neither needed
by default chat nor included as recovery state in the full portable ZIP.

### Conversion work

`outputs/laptop_export/llama.cpp-b11429/` and `windows/` contain converter/native
tools used for export and desktop testing. They are not used by the original
desktop chat launcher or by the copied M2 package. They help repeat conversion.

The large merged-HF and intermediate BF16 GGUF working copies have already been
removed after the verified export; there is no further 16 GB saving to claim
from those files. Keep export metadata, tool provenance, merge receipt and small
quality/restore reports. If moving conversion tools, record how to restore them.

## 8. What can reduce size, and what each choice costs

This is a review plan, not a deletion command list. Before removing the only local
copy of valuable artifacts, transfer the appropriate files to another device/disk
and verify their checksums. Stop related services/training before moving their
working files. A backup folder on the same disk does not free that disk's space.

| Candidate | Potential local saving | Condition and consequence |
| --- | ---: | --- |
| Model-only ZIP | 8.315 GB | Full portable build includes the same frozen base/adapter; verify that independent backup and test restoration first. The standalone convenience ZIP can then live externally. |
| Full portable ZIP | 8.392 GB | Move ZIP and checksum to verified external storage. Working source/weights still run; keep an accessible recovery copy. |
| Optional Q5 export and matching metadata | About 2.890 GB | Default Q4 inference/build does not need it. Preserve elsewhere if keeping the diagnostic experiment. |
| M2 ZIP after transfer | 2.509 GB | Move with its checksum once the laptop copy is verified. Keep it local if you need quick redistribution. |
| Older `outputs/checkpoints/` | 3.735 GB | Back up the complete older experiment separately. It is not present in the current model/data transfer ZIP. |
| Current training recovery folders | Up to 2.691 GB | Only retire after backing up run chain/checkpoints or explicitly deciding that continuation/recovery is unnecessary. Frozen serving weights remain separate. |
| Conversion tooling under `outputs/laptop_export/` | Less than 0.299 GB | Keep small evidence/provenance. Export/testing needs tools restored; routine packaged inference does not. |
| `.venv/` | 5.924 GB | Only if retiring this desktop environment. Scripts need dependency installation again; do not do this to the currently useful seminar setup. |
| Python/test caches and old server logs | Small | Regenerable caches and unwanted logs are low-value space targets. Preserve logs/results you still need to diagnose or present. |

### Practical options

**Keep desktop development and recovery:** move the full and model-only ZIPs to
a verified external drive. This frees about **16.71 GB** locally while leaving
working models, the existing environment, data and training states present. If
you also move the optional Q5 export, the potential saving is about **19.60 GB**.

**Use only the laptop for show day:** transfer the M2 ZIP/checksum, install into a
fresh laptop folder and rehearse. Leave the development/history copy on the
original machine or external storage. Copying only the M2 package avoids carrying
the full working folder to the Air.

**Retire older experiments later:** once separately backed up and reviewed,
archive the 3.735 GB older checkpoint tree. Keep the current run's recovery state
until you decide whether further training/continuation is needed.

Do not add the table's savings together indiscriminately: choices depend on which
workflow you keep and where backups live. Moving a ZIP within this project, such
as `archives/` to `builds/`, organizes it but does not reduce total size.

## 9. Verify backups and keep a review record

For each ZIP and matching checksum sidecar, use:

```powershell
Get-FileHash builds/BEACON-v1.0.0-portable.zip -Algorithm SHA256
Get-Content builds/BEACON-v1.0.0-portable.zip.sha256
```

Repeat against the **destination copy**, not just the source. Both hashes must
match the sidecar. The full portable and M2 packages use `MANIFEST.json` and the
standard verifier:

```powershell
.\.venv\Scripts\python.exe scripts/verify_package.py --zip builds/BEACON-v1.0.0-portable.zip
.\.venv\Scripts\python.exe scripts/verify_package.py --zip builds/BEACON-v1.0.0-m2-air.zip
```

Run the model-only package's included restoration/verification instructions for
that package's manifest; do not assume all archive formats are identical.
Verify a fresh extracted copy before editing it. ZIP verification needs no GPU
but reads all archived bytes. A checksum proves byte identity, not successful
setup: also run the intended artifact check and real chat on the destination.

The RAG/history archives have their own manifest/instructions. Preserve
`archives/rag_5th_year_2026-10-04.zip` for the fifth-year project. Its approximately
95 MB size is small compared with model builds.

### Review worksheet

| Area | Decision/status | Reviewer and date | Backup location or evidence |
| --- | --- | --- | --- |
| Source and API | Pending | | |
| Visitor UI and members' panel | Pending | | |
| Original serving weights | Pending | | |
| M2 real-device rehearsal | Pending | | |
| Burmese fluency and IT answers | Pending | | |
| Frozen dataset integrity/content | Pending | | |
| Full and model-only ZIP copies | Pending | | |
| Older checkpoint experiment | Pending | | |
| Current training recovery states | Pending | | |
| RAG and dataset/project history | Pending | | |

Record whether an item is **keep locally**, **backed up externally**, **restore
tested**, or **approved for removal**. A trained model, training recovery state,
training data and application source are four separate kinds of artifacts.

## 10. Final project review checklist

- [ ] Read the release metadata and confirm the intended model/runtime.
- [ ] Review source changes with `git status --short` and `git diff`; confirm
  large artifacts, credentials, datasets and visitor logs remain excluded from Git.
- [ ] Run the appropriate frozen-model file check.
- [ ] Review the API stream contract and cancellation/shutdown behavior.
- [ ] Rehearse real chats and monitoring on the target device.
- [ ] Review Burmese/IT quality evidence and record remaining content limitations.
- [ ] Confirm the selected dataset and its frozen split hashes.
- [ ] Distinguish current recovery states from older experiments.
- [ ] Verify external backup hashes and test the restoration you will depend on.
- [ ] Record any retention/removal decisions before cleanup.
- [ ] Refresh the inventory afterward and compare actual space savings.

Existing tests cover datasets, training configuration/recovery, streaming,
cancellation, context limits, previews, show-day behavior, GGUF isolation and
packaging. Use checks appropriate to any changes; the full suite requires the
full development dependencies and local dataset artifacts. A project review does
not require retraining or rerunning the entire evaluation experiment.

Current source guides use `builds/`. Existing ZIPs were moved without changing
their contents, so embedded guides may mention their former `archives/` location
or omit this newer review guide. Rebuild intentionally when distributing a new
documented source snapshot; do not confuse an older verified ZIP with today's
working tree.
