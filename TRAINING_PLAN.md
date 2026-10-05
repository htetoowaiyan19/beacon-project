# IT seminar training plan

Updated 2026-10-05 for both new submissions and the whole-release final scan. See [FINAL_DATASET_REVIEW.md](FINAL_DATASET_REVIEW.md) for corrections and review limits.

For desktop controls, launch `train_ui.bat`. See [TRAINING_UI.md](TRAINING_UI.md)
for isolated runs, progress/stats, safe stopping, logs, and selecting a trained
adapter for the model-only chat server. The UI reads `datasets/active.json`.

The objective is **accurate IT explanations in fluent Burmese**, with polite, casual, and BFF styles adapted to the user. Use the full combined dataset. IT content is the core of this project.

## Shared system prompt

Training, evaluation, CLI chat, and the API use `scripts/utils/persona.py`:

> You are a helpful Burmese AI companion. Reply in the user's language unless they request another. Match their tone: polite, casual, or close-friend. Be accurate and admit when you don't know.

This is 44 tokens with the local Qwen3-4B tokenizer, compared with 255 for the previous Burmese prompt. English requests should receive English replies; explicit language requests take precedence. The training examples remain primarily Burmese, so check both English and Burmese behavior in the final evaluation. Reduced prompt length saves token processing but does not establish a measured training speedup. Previous dataset releases preserve their historical prompts.

## 1. Current release

`datasets/releases/it_seminar_v3/` combines the team's new files, the recovered older corpus, and the authored multi-turn conversations.

| Split | Team-collected | Recovered older | Authored style | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 8,519 | 4,105 | 135 | 12,759 |
| Validation | 485 | 256 | 8 | 749 |
| Test | 499 | 243 | 7 | 749 |

The nine team files contain **9,647 recoverable records**, including 330 in `041026data330.jsonl` and 1,129 in `120926data339 (2).jsonl`. The merge removes 144 exact duplicate conversations, supersedes 35 older answers with team answers to matching prompts, and excludes 8 defective legacy records. Total retained: **14,257**. Prior v1/v2 releases remain unchanged. The final scan corrects text in 62 records and tags in 49 records (one overlaps), with saved before/after evidence.

Human-collected describes the provenance you supplied. It does not automatically mean every answer has passed technical or native-language verification. There are **394 groups with alternative answers**; some are valid paraphrases, others may conflict. Review [the readable comparison](datasets/releases/it_seminar_v3/review/alternative_answers.md) first.

Most data are single user–assistant exchanges. The team small-talk file contributes 280 multi-turn conversations and the authored set adds 150; nothing was padded with invented follow-up turns or duplicated to increase its weight.

## 2. Review, rebuild, then freeze

The originals remain untouched in `datasets/freshes/`. Normalized copies are in `datasets/sources/human_collected_v3/`. Repairs were limited to unescaped quotation marks in 26 cybersecurity records, JSON boundaries, one literal newline in a JSON string, five repeated system/user prefixes, two tag typos, tag-list normalization, the common system prompt, and NFC/outer-whitespace normalization. All final text/tag corrections and exclusions are recorded in `datasets/reviews/it_seminar_v3.final_scan_changes.json`; originals are preserved.

Review the alternative-answer groups, particularly operating-system definitions and networking statements with words such as “always,” “never,” or absolute guarantees. Check Burmese clarity independently from IT correctness. Human alternative answers are preserved in the candidate rather than automatically labeled wrong.

To approve, reject, or correct records, copy only the relevant entries from `review/decisions.template.json` into `datasets/reviews/it_seminar_v3.decisions.json`. Each entry needs its stable `id`, `content_sha256`, and `status` (`pending`, `approved`, or `rejected`). An approved entry can include `replacement_messages`, a complete ordered message list. Notes alone do not change the training answer. Saved v1/v2 reviews are also read; v3 decisions override matching IDs, and all hashes are rechecked. A rejected conversation and its identical copies are excluded; stale hashes fail the build.

```powershell
.\.venv\Scripts\python.exe scripts/build_seminar_dataset.py
```

The builder creates fresh, category-stratified approximately 90/5/5 splits. Exact opening-question groups, high character-similarity groups (5-gram Jaccard ≥ 0.85 for sufficiently long questions), and authored scenario families stay together. Old split assignments are retired; do not mix files from different releases. This limits detectable leakage, not all semantic overlap.

After review, freeze the source files, decisions file, and release manifest hashes for the experiment. Do not rebuild or move examples between splits during training. New additions should produce a separately versioned experiment. Keep the test split untouched during parameter selection.

## 3. Baseline and smoke test

Verified environment: RTX 5060 Ti, 15.9 GiB VRAM, BF16 available; PyTorch 2.11.0+cu128, Transformers 5.11.0, TRL 1.6.0, PEFT 0.19.1.

Start from `models/qwen3-4b`, not the old adapter. The old adapter was trained on parts of the original corpus and is not an unseen-data baseline. Stop the inference server while training to avoid two model copies occupying VRAM.

First save a small base-model validation run:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_model.py --base --prompts_file datasets/releases/it_seminar_v3/evaluation/validation/human_collected.jsonl --max_samples 30
```

Then run a diagnostic training job using 32 of the longest training conversations and 16 validation records:

```powershell
.\.venv\Scripts\python.exe scripts/train_lora.py --train_file datasets/releases/it_seminar_v3/smoke/train.jsonl --val_file datasets/releases/it_seminar_v3/smoke/validation.jsonl --output_dir outputs/checkpoints-seminar-v3-smoke --max_steps 10 --batch_size 1 --eval_batch_size 1 --grad_accum 8 --learning_rate 1e-4 --lora_r 32 --lora_alpha 64 --max_seq_length 2048
```

This command is provided for you to run; no GPU training was started while arranging data. The smoke adapter is disposable and must not be deployed. Check finite loss, nonempty assistant labels, successful evaluation/checkpoint saving, and VRAM headroom. The tiny-model trainer test verifies software behavior but does not replace this real 4B memory check.

## 4. Main training run

Proposed first experiment, not a claim of optimal hyperparameters:

| Setting | Value | Purpose |
| --- | --- | --- |
| Base model | Local Qwen3-4B | Fresh base rather than the legacy adapter |
| Epochs | 2 | Short first run; choose the best validation checkpoint |
| Learning rate | `1e-4` | Starting point for adapter training |
| LoRA rank / alpha | 32 / 64 | Lower adapter size than the earlier rank-64 setup |
| Train/eval batch | 1 / 1 | Limit VRAM pressure |
| Gradient accumulation | 8 | Effective batch of 8 on one GPU |
| Maximum length | 2,048 | All retained examples fit; no truncation is needed |
| Precision | BF16 automatically | Supported by this GPU |
| Supervision | Assistant-only | Do not train the model to reproduce user/system turns |
| Checkpointing | Enabled; retain 3 | Bound memory and checkpoint storage |
| Evaluation | Every 250 optimizer steps | Select lowest validation loss |

```powershell
.\.venv\Scripts\python.exe scripts/train_lora.py --output_dir outputs/checkpoints-it-seminar-v3 --epochs 2 --batch_size 1 --eval_batch_size 1 --grad_accum 8 --learning_rate 1e-4 --lora_r 32 --lora_alpha 64 --max_seq_length 2048 --save_total_limit 3 --early_stopping_patience 1
```

These settings are also the updated trainer defaults. At 12,759 training examples and effective batch 8, expect approximately **1,595 optimizer updates per epoch**, or **3,190 for two epochs** before early stopping. Estimate runtime from the measured smoke/main-run step speed plus evaluation and save time; no reliable wall-clock estimate is available yet. The long-example smoke is deliberately more demanding than the average batch.

If memory fails, first ensure the inference server is stopped. Lower LoRA rank to 16 with alpha 32 if necessary. Reducing gradient accumulation alone does not reduce the memory needed by one sequence. Do not silently shorten context below dataset lengths; rebuilding a shorter-context candidate would exclude affected examples and require renewed accounting.

The project uses the installed TRL training template and assistant masks, gradient checkpointing, and chunked NLL. TRL documents conversational assistant-only supervision and notes that adapter training commonly uses a learning rate around `1e-4`; the exact rank, epoch count, and accumulation above are this project's experimental choices. [Official TRL SFT documentation](https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.md)

## 5. Evaluate IT correctness and Burmese separately

Use the same prompts and inference settings for base and adapter. Keep thinking mode off for the seminar's direct-answer comparison; the inference service already defaults to `think=false`. Qwen documents the thinking-mode switch in its chat template. [Official Qwen quickstart](https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html)

During development, use the `evaluation/validation/` subsets: operating systems, networking, NLP, hardware, cybersecurity, human-collected data, and conversation style. These are views of the same validation split, not extra training records. Aggregate loss can hide a weak topic.

After choosing the checkpoint, run the held-out comparison:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_model.py --compare --lora_path outputs/checkpoints-it-seminar-v3 --test_set
```

For a time-limited seminar review, use matching category subsets and a fixed sample count:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_model.py --compare --lora_path outputs/checkpoints-it-seminar-v3 --prompts_file datasets/releases/it_seminar_v3/evaluation/test/operating_systems.jsonl --max_samples 15
```

Repeat for `networking.jsonl`, `nlp.jsonl`, `hardware.jsonl`, and `cybersecurity.jsonl`; sample `conversation_style.jsonl` and add fresh team-written follow-ups. This style subset now includes team small-talk data as well as authored dialogues. Existing reference matching/chrF metrics measure wording overlap, not factual correctness.

Have two teammates score blinded outputs from 1–5 for technical correctness, Burmese clarity, relevance, and instruction/tone adherence. Separately flag fabricated claims and incorrect technical definitions. Discuss disagreements. Report sample sizes, per-topic results, base-versus-adapter differences, and representative failures. A low validation loss alone is not seminar evidence of an accurate assistant.

## 6. Deploy only the selected adapter

Training outputs are separate from `outputs/checkpoints`, which the inference server uses by default. A new run does not automatically replace the serving adapter. After evaluating the chosen checkpoint, set `BEACON_LORA_PATH` to its adapter folder, restart the server, and smoke-test model-only chat. The retired retrieval project is preserved in `archives/rag_5th_year_2026-10-04.zip`.

For the deadline, prioritize one reviewed full-data run and a transparent IT evaluation over multiple unmeasured hyperparameter experiments.

## Stopping and resuming

The trainer now saves full checkpoints every 25 optimizer steps (configurable with `--save_steps`). See [TRAINING_UI.md](TRAINING_UI.md) for safe stop, power-cut recovery, and Resume run. Resume keeps the original total epoch target and requires unchanged data/settings.
