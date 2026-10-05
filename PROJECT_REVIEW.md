> Historical review: the RAG/variant paths and commands below describe the archived implementation. As of 2026-10-04, the active project is model-only. Use README.md, API.md and TRAINING_UI.md for current instructions. The complete retrieval snapshot is in archives/rag_5th_year_2026-10-04.zip.

> Current dataset: [burmese_fluency_v2](datasets/README.md), a 150-conversation review candidate, supersedes the v1 paths and counts below. Source text, versioned releases, user reviews, and archived data now have separate directories.

> Dataset update (2026-10-03): The legacy corpus and generator are archived. Current training/evaluation defaults use `datasets/burmese_fluency_v1/`, a 50-conversation synthetic style seed awaiting native-speaker review and expansion. Run `python scripts/build_fluency_dataset.py` to rebuild it. New checkpoints default to `outputs/checkpoints-fluency-v1`. Dataset counts, paths, ingestion instructions, and training commands below describe the previous corpus and are historical. See [DATASET_QUALITY_REPORT.md](DATASET_QUALITY_REPORT.md) for current findings and limitations.

# BEACON project review

Review date: 2026-10-03 (Asia/Rangoon).

The project now has two separately launchable application variants, a Vercel-compatible chat output stream, and fixes to dataset preparation, training supervision and evaluation. The existing adapter is usable for experimentation. Its factual accuracy and production readiness are **not established** by the saved reports.

## 1. RAG and non-RAG applications

| Variant | Entry point | Dependency file | Features |
| --- | --- | --- | --- |
| With RAG | `variants/rag/app.py` | `variants/rag/requirements.txt` | Model chat, document upload, ChromaDB retrieval, citations |
| Without RAG | `variants/no_rag/app.py` | `variants/no_rag/requirements.txt` | Model chat and base/LoRA settings |

Both share the backend, frontend, model files and adapter. This avoids duplicating several gigabytes of weights. They can use separate Python environments. The non-RAG application has no document API routes and does not import or initialize ChromaDB, the PDF pipeline or OCR. Passing `use_rag: true` cannot activate retrieval in that application. Its interface hides retrieval controls and document management.

Run either command from the repository root, using the existing CUDA-enabled environment:

```powershell
.\.venv\Scripts\python.exe scripts/run_server.py --variant rag --port 8000
.\.venv\Scripts\python.exe scripts/run_server.py --variant no-rag --port 8001
```

Run one at a time on the 16 GB GPU: separate server processes each allocate their own model. `backend.app:app` remains the RAG entry point for existing launchers. Installation details are in each variant's README; scanned-PDF support has a separate optional OCR requirements file.

## 2. Vercel chat streaming

`POST /api/chat/stream` now returns the current AI SDK **UI Message Stream v1** data protocol, with `Content-Type: text/event-stream` and `x-vercel-ai-ui-message-stream: v1`. It emits message/step boundaries, text start/delta/end parts, document sources, custom `data-sources`/`data-metrics` parts, errors, a finish reason, and the `[DONE]` terminator. This follows [Vercel's stream specification](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol).

The request body still uses this project's `message`, `history`, and generation settings. An AI SDK frontend must map its outgoing UI messages to that request schema. Output compatibility does not imply accepting the default `useChat` request body automatically. This implements the current SSE protocol, rather than the obsolete AI SDK 4 line-prefix format.

The shared browser client handles split UTF-8 characters, transport boundaries, stream errors and premature disconnection. It records a reply in history only after stream completion. Retrieved text and filenames are escaped before rendering.

The inference service now serializes generation and adapter switching, explicitly disables PEFT for base requests, propagates worker failures, cancels generation when its stream closes, and counts **generated token IDs** rather than decoded text chunks. Thinking mode is controlled through `enable_thinking`, as documented in the [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-4B).

## 3. Training efficiency and accuracy

### Evidence from the existing run

The saved trainer state confirms 4,875 optimizer steps across three epochs. Validation results were:

| Epoch | Validation loss | Teacher-forced token accuracy |
| --- | --- | --- |
| 1 | 0.3668 | 89.50% |
| 2 | **0.3230** | 90.73% |
| 3 | 0.3320 | 90.92% |

Loss worsened after epoch 2 while training loss continued falling, indicating some overfitting. SHA-256 hashes confirm that `outputs/checkpoints/adapter_model.safetensors` matches `checkpoint-3250`, the best epoch-2 adapter. The root export is not the final epoch-3 adapter.

BF16, LoRA, SDPA, gradient checkpointing and chunked NLL are sensible memory controls for this machine. The installed GPU was confirmed as an RTX 5060 Ti. The preserved state has no total training runtime or peak-memory trace, so the claimed sub-12 GB training peak and overall training throughput cannot be independently confirmed. Rank 64 is fairly expensive; rank 16/32 should be compared using the same clean validation set before assuming that 64 is necessary.

### Problems found

1. **Repeated questions across splits:** the original train/validation/test counts are 6,498/361/361. There are no identical complete conversations across splits, but train and validation share 45 opening questions; train and test share 45; validation and test share 5. Different answers to the same question are not independent unseen examples.
2. **Full-sequence supervision:** the old trainer flattened conversations into text and trained on system, user and assistant tokens. The reported 95.82% training token accuracy is not an answer-quality score and includes predictable prompt tokens. It was recorded at step 4,870, just before the final step.
3. **Truncated targets:** with the installed tokenizer and TRL training template, 924/6,498 training rows (14.2%) exceed 2,048 tokens. Training lengths have median 384, 95th percentile 3,248, and maximum 5,795. Validation/test have 58/51 overlong rows. No sampled row loses every assistant token, but long answers can lose their ending and stop token. Shorten or segment these examples at conversation boundaries, or benchmark a longer context against GPU memory, before the next full run.
4. **Insufficient accuracy evaluation:** the latest report generated answers for only 10 questions and reported speed. Earlier reports cover 25 prompts. Neither measures factual correctness, citation support, or native-speaker fluency. Reference answers were loaded but unused. The old base/LoRA comparison also reused the PEFT-modified base without disabling its adapter.
5. **Silent configuration loss:** the old compatibility helper silently discarded unsupported training options. For example, `group_by_length` is absent from the installed TRL configuration, so that claimed optimization was not active.

### Changes made

- Deduplicate complete conversations; group identical normalized opening questions into the same split.
- Preserve existing holdouts on ordinary rebuilds, and require an explicit `--resplit` to retire them. Reject overlapping splits before training.
- Reject malformed conversations and prevent the inserted Burmese system prompt from making English-only data pass language filtering.
- Preserve conversational datasets and enable assistant-only loss. A real installed-TRL test confirms that the user prompt is masked, the Burmese answer and end-of-turn token are supervised, and chunked loss supports backpropagation.
- Bound evaluation batches to one, retain gradient checkpointing/chunked loss, add early stopping, and save training metrics and a manifest. Unsupported training options now fail visibly.
- Disable the adapter for base comparisons; preserve multi-turn evaluation history and gold references; export machine-readable results and normalized exact match. Installing `requirements-evaluation.txt` additionally enables chrF. These are reference-agreement scores, not factual accuracy or a substitute for native-speaker review.

A separate prepared dataset exists at `datasets/clean_grouped/`: **6,496 train / 361 validation / 363 test**, with zero shared opening prompts across those splits. The original data and adapter remain available. This regrouped data contains examples already seen by the existing adapter; it is suitable for a **new run starting from the original base**, not an unbiased retest of the existing adapter.

Example next training experiment, saving to a new adapter directory:

```powershell
python scripts/train_lora.py --train_file datasets/clean_grouped/train/train_combined.jsonl --val_file datasets/clean_grouped/validation/validation_combined.jsonl --test_file datasets/clean_grouped/test/test_combined.jsonl --output_dir outputs/checkpoints-assistant-only --epochs 2 --lora_r 32 --lora_alpha 64
```

This command has not been run. Rank 32 and two epochs are starting points for a controlled comparison, not proven optimal settings. Full retraining remains necessary to measure the effect of the changes. Revised assistant-only loss values should not be compared directly to historical full-sequence loss.

Reproduce the data audit:

```powershell
python scripts/audit_training.py --output outputs/evaluations/training_audit.json
python scripts/audit_training.py --data-dir datasets/clean_grouped --output outputs/evaluations/grouped_training_audit.json
python scripts/audit_training.py --tokenizer models/qwen3-4b --output outputs/evaluations/training_audit_with_lengths.json
```

For a meaningful quality assessment, use fresh native-reviewed Burmese questions plus the public benchmark below. Include technical QA, translation, colloquial/formal conversation, instruction following and code switching. Evaluate RAG separately with known supporting documents, unanswerable questions, retrieval recall and citation faithfulness.

## 4. Is Qwen3-4B good for Burmese?

**It is a reasonable compact baseline, particularly with thinking enabled.** Qwen explicitly lists Burmese among Qwen3's supported languages, but support alone does not guarantee fluent or accurate output. [Qwen's release documentation](https://qwenlm.github.io/blog/qwen3/).

The native-reviewed BURMESE-SAN paper reports these aggregate MY scores:

| Model and mode | MY score |
| --- | --- |
| Qwen3-4B, thinking | 34.78 |
| Qwen3-8B, thinking | 37.12 |
| Gemma 3 12B, instruct | 42.46 |
| Gemma SEA-LION v4 4B VL, instruct | 26.24 |
| Qwen SEA-LION v4 4B VL, instruct | 23.31 |
| Gemma SEA-LION v4 27B, instruct | 47.18 |

These are aggregate benchmark scores, not percentages of correct chat answers. Generation modes and budgets differ. They do not score this project's LoRA or establish the non-thinking Qwen3-4B ranking. [BURMESE-SAN, Tables 3–4](https://arxiv.org/html/2602.18788v1).

For this GPU, first compare the existing base and LoRA with both thinking settings. Qwen3-8B is the simplest architecture upgrade; quantization or offloading is needed for practical 16 GB headroom. Gemma 3 12B is a stronger benchmark candidate but needs quantization and a different model loader. For a compact non-thinking candidate, test [Gemma SEA-LION v4 4B VL](https://huggingface.co/aisingapore/Gemma-SEA-LION-v4-4B-VL), which explicitly includes Burmese post-training. The [Qwen SEA-LION 4B VL card](https://huggingface.co/aisingapore/Qwen-SEA-LION-v4-4B-VL) also includes Burmese, but its smaller benchmark score does not justify an automatic replacement.

Both VL families require their appropriate conditional-generation model and processor, rather than simply changing this application's causal-LM path. The current Qwen3-4B LoRA cannot be reused on a different model. Larger SEA-LION models, including the newer [Qwen SEA-LION v4.5 27B](https://huggingface.co/aisingapore/Qwen-SEA-LION-v4.5-27B-IT), warrant separate hardware and task evaluation; no comparable MY score for that newer release was established in this review.

## 5. Remaining RAG limitations

The existing Chroma collection uses its default MiniLM embedding function. Burmese semantic retrieval has not been benchmarked. Compare a multilingual embedding model such as [BGE-M3](https://huggingface.co/BAAI/bge-m3) on a Burmese retrieval set; changing embeddings requires a new collection and re-indexing. Cosine similarity and the 0.35 threshold are not calibrated probabilities of correctness.

The OCR path uses standard `yolov8n.pt`, an object detector, and `microsoft/trocr-base-printed`, which was fine-tuned on SROIE. These do not establish a Burmese text-line detector/recognizer. Burmese scanned-PDF accuracy is unverified; digital PDF/TXT/MD is the appropriate starting point. [YOLOv8 model documentation](https://docs.ultralytics.com/models/yolov8/), [TrOCR model card](https://huggingface.co/microsoft/trocr-base-printed).

RAG currently falls back to general model generation when retrieval has no qualifying results. It is not a strict document-only answering system. Retrieval scores and displayed sources do not prove that every generated statement is supported.

## 6. Validation and limits

- 26 Python tests passed: isolated document CRUD, pipeline logic, no-RAG import isolation, streaming/errors, grouped splits, PEFT base isolation and actual-token counting, multi-turn references/scoring, and real TRL assistant masking/loss/configuration.
- Four Node tests passed: fragmented Burmese UTF-8/CRLF streams, stream errors, premature termination, and non-RAG interface initialization. JavaScript syntax and Python compilation passed.
- Real local Qwen3-4B GPU chat output was exercised through the non-RAG API with adapter disabled and enabled. Both emitted the required header, text, metrics and finish events. The 32-token transport checks ended with `finishReason: length`; they are not accuracy measurements. Results are in `outputs/evaluations/inference_smoke.json`.
- The connected browser was unavailable, so full visual/browser interaction verification was not possible. No full retraining, public benchmark run, alternative-model download or Burmese OCR evaluation was performed.

Machine-readable evidence: `outputs/evaluations/training_audit.json`, `grouped_training_audit.json`, `token_length_audit.json`, and `inference_smoke.json`.
