> 2026-10-05: The current scanned release is it_seminar_v3 (14,257 total; 12,759 train). See [FINAL_DATASET_REVIEW.md](FINAL_DATASET_REVIEW.md) and [TRAINING_PLAN.md](TRAINING_PLAN.md). Earlier release details below are historical.

> 2026-10-04 update: `it_seminar_v2` includes all seven team files and contains 12,813 retained conversations. Follow [TRAINING_PLAN.md](TRAINING_PLAN.md); earlier counts below are historical.

> Current seminar release: **it_seminar_v1**, 8,250 conversations including the new team-collected files. Follow [TRAINING_PLAN.md](TRAINING_PLAN.md) for current commands. Earlier paths/counts below document prior reviews.

> Recovery update: [burmese_recovered_v1](datasets/releases/burmese_recovered_v1/README.md) restores 4,647 older examples alongside the 150 authored dialogues. It is now the default training/evaluation candidate; original archive files remain unchanged.

> Current dataset: [burmese_fluency_v2](datasets/README.md), a 150-conversation review candidate, supersedes the v1 paths and counts below. Source text, versioned releases, user reviews, and archived data now have separate directories.

# Burmese dataset quality review — 2026-10-03

The previous corpus is **not reliable enough to use unchanged for Burmese fluency training**. Structural validity and low training loss did not establish correct, natural Burmese.

## Evidence and disposition

All 7,220 original split rows were structurally inspected; language/content inspection was targeted, not an exhaustive native-speaker assessment. All original examples were single-turn. Confirmed examples below use one-based lines in the archived original files:

| File / line | Prompt | Incorrect answer |
| --- | --- | --- |
| train / 196 | Thank you very much | မစားရသေးပါဘူး (not eaten yet) |
| train / 2312 | No, I haven't eaten yet | ကျေးဇူးအများကြီးတင်ပါတယ် (thank you very much) |
| test / 145 | Congratulations | ဟိုတယ် (hotel) |
| train / 1064 | Restaurant | အစားအစာ (food) |
| train / 2399 | Maybe | ဖြစ်နိုင်ပပါ (malformed text) |

The conversation generator also produced repetitive combinations, unrelated greeting fragments, and promises of physical meetups. Its 3,200-row output combined 850 template-generated close-friend examples and 2,350 translated instruction examples. It did not provide adequate semantic quality control. The original splits shared 45 opening prompts between train/validation, 45 between train/test, and 5 between validation/test. The subsequently grouped split fixed that leakage but retained content defects.

The original `clean`, `clean_grouped`, and `raw` directories and the old conversation generator were moved to `datasets/archive/pre_fluency_2026-10-03/`. No original content was permanently deleted. A SHA-256 inventory and row-level decisions make the change auditable and reversible:

| Exclusion reason | Original rows |
| --- | ---: |
| Individually confirmed translation/text errors | 5 |
| Unreliable synthetic batches | 3,200 |
| Awaiting individual language and factual review | 4,015 |

Exclusion is conservative: the remaining 4,015 rows are **not all proven incorrect**. Technical material may be useful after review, but was not automatically carried into the conversational seed. The raw source is archived as unreviewed, not certified wholly unhealthy. Original model checkpoints remain unchanged.

## Replacement seed

`datasets/authored/burmese_fluency_v1.txt` contains 50 individually authored synthetic conversations (100 assistant responses), with two exchanges each. The active JSONL files are under `datasets/burmese_fluency_v1/`.

| Split | Conversations |
| --- | ---: |
| Train | 38 |
| Validation | 6 |
| Test | 6 |

There are 21 polite, 11 casual, and 18 close-friend examples across 41 scenario families. Coverage includes tone switching, practical planning, rewriting, translation, simple explanations, following formatting requests, and admitting missing information. Paired polite/BFF scenarios remain in the same split. Tone labels describe scenarios; individual conversations can explicitly change tone.

Validation found no duplicate conversations, shared opening prompts, or shared scenario families across splits. With the local Qwen3-4B tokenizer and TRL training template, all examples fit within 2,048 tokens; the maximum is 1,017. Every conversation has assistant supervision tokens. These checks do not certify semantics, grammar, or native fluency.

The builder preserves quoted text, numbers, and multiline responses, applies NFC normalization, rejects malformed role sequences and damaged text, and records source/output hashes. It reads only the explicit authored source, never the archive. Training, evaluation `--test_set`, and audit defaults now use this seed. New training defaults to `outputs/checkpoints-fluency-v1`.

## Limits and next training decision

This is a **small style seed, not a complete fluency corpus**. It was authored by an AI assistant and has not been independently reviewed by native Burmese speakers. Six validation and six test conversations cannot establish generalization. Both holdouts share the same authoring process as training; family separation is not an independent benchmark.

No new training run or claim of improved model fluency has been made. Do not treat this replacement as equivalent in coverage to the retired corpus or compensate for its small size by repeating it many times.

Before a substantial training run, native reviewers should correct and approve examples, expand natural dialogues across varied speakers and topics using consented or appropriately licensed sources, and create a separate human-written evaluation set. Score grammatical correctness, natural phrasing, relevance, tone matching, multi-turn consistency, and unsupported claims separately. Compare the base model and new adapter blindly on identical prompts. A fluent-looking answer can still be false.

## Reproduce

```powershell
python scripts/build_fluency_dataset.py
python scripts/audit_training.py --data-dir datasets/burmese_fluency_v1 --tokenizer models/qwen3-4b --output datasets/burmese_fluency_v1/audit.json
python -m pytest tests/test_fluency_dataset.py tests/test_dataset_checks.py tests/test_training_config.py tests/test_evaluation.py -q -p no:cacheprovider
```

The first command requires the installed Transformers/TRL environment and the local tokenizer. No external dataset download or model training is performed. `build_dataset.py` and `build_curated_conversations.py` without arguments now build the explicit seed. The old generic compiler requires `--legacy`; archived sources are not scanned by default.
