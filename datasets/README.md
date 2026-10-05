# IT seminar datasets

The active release is **[it_seminar_v3](releases/it_seminar_v3/README.md)**: **14,257 conversations**, split into **12,759 train / 749 validation / 749 test**. Both newly collected files are included. IT knowledge and fluent Burmese remain the objective.

Read [FINAL_DATASET_REVIEW.md](../FINAL_DATASET_REVIEW.md) for the full scan, corrections, exclusions, references and review limits. The [training plan](../TRAINING_PLAN.md) covers smoke training and final evaluation. Training has not been started during this preparation.

## Folder organization

| Folder | Contents |
| --- | --- |
| `freshes/` | Nine original team submissions, preserved unchanged |
| `sources/human_collected_v3/` | Generated normalized source copies before the final review overlay |
| `releases/it_seminar_v3/` | Active train, validation, test, evaluation views, smoke files and review evidence |
| `reviews/it_seminar_v3.decisions.json` | Saved final-scan changes and project content decisions |
| `reviews/it_seminar_v3.final_scan_changes.json` | Before/after text, tags, exclusions and evidence URLs |
| `releases/it_seminar_v1/`, `releases/it_seminar_v2/` | Previous combined releases, preserved for comparison |
| `releases/burmese_recovered_v1/` | Preserved legacy component |
| `sources/burmese_fluency_v2/`, `releases/burmese_fluency_v2/` | Authored multi-turn source and component |
| `archive/` | Original roughly 7,000-row legacy corpus and earlier layout |

`active.json` selects v3. Desktop training, CLI training, test-set evaluation, and audits follow this descriptor. Model checkpoints belong under `outputs/`.

## Counts and cleanup

Nine team files contain 9,647 parsed records. The two latest files contribute **330 + 1,129 = 1,459**, regardless of the number in the second filename. Retained composition: 9,503 team-collected examples, 4,604 recovered older examples, and 150 authored dialogues. There are 430 multi-turn conversations.

144 exact duplicate conversations, 35 superseded legacy answers, and 8 defective legacy examples were excluded from this release. The final review corrected text in 62 source records and retagged 49 automotive cooling examples (one record received both). Originals remain unchanged; decisions and original text are preserved. No records were invented or duplicated to meet the count.

## Final review and training files

Train only `releases/it_seminar_v3/train/train_combined.jsonl`. Validation/test files are reserved for selection/evaluation. Every retained record fits 2,048 tokens with the local Qwen/TRL training template, and every assistant turn has labels. The complete scan reports zero mechanical failures or detected split leakage.

Review [394 alternative-answer groups](releases/it_seminar_v3/review/alternative_answers.md) for actual disagreement versus legitimate paraphrases. Semantic correctness and native Burmese approval are still pending. The four remaining heuristic flags are valid HTTP permanent-redirect descriptions; see the root final review. A mechanical pass does not certify fluent or accurate model behavior.

To add project decisions, edit `reviews/it_seminar_v3.decisions.json`, preserving existing entries. Use stable IDs, `content_sha256`, and a status of pending/approved/rejected. Approved corrections can include the complete `replacement_messages` list and/or `replacement_tags`. Notes alone do not change training text. Assistant-generated entries are explicitly marked with their reviewer; they are not native-speaker approval.

```powershell
.\.venv\Scripts\python.exe scripts/build_seminar_dataset.py
.\.venv\Scripts\python.exe scripts/audit_final_dataset.py
```

The builder applies saved v1/v2/v3 reviews in that order, propagates corrections to identical copies, rejects stale/conflicting decisions, and rebuilds grouped splits. Freeze the inputs, review overlay and final split hashes before training. Do not rebuild mid-experiment or mix old/new splits.

## Legacy conversations remain available

The original 7,220 records remain under `archive/pre_fluency_2026-10-03/clean/`. This active release retains 4,604 recovered legacy records. The recovery's excluded examples remain available with reasons for later adjudication. IT content was not excluded merely for being technical.

Do not append evaluation subsets, smoke files, component releases, normalized copies or originals to the train file; they overlap with the release. Adding future collection data should create a separately versioned release.
