# Final dataset review - 2026-10-05

> **Content review update:** the repeated mechanical audit still passes, but a deeper
> contextual check found additional content problems. See the
> [prioritized human review packet](datasets/releases/it_seminar_v3/review/human_review/README.md)
> and [searchable conversations](datasets/releases/it_seminar_v3/review/human_review/review.html).
> The four rule-based warnings below are not a complete list of content defects.
> **Update after user delegation:** all 16 priority factual/contextual issues have
> been corrected and the release rebuilt. Exactly 16 answers changed; IDs and split
> assignments were preserved. See the [before/after log](datasets/reviews/it_seminar_v3.delegated_16_changes.json).
> The earlier correction counts below describe the initial scan. The saved overlay
> now contains 120 corrected records and 8 exclusions. Broader native-language review
> remains available; these delegated corrections do not claim independent human approval.

The active release is [it_seminar_v3](datasets/releases/it_seminar_v3/README.md).
It contains **14,257 unique conversations**, including **12,759 training records**,
749 validation records, and 749 held-out test records. Both new files are included:
`041026data330.jsonl` has 330 records; `120926data339 (2).jsonl` has 1,129.
No padding or extra invented conversations were added to meet the minimum.

## What was checked

Every record and every message in all three splits was scanned. The full local
Qwen3 tokenizer and the installed TRL training template were used, without loading
the model. Checks cover JSON/schema, ID uniqueness, alternating roles, the shared
persona, empty text, damaged controls, Unicode NFC, mixed scripts, mojibake,
formatting marks, chat-control tokens, placeholders, unclosed code blocks,
question copying, repeated answers, source/release hashes, token length, and
assistant labels for every turn. Exact prompts, near prompts and saved components
were checked for split leakage. All 139 detected near-prompt pairs stay together.

**Zero mechanical errors remain.** No detected exact conversation duplicates or
opening-prompt/near-prompt split leakage remain. Every retained example fits 2,048
tokens; the maximum is 1,835. All assistant turns have training labels. Original
input hashes and final split hashes match. Emoji ZWJ sequences were retained.

Detailed evidence: [scan summary](datasets/releases/it_seminar_v3/review/final_audit/summary.json),
[scan guide](datasets/releases/it_seminar_v3/review/final_audit/README.md), and
[pinned split hashes](datasets/releases/it_seminar_v3/release.sha256).

## Corrections and exclusions

The saved overlay affects 118 source records: 110 corrected and 8 excluded.
Text was corrected in **62** records, and **49** automotive cooling records were
retagged from computer hardware to `automotive_basics`; one record received both.
Those automotive examples remain available for general technical language learning.
Text corrections include accidental Thai/Chinese fragments, stray Burmese ZWNJ/ZWSP
marks and associated vowel-order problems, two garbled question/particle typos,
and narrowly verified IT claims. Valid emoji joiners were not removed.

The eight excluded legacy examples contain two incoherent translated jokes and
six unsupported physical-world/personal or current-school claims by the assistant.
They are preserved in the source component and before/after evidence. The merge
also excludes 144 exact duplicate conversations and 35 legacy answers superseded
by team answers to matching questions. Every exclusion has a recorded reason.

The [complete change log](datasets/reviews/it_seminar_v3.final_scan_changes.json)
contains original and corrected messages, tags, reasons, and technical references.
The [review overlay](datasets/reviews/it_seminar_v3.decisions.json) makes those
changes reproducible. Its reviewer is explicitly `codex_final_scan`; this does
not assert teammate/native-speaker approval. Corrections propagate to identical
copies, and conflicting or stale corrections fail the build.

Examples of factual corrections:

- Bit stuffing inserts zero after five consecutive ones; a six/five contradiction
  was corrected. [RFC 1662, section 5.2](https://www.rfc-editor.org/rfc/rfc1662)
- TLS answers now distinguish a normal 1-RTT handshake from optional 0-RTT early
  data and its replay risk. [RFC 8446](https://www.rfc-editor.org/rfc/rfc8446.html)
- HSTS answers now state the cached policy lifetime and first-visit/preload scope.
  [RFC 6797](https://www.rfc-editor.org/rfc/rfc6797.html)
- WPA3-Personal/SAE and Enhanced Open/OWE are distinguished.
  [Android implementation documentation](https://source.android.com/docs/core/connect/wifi-wpa3-owe)
- The quantum-computing answer no longer promises current RSA/ECC breaking in
  seconds. [NIST post-quantum FAQ](https://csrc.nist.gov/projects/post-quantum-cryptography/faqs)
- SameSite descriptions now distinguish same-site traffic, cross-site navigation,
  and the Secure requirement for None.
  [Mozilla Set-Cookie reference](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Set-Cookie)
- The layer-abstraction answer no longer guarantees unchanged packet bits across
  routers; it describes IPv4 TTL/header changes.
  [RFC 1812](https://www.rfc-editor.org/rfc/rfc1812)

## What remains for content approval

The final heuristic scan has **four flags**. They are HTTP 301/308 answers using
the valid term "permanent", rather than unsupported absolute security promises.
These are retained; their full messages are in `review/final_audit/content_flags.jsonl`.
[HTTP semantics, sections 15.4.2 and 15.4.9](https://www.rfc-editor.org/rfc/rfc9110.html)

There are also **394 groups sharing an opening question** with different full
conversations. Some are paraphrases or different later turns; others may disagree.
They are kept in the same split and preserved for comparison in
[alternative_answers.md](datasets/releases/it_seminar_v3/review/alternative_answers.md).
This scan does not certify every answer's factual accuracy, every Burmese expression,
or the absence of subtle semantic overlap. Native-language and IT content approval
is still pending. Structural readiness and fluent/accurate model behavior are
different things; evaluate the trained adapter on the untouched test split.

## Using the release

`datasets/active.json` now selects v3. Open a fresh training panel with `train_ui.bat`;
its defaults, the CLI trainer, test-set evaluation, and audit tools use that release.
Train only `train/train_combined.jsonl`. Evaluation subsets and smoke files are
views of their corresponding splits, not extra records to append. Keep all original
submissions and prior releases unchanged. The original roughly 7,000-record corpus
remains archived, and 4,604 recovered legacy records are retained here.

The two-epoch seminar run has since completed; see [RELEASE_V1.md](RELEASE_V1.md)
for its adapter and held-out results. Keep the trained release and its split hashes
frozen. Further review changes should belong to a new experiment, followed by
smoke training and validation using [TRAINING_UI.md](TRAINING_UI.md).
Do not move hold-out records into training to increase the count.
