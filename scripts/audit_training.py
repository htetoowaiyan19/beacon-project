"""Read-only audit of dataset leakage and saved training evidence; no model loading."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
if __package__:
    from .utils.dataset_checks import conversation_key, prompt_key
    from .utils.active_dataset import active_release
else:
    from utils.dataset_checks import conversation_key, prompt_key
    from utils.active_dataset import active_release

ROOT = Path(__file__).resolve().parents[1]


def read_samples(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def audit(root=ROOT, data_dir=None):
    data_dir = data_dir or active_release(root)
    result = {"data_dir": str(data_dir), "training_evidence_source": "Existing checkpoint; not a new training run",
              "splits": {}, "overlap": {}, "training": {}, "benchmarks": []}
    samples, keys, prompts = {}, {}, {}
    for name, file in [("train", "train_combined.jsonl"), ("validation", "validation_combined.jsonl"), ("test", "test_combined.jsonl")]:
        path = data_dir / name / file
        if not path.exists():
            continue
        rows = read_samples(path)
        samples[name] = rows
        keys[name] = {conversation_key(s) for s in rows}
        prompts[name] = {prompt_key(s) for s in rows}
        answers = Counter(m["content"] for s in rows for m in s["messages"] if m["role"] == "assistant")
        result["splits"][name] = {
            "rows": len(rows), "unique_conversations": len(keys[name]), "unique_opening_prompts": len(prompts[name]),
            "unique_assistant_answers": len(answers), "most_repeated_answer_count": max(answers.values(), default=0),
            "assistant_messages": sum(answers.values()),
        }
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if a in keys and b in keys:
            result["overlap"][f"{a}/{b}"] = {
                "identical_conversations": len(keys[a] & keys[b]),
                "identical_opening_prompts": len(prompts[a] & prompts[b]),
            }
    states = sorted((root / "outputs" / "checkpoints").glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[-1]))
    if states:
        state = json.loads(states[-1].read_text(encoding="utf-8"))
        logs = state.get("log_history", [])
        result["training"] = {
            "global_step": state.get("global_step"), "epoch": state.get("epoch"),
            "best_metric": state.get("best_metric"), "best_model_checkpoint": state.get("best_model_checkpoint"),
            "evaluations": [r for r in logs if "eval_loss" in r],
            "last_training_log": next((r for r in reversed(logs) if "loss" in r), {}),
            "run_summary": next((r for r in reversed(logs) if "train_runtime" in r), {}),
        }
    for path in sorted((root / "outputs" / "evaluations" / "results").glob("*.md")):
        result["benchmarks"].append({"file": path.name, "header": path.read_text(encoding="utf-8").split("---")[0].strip()})
    return result


def token_lengths(data_dir, model_path, max_length):
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    template = get_training_chat_template(tokenizer)
    report = {}
    for split, filename in [("train", "train_combined.jsonl"), ("validation", "validation_combined.jsonl"), ("test", "test_combined.jsonl")]:
        path = data_dir / split / filename
        if not path.exists():
            continue
        sizes, no_targets = [], 0
        for row in read_samples(path):
            encoded = tokenizer.apply_chat_template(row["messages"], chat_template=template, tokenize=True,
                return_dict=True, return_assistant_tokens_mask=True)
            sizes.append(len(encoded["input_ids"]))
            no_targets += not any(encoded["assistant_masks"][:max_length])
        sizes.sort()
        if sizes:
            report[split] = {"count": len(sizes), "p50_tokens": sizes[len(sizes)//2],
                "p95_tokens": sizes[int(len(sizes)*0.95)], "max_tokens": sizes[-1],
                "over_max_length": sum(n > max_length for n in sizes),
                "no_assistant_tokens_after_truncation": no_targets, "max_length": max_length}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=active_release())
    parser.add_argument("--tokenizer", type=Path, help="Also audit token lengths using the local training tokenizer")
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    report = audit(data_dir=args.data_dir)
    if args.tokenizer:
        report["token_lengths"] = token_lengths(args.data_dir, args.tokenizer, args.max_length)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
