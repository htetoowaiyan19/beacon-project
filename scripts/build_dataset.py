"""Compile, clean, deduplicate, and merge diverse Burmese training datasets (.jsonl, .csv, .tsv, .parquet)."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.text_normalizer import (
    is_myanmar_text,
    normalize_myanmar_text,
    standardize_system_prompt,
)
from utils.dataset_checks import assert_disjoint, conversation_key, grouped_split, prompt_key

from utils.persona import DEFAULT_SYSTEM_PROMPT


def valid_conversation(messages: list[dict]) -> bool:
    turns = messages[1:] if messages and messages[0]["role"] == "system" else messages
    return bool(turns) and len(turns) % 2 == 0 and all(
        message["role"] == ("user" if i % 2 == 0 else "assistant")
        for i, message in enumerate(turns)
    )


def parse_sample_to_messages(item: dict) -> list[dict] | None:
    """Standardize different dataset schemas into a standard ShareGPT / OpenAI chat messages list.
    
    Supported Schemas:
    1. Chat format: {'messages': [{'role': 'user', 'content': ...}, {'role': 'assistant', ...}]}
    2. Alpaca format: {'instruction': ..., 'input': ..., 'output': ...}
    3. QA format: {'question': ..., 'answer': ...} or {'prompt': ..., 'response': ...}
    4. Two-column format: {'user': ..., 'assistant': ...}
    5. Translation format: {'en': ..., 'my': ...} or {'source': ..., 'target': ...}
    """
    if not isinstance(item, dict):
        return None

    # Case 1: Already messages format
    if "messages" in item and isinstance(item["messages"], list):
        cleaned_messages = []
        has_user = False
        has_assistant = False

        for msg in item["messages"]:
            if not isinstance(msg, dict):
                continue
            if not isinstance(msg.get("role"), str) or not isinstance(msg.get("content"), str):
                return None
            role = msg["role"].strip().lower()
            content = msg["content"].strip()
            if not role or not content:
                continue

            if role == "system":
                content = standardize_system_prompt(content)
            else:
                content = normalize_myanmar_text(content)

            if not content:
                continue

            if role == "user":
                has_user = True
            elif role == "assistant":
                has_assistant = True

            cleaned_messages.append({"role": role, "content": content})

        if has_user and has_assistant:
            if cleaned_messages[0]["role"] != "system":
                cleaned_messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
            return cleaned_messages if valid_conversation(cleaned_messages) else None
        return None

    # Case 2: Alpaca format (instruction, input, output)
    instruction = str(item.get("instruction") or "").strip()
    input_text = str(item.get("input") or "").strip()
    output_text = str(item.get("output") or "").strip()

    if instruction and output_text:
        user_content = f"{instruction}\n\n{input_text}".strip() if input_text else instruction
        user_content = normalize_myanmar_text(user_content)
        asst_content = normalize_myanmar_text(output_text)
        if user_content and asst_content:
            return [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": asst_content},
            ]

    # Case 3: Prompt/Response or Question/Answer or User/Assistant
    user_val = str(item.get("prompt") or item.get("question") or item.get("user") or item.get("query") or "").strip()
    asst_val = str(item.get("response") or item.get("answer") or item.get("assistant") or item.get("completion") or "").strip()

    if user_val and asst_val:
        user_clean = normalize_myanmar_text(user_val)
        asst_clean = normalize_myanmar_text(asst_val)
        if user_clean and asst_clean:
            return [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_clean},
                {"role": "assistant", "content": asst_clean},
            ]

    # Case 4: Translation (en/my or source/target)
    en_val = str(item.get("en") or item.get("english") or item.get("source") or "").strip()
    my_val = str(item.get("my") or item.get("myanmar") or item.get("burmese") or item.get("target") or "").strip()

    if en_val and my_val:
        user_clean = f"'{normalize_myanmar_text(en_val)}' ကို မြန်မာလို ဘာသာပြန်ပေးပါ။"
        asst_clean = normalize_myanmar_text(my_val)
        if user_clean and asst_clean:
            return [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": user_clean},
                {"role": "assistant", "content": asst_clean},
            ]

    return None


def load_file_records(file_path: Path) -> list[dict]:
    """Load raw records from JSONL, JSON, CSV, TSV, or Parquet files."""
    suffix = file_path.suffix.lower()
    records = []

    try:
        if suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            pass

        elif suffix == ".json":
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    records = [data]

        elif suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    records.append(row)

        elif suffix == ".parquet":
            try:
                import pandas as pd
                df = pd.read_parquet(file_path)
                records = df.to_dict(orient="records")
            except Exception as e:
                print(f"  [!] Failed to read parquet {file_path.name}: {e}")

    except Exception as e:
        print(f"  [!] Error reading {file_path.name}: {e}")

    return records


def load_and_clean_file(file_path: Path) -> list[dict]:
    """Parse, clean, format-normalize, and filter any dataset file."""
    valid_samples = []
    if not file_path.exists() or file_path.stat().st_size == 0:
        return valid_samples

    raw_records = load_file_records(file_path)

    for item in raw_records:
        messages = parse_sample_to_messages(item)
        if not messages:
            continue

        # The inserted Burmese persona must not make an English-only row appear Burmese.
        full_text = " ".join(m["content"] for m in messages if m["role"] != "system")
        if not is_myanmar_text(full_text):
            continue

        tags = item.get("tags", "general") if isinstance(item, dict) else "general"
        valid_samples.append({"messages": messages, "tags": str(tags)})

    return valid_samples


def build_unified_dataset(
    output_dir: Path | None = None,
    val_split_ratio: float = 0.05,
    test_split_ratio: float = 0.05,
    seed: int = 42,
    include_existing_combined: bool = True,
    resplit: bool = False,
) -> tuple[Path, Path, Path, int, int, int]:
    """Gather all dataset files, clean, deduplicate, back up, and export to train, validation, and test sets."""
    output_dir = output_dir or PROJECT_ROOT / "datasets" / "clean"
    train_dir = output_dir / "train"
    val_dir = output_dir / "validation"
    test_dir = output_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_file = train_dir / "train_combined.jsonl"
    val_file = val_dir / "validation_combined.jsonl"
    test_file = test_dir / "test_combined.jsonl"

    search_dirs = [
        PROJECT_ROOT / "datasets" / "raw" / "train",
        PROJECT_ROOT / "datasets" / "clean" / "train",
        PROJECT_ROOT / "datasets" / "trained" / "train",
    ]

    supported_extensions = {"*.jsonl", "*.json", "*.csv", "*.tsv", "*.parquet"}
    candidate_files: list[Path] = []
    for s_dir in search_dirs:
        if s_dir.exists():
            for ext in supported_extensions:
                for f in s_dir.glob(ext):
                    if not f.name.endswith(".bak"):
                        if include_existing_combined or "combined" not in f.name:
                            candidate_files.append(f)

    # Sort uniquely
    candidate_files = sorted(list(set(candidate_files)), key=lambda x: x.name)
    if resplit:
        # Explicitly retire the old split and recover its held-out rows for the new one.
        for split_name, filename in [("validation", "validation_combined.jsonl"), ("test", "test_combined.jsonl")]:
            for root in {output_dir, PROJECT_ROOT / "datasets" / "clean"}:
                path = root / split_name / filename
                if path.exists() and path not in candidate_files:
                    candidate_files.append(path)

    print("=" * 70)
    print("BEACON MULTI-FORMAT MYANMAR DATASET PIPELINE")
    print("=" * 70)
    print(f"Found {len(candidate_files)} dataset files to process.")

    all_samples: list[dict] = []
    for f in candidate_files:
        samples = load_and_clean_file(f)
        all_samples.extend(samples)
        print(f"  [+] {f.name:<34} ({f.suffix:<8}) -> {len(samples):>5} valid Burmese samples")

    print("-" * 70)
    print(f"Total raw samples collected: {len(all_samples):,}")

    # Deduplicate complete conversations without discarding different later turns.
    seen_pairs: set[str] = set()
    deduped_samples: list[dict] = []

    for s in all_samples:
        key = conversation_key(s)

        if key not in seen_pairs:
            seen_pairs.add(key)
            deduped_samples.append(s)

    duplicates_removed = len(all_samples) - len(deduped_samples)
    print(f"Duplicates removed:          {duplicates_removed:,}")
    print(f"Unique clean samples:        {len(deduped_samples):,}")

    if not resplit and (val_file.exists() or test_file.exists()):
        if not (val_file.exists() and test_file.exists()):
            raise ValueError("Both held-out files are required. Use --resplit to create a new split.")
        # Preserve existing held-out rows exactly, including metadata.
        val_samples = load_file_records(val_file)
        test_samples = load_file_records(test_file)
        assert_disjoint([], val_samples, test_samples)
        excluded = {prompt_key(s) for s in [*val_samples, *test_samples]}
        train_samples = [s for s in deduped_samples if prompt_key(s) not in excluded]
    else:
        train_samples, val_samples, test_samples = grouped_split(
            deduped_samples, val_split_ratio, test_split_ratio, seed
        )
    assert_disjoint(train_samples, val_samples, test_samples)
    if not train_samples:
        raise ValueError("No training samples remain after excluding held-out prompt groups.")
    # Back up all artifacts only after validation succeeds.
    for path in (train_file, val_file, test_file):
        if path.exists() and path.stat().st_size:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

    with train_file.open("w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with val_file.open("w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with test_file.open("w", encoding="utf-8") as f:
        for s in test_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("=" * 70)
    print("DATASET COMPILATION COMPLETE")
    print("=" * 70)
    print(f"Train File:      {train_file} ({len(train_samples):,} samples)")
    print(f"Validation File: {val_file} ({len(val_samples):,} samples)")
    print(f"Test File:       {test_file} ({len(test_samples):,} samples)")
    print("=" * 70)

    return train_file, val_file, test_file, len(train_samples), len(val_samples), len(test_samples)


if __name__ == "__main__":
    # Legacy ingestion is opt-in: never silently regenerate the retired corpus.
    if "--legacy" not in sys.argv:
        if len(sys.argv) > 1:
            raise SystemExit("Use scripts/build_seminar_dataset.py for the active release; legacy options require --legacy.")
        from build_seminar_dataset import build as build_seminar
        build_seminar()
        raise SystemExit(0)
    sys.argv.remove("--legacy")
    parser = argparse.ArgumentParser(description="Build clean combined Burmese dataset from diverse formats (train, validation, test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio (default: 0.05)")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test split ratio (default: 0.05)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination clean dataset directory")
    parser.add_argument("--resplit", action="store_true", help="Explicitly retire old holdouts and create a grouped split")
    parser.add_argument(
        "--no-existing",
        action="store_true",
        help="Exclude existing train_combined.jsonl from compilation",
    )
    args = parser.parse_args()

    build_unified_dataset(
        output_dir=args.output_dir,
        resplit=args.resplit,
        val_split_ratio=args.val_ratio,
        test_split_ratio=args.test_ratio,
        seed=args.seed,
        include_existing_combined=not args.no_existing,
    )
