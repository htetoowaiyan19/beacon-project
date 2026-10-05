"""Deterministic dataset identity, grouped splits and leakage checks."""
from __future__ import annotations
import hashlib
import json
import random
import unicodedata


def normalize_identity(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split()).casefold()


def conversation_key(sample: dict) -> str:
    turns = [(m["role"], normalize_identity(m["content"]))
             for m in sample["messages"] if m["role"] != "system"]
    return hashlib.sha256(json.dumps(turns, ensure_ascii=False).encode("utf-8")).hexdigest()


def prompt_key(sample: dict) -> str:
    # Group all conversations sharing their opening question, including different answers.
    prompt = next(m["content"] for m in sample["messages"] if m["role"] == "user")
    return hashlib.sha256(normalize_identity(prompt).encode("utf-8")).hexdigest()


def assert_disjoint(train, validation, test=()) -> None:
    names = {"train": train, "validation": validation, "test": test}
    keys = {name: {prompt_key(s) for s in samples} for name, samples in names.items()}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = keys[left] & keys[right]
        if overlap:
            raise ValueError(f"Dataset leakage: {len(overlap)} shared opening prompts between {left} and {right}. "
                             "Build a new grouped split before training; old holdouts must be retired.")


def grouped_split(samples, val_ratio: float, test_ratio: float, seed: int):
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("Validation and test ratios must be positive and sum to less than 1.")
    groups = {}
    for sample in samples:
        groups.setdefault(prompt_key(sample), []).append(sample)
    ordered = sorted(groups)
    if len(ordered) < 3:
        raise ValueError("At least three distinct opening prompts are required for three splits.")
    random.Random(seed).shuffle(ordered)
    targets = [max(1, int(len(samples) * val_ratio)), max(1, int(len(samples) * test_ratio))]
    partitions = [[], [], []]  # validation, test, train
    index = 0
    for offset, key in enumerate(ordered):
        if index < 2 and len(partitions[index]) >= targets[index]:
            index += 1
        # Always leave at least one group for each remaining partition.
        if len(ordered) - offset <= 2 - index:
            index += 1
        partitions[index].extend(groups[key])
    validation, test, train = partitions
    assert_disjoint(train, validation, test)
    return train, validation, test
