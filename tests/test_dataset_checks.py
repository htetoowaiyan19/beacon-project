import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from utils.dataset_checks import assert_disjoint, conversation_key, grouped_split


def sample(prompt, answer="answer"):
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}


def test_different_answers_to_same_prompt_stay_together():
    samples = [sample(str(i)) for i in range(30)] + [sample("1", "other answer")]
    train, val, test = grouped_split(samples, 0.1, 0.1, 42)
    assert_disjoint(train, val, test)
    assert len(train) + len(val) + len(test) == len(samples)
    assert (train, val, test) == grouped_split(list(reversed(samples)), 0.1, 0.1, 42) or (
        [{conversation_key(x) for x in part} for part in (train, val, test)] ==
        [{conversation_key(x) for x in part} for part in grouped_split(list(reversed(samples)), 0.1, 0.1, 42)]
    )


def test_small_dataset_keeps_all_three_splits():
    train, val, test = grouped_split([sample(str(i)) for i in range(3)], 0.05, 0.05, 42)
    assert len(train) == len(val) == len(test) == 1


def test_leakage_detected_even_when_answers_differ():
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint([sample(" Same  prompt ")], [sample("same prompt", "different")])


def test_full_conversation_identity_keeps_later_turns():
    a = sample("opening")
    b = {"messages": a["messages"] + sample("follow up")["messages"]}
    assert conversation_key(a) != conversation_key(b)


def test_burmese_persona_does_not_admit_english_only_data(tmp_path):
    import json
    from build_dataset import load_and_clean_file
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(sample("hello", "good morning")), encoding="utf-8")
    assert load_and_clean_file(path) == []


def test_malformed_conversation_is_rejected():
    from build_dataset import parse_sample_to_messages
    row = sample("မေးခွန်း", "အဖြေ")
    row["messages"].append({"role": "user", "content": "မပြီးသေး"})
    assert parse_sample_to_messages(row) is None
