import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_multiturn_eval_preserves_context_and_final_reference(tmp_path):
    from evaluate_model import load_prompts
    messages = [{"role": "user", "content": "opening"}, {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "follow-up"}, {"role": "assistant", "content": "gold answer"}]
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"messages": messages}) + "\n", encoding="utf-8")
    prompt = load_prompts(path)[0]
    assert prompt["prompt"] == "follow-up"
    assert prompt["messages"] == messages[:-1]
    assert prompt["reference"] == "gold answer"


def test_reference_scores_are_explicit_and_missing_references_are_unscored():
    from evaluate_model import score_reference
    assert score_reference("မြန်မာ  စာ", "မြန်မာ စာ")["normalized_exact_match"] == 1
    assert score_reference("wrong", "right")["normalized_exact_match"] == 0
    assert score_reference("answer", "") == {}
