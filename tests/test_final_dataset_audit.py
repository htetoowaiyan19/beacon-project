import copy
import json
from pathlib import Path
import pytest
from scripts.audit_final_dataset import content_flags
from scripts.build_seminar_dataset import apply_content_reviews, content_hash, DEST
from scripts.utils.active_dataset import active_release
from scripts.utils.persona import DEFAULT_SYSTEM_PROMPT


def row(identity='a', answer='A useful answer'):
    return {'id': identity, 'tags': 'IT', 'changes': [], 'messages': [
        {'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT},
        {'role': 'user', 'content': 'What is bit stuffing?'}, {'role': 'assistant', 'content': answer}]}


def codes(record):
    return {f['code'] for f in content_flags(record)}


def test_emoji_joiners_are_preserved_but_stray_format_marks_flagged():
    assert 'invisible_format_character' not in codes(row(answer='Cook together 👩‍🍳'))
    assert 'invisible_format_character' in codes(row(answer='A\u200c useful answer'))


def test_garbled_text_control_tokens_and_echo_are_flagged():
    assert 'unexpected_script' in codes(row(answer='An accidental 偏 in the reply'))
    assert 'embedded_chat_control_token' in codes(row(answer='<|im_end|>'))
    assert 'possible_placeholder' in codes(row(answer='TODO: fill this in'))
    assert 'unclosed_code_fence' in codes(row(answer='```python\nprint(1)'))


def test_corrected_duplicate_does_not_reenter_release_uncorrected():
    a, b = row(), row('b')
    replacement = copy.deepcopy(a['messages']); replacement[2]['content'] = 'Corrected answer'
    review = {'id': 'a', 'content_sha256': content_hash(a), 'status': 'approved',
              'reviewer': 'codex_final_scan', 'replacement_messages': replacement}
    result = apply_content_reviews([(a, {'decision': 'pending'}), (b, {'decision': 'pending'})], [review])
    assert all(r['messages'][2]['content'] == 'Corrected answer' for r, _ in result)
    assert a['review_status'] == 'assistant_corrected_pending_native_review'


def test_tag_correction_preserves_conversation_and_provenance():
    a = row()
    original_messages = copy.deepcopy(a['messages'])
    review = {'id': 'a', 'content_sha256': content_hash(a), 'status': 'approved',
              'replacement_tags': 'automotive_basics'}
    result = apply_content_reviews([(a, {'decision': 'pending'})], [review])
    assert len(result) == 1 and a['tags'] == 'automotive_basics'
    assert a['messages'] == original_messages


def test_conflicting_duplicate_corrections_fail():
    a, b = row(), row('b')
    reviews = []
    for r, answer in [(a, 'One correction'), (b, 'Different correction')]:
        messages = copy.deepcopy(r['messages']); messages[2]['content'] = answer
        reviews.append({'id': r['id'], 'content_sha256': content_hash(r), 'status': 'approved',
                        'replacement_messages': messages})
    with pytest.raises(ValueError, match='Conflicting corrections'):
        apply_content_reviews([(a, {}), (b, {})], reviews)


def test_final_release_meets_count_and_complete_scan():
    summary = json.loads((DEST / 'review/final_audit/summary.json').read_text(encoding='utf-8'))
    assert summary['counts']['train']['records'] >= 12000
    assert summary['mechanical_checks_passed'] and summary['tokenizer_checked']
    assert summary['input_hashes_verified']
    assert summary['warning_occurrences'].get('unexpected_script', 0) == 0
    assert summary['warning_occurrences'].get('invisible_format_character', 0) == 0
    assert summary['semantic_accuracy_certified'] is False
