import copy
import json
from scripts.build_seminar_dataset import (
    decode_records, normalize_record, grouped_components, assign_splits, prompt_identity,
    content_hash, apply_content_reviews, DEST,
)
import pytest


def row(identity, question, answer='မြန်မာလို အဖြေပါ။'):
    return {'id': identity, 'tags': 'IT', 'changes': [], 'messages': [
        {'role': 'system', 'content': 'system'}, {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': answer}]}


def test_repairs_literal_newline_and_concatenated_records():
    data = '{"text":"first\nsecond"}{"text":"third"}'
    records = decode_records(data)
    assert len(records) == 2 and records[0][0]['text'] == 'first\nsecond'
    assert len(records[0][1]['repairs']) == 2
    with pytest.raises(json.JSONDecodeError): decode_records('{"broken":')


def test_repeated_prefix_requires_identical_content():
    r = row('a', 'မေးခွန်း')
    r['messages'] = copy.deepcopy(r['messages'][:2]) + r['messages']
    messages, _, changes = normalize_record(r)
    assert len(messages) == 3
    assert 'removed_identical_repeated_system_user_prefix' in changes
    r['messages'][1]['content'] = 'different'
    with pytest.raises(ValueError, match='invalid_role_order'): normalize_record(r)


def test_symbols_and_group_leakage():
    assert prompt_identity(row('a', 'C++ ဆိုတာဘာလဲ')) != prompt_identity(row('b', 'C ဆိုတာဘာလဲ'))
    records = [row(str(i), f'This is an intentionally very long networking question number {i}') for i in range(8)]
    records += [row('same-question', records[0]['messages'][1]['content'], 'တခြားအဖြေ')]
    components, links = grouped_components(records)
    assert any({0, 8} <= set(group) for group in components)
    splits, mapping = assign_splits(records, components)
    for link in links: assert mapping[link['left']]['split'] == mapping[link['right']]['split']
    assert sum(map(len, splits.values())) == len(records)


def test_stale_review_and_rejected_duplicate():
    a, b = row('a', 'မေးခွန်း'), row('b', 'မေးခွန်း')
    decision = {'id': 'a', 'content_sha256': content_hash(a), 'status': 'rejected'}
    assert apply_content_reviews([(a, {}), (b, {})], [decision]) == []
    decision['content_sha256'] = 'stale'
    with pytest.raises(ValueError, match='Stale'): apply_content_reviews([(a, {})], [decision])


def test_release_accounting_groups_and_human_provenance():
    manifest = json.loads((DEST / 'manifest.json').read_text(encoding='utf-8'))
    assert sum(manifest['fresh_records_by_file'].values()) == 9647
    assert manifest['fresh_records_by_file']['041026data330.jsonl'] == 330
    assert manifest['fresh_records_by_file']['120926data339 (2).jsonl'] == 1129
    decisions = [json.loads(x) for x in (DEST / 'review/decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(d['decision'] == 'included' for d in decisions) == sum(manifest['counts'].values())
    assert all(d['decision'] in ('included', 'excluded') for d in decisions)
    mapping = json.loads((DEST / 'review/split_groups.json').read_text(encoding='utf-8'))
    seen = {}
    for value in mapping.values():
        assert seen.setdefault(value['group_id'], value['split']) == value['split']
    for split in manifest['counts']:
        records = [json.loads(x) for x in (DEST / split / f'{split}_combined.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(records) == manifest['counts'][split]
        assert all(mapping[r['id']]['split'] == split for r in records)


def test_unescaped_content_quotes_are_preserved():
    text = '{"messages":[{"role":"user","content":"What does "Reset" mean?"},{"role":"assistant","content":"Use "Reset" carefully."}],"tags":"IT"}'
    rows = decode_records(text)
    assert rows[0][0]['messages'][0]['content'] == 'What does "Reset" mean?'
    assert rows[0][0]['messages'][1]['content'] == 'Use "Reset" carefully.'
    assert rows[0][1]['repairs'] == ['escaped_unescaped_quotes_in_content']
