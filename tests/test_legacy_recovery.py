import json
from pathlib import Path
from scripts.recover_legacy_dataset import identity, quality_reason, recover_friend, DEST
from scripts.utils.dataset_checks import assert_disjoint


def test_canonical_prompt_and_observed_defects():
    assert identity('Hello, WORLD!') == identity('hello world')
    assert identity('မြန်မာ') != identity('မနမာ')
    assert quality_reason('test', 'မစားရသေးပါဘူး', 'translation')
    assert quality_reason('ရာထူးက ဘာလဲ', 'ပါမောက္ခပါ', 'education')
    assert quality_reason('ဘာလုပ်မလဲ', 'ငါ မင်းဆီ ဖုန်းဆက်မယ်', 'daily')
    assert quality_reason('မင်္ဂလာပါ', 'မင်္ဂလာပါ', 'greeting') is None


def test_friend_recovery_requires_original_pair():
    templates = [('question', 'answer')]
    assert recover_friend('hello question?', 'prefix answer closing', templates) == ('question', 'answer')
    assert recover_friend('hello question?', 'different reply', templates) is None


def test_recovered_release_accounting_and_splits():
    manifest = json.loads((DEST / 'manifest.json').read_text(encoding='utf-8'))
    decisions = [json.loads(x) for x in (DEST / 'review/recovery_decisions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(decisions) == 7220
    included = {d['id'] for d in decisions if d['decision'] == 'included_candidate'}
    assert len(included) == manifest['legacy_recovered_rows']
    assert manifest['legacy_recovered_rows'] + sum(manifest['exclusion_counts'].values()) == 7220
    splits = {s: [json.loads(x) for x in (DEST / s / f'{s}_combined.jsonl').read_text(encoding='utf-8').splitlines()]
              for s in ('train', 'validation', 'test')}
    assert_disjoint(**splits)
    legacy_ids = {r['id'] for rows in splits.values() for r in rows if r['source'] == 'recovered_legacy'}
    assert legacy_ids == included
    for s, rows in splits.items():
        assert len(rows) == manifest['counts'][s]
        for r in rows:
            if r['source'] == 'burmese_fluency_v2': assert r['original_split'] == s
    assert manifest['max_tokens'] <= 2048
