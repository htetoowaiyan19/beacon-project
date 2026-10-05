import copy
import json
import pytest
from scripts.build_fluency_dataset import SOURCE, parse_source, validate, build, read_source, apply_reviews, review_hash


def test_authored_corpus_and_reproducible_build(tmp_path):
    rows = read_source()[0]
    assert len(rows) == 150
    assert all(len(r['messages']) >= 5 for r in rows)
    first = build(destination=tmp_path)
    second = build(destination=tmp_path)
    assert first == second


def test_family_leakage_rejected():
    rows = read_source()[0]
    paired = next(r for r in rows if r['id'] == 'tone01c')
    paired['split'] = 'test'
    with pytest.raises(ValueError, match='family leaks'):
        validate(rows)


def test_bad_role_and_duplicate_rejected():
    rows = read_source()[0]
    broken = copy.deepcopy(rows)
    broken[0]['messages'][2]['role'] = 'user'
    with pytest.raises(ValueError, match='role order'):
        validate(broken)
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match='Duplicate id'):
        validate(rows)


def test_quotes_and_multiline_preserved():
    rows = read_source()[0]
    assert '“' in rows[0]['messages'][1]['content']
    assert '\n' in next(r for r in rows if r['id'] == 'format01')['messages'][2]['content']


def test_rejected_rows_excluded_and_stale_reviews_fail(tmp_path):
    rows, _ = read_source()
    review = {'id': rows[0]['id'], 'content_sha256': review_hash(rows[0]), 'status': 'rejected', 'notes': 'Needs correction'}
    path = tmp_path / 'decisions.json'
    path.write_text(json.dumps([review]), encoding='utf-8')
    kept = apply_reviews(rows, path)
    assert len(kept) == 149 and review['id'] not in {r['id'] for r in kept}
    rows[0]['messages'][2]['content'] += ' edited'
    with pytest.raises(ValueError, match='Stale review'):
        apply_reviews(rows, path)


def test_duplicate_review_and_invalid_tone_fail(tmp_path):
    rows, _ = read_source()
    review = {'id': rows[0]['id'], 'content_sha256': review_hash(rows[0]), 'status': 'approved', 'notes': ''}
    path = tmp_path / 'reviews.json'
    path.write_text(json.dumps([review, review]), encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate or unknown'):
        apply_reviews(rows, path)
    rows[0]['tone'] = 'unknown'
    with pytest.raises(ValueError, match='Invalid tone'):
        validate(rows)
