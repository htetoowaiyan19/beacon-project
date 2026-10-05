"""Build the explicitly authored Burmese style seed; never ingest archived corpora."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata
from difflib import SequenceMatcher

if __package__:
    from .utils.dataset_checks import assert_disjoint, conversation_key
else:
    from utils.dataset_checks import assert_disjoint, conversation_key

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'datasets/sources/burmese_fluency_v2'
DEST = ROOT / 'datasets/releases/burmese_fluency_v2'
DEFAULT_REVIEW = ROOT / 'datasets/reviews/burmese_fluency_v2.decisions.json'
if __package__:
    from .utils.persona import DEFAULT_SYSTEM_PROMPT as SYSTEM
else:
    from utils.persona import DEFAULT_SYSTEM_PROMPT as SYSTEM


def parse_source(text):
    rows, current = [], None
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        if line.startswith('@@ '):
            fields = [v.strip() for v in line[3:].split('|')]
            if len(fields) != 4:
                raise ValueError('Header needs id, tone, family, split')
            identity, tone, family, split = fields
            current = {'id': identity, 'tone': tone, 'scenario_family': family,
                       'split': split, 'source': 'assistant_authored_synthetic',
                       'review_status': 'requires_native_speaker_review',
                       'messages': [{'role': 'system', 'content': SYSTEM}]}
            rows.append(current)
        elif current is None:
            raise ValueError('Content before first header')
        elif line.startswith(('U: ', 'A: ')):
            current['messages'].append({'role': 'user' if line[0] == 'U' else 'assistant',
                                        'content': unicodedata.normalize('NFC', line[3:].strip())})
        else:
            if len(current['messages']) < 2:
                raise ValueError('Continuation without a turn')
            current['messages'][-1]['content'] += '\n' + unicodedata.normalize('NFC', line.rstrip())
    validate(rows)
    return rows


def read_source(source=SOURCE):
    paths = sorted(source.glob('*.txt')) if source.is_dir() else [source]
    if not paths:
        raise ValueError('No authored source files found')
    return parse_source('\n'.join(p.read_text(encoding='utf-8') for p in paths)), paths


def review_hash(row):
    # Includes system text, tone, and split as well as all conversation turns.
    content = {k: row[k] for k in ('id', 'tone', 'scenario_family', 'split', 'messages')}
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def apply_reviews(rows, review_file=None):
    by_id = {r['id']: r for r in rows}
    decisions = {}
    if review_file:
        reviews = json.loads(review_file.read_text(encoding='utf-8'))
        if not isinstance(reviews, list):
            raise ValueError('Review file must contain a JSON array')
        for review in reviews:
            identity = review['id']
            if identity in decisions or identity not in by_id:
                raise ValueError('Duplicate or unknown reviewed ID')
            if review['content_sha256'] != review_hash(by_id[identity]):
                raise ValueError(f'Stale review after source changes: {identity}')
            if review['status'] not in ('pending', 'approved', 'rejected'):
                raise ValueError('Invalid review status')
            if not isinstance(review.get('notes', ''), str):
                raise ValueError('Review notes must be text')
            decisions[identity] = review
    for row in rows:
        decision = decisions.get(row['id'], {})
        row['review_status'] = decision.get('status', 'pending')
        row['review_notes'] = decision.get('notes', '')
    return [r for r in rows if r['review_status'] != 'rejected']


def validate(rows):
    ids, conversations, families = set(), set(), {}
    splits = {name: [] for name in ('train', 'validation', 'test')}
    for row in rows:
        if row['id'] in ids or row['split'] not in splits:
            raise ValueError('Duplicate id or invalid split')
        ids.add(row['id'])
        if row['tone'] not in ('polite', 'casual', 'bff'):
            raise ValueError('Invalid tone')
        turns = row['messages'][1:]
        if len(turns) < 4 or len(turns) % 2:
            raise ValueError('Each example needs at least two complete exchanges')
        for i, turn in enumerate(turns):
            text = turn['content']
            if turn['role'] != ('user' if i % 2 == 0 else 'assistant') or not text.strip():
                raise ValueError('Invalid role order or empty content')
            if '\ufffd' in text or any(ord(c) < 32 and c not in '\n\t' for c in text):
                raise ValueError('Damaged text or control character')
            if any('\u4e00' <= c <= '\u9fff' for c in text):
                raise ValueError('Unexpected CJK text; review the source')
            if turn['role'] == 'assistant' and not any('\u1000' <= c <= '\u109f' for c in text):
                raise ValueError('Assistant response has no Burmese text')
        key = conversation_key(row)
        if key in conversations:
            raise ValueError('Duplicate conversation')
        conversations.add(key)
        family = row['scenario_family']
        if families.setdefault(family, row['split']) != row['split']:
            raise ValueError('Scenario family leaks across splits')
        splits[row['split']].append(row)
    if not all(splits.values()):
        raise ValueError('All three splits must contain examples')
    assert_disjoint(**splits)
    return splits


def build(source=SOURCE, destination=DEST, tokenizer_path=None, max_length=2048, review_file=None):
    if review_file is None and source == SOURCE and DEFAULT_REVIEW.exists():
        review_file = DEFAULT_REVIEW
    all_rows, paths = read_source(source)
    rows = apply_reviews(all_rows, review_file)
    splits = validate(rows)
    lengths = {}
    if tokenizer_path:
        from transformers import AutoTokenizer
        from trl.chat_template_utils import get_training_chat_template
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        template = get_training_chat_template(tokenizer)
        for row in rows:
            encoded = tokenizer.apply_chat_template(row['messages'], chat_template=template,
                tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
            lengths[row['id']] = len(encoded['input_ids'])
            if lengths[row['id']] > max_length or not any(encoded['assistant_masks']):
                raise ValueError(f"Invalid token length or missing targets: {row['id']}")
            # Verify each assistant reply contributes labels, not only the conversation as a whole.
            previous = 0
            for end in range(3, len(row['messages']) + 1, 2):
                prefix = tokenizer.apply_chat_template(row['messages'][:end], chat_template=template,
                    tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
                targets = sum(prefix['assistant_masks'])
                if targets <= previous:
                    raise ValueError(f"Unsupervised assistant reply: {row['id']}, turn {end}")
                previous = targets
    manifest = {'version': 'burmese_fluency_v2',
                'sources': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'status': 'reviewed_candidate' if all(r['review_status'] == 'approved' for r in rows) else 'candidate_pending_user_review',
                'review_counts': dict(Counter(r['review_status'] for r in all_rows)),
                'review_file_sha256': hashlib.sha256(review_file.read_bytes()).hexdigest() if review_file else None,
                'counts': {s: len(r) for s, r in splits.items()},
                'tones': dict(Counter(r['tone'] for r in rows)),
                'tones_by_split': {s: dict(Counter(r['tone'] for r in rs)) for s, rs in splits.items()},
                'assistant_replies': sum((len(r['messages']) - 1) // 2 for r in rows),
                'scenario_families': len({r['scenario_family'] for r in rows}),
                'tokenizer_checked': str(tokenizer_path) if tokenizer_path else None,
                'max_tokens': max(lengths.values()) if lengths else None,
                'max_length': max_length, 'files': {}}
    for split, records in splits.items():
        path = destination / split / f'{split}_combined.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        manifest['files'][split] = {'path': str(path.relative_to(destination)),
                                   'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    review_dir = destination / 'review'
    review_dir.mkdir(exist_ok=True)
    review_records = [{**r, 'content_sha256': review_hash(r), 'tokens': lengths.get(r['id'])} for r in all_rows]
    (review_dir / 'conversations.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in review_records), encoding='utf-8')
    template = [{'id': r['id'], 'content_sha256': review_hash(r), 'status': r['review_status'],
                 'notes': r.get('review_notes', '')} for r in all_rows]
    (review_dir / 'decisions.template.json').write_text(json.dumps(template, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    # Similarity flags are review aids, not semantic accuracy judgments or automatic deletions.
    near = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            a, b = left['messages'][1]['content'], right['messages'][1]['content']
            score = SequenceMatcher(None, a, b, autojunk=False).ratio()
            if score >= 0.80:
                near.append({'left': left['id'], 'right': right['id'], 'opening_similarity': round(score, 3),
                             'cross_split': left['split'] != right['split']})
    answers = Counter(m['content'] for r in rows for m in r['messages'] if m['role'] == 'assistant')
    quality = {'near_opening_prompts': near,
               'repeated_exact_answers': [{'text': t, 'count': n} for t, n in answers.items() if n > 1],
               'tokens_by_id': lengths, 'automatic_checks': ['alternating_roles', 'nonempty_turns',
                   'no_duplicate_conversations', 'no_opening_prompt_split_overlap', 'no_family_split_overlap',
                   'no_replacement_or_control_characters', 'no_unexpected_CJK'],
               'token_checks_passed': bool(tokenizer_path),
               'semantic_or_native_fluency_certified': False}
    (review_dir / 'quality.json').write_text(json.dumps(quality, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    page_template = (ROOT / 'scripts/templates/dataset_review.html').read_text(encoding='utf-8')
    payload = json.dumps(review_records, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    (review_dir / 'index.html').write_text(page_template.replace('__RECORDS_JSON__', payload), encoding='utf-8')
    review_md = ['# Burmese fluency v2: conversation review', '',
                 'Edit source text, then rebuild. Record approval in decisions JSON or the HTML reviewer.', '']
    for row in all_rows:
        review_md += [f"## {row['id']} — {row['split']} / {row['tone']}", '',
                      f"Family: {row['scenario_family']} | Status: {row['review_status']}", '']
        for message in row['messages']:
            review_md += [f"**{message['role']}**", '', message['content'], '']
    (review_dir / 'conversations.md').write_text('\n'.join(review_md), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'models/qwen3-4b')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--review-file', type=Path, help='Exported review decisions; rejects are excluded and stale decisions fail')
    args = parser.parse_args()
    print(json.dumps(build(tokenizer_path=args.tokenizer, max_length=args.max_length, review_file=args.review_file), indent=2))
