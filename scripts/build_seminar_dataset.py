"""Combine human-collected IT data, recovered legacy data, and authored Burmese dialogues."""
from __future__ import annotations
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

if __package__:
    from .build_fluency_dataset import ROOT, SYSTEM
    from .utils.dataset_checks import conversation_key, assert_disjoint, normalize_identity
else:
    from build_fluency_dataset import ROOT, SYSTEM
    from utils.dataset_checks import conversation_key, assert_disjoint, normalize_identity

VERSION = 'it_seminar_v3'
DEST = ROOT / 'datasets/releases' / VERSION
NORMALIZED = ROOT / 'datasets/sources/human_collected_v3'
RECOVERED = ROOT / 'datasets/releases/burmese_recovered_v1'
REVIEW_FILE = ROOT / 'datasets/reviews/it_seminar_v3.decisions.json'


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def decode_records(text):
    """Recover concatenated objects/literal newlines, never skip malformed data silently."""
    decoder = json.JSONDecoder(strict=False)
    newlines = [i for i, c in enumerate(text) if c == '\n']
    records, offset = [], 0
    while offset < len(text):
        if text[offset].isspace():
            offset += 1
            continue
        quote_repair = False
        try:
            value, end = decoder.raw_decode(text, offset)
        except json.JSONDecodeError:
            end = text.find('\n', offset)
            if end < 0: end = len(text)
            raw_line = text[offset:end]
            fixed = re.sub(r'("content"\s*:\s*")(.*?)("\s*})',
                           lambda m: m[1] + re.sub(r'(?<!\\)"', r'\\"', m[2]) + m[3], raw_line)
            if fixed == raw_line:
                raise
            value = json.loads(fixed)
            quote_repair = True
        if not isinstance(value, dict):
            raise ValueError('Every record must be a JSON object')
        raw = text[offset:end]
        repairs = ['escaped_unescaped_quotes_in_content'] if quote_repair else []
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            if not quote_repair: repairs.append('escaped_literal_control_character_in_JSON_string')
        start_line = bisect_right(newlines, offset) + 1
        if records and records[-1][1]['end_line'] == start_line:
            repairs.append('separated_concatenated_JSON_objects')
            records[-1][1]['repairs'].append('separated_concatenated_JSON_objects')
        records.append((value, {'start_line': start_line, 'end_line': bisect_right(newlines, end - 1) + 1,
                               'original_record_sha256': sha(raw), 'repairs': repairs}))
        offset = end
    return records


def normalize_record(row):
    messages = row['messages']
    changes = []
    if ([m['role'] for m in messages] == ['system', 'user', 'system', 'user', 'assistant']
            and messages[:2] == messages[2:4]):
        messages = messages[2:]
        changes.append('removed_identical_repeated_system_user_prefix')
    turns = messages[1:] if messages and messages[0]['role'] == 'system' else messages
    if not turns or len(turns) % 2 or any(m['role'] != ('user' if i % 2 == 0 else 'assistant') for i, m in enumerate(turns)):
        raise ValueError('invalid_role_order')
    normalized = [{'role': 'system', 'content': SYSTEM}]
    for message in turns:
        content = unicodedata.normalize('NFC', message['content']).strip()
        if not content or '\ufffd' in content or any(ord(c) < 32 and c not in '\n\t' for c in content):
            raise ValueError('empty_or_damaged_text')
        normalized.append({'role': message['role'], 'content': content})
    tags = row.get('tags', 'unclassified')
    if isinstance(tags, list):
        tags = '|'.join(tags)
        changes.append('normalized_tag_list_to_string')
    fixed = {'Operatin်g System': 'Operating System', 'network_physical': 'networking_physical'}.get(tags, tags)
    if fixed != tags: changes.append('corrected_tag_typo')
    changes += ['standardized_system_prompt', 'NFC_and_outer_whitespace']
    return normalized, fixed, changes


def prompt_identity(row):
    # Preserve meaningful symbols such as C++, decimals and operators.
    return normalize_identity(row['messages'][1]['content']).rstrip('။?! .')


def content_hash(row):
    return sha(json.dumps({'messages': row['messages'], 'tags': row['tags']}, ensure_ascii=False, sort_keys=True))


def apply_content_reviews(candidates, reviews):
    by_id = {r['id']: (r, d) for r, d in candidates}
    original_keys = {r['id']: conversation_key(r) for r, _ in candidates}
    corrections = {}
    seen, rejected_content = set(), set()
    for review in reviews:
        identity = review['id']
        if identity in seen or identity not in by_id:
            raise ValueError('Duplicate or unknown review ID: ' + identity)
        seen.add(identity)
        row, decision = by_id[identity]
        if review['content_sha256'] != content_hash(row):
            raise ValueError('Stale review: ' + identity)
        if review['status'] not in ('pending', 'approved', 'rejected'):
            raise ValueError('Invalid review status')
        if review['status'] == 'rejected':
            rejected_content.add(conversation_key(row))
        elif review['status'] == 'approved':
            if 'replacement_tags' in review:
                tag = review['replacement_tags']
                if not isinstance(tag, str) or not tag.strip():
                    raise ValueError('Invalid replacement tag: ' + identity)
                row['tags'] = tag.strip()
                row['changes'].append('review_overlay_tag_correction')
            if 'replacement_messages' in review:
                messages, _, changes = normalize_record({'messages': review['replacement_messages'], 'tags': row['tags']})
                key = original_keys[identity]
                if key in corrections and corrections[key] != messages:
                    raise ValueError('Conflicting corrections for identical conversation: ' + identity)
                corrections[key] = messages
                row['messages'] = messages
                row['changes'] += ['review_overlay_text_correction']
            row['review_status'] = ('assistant_corrected_pending_native_review' if review.get('reviewer') == 'codex_final_scan'
                                    else 'approved_by_project_reviewer')
    for row, decision in candidates:
        key = original_keys[row['id']]
        if key in rejected_content:
            decision.update(decision='excluded', reason='rejected_by_project_reviewer_or_identical_copy')
        elif key in corrections and row['messages'] != corrections[key]:
            row['messages'] = [dict(message) for message in corrections[key]]
            row['changes'].append('propagated_review_correction_to_identical_copy')
            row['review_status'] = 'corrected_identical_copy_pending_native_review'
    return [(r, d) for r, d in candidates if d['decision'] != 'excluded']


def grouped_components(rows, families=None, threshold=0.85):
    parent = list(range(len(rows)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)
    exact, family_ids, postings, grams_by_id = {}, {}, defaultdict(list), {}
    links = []
    for i, row in enumerate(rows):
        key = prompt_identity(row)
        if key in exact: union(i, exact[key])
        else: exact[key] = i
        family = (families or {}).get(row['id'])
        if family:
            if family in family_ids: union(i, family_ids[family])
            else: family_ids[family] = i
        text = ''.join(key.split())
        if len(text) < 25: continue
        grams = {text[j:j+5] for j in range(len(text) - 4)}
        matches = Counter(j for gram in grams for j in postings[gram])
        for j, intersection in matches.items():
            score = intersection / (len(grams) + len(grams_by_id[j]) - intersection)
            if score >= threshold:
                union(i, j)
                if prompt_identity(rows[j]) != key:
                    links.append({'left': rows[j]['id'], 'right': row['id'], 'jaccard_5gram': round(score, 4)})
        grams_by_id[i] = grams
        for gram in grams: postings[gram].append(i)
    groups = defaultdict(list)
    for i in range(len(rows)): groups[find(i)].append(i)
    return list(groups.values()), links


def assign_splits(rows, components, seed=42):
    # Stratify by the main category within each indivisible prompt/family component.
    strata = defaultdict(list)
    for group in components:
        tags = Counter(rows[i]['tags'] for i in group)
        tag = sorted(tags, key=lambda k: (-tags[k], k))[0]
        strata[tag].append(group)
    splits = {s: [] for s in ('train', 'validation', 'test')}
    group_map = {}
    for tag, groups in sorted(strata.items()):
        groups.sort(key=lambda g: min(rows[i]['id'] for i in g))
        random.Random(f'{seed}:{tag}').shuffle(groups)
        total = sum(map(len, groups))
        targets = [max(1, round(total * .05)), max(1, round(total * .05))] if len(groups) >= 3 else [0, 0]
        index, filled = 0, [0, 0]
        for position, group in enumerate(groups):
            while index < 2 and (filled[index] >= targets[index] or len(groups) - position <= 2 - index):
                index += 1
            split = ('validation', 'test', 'train')[index]
            if index < 2: filled[index] += len(group)
            gid = sha('|'.join(sorted(rows[i]['id'] for i in group)))[:16]
            for i in group:
                splits[split].append(rows[i])
                group_map[rows[i]['id']] = {'group_id': gid, 'split': split}
    for records in splits.values(): records.sort(key=lambda r: r['id'])
    assert_disjoint(**splits)
    return splits, group_map


def build():
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/qwen3-4b'), local_files_only=True)
    template = get_training_chat_template(tokenizer)
    input_files, candidates, decisions, repairs = {}, [], [], []
    fresh_counts = {}
    for path in sorted((ROOT / 'datasets/freshes').glob('*.jsonl')):
        input_files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = decode_records(path.read_text(encoding='utf-8-sig'))
        fresh_counts[path.name] = len(records)
        normalized_file = []
        for number, (original, provenance) in enumerate(records, 1):
            identity = f'human-{path.stem}-{number:05d}'
            decision = {'id': identity, 'source_file': str(path.relative_to(ROOT)), **provenance,
                        'decision': 'pending', 'reason': None}
            decisions.append(decision)
            try:
                messages, tags, changes = normalize_record(original)
            except (KeyError, TypeError, ValueError) as exc:
                decision.update(decision='excluded', reason=str(exc)); continue
            row = {'id': identity, 'messages': messages, 'tags': tags, 'source': 'team_human_collected',
                   'source_id': identity, 'source_file': str(path.relative_to(ROOT)),
                   'source_line': provenance['start_line'], 'source_sha256': provenance['original_record_sha256'],
                   'review_status': 'human_collected_per_user_not_independently_fact_checked',
                   'changes': changes + provenance['repairs']}
            normalized_file.append(row)
            candidates.append((row, decision))
            significant = [c for c in row['changes'] if c not in ('NFC_and_outer_whitespace', 'standardized_system_prompt', 'normalized_tag_list_to_string')]
            if significant: repairs.append({'id': identity, 'source_line': provenance['start_line'], 'changes': significant})
        write_rows(NORMALIZED / path.name, normalized_file)
    for split in ('train', 'validation', 'test'):
        path = RECOVERED / split / f'{split}_combined.jsonl'
        input_files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text(encoding='utf-8').splitlines():
            old = json.loads(line)
            row = {k: old[k] for k in ('id', 'messages', 'tags', 'source', 'source_id', 'source_sha256', 'review_status', 'changes')}
            row['messages'] = [{'role': 'system', 'content': SYSTEM}] + row['messages'][1:]
            row.update(source_file=str(path.relative_to(ROOT)), source_line=None)
            decision = {'id': row['id'], 'source_file': row['source_file'], 'decision': 'pending', 'reason': None}
            decisions.append(decision)
            candidates.append((row, decision))
    review_template = [{'id': r['id'], 'content_sha256': content_hash(r), 'status': 'pending', 'notes': ''} for r, _ in candidates]
    saved_reviews = {}
    for review_path in (ROOT / 'datasets/reviews/it_seminar_v1.decisions.json',
                        ROOT / 'datasets/reviews/it_seminar_v2.decisions.json', REVIEW_FILE):
        if review_path.exists():
            input_files[str(review_path.relative_to(ROOT))] = hashlib.sha256(review_path.read_bytes()).hexdigest()
            records = json.loads(review_path.read_text(encoding='utf-8'))
            if len({r['id'] for r in records}) != len(records):
                raise ValueError('Duplicate review IDs in ' + str(review_path))
            for review in records: saved_reviews[review['id']] = review
    if saved_reviews:
        candidates = apply_content_reviews(candidates, list(saved_reviews.values()))
    # Evaluate valid human records before letting them supersede a matching legacy prompt.
    eligible, lengths = [], {}
    for row, decision in candidates:
        encoded = tokenizer.apply_chat_template(row['messages'], chat_template=template, tokenize=True,
                                                return_dict=True, return_assistant_tokens_mask=True)
        lengths[row['id']] = len(encoded['input_ids'])
        if lengths[row['id']] > 2048:
            decision.update(decision='excluded', reason='over_2048_tokens_not_truncated'); continue
        if not any(encoded['assistant_masks']):
            decision.update(decision='excluded', reason='no_assistant_labels'); continue
        if len(row['messages']) > 3:
            previous = 0
            for end in range(3, len(row['messages']) + 1, 2):
                prefix = tokenizer.apply_chat_template(row['messages'][:end], chat_template=template,
                    tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
                supervised = sum(prefix['assistant_masks'])
                if supervised <= previous:
                    raise ValueError('Missing labels for assistant turn: ' + row['id'])
                previous = supervised
        eligible.append((row, decision))
    human_prompts = {prompt_identity(r) for r, _ in eligible if r['source'] == 'team_human_collected'}
    seen, rows = {}, []
    for row, decision in eligible:
        if row['source'] == 'recovered_legacy' and prompt_identity(row) in human_prompts:
            decision.update(decision='excluded', reason='superseded_by_team_answer_for_same_prompt'); continue
        key = conversation_key(row)
        if key in seen:
            decision.update(decision='excluded', reason='exact_duplicate_conversation', retained_id=seen[key]); continue
        seen[key] = row['id']
        rows.append(row)
        decision.update(decision='included', reason='passed_structural_and_token_checks')
    families = {}
    for path in (ROOT / 'datasets/releases/burmese_fluency_v2').glob('*/*_combined.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            families['style-' + row['id']] = row['scenario_family']
    components, near_links = grouped_components(rows, families)
    splits, group_map = assign_splits(rows, components)
    alternatives = defaultdict(list)
    for row in rows: alternatives[prompt_identity(row)].append(row)
    alternatives = [{'prompt': records[0]['messages'][1]['content'], 'ids': [r['id'] for r in records],
                     'answers': [r['messages'][2]['content'] for r in records],
                     'note': 'Alternative wording or possible conflict; review, not automatically judged incorrect.'}
                    for records in alternatives.values() if len(records) > 1]
    manifest = {'version': VERSION, 'status': 'training_candidate_pending_content_review',
        'objective': 'Accurate IT explanations in fluent, tone-adaptive Burmese',
        'system_prompt': SYSTEM, 'input_sha256': input_files,
        'fresh_records_by_file': fresh_counts, 'normalized_human_records': sum(fresh_counts.values()),
        'counts': {s: len(rs) for s, rs in splits.items()},
        'sources_by_split': {s: dict(Counter(r['source'] for r in rs)) for s, rs in splits.items()},
        'multi_turn_by_split': {s: sum(len(r['messages']) > 3 for r in rs) for s, rs in splits.items()},
        'tags_by_split': {s: dict(Counter(r['tags'] for r in rs)) for s, rs in splits.items()},
        'excluded_reasons': dict(Counter(d['reason'] for d in decisions if d['decision'] == 'excluded')),
        'repairs': repairs, 'prompt_components': len(components), 'near_prompt_links': len(near_links),
        'alternative_answer_groups': len(alternatives), 'seed': 42, 'max_length': 2048,
        'max_tokens': max(lengths[r['id']] for r in rows),
        'split_policy': 'Fresh grouped stratified ~90/5/5; exact prompts, >=0.85 character-5gram similarity, and authored scenario families stay together; prior split assignments retired.',
        'semantic_accuracy_certified': False, 'files': {}}
    for split, records in splits.items():
        path = DEST / split / f'{split}_combined.jsonl'
        write_rows(path, records)
        manifest['files'][split] = {'path': str(path.relative_to(DEST)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if split != 'train':
            for category, tags in {'operating_systems': {'Operating System'},
                'networking': {'networking_physical', 'networking_datalink', 'networking_cryptography', 'Computer Networking',
                              'network_application', 'network_transport', 'network_application_http'},
                'nlp': {'NLP'}, 'hardware': {'computer hardware'}, 'cybersecurity': {'cybersecurity basics'}}.items():
                write_rows(DEST / 'evaluation' / split / f'{category}.jsonl', [r for r in records if r['tags'] in tags])
            write_rows(DEST / 'evaluation' / split / 'human_collected.jsonl', [r for r in records if r['source'] == 'team_human_collected'])
            write_rows(DEST / 'evaluation' / split / 'conversation_style.jsonl', [r for r in records if r['source'] == 'burmese_fluency_v2' or r['tags'].startswith('smalltalk_')])
    # Small diagnostic files: no holdout added to training; long train records exercise memory use.
    write_rows(DEST / 'smoke/train.jsonl', sorted(splits['train'], key=lambda r: lengths[r['id']], reverse=True)[:32])
    write_rows(DEST / 'smoke/validation.jsonl', splits['validation'][:16])
    write_json(DEST / 'manifest.json', manifest)
    write_rows(DEST / 'review/decisions.jsonl', decisions)
    write_json(DEST / 'review/split_groups.json', group_map)
    write_json(DEST / 'review/near_prompt_links.json', near_links)
    write_json(DEST / 'review/alternative_answers.json', alternatives)
    included_ids = {r['id'] for r in rows}
    write_json(DEST / 'review/decisions.template.json', [r for r in review_template if r['id'] in included_ids])
    alt_md = ['# Alternative answers requiring comparison', '',
              'These are possible paraphrases or conflicts, not automatically declared errors. Keep valid alternatives; reject or correct incorrect answers through the saved review decisions file.', '']
    lookup = {r['id']: r for r in rows}
    for i, group in enumerate(alternatives, 1):
        alt_md += [f'## Group {i}', '', group['prompt'], '']
        for identity in group['ids']:
            row = lookup[identity]
            alt_md += [f"**{identity}** — {row['tags']} — {group_map[identity]['split']}", '',
                       row['messages'][2]['content'], '']
    (DEST / 'review/alternative_answers.md').write_text('\n'.join(alt_md), encoding='utf-8')
    write_json(DEST / 'review/token_lengths.json', {r['id']: lengths[r['id']] for r in rows})
    write_json(NORMALIZED / 'manifest.json', {'input_sha256': {k: v for k, v in input_files.items() if '/freshes/' in k.replace('\\', '/')},
                                            'records': fresh_counts, 'provenance': 'Human-collected per project owner; authorship and technical correctness not independently certified.'})
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('tags_by_split', 'repairs', 'input_sha256', 'files')}, indent=2))
    return manifest


if __name__ == '__main__':
    build()
