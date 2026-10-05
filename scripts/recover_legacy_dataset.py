"""Recover archived examples into an auditable mixed training candidate, without modifying originals."""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

if __package__:
    from .build_fluency_dataset import ROOT, SYSTEM, DEST as STYLE_DIR
    from .utils.dataset_checks import assert_disjoint
else:
    from build_fluency_dataset import ROOT, SYSTEM, DEST as STYLE_DIR
    from utils.dataset_checks import assert_disjoint

ARCHIVE = ROOT / 'datasets/archive/pre_fluency_2026-10-03'
DEST = ROOT / 'datasets/releases/burmese_recovered_v1'


def identity(text):
    return ''.join(c for c in unicodedata.normalize('NFC', text).casefold()
                   if unicodedata.category(c)[0] in 'LMN')


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def literal_templates():
    # Parse constants without importing/executing the retired generator.
    tree = ast.parse((ARCHIVE / 'build_curated_conversations.py').read_text(encoding='utf-8'))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('FRIEND_BASE_SCENARIOS', 'inquiries'):
                    found[target.id] = ast.literal_eval(node.value)
    return [(u, a) for u, a, _ in found['FRIEND_BASE_SCENARIOS']] + found['inquiries']


def recover_friend(prompt, answer, templates):
    # Only retain an underlying pair demonstrably present in the archived row.
    for user, assistant in templates:
        if identity(user) in identity(prompt) and identity(assistant) in identity(answer):
            return user, assistant
    return None


def quality_reason(prompt, answer, tag):
    if tag == 'translation':
        return 'translation_batch_pending_pairwise_review'
    if tag in {'health', 'healthcare', 'department', 'achievement', 'osi_vs_tcpip'}:
        return 'domain_claims_pending_review'
    if any(term in prompt for term in ('ရာထူး', 'ဒီနှစ် TU', 'လက်ရှိပါမောက္ခ')):
        return 'unverified_current_institutional_claim'
    if 'GDDR' in prompt and 'DDR5' in prompt:
        return 'oversimplified_hardware_comparison_pending_review'
    if not prompt.strip() or not answer.strip():
        return 'empty_text'
    if any('\ufffd' in t or any(ord(c) < 32 and c not in '\n\t' for c in t) for t in (prompt, answer)):
        return 'damaged_text'
    if re.search(r'[\u4e00-\u9fff]', prompt + answer):
        return 'unexpected_CJK'
    if not any('\u1000' <= c <= '\u109f' for c in answer):
        return 'no_Burmese_answer'
    if re.search(r'(.)\1{12,}', answer):
        return 'repeated_character_noise'
    # Narrow, observed defects; these do not certify absence of other hallucinations.
    if any(term in answer for term in ('ငါ မင်းဆီ ဖုန်းဆက်', 'ငါနဲ့ အတူတူ သွား',
           'ငါတို့ အဆာပြေ', 'ငါနေကောင်းပါတယ်', 'ငါနေကောင်း',
           'ငါ နေကောင်း', 'အေး နေကောင်းပါတယ်', 'ဝါသနာပါတာပေါ့', 'အေး သွားကြည့်ရအောင်',
           'ဖုန်းဆက်လိုက်', 'ဆုံကြတာပေါ့', 'မင်းနဲ့မတွေ့တာ', 'ငါတို့ အတူတူ',
           'လက်ဖက်ရည်ဆိုင်မှာ စကားစမြည်', 'ခဏနေရင် အိပ်ပျော်သွားလိမ့်မယ်')):
        return 'unsupported_personal_claim_or_promise'
    if tag == 'conversation_casual_close_friend' and any(term in prompt for term in ('Gym', 'အိပ်မပျော်', 'ခရီးတို', 'ဂိမ်း')):
        return 'friend_scenario_pending_review'
    return None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-length', type=int, default=2048)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/qwen3-4b'), local_files_only=True)
    template = get_training_chat_template(tokenizer)
    templates = literal_templates()
    bad_hashes = {r['sha256'] for r in map(json.loads, (ARCHIVE / 'row_decisions.jsonl').read_text(encoding='utf-8').splitlines())
                  if r.get('reason') == 'confirmed_translation_error'}
    styles, decisions, candidates, token_sizes = {}, [], [], {}
    protected_prompts = set()
    protected_answers = set()
    for split in ('train', 'validation', 'test'):
        path = STYLE_DIR / split / f'{split}_combined.jsonl'
        styles[split] = []
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('review_status') == 'rejected':
                continue
            styles[split].append({'id': 'style-' + row['id'], 'messages': row['messages'],
                'tags': 'authored_' + row['tone'], 'source': 'burmese_fluency_v2',
                'source_id': row['id'], 'source_sha256': digest(line), 'original_split': split,
                'original_line': None, 'review_status': row['review_status'], 'changes': []})
            for m in row['messages']:
                if m['role'] == 'user': protected_prompts.add(identity(m['content']))
                if m['role'] == 'assistant': protected_answers.add(identity(m['content']))
    for split in ('train', 'validation', 'test'):
        path = ARCHIVE / 'clean' / split / f'{split}_combined.jsonl'
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip(): continue
            original = json.loads(line)
            tag = original.get('tags', '')
            decision = {'id': f'legacy-{split}-{number:05d}', 'source': str(path.relative_to(ROOT)),
                        'line': number, 'sha256': digest(line), 'tag': tag, 'decision': 'excluded', 'reason': None}
            decisions.append(decision)
            turns = [m for m in original['messages'] if m['role'] != 'system']
            if [m['role'] for m in turns] != ['user', 'assistant']:
                decision['reason'] = 'unexpected_legacy_turn_structure'; continue
            prompt, answer = [unicodedata.normalize('NFC', m['content']).strip() for m in turns]
            changes = ['standardized_system_prompt', 'NFC_and_outer_whitespace']
            if digest(line) in bad_hashes:
                decision['reason'] = 'confirmed_error'; continue
            if tag == 'conversation_casual_close_friend':
                pair = recover_friend(prompt, answer, templates)
                if pair is None:
                    decision['reason'] = 'unmatched_friend_template'; continue
                prompt, answer = pair
                changes.append('removed_generator_openers_and_closings')
            elif tag == 'conversation_curated_inquiry' and answer.startswith('ဟုတ်ကဲ့ပါခင်ဗျာ။'):
                answer = answer[len('ဟုတ်ကဲ့ပါခင်ဗျာ။'):].strip()
                changes.append('removed_forced_gendered_acknowledgment')
            reason = quality_reason(prompt, answer, tag)
            if reason:
                decision['reason'] = reason; continue
            if identity(prompt) in protected_prompts or identity(answer) in protected_answers:
                decision['reason'] = 'overlap_with_authored_conversations'; continue
            row = {'id': decision['id'], 'messages': [{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}],
                   'tags': tag, 'source': 'recovered_legacy', 'source_id': decision['id'],
                   'source_sha256': digest(line), 'original_split': split, 'original_line': number,
                   'review_status': 'automatic_filters_only_not_semantically_verified', 'changes': changes}
            encoded = tokenizer.apply_chat_template(row['messages'], chat_template=template,
                        tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
            size = len(encoded['input_ids'])
            decision['tokens'] = size
            if size > args.max_length:
                decision['reason'] = 'over_context_budget_not_truncated'; continue
            if not any(encoded['assistant_masks']):
                decision['reason'] = 'missing_assistant_labels'; continue
            token_sizes[row['id']] = size
            candidates.append((row, decision))
    # Exact normalized prompt conflicts are held out for review, not resolved by guessing an answer.
    groups = defaultdict(list)
    for row, decision in candidates:
        groups[identity(row['messages'][1]['content'])].append((row, decision))
    recovered, seen_answers = [], set()
    for prompt_id, records in sorted(groups.items()):
        if len({identity(r['messages'][2]['content']) for r, _ in records}) > 1:
            for _, d in records: d['reason'] = 'conflicting_answers_for_same_prompt'
            continue
        for index, (row, decision) in enumerate(records):
            answer_id = identity(row['messages'][2]['content'])
            if index or answer_id in seen_answers:
                decision['reason'] = 'duplicate_prompt_or_answer_after_cleanup'; continue
            seen_answers.add(answer_id)
            recovered.append(row)
            decision['decision'] = 'included_candidate'
            decision['reason'] = 'passed_automatic_filters_not_semantic_review'
    # Stable prompt-hash split; old legacy splits are retired because they leaked.
    splits = {name: list(rows) for name, rows in styles.items()}
    for row in recovered:
        bucket = int(digest(identity(row['messages'][1]['content']))[:8], 16) % 100
        split = 'validation' if bucket < 5 else 'test' if bucket < 10 else 'train'
        splits[split].append(row)
    assert_disjoint(**splits)
    canonical = {s: {identity(r['messages'][1]['content']) for r in rows} for s, rows in splits.items()}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not canonical[a] & canonical[b], f'Canonical prompt overlap: {a}/{b}'
    for split, rows in styles.items():
        for row in rows:
            encoded = tokenizer.apply_chat_template(row['messages'], chat_template=template, tokenize=True,
                        return_dict=True, return_assistant_tokens_mask=True)
            if len(encoded['input_ids']) > args.max_length or not any(encoded['assistant_masks']):
                raise ValueError('Authored row fails token checks: ' + row['id'])
            token_sizes[row['id']] = len(encoded['input_ids'])
    included_ids = {r['id'] for rows in splits.values() for r in rows}
    manifest = {'version': 'burmese_recovered_v1', 'status': 'candidate_pending_language_and_fact_review',
        'legacy_original_rows': len(decisions), 'legacy_recovered_rows': len(recovered),
        'authored_rows': sum(map(len, styles.values())), 'counts': {s: len(r) for s, r in splits.items()},
        'composition_by_split': {s: dict(Counter(r['source'] for r in rows)) for s, rows in splits.items()},
        'recovered_tags': dict(Counter(r['tags'] for r in recovered)),
        'exclusion_counts': dict(Counter(d['reason'] for d in decisions if d['decision'] == 'excluded')),
        'max_length': args.max_length, 'max_tokens': max(token_sizes[i] for i in included_ids),
        'source_inventory_sha256': digest((ARCHIVE / 'inventory.json').read_text(encoding='utf-8')),
        'style_manifest_sha256': digest((STYLE_DIR / 'manifest.json').read_text(encoding='utf-8')),
        'split_policy': 'Legacy prompt hash 90/5/5; authored split assignments preserved; original legacy holdouts retired',
        'semantic_accuracy_certified': False, 'files': {}}
    for split, rows in splits.items():
        path = DEST / split / f'{split}_combined.jsonl'
        write_jsonl(path, rows)
        manifest['files'][split] = {'path': str(path.relative_to(DEST)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(DEST / 'manifest.json', manifest)
    write_jsonl(DEST / 'review/recovery_decisions.jsonl', decisions)
    write_json(DEST / 'review/token_lengths.json', {i: token_sizes[i] for i in sorted(included_ids)})
    # Separate subsets allow a short style-only follow-up without repeating that data in the main corpus.
    for source, filename in [('recovered_legacy', 'legacy_train.jsonl'), ('burmese_fluency_v2', 'style_train.jsonl')]:
        write_jsonl(DEST / 'subsets' / filename, [r for r in splits['train'] if r['source'] == source])
    conversational_tags = {'conversation_curated_inquiry', 'conversation_casual_close_friend',
        'daily', 'daily_life', 'greeting', 'travel', 'foodandshop', 'food_shopping', 'shopping', 'food'}
    conversational = [r for r in splits['train'] if r['source'] == 'burmese_fluency_v2' or r['tags'] in conversational_tags]
    write_jsonl(DEST / 'subsets/conversation_train.jsonl', conversational)
    manifest['conversation_train_subset_rows'] = len(conversational)
    write_json(DEST / 'manifest.json', manifest)
    sample = ['# Recovered training examples: quick review', '',
        'Two retained training examples per legacy tag (or fewer where unavailable). This is a review sample, not a quality certification.', '']
    by_tag = defaultdict(list)
    for row in splits['train']:
        if row['source'] == 'recovered_legacy': by_tag[row['tags']].append(row)
    for tag, records in sorted(by_tag.items()):
        for row in records[:2]:
            sample += [f"## {row['id']} — {tag}", '',
                f"Original: {row['original_split']}, line {row['original_line']}. Changes: {', '.join(row['changes'])}.", '']
            for message in row['messages'][1:]:
                sample += [f"**{message['role']}**", '', message['content'], '']
    (DEST / 'review/sample.md').write_text('\n'.join(sample), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
