"""Scan every release record; distinguish mechanical failures from review heuristics.

No network calls, model generation, or edits to answers. Reports are reproducible.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

if __package__:
    from .utils.active_dataset import active_release, ROOT
    from .utils.dataset_checks import conversation_key, normalize_identity
    from .build_seminar_dataset import grouped_components, prompt_identity
    from .utils.persona import DEFAULT_SYSTEM_PROMPT
else:
    from utils.active_dataset import active_release, ROOT
    from utils.dataset_checks import conversation_key, normalize_identity
    from build_seminar_dataset import grouped_components, prompt_identity
    from utils.persona import DEFAULT_SYSTEM_PROMPT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_flags(row):
    """High recall heuristics; these are NOT automatic factual verdicts."""
    flags = []
    question = row['messages'][1]['content']
    if row.get('tags') == 'computer hardware' and re.search(r'ကား|ရေခဲသေတ္တာ|အဝတ်လျှော်|Coolant|Thermostat', question) and not re.search(r'ဘာသာစကား|လက်ချောင်း', question):
        flags.append({'code': 'possible_off_topic_hardware_tag', 'message_index': 1, 'role': 'user'})
    for index, m in enumerate(row['messages'][1:], 1):
        text = m['content']
        found = []
        if re.search(r'[\u4e00-\u9fff\u0e00-\u0e7f\u0900-\u097f]', text):
            found.append('unexpected_script')
        if re.search(r'(?:Ã.|Â.|â€|á€)', text):
            found.append('possible_mojibake')
        # ZWJ is required inside emoji such as woman cook / face exhaling.
        no_emoji_joiners = re.sub(r'([\U0001f000-\U0001faff]\ufe0f?)\u200d(?=[\U0001f000-\U0001faff])', r'\1', text)
        if any(unicodedata.category(c) == 'Cf' for c in no_emoji_joiners):
            found.append('invisible_format_character')
        if re.search(r'<\|(?:im_start|im_end|endoftext)\|>', text):
            found.append('embedded_chat_control_token')
        if re.search(r'(?:\bTODO\b|\bTBD\b|lorem ipsum|insert (?:answer|response) here)', text, re.I):
            found.append('possible_placeholder')
        if re.search(r'([^\W\d_])\1{19,}', text):
            found.append('repeated_character_noise')
        if re.search(r'(?:^|[\s\u200b\u200c])\u1031[\u1000-\u1021]', text):
            found.append('possible_prebase_vowel_order')
        if m['role'] == 'assistant':
            if any('\u1000' <= c <= '\u109f' for c in row['messages'][index-1]['content']) and not any('\u1000' <= c <= '\u109f' for c in text):
                found.append('burmese_question_without_burmese_answer')
            if len(text) >= 40 and normalize_identity(text) == normalize_identity(row['messages'][index-1]['content']):
                found.append('answer_copies_question')
            if text.count('```') % 2:
                found.append('unclosed_code_fence')
            if re.search(r'(?i)\b(?:MD5|SHA-?1|SHA-?256|SHA-?512)\b', text) and re.search(r'(?i)encrypt|decrypt|encryption|decryption|စာဝှက်', text):
                found.append('hashing_vs_encryption_review')
            if re.search(r'(?i)\bDNS\b', text) and re.search(r'(?i)UDP\s+only|only\s+UDP|UDP ပဲ|UDP ကိုသာ', text):
                found.append('dns_transport_absolute_review')
            if 'HTTPS' in text and re.search(r'100\s*%|ရာနှုန်းပြည့်', text):
                found.append('https_absolute_security_review')
            if re.search(r'(?i)\b(?:TCP|UDP|HTTP|HTTPS|DNS|SMTP|FTP|SSH|TLS|SSL)\b', text) and re.search(r'အမြဲတမ်း|လုံးဝ|100\s*%|(?i:always|never)', text):
                found.append('technical_absolute_review')
            if re.search(r'ငါ.{0,12}(?:လိုက်ခဲ့|စားခဲ့|ဖုန်းဆက်ပေး|သွားပေး|အိပ်နေ|ပင်ပန်းနေ)|ကျွန်တော်.{0,8}စားခဲ့', text):
                found.append('possible_unfounded_personal_claim')
        flags.extend({'code': code, 'message_index': index, 'role': m['role']} for code in found)
    return flags


def audit(directory, tokenize=True, minimum_train=12000):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    errors, warnings, records, split_for = [], [], [], {}
    counts, input_hashes, lengths, sources, roles = {}, {}, {}, Counter(), Counter()
    seen_ids, seen_conversations, prompt_splits = set(), {}, defaultdict(set)
    answer_ids = defaultdict(list)
    group_map = json.loads((directory / 'review/split_groups.json').read_text(encoding='utf-8'))
    group_splits = defaultdict(set)
    tokenizer = template = None
    if tokenize:
        from transformers import AutoTokenizer
        from trl.chat_template_utils import get_training_chat_template
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models/qwen3-4b'), local_files_only=True)
        template = get_training_chat_template(tokenizer)
    for split in ('train', 'validation', 'test'):
        path = directory / split / f'{split}_combined.jsonl'
        count, sizes, target_counts = 0, [], []
        if sha(path) != manifest['files'][split]['sha256']:
            errors.append({'code': 'release_hash_mismatch', 'split': split})
        for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            try:
                row = json.loads(line)
                identity, messages = row['id'], row['messages']
                assert isinstance(identity, str) and identity
                assert identity not in seen_ids, 'duplicate_id'
                seen_ids.add(identity)
                assert len(messages) >= 3 and len(messages) % 2 == 1, 'message_count'
                assert messages[0] == {'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT}, 'system_prompt'
                assert isinstance(row['tags'], str) and row['tags'].strip(), 'tags'
                for i, message in enumerate(messages[1:], 1):
                    assert message['role'] == ('user' if i % 2 else 'assistant'), 'role_order'
                    text = message['content']
                    assert isinstance(text, str) and text.strip(), 'empty_text'
                    assert text == text.strip() == unicodedata.normalize('NFC', text), 'normalization'
                    assert '\ufffd' not in text, 'replacement_character'
                    assert not any((unicodedata.category(c) in {'Cc', 'Cs'} and c not in '\n\t') for c in text), 'damaged_control_character'
                key = conversation_key(row)
                assert key not in seen_conversations, 'duplicate_conversation'
                seen_conversations[key] = identity
                assert identity in group_map and group_map[identity]['split'] == split, 'group_mapping'
                group_splits[group_map[identity]['group_id']].add(split)
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                errors.append({'code': 'invalid_record', 'split': split, 'line': line_no, 'detail': str(exc)})
                continue
            count += 1; records.append(row); split_for[identity] = split
            sources[row['source']] += 1
            roles.update(m['role'] for m in messages)
            prompt_splits[prompt_identity(row)].add(split)
            flags = content_flags(row)
            if flags:
                warnings.append({'id': identity, 'split': split, 'source_file': row.get('source_file'),
                    'source_line': row.get('source_line'), 'tags': row['tags'], 'flags': flags,
                    'messages': messages[1:]})
            for message in messages:
                if message['role'] == 'assistant' and len(message['content']) >= 50:
                    answer_ids[normalize_identity(message['content'])].append(identity)
            if tokenizer:
                encoded = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=True,
                    return_dict=True, return_assistant_tokens_mask=True)
                size = len(encoded['input_ids']); sizes.append(size); lengths[identity] = size
                mask = encoded['assistant_masks']; target_counts.append(sum(mask))
                if size > 2048 or not any(mask[:2048]):
                    errors.append({'code': 'token_overflow_or_no_labels', 'id': identity, 'tokens': size})
                previous = 0
                for end in range(3, len(messages)+1, 2):
                    prefix = encoded if end == len(messages) else tokenizer.apply_chat_template(
                        messages[:end], chat_template=template, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
                    supervised = sum(prefix['assistant_masks'])
                    if supervised <= previous:
                        errors.append({'code': 'unsupervised_assistant_turn', 'id': identity, 'turn_end': end})
                    previous = supervised
        counts[split] = {'records': count}
        if sizes:
            sizes.sort()
            counts[split].update(max_tokens=max(sizes), p50_tokens=sizes[len(sizes)//2], p95_tokens=sizes[int(.95*len(sizes))],
                assistant_target_tokens=sum(target_counts))
        if count != manifest['counts'][split]:
            errors.append({'code': 'manifest_count_mismatch', 'split': split})
    if counts['train']['records'] < minimum_train:
        errors.append({'code': 'minimum_training_count_not_met', 'minimum': minimum_train})
    for key, splits in prompt_splits.items():
        if len(splits) > 1: errors.append({'code': 'opening_prompt_leakage', 'prompt': key, 'splits': sorted(splits)})
    for key, splits in group_splits.items():
        if len(splits) > 1: errors.append({'code': 'component_leakage', 'group_id': key})
    for filename, expected in manifest['input_sha256'].items():
        path = ROOT / filename
        actual = sha(path) if path.exists() else None
        input_hashes[filename] = actual == expected
        if actual != expected: errors.append({'code': 'input_hash_mismatch', 'file': filename})
    # Recompute near-prompt groups independently of saved split assignments.
    _, near_links = grouped_components(records)
    for link in near_links:
        if split_for[link['left']] != split_for[link['right']]:
            errors.append({'code': 'near_prompt_leakage', **link})
    repeated = [{'count': len(ids), 'ids': ids, 'answer': answer} for answer, ids in answer_ids.items() if len(ids) >= 5]
    repeated.sort(key=lambda r: (-r['count'], r['answer']))
    summary = {'release': directory.name, 'scope': 'Every row and every message in train, validation and test',
        'tokenizer_checked': bool(tokenizer), 'minimum_training_records': minimum_train,
        'counts': counts, 'total_records': len(records), 'messages_by_role': dict(roles), 'sources': dict(sources),
        'mechanical_errors': len(errors), 'mechanical_checks_passed': not errors,
        'flagged_records': len(warnings), 'warning_occurrences': dict(Counter(f['code'] for w in warnings for f in w['flags'])),
        'alternative_answer_groups': manifest['alternative_answer_groups'],
        'near_prompt_pairs_recomputed': len(near_links), 'repeated_answer_groups_at_least_5': len(repeated),
        'input_hashes_verified': all(input_hashes.values()), 'semantic_accuracy_certified': False,
        'limits': 'Rules can miss factual mistakes, unnatural Burmese, subtle contradictions and semantic leakage. Flags are review candidates, not automatic errors. No independent native-speaker approval is asserted.'}
    report_dir = directory / 'review/final_audit'; report_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [('summary.json', summary), ('mechanical_errors.json', errors), ('repeated_answers.json', repeated)]:
        (report_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (report_dir / 'content_flags.jsonl').write_text(''.join(json.dumps(w,ensure_ascii=False)+'\n' for w in warnings), encoding='utf-8')
    lines = ['# Final dataset scan', '', f"Release: {directory.name}. Scanned all {len(records):,} records.", '',
        '| Split | Records | Maximum tokens |', '| --- | ---: | ---: |']
    for split, count in counts.items(): lines.append(f"| {split} | {count['records']:,} | {count.get('max_tokens', 'not checked')} |")
    lines += ['', f"Mechanical errors: **{len(errors)}**. Content flags: **{len(warnings)} records**.",
        f"Alternative-answer groups requiring comparison: **{manifest['alternative_answer_groups']}**.", '',
        summary['limits'], '', '## Content flag counts', '']
    lines += [f"- {code}: {number}" for code,number in sorted(summary['warning_occurrences'].items())]
    lines += ['', 'See content_flags.jsonl for complete flagged messages, repeated_answers.json for answer reuse,',
        '../alternative_answers.md for same-question answers, and mechanical_errors.json for failures.', '',
        'No original submission or prior release is changed by this audit.']
    (report_dir/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--skip-tokenizer', action='store_true')
    parser.add_argument('--minimum-train', type=int, default=12000)
    args = parser.parse_args()
    result = audit(args.data_dir or active_release(), not args.skip_tokenizer, args.minimum_train)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['mechanical_checks_passed'] else 1)
