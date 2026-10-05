"""Generate an auditable human content-review queue without editing training data."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from scripts.utils.active_dataset import active_release, ROOT
from scripts.build_seminar_dataset import content_hash
from scripts.finalize_seminar_review import DELEGATED_FACTS

# Content findings manually triaged against the complete conversation on 2026-10-05.
PRIORITY_FINDINGS = {
    'human-031026data372-00030': 'Update Kyber/ML-KEM terminology: FIPS 203 standardizes ML-KEM, derived from Kyber. Describe establishing a shared secret using a KEM, rather than general payload encryption.',
    'human-031026data372-00275': 'Replace the bare cryptographic hash definition of the JWT signature with a digital signature or keyed MAC. Qualify the three-part format as JWS compact serialization.',
    'human-041026data330-00229': 'Replace the bare hash definition of the JWT signature with a digital signature or keyed MAC. Qualify the three-part format as JWS compact serialization.',
    'human-031026data980-00104': 'Qualify RAID protection by level: RAID 0 has no redundancy. Avoid implying RAID guarantees protection from all data loss.',
    'human-041026data330-00139': 'Remove the unsupported claim that quantum computers can break RSA/ECC in seconds. Explain the future threat from sufficiently capable fault-tolerant quantum computers; repair the garbled Burmese wording.',
    'legacy-train-00623': 'The factory-reset instructions omit data erasure and backup precautions and assume an unspecified phone platform. Ask for the device and explain consequences before steps.',
    'legacy-train-04941': 'Replace the unqualified claim that Windows Defender is the best and safest antivirus. Explain suitability and limitations without a universal ranking.',
    'legacy-train-00406': 'SSD boot time is not zero or universally a few seconds. Qualify the answer by hardware, operating system and startup workload.',
    'legacy-train-05561': 'The fare is invented without a starting location, route or date. Ask for missing context and recommend checking the current fare.',
}
for _identity in ('legacy-test-00308', 'legacy-train-02049', 'legacy-train-02106',
                  'legacy-train-06061', 'legacy-validation-00026'):
    PRIORITY_FINDINGS[_identity] = 'The assistant invents a nearby shop, availability, quality or delivery policy without location or evidence. Ask for location and avoid unsupported local claims.'
for _identity in ('legacy-train-02455', 'legacy-train-03677'):
    PRIORITY_FINDINGS[_identity] = 'The answer invents a restaurant reservation policy of calling at least one hour ahead. Explain that booking policies vary and advise checking with the restaurant.'


def signals(row):
    """Review triggers, not automatic error judgments; retain matched evidence."""
    results = []
    rules = [
        ('P1', 'unverified_personal_or_local_claim',
         r'ငါ့(?:အိမ်|ဖုန်း|အခန်း|ဆိုင်|ကျောင်း)|အိမ်မှာပဲ.{0,12}(?:နား|ရှိ)|စားပြီးပြီ|ငါ.{0,12}(?:စားနေ|သွားနေ|ပြန်လာ|ဝယ်ထား)|စားပွဲပေါ်မှာ.{0,16}(?:ရှိ|ယူ)|လမ်းထိပ်က.{0,20}(?:ဆိုင်|ရှိ)|တစ်နာရီကြိုဆက်',
         'Check whether the assistant invents physical possessions, activities, a nearby place, or business availability. Roleplay/quoted examples may be valid.'),
        ('P1', 'unverified_current_information',
         r'(?:မနက်ဖြန်|ဒီနေ့|လာမယ့်).{0,35}(?:ကျောင်းပိတ်|အတန်းရှိ|စာမေးပွဲ|သင်တန်းပိတ်|ဖွင့်ပါတယ်|ပိတ်ပါတယ်)|(?:ကားခ|ဘတ်စ်ကားခ|လိုင်းကားနဲ့).{0,35}[၀-၉0-9]',
         'Verify time/place-specific school, opening-time, fare or schedule claims; the conversation may lack sufficient context.'),
        ('P1', 'overconfident_claim',
         r'အကောင်းဆုံး(?:နဲ့|နှင့်)|စိတ်အချရဆုံး|စက္ကန့်ပိုင်းအတွင်း|(?:အမြဲတမ်း|လုံးဝ).{0,12}(?:လုံခြုံ|မှန်ကန်)|100\s*%',
         'Check the scope and evidence for the absolute/best/guaranteed claim; qualify it if exceptions exist.'),
        ('P2', 'possible_bad_translation',
         r'ရိုးရိုးနံပါတ်ရေ|တင်းရေ|မာတီမီဂါ|အေးခဲပေးစနစ်|လှိုင်းနှုန်းကို 1-RTT',
         'A native Burmese reviewer should verify the terminology against the intended technical meaning.'),
    ]
    for index,m in enumerate(row['messages']):
        if m['role'] == 'system': continue
        for priority, code, pattern, reason in rules:
            if m['role'] != 'assistant' and code != 'possible_bad_translation': continue
            match = re.search(pattern, m['content'])
            if match:
                results.append(dict(priority=priority, code=code, reason=reason,
                                    message_index=index, matched_text=match.group()))
    question = row['messages'][1]['content']
    answer = row['messages'][2]['content']
    if 'JWT' in question and re.search(r'Signature.{0,50}(?:Hash|hash)', answer):
        results.append(dict(priority='P1', code='jwt_signature_definition', reason='A JWT signature/MAC is described as a hash. Check JWS/JWE scope, signature versus keyed MAC, and avoid teaching a bare hash as authentication.'))
    if re.search(r'RAID', question) and re.search(r'ပျက်စီးမှုမှ ကာကွယ်|ပျက်စီး.*မရှိ',answer):
        results.append(dict(priority='P1', code='raid_guarantee', reason='Check RAID level: RAID 0 has no redundancy, and RAID is not a substitute for backup.'))
    if 'CRYSTALS-Kyber' in question:
        results.append(dict(priority='P1', code='pqc_standard_name', reason='Verify Kyber versus standardized ML-KEM naming and distinguish key encapsulation from general data encryption.'))
    if 'factory reset' in question.lower():
        results.append(dict(priority='P1', code='destructive_reset_advice', reason='Verify device-specific steps, data-erasure warning and backup guidance before recommending factory reset.'))
    return results


def load_rows(directory):
    rows = {}
    for split in ('train', 'validation', 'test'):
        path = directory / split / f'{split}_combined.jsonl'
        for line, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            row = json.loads(raw)
            rows[row['id']] = dict(row, split=split, release_file=path.relative_to(ROOT).as_posix(), release_line=line)
    return rows


def build():
    directory = active_release()
    rows = load_rows(directory)
    queue = defaultdict(list)
    for identity,row in rows.items(): queue[identity].extend(signals(row))
    groups = json.loads((directory/'review/alternative_answers.json').read_text(encoding='utf-8'))
    group_details = []
    for number,group in enumerate(groups,1):
        answers = [rows[i]['messages'][2]['content'] for i in group['ids']]
        numbers = [sorted(set(re.findall(r'\d+(?:\.\d+)?', a.translate(str.maketrans('၀၁၂၃၄၅၆၇၈၉','0123456789'))))) for a in answers]
        differing_numbers = len({tuple(n) for n in numbers}) > 1
        same_first_answer = len({' '.join(a.split()) for a in answers}) == 1
        # All five numeric-difference groups were inspected: examples, list indices,
        # and optional details are not established contradictions.
        priority = 'P2'
        code = 'alternative_answers_different_numbers' if differing_numbers else 'alternative_answers'
        reason = ('Answers contain different numbers/versions. These may be examples, not contradictions; compare the complete answers.' if differing_numbers else
                  'Same opening question has alternative answers or later turns. Check consistency; keep valid paraphrases.')
        for identity in group['ids']:
            queue[identity].append(dict(priority=priority,code=code,reason=reason,group=number))
        group_details.append(dict(group=number,priority=priority,ids=group['ids'],prompt=group['prompt'],
                                  differing_numbers=differing_numbers,same_first_answer=same_first_answer))
    # Explicit source-batch checks cover defects that sparse rules cannot establish.
    for identity,row in rows.items():
        if row['tags'] == 'conversation_curated_inquiry':
            queue[identity].append(dict(priority='P2',code='legacy_conversation_batch',reason='Review natural Burmese, missing context, invented personal experiences, and whether the reply suits an AI companion. This batch has not had full native-speaker approval.'))
    existing = json.loads((ROOT/'datasets/reviews/it_seminar_v3.decisions.json').read_text(encoding='utf-8'))
    for review in existing:
        if review['id'] in rows and review.get('reviewer') == 'codex_final_scan':
            queue[review['id']].append(dict(priority='P2',code='assistant_correction_needs_human_signoff',reason=review.get('notes','Check assistant-authored correction.')))
    triage = []
    for identity, flags in queue.items():
        retained = []
        for flag in flags:
            # Full-context inspection: these smalltalk matches ask about the user
            # eating, or explicitly say that the AI does not eat.
            if identity.startswith('human-smalltalk_2000-') and flag['code'] == 'unverified_personal_or_local_claim':
                triage.append(dict(id=identity, code=flag['code'], disposition='dismissed',
                                   reason='User-directed food question or explicit AI-does-not-eat context; no invented personal meal.'))
                continue
            flag['priority'] = 'P2'
            retained.append(flag)
        queue[identity] = retained
    resolved = []
    for identity, reason in PRIORITY_FINDINGS.items():
        if identity not in rows:
            raise ValueError(f'Manual finding refers to a missing record: {identity}')
        if identity in DELEGATED_FACTS and rows[identity]['messages'][2]['content'] == DELEGATED_FACTS[identity][0]:
            queue[identity] = [f for f in queue[identity] if f['code'] in
                               ('alternative_answers', 'alternative_answers_different_numbers',
                                'legacy_conversation_batch', 'assistant_correction_needs_human_signoff')]
            resolved.append(dict(id=identity, issue=reason, resolution='Assistant corrected under user delegation; not independent human approval.',
                                 current_content_sha256=content_hash(rows[identity])))
        else:
            queue[identity].append(dict(priority='P1', code='manually_triaged_content_issue', reason=reason))
    items=[]
    for identity, flags in queue.items():
        if not flags: continue
        row=rows[identity]
        items.append(dict(row,priority=min(f['priority'] for f in flags),review_reasons=flags,
                          current_content_sha256=content_hash(row),review_status='pending'))
    items.sort(key=lambda r:(r['priority'],r['id']))
    output=directory/'review/human_review';output.mkdir(exist_ok=True)
    (output/'queue.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf-8')
    (output/'groups.json').write_text(json.dumps(group_details,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary=dict(total_release_records=len(rows),queued_records=len(items),by_priority=dict(Counter(r['priority'] for r in items)),
                 by_reason=dict(Counter(f['code'] for r in items for f in r['review_reasons'])),
                 alternative_groups=len(groups),alternative_records=len({i for g in groups for i in g['ids']}),
                 groups_with_different_numbers=sum(g['differing_numbers'] for g in group_details),
                 training_files_changed=False, resolved_priority_findings=len(resolved), semantic_accuracy_certified=False,
                 reviewed_on='2026-10-05',
                 scope='All rows mechanically scanned; targeted content inspection, not a full human semantic review.',
                 split_sha256={split: hashlib.sha256((directory/split/f'{split}_combined.jsonl').read_bytes()).hexdigest()
                               for split in ('train','validation','test')})
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (output/'dismissed_signals.json').write_text(json.dumps(triage,indent=2)+'\n',encoding='utf-8')
    (output/'resolved_findings.json').write_text(json.dumps(resolved,indent=2)+'\n',encoding='utf-8')
    write_report(output, items, summary)
    print(json.dumps(summary,indent=2))
    return directory,rows,items,group_details


def write_report(output, items, summary):
    sources = '''- [JWT: RFC 7519](https://www.rfc-editor.org/rfc/rfc7519.html)
- [ML-KEM: NIST FIPS 203](https://csrc.nist.gov/pubs/fips/203/final)
- [Quantum threat: NIST explanation](https://www.nist.gov/cybersecurity-and-privacy/what-post-quantum-cryptography)
- [RAID levels: Dell](https://www.dell.com/support/kbdoc/en-us/000128635/dell-servers-what-are-the-raid-levels-and-their-specifications)
- [Factory reset erases phone data: Android Help](https://support.google.com/android/answer/6088915)
'''
    lines = [f'''# Human review before final training — 2026-10-05

**Content sign-off is still pending. Do not treat this release as perfect or fully fact-checked.**
The repeated full mechanical audit passed: 14,257 conversations (12,759 train,
749 validation, 749 test), maximum 1,835 tokens, no detected mechanical errors.
This read-only packet does not change training files. The user-delegated correction pass resolved
**{summary['resolved_priority_findings']} previous P1 findings**; see [resolved_findings.json](resolved_findings.json).
Hashes of the current rebuilt dataset are pinned in [summary.json](summary.json).

Open [review.html](review.html) locally in a browser to search the complete review queue.
It is read-only; record decisions and replacement text separately. [queue.jsonl](queue.jsonl)
includes full conversations, exact split paths and line numbers, reasons, and current content hashes.
Legacy IDs describe their original split; use `split` and `release_line` for their current location.

## Review order

1. **{summary['by_priority'].get('P1',0)} P1 records below:** specific content problems identified during contextual inspection. An IT reviewer should correct technical claims; a native Burmese reviewer should approve the replacement wording.
2. **394 alternative-answer groups / 879 records:** compare answers and follow-up turns. Different wording is valid; these are not 394 confirmed contradictions. The five numeric-difference groups use examples, list numbering or optional details, so number differences alone are not errors.
3. **1,199 `conversation_curated_inquiry` records:** native review for natural phrasing, coherent replies, missing context, invented physical experiences and unsupported local information. Keep polite, casual and close-friend tones when prompted; familiarity should not invent real-world experiences.
4. **{summary['by_reason'].get('assistant_correction_needs_human_signoff',0)} assistant-corrected records:** review factual/text corrections and category changes as desired. The `codex_final_scan` decision is assistant review, not independent human approval. The user delegated the 16 priority factual/contextual fixes to the assistant.
5. **Other P2 candidates:** terminology, suspicious translation, and claims needing context. A rule match can be valid; do not delete a record just because it is flagged.

These categories overlap. The queue contains **{summary['queued_records']:,} distinct records**;
absence from it is not certification. A native Burmese IT reviewer should also inspect a
stratified sample from every remaining source and topic and all retained multi-turn examples.
Expand review to the affected source/topic if that sample exposes recurring defects.
For the strongest final sign-off, review every retained conversation, including validation and test.

## How to record decisions

For each record save ID, current content hash, reviewer name, date, decision
(`keep`, `correct`, or `exclude`), explanation, and complete replacement messages when correcting.
Preserve multi-turn context. Recheck matching prompt groups after edits. Keep usable IT examples:
correct a weak definition rather than discarding the topic. Preserve raw human submissions.
Do not directly overwrite the existing build overlay with this queue: its hashes refer to
the current release, whereas the build overlay validates source-stage records.
Apply approved corrections through the dataset builder, regenerate the release, then rerun
the full audit. Freeze hashes and keep test examples out of training. Evaluation-set edits
must be finished before comparing models; do not tune to their answers.
No training was started and no human approval is asserted.

## P1 records: inspect and resolve before the final run
''']
    for row in items:
        if row['priority'] != 'P1': continue
        reason = next(f['reason'] for f in row['review_reasons'] if f['code']=='manually_triaged_content_issue')
        lines.append(f"### {row['id']}\n\n{row['split']}, line {row['release_line']} — `{row['release_file']}`\n\n{reason}\n")
        for message in row['messages']:
            if message['role'] != 'system':
                lines.append(f"**{message['role']}**\n\n{message['content']}\n")
    lines.append('## Primary references for the technical findings\n\n'+sources)
    lines.append('## Limits\n\nAutomated checks cannot prove natural Burmese, factual accuracy, absence of all semantic leakage, or future model fluency. The earlier four audit warnings came from narrower rules and did not cover the content defects listed here. The full 14,257 rows were mechanically scanned, not individually approved by a human.\n')
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    template = (ROOT/'scripts/templates/human_content_review.html').read_text(encoding='utf-8')
    payload = json.dumps(items,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    (output/'review.html').write_text(template.replace('__RECORDS_JSON__',payload),encoding='utf-8')


if __name__=='__main__':
    build()
