"""One-time, reversible retirement of the pre-fluency corpus, with row-level decisions."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'datasets/archive/pre_fluency_2026-10-03'


def main():
    if ARCHIVE.exists():
        raise RuntimeError('Archive already exists; refusing to overwrite it')
    sources = [ROOT / 'datasets/clean', ROOT / 'datasets/clean_grouped', ROOT / 'datasets/raw',
               ROOT / 'scripts/build_curated_conversations.py']
    # Resolve every source and destination before any move, including symlinks.
    for path in [ARCHIVE, *sources]:
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError('Archive path escapes workspace')
    decisions, counts, inventory = [], Counter(), []
    known_bad = {('train', 196): 'Thank you very much translated as not eaten yet',
                 ('train', 2312): 'Not eaten yet translated as thank you very much',
                 ('test', 145): 'Congratulations translated as hotel',
                 ('train', 1064): 'Restaurant translated as food',
                 ('train', 2399): 'Malformed Burmese translation of Maybe'}
    for split in ('train', 'validation', 'test'):
        path = ROOT / f'datasets/clean/{split}/{split}_combined.jsonl'
        for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            tag = str(row.get('tags', row.get('tag', row.get('category', ''))))
            evidence = known_bad.get((split, line_number))
            reason = ('confirmed_translation_error' if evidence else
                      'unreliable_synthetic_batch' if tag.startswith('conversation_') else
                      'pending_individual_language_and_fact_review')
            counts[reason] += 1
            decisions.append({'source': f'clean/{split}/{split}_combined.jsonl', 'line': line_number,
                              'sha256': hashlib.sha256(line.encode('utf-8')).hexdigest(),
                              'tag': tag, 'decision': 'exclude_from_active_training',
                              'reason': reason, 'evidence': evidence})
    for source in sources:
        for path in ([source] if source.is_file() else sorted(source.rglob('*'))):
            if path.is_file():
                inventory.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    ARCHIVE.mkdir(parents=True)
    (ARCHIVE / 'row_decisions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in decisions), encoding='utf-8')
    (ARCHIVE / 'inventory.json').write_text(json.dumps({'counts': dict(counts), 'files': inventory}, indent=2), encoding='utf-8')
    for source in sources:
        shutil.move(str(source), str(ARCHIVE / source.name))
    print(json.dumps({'archive': str(ARCHIVE), 'decisions': dict(counts)}, indent=2))


if __name__ == '__main__':
    main()
