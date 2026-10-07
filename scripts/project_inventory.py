"""Measure project files without loading models or modifying existing artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]


def inventory(root: Path) -> dict:
    root = root.resolve()
    totals: dict[str, dict] = {}
    subfolders: dict[str, dict] = {}
    large_files = []
    errors = []
    skipped_links = []

    def walk_error(error):
        errors.append(str(error))

    for directory, folders, files in os.walk(root, followlinks=False, onerror=walk_error):
        base = Path(directory)
        for name in list(folders):
            path = base / name
            try:
                linked = path.is_symlink() or getattr(path.lstat(), 'st_reparse_tag', 0) == getattr(stat, 'IO_REPARSE_TAG_MOUNT_POINT', -1)
            except OSError as error:
                folders.remove(name)
                errors.append(f'{path.relative_to(root).as_posix()}: {error}')
                continue
            if linked:
                folders.remove(name)
                skipped_links.append(path.relative_to(root).as_posix())
        for name in files:
            path = base / name
            relative = path.relative_to(root)
            if path.is_symlink():
                skipped_links.append(relative.as_posix())
                continue
            try:
                size = path.stat().st_size
            except OSError as error:
                errors.append(f'{relative.as_posix()}: {error}')
                continue
            top = relative.parts[0]
            row = totals.setdefault(top, {'path': top, 'bytes': 0, 'files': 0})
            row['bytes'] += size
            row['files'] += 1
            if len(relative.parts) > 2:
                group = '/'.join(relative.parts[:2])
                row = subfolders.setdefault(group, {'path': group, 'bytes': 0, 'files': 0})
                row['bytes'] += size
                row['files'] += 1
            if size >= 100_000_000:
                large_files.append({'path': relative.as_posix(), 'bytes': size})
    by_size = lambda rows: sorted(rows, key=lambda row: row['bytes'], reverse=True)
    return {
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'measurement': 'Logical file bytes; includes hidden folders, excludes symlinks/junctions. Not allocated disk space or external caches.',
        'total_bytes': sum(row['bytes'] for row in totals.values()),
        'items': by_size(totals.values()),
        'subfolders': by_size(subfolders.values()),
        'large_files': by_size(large_files),
        'errors': errors,
        'skipped_links': skipped_links,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, help='JSON report; default: ROOT/outputs/project_review/inventory.json')
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error('Project root must be an existing directory.')
    report = inventory(root)
    output = args.output or root / 'outputs/project_review/inventory.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for row in report['items']:
        print(f"{row['bytes'] / 1e9:8.3f} GB  {row['files']:6d} files  {row['path']}")
    print(f"Total: {report['total_bytes'] / 1e9:.3f} GB ({report['total_bytes'] / 1024**3:.3f} GiB)")
    print(f"Read errors: {len(report['errors'])}; skipped links: {len(report['skipped_links'])}")
    print(f'JSON report: {output}')


if __name__ == '__main__':
    main()
