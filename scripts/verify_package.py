"""Verify a transfer ZIP or extracted files using only Python's standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def check_names(manifest: dict) -> None:
    if manifest.get('format_version') != 1 or not manifest.get('files'):
        raise ValueError('Unsupported or empty package manifest')
    for name in manifest['files']:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
            raise ValueError(f'Unsafe manifest path: {name}')


def check_file(handle, name: str, info: dict, size: int) -> None:
    if size != info['bytes']:
        raise ValueError(f'File size mismatch: {name}')
    if hashlib.file_digest(handle, 'sha256').hexdigest() != info['sha256']:
        raise ValueError(f'File checksum mismatch: {name}')


def verify_zip(path: Path, expected_manifest: dict | None = None) -> int:
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read('MANIFEST.json'))
        check_names(manifest)
        if expected_manifest is not None and manifest != expected_manifest:
            raise ValueError('Archived manifest differs from source manifest')
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest['files']) | {'MANIFEST.json'}:
            raise ValueError('Unexpected, missing or duplicate ZIP entries')
        for name, info in manifest['files'].items():
            with archive.open(name) as handle:
                check_file(handle, name, info, archive.getinfo(name).file_size)
    return len(manifest['files'])


def verify_directory(root: Path) -> int:
    root = root.resolve()
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    check_names(manifest)
    for name, info in manifest['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError(f'Unsafe extracted path: {name}')
        with path.open('rb') as handle:
            check_file(handle, name, info, path.stat().st_size)
    return len(manifest['files'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, help='Verify ZIP instead of extracted files')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        count = verify_zip(args.zip) if args.zip else verify_directory(args.root)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Package verification failed: {exc}\n')
    print(f'Package verified: {count} files, sizes and SHA-256 checksums match.')


if __name__ == '__main__':
    main()
