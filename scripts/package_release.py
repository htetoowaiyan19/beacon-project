"""Build a verified transfer ZIP with source, base weights, adapter and datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.release import RELEASE, RELEASE_ADAPTER
from scripts.run_release import verify_release


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def package_files(root: Path = ROOT) -> dict[str, Path]:
    """Allow only project files; never collect environments, outputs or credentials."""
    files = [p for p in root.iterdir() if p.is_file() and (
        p.suffix in {'.md', '.bat', '.command'} or p.name in {
            'LICENSE', 'beacon-release.json', '.gitignore', '.gitattributes'}
        or (p.name.startswith('requirements-') and p.suffix == '.txt'))]
    for directory, suffixes in (
        ('backend', {'.py'}), ('scripts', {'.py', '.html'}),
        ('frontend', {'.md', '.html', '.css', '.js', '.svg'}),
        ('tests', {'.py', '.cjs'}), ('prompts', {'.json'}),
        ('datasets', {'.md', '.json', '.jsonl', '.csv', '.tsv', '.txt',
                      '.sha256', '.html', '.parquet'}),
        ('models/qwen3-4b', {'.json', '.jinja', '.safetensors'}),
        ('models/adapters/' + RELEASE_ADAPTER.name, {'.json', '.jinja', '.safetensors'}),
    ):
        files.extend(p for p in (root / directory).rglob('*')
                     if p.is_file() and p.suffix in suffixes
                     and not any(part.startswith('.') or part == '__pycache__'
                                 for part in p.relative_to(root).parts))
    entries = {}
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Unsafe package source: {path}')
        entries[path.relative_to(root).as_posix()] = path
    required = ['README.md', 'TRANSFER.md', 'scripts/setup_device.py',
                'scripts/verify_package.py', 'datasets/active.json',
                'models/qwen3-4b/model.safetensors',
                f'models/adapters/{RELEASE_ADAPTER.name}/adapter_model.safetensors']
    for name in required:
        if name not in entries:
            raise ValueError(f'Missing package source: {name}')
    return dict(sorted(entries.items()))


def main() -> None:
    verify_release()
    entries = package_files()
    output = ROOT / 'archives' / f"BEACON-v{RELEASE['version']}-portable.zip"
    output.parent.mkdir(exist_ok=True)
    size = sum(p.stat().st_size for p in entries.values())
    if shutil.disk_usage(output.parent).free < size + 1024**3:
        raise OSError('Not enough space for the transfer ZIP and 1 GiB headroom')
    print(f'Preparing {len(entries)} files ({size / 1024**3:.2f} GiB); hashing sources...', flush=True)
    manifest = {'format_version': 1, 'release': RELEASE,
                'base_weights_included': True, 'datasets_included': True,
                'optimizer_checkpoints_included': False,
                'files': {name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                          for name, p in entries.items()}}
    temporary = output.with_suffix('.zip.tmp')
    try:
        with ZipFile(temporary, 'w', allowZip64=True) as archive:
            for name, path in entries.items():
                if path.suffix == '.safetensors':
                    print(f'Packing weights: {name}', flush=True)
                # Storing weights avoids a lengthy CPU compression pass.
                compression = ZIP_STORED if path.suffix == '.safetensors' else ZIP_DEFLATED
                archive.write(path, name, compress_type=compression, compresslevel=1)
            archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n',
                             compress_type=ZIP_DEFLATED)
        print('Checking every archived file against its source SHA-256...', flush=True)
        from scripts.verify_package import verify_zip
        verify_zip(temporary, expected_manifest=manifest)
        # Catch files changed during packing; do not publish a mixed snapshot.
        for name, path in entries.items():
            if sha256(path) != manifest['files'][name]['sha256']:
                raise ValueError(f'Source changed while packaging: {name}')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    checksum = sha256(output)
    output.with_suffix('.zip.sha256').write_text(
        f'{checksum}  {output.name}\n', encoding='utf-8')
    print(f'Verified {len(entries)} files: {output} ({output.stat().st_size / 1024**3:.2f} GiB)', flush=True)


if __name__ == '__main__':
    main()
