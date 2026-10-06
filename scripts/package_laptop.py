"""Create the small M2 Air package, excluding BF16 weights and training data."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.release import RELEASE
from scripts.utils.llama_runtime import verify_model
from scripts.verify_package import verify_zip


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q5', action='store_true', help='Package the optional five-bit diagnostic export')
    args = parser.parse_args()
    metadata_name = 'export-Q5_K_M.json' if args.q5 else 'export.json'
    model, metadata = verify_model(metadata_name=metadata_name)
    runtime = ROOT / 'runtime/macos-arm64.tar.gz'
    provenance = json.loads((ROOT / 'runtime/provenance.json').read_text(encoding='utf-8'))
    if digest(runtime) != provenance['sha256']:
        raise ValueError('Bundled runtime checksum mismatch')
    names = ['beacon-release.json', 'LICENSE', 'requirements-laptop.txt',
             'setup_laptop.command', 'start_laptop.command', 'chat_laptop.command',
             'M2_AIR.md', 'SHOW_DAY.md', 'API.md', 'scripts/setup_laptop.py',
             'scripts/run_laptop.py', 'scripts/smoke_laptop.py', 'scripts/verify_package.py',
             'scripts/show_day_ui.py', 'scripts/show_day_backend.py', 'scripts/show_day_frontend.py',
             'scripts/utils/show_day_control.py', 'scripts/utils/llama_runtime.py',
             'scripts/utils/persona.py', 'models/gguf/export.json',
             'runtime/macos-arm64.tar.gz', 'runtime/provenance.json']
    entries = {name: ROOT / name for name in names}
    entries['models/gguf/export.json'] = ROOT / 'models/gguf' / metadata_name
    entries['README.md'] = ROOT / 'M2_AIR.md'
    entries[model.relative_to(ROOT).as_posix()] = model
    for path in (ROOT / 'backend').rglob('*.py'):
        if '__pycache__' not in path.parts:
            entries[path.relative_to(ROOT).as_posix()] = path
    for name in ('index.html', 'app.js', 'stream.js', 'show-day.js', 'style.css', 'favicon.svg'):
        entries['frontend/' + name] = ROOT / 'frontend' / name
    smoke = ROOT / 'outputs/laptop_export/smoke.json'
    if smoke.exists():
        entries['checks/desktop-gguf-smoke.json'] = smoke
    comparison = ROOT / 'outputs/laptop_export/quality-comparison.json'
    if comparison.exists():
        entries['checks/desktop-quality-comparison.json'] = comparison
    for source, target in [('validation-quantization.json', 'validation-quantization.json'),
                           ('http-check.json', 'desktop-http-check.json'),
                           ('template-check.json', 'template-token-check.json')]:
        evidence = ROOT / 'outputs/laptop_export' / source
        if evidence.exists():
            entries['checks/' + target] = evidence
    manifest = {'format_version': 1, 'release': RELEASE, 'edition': 'M2 Air 8 GB / Metal',
                'base_weights_included': False, 'merged_trained_gguf_included': True,
                'datasets_included': False, 'optimizer_checkpoints_included': False,
                'model': metadata, 'runtime': provenance,
                'files': {n: {'bytes': p.stat().st_size, 'sha256': digest(p)} for n, p in sorted(entries.items())}}
    output = ROOT / ('archives/BEACON-v1.0.0-m2-air-q5.zip' if args.q5 else 'archives/BEACON-v1.0.0-m2-air.zip')
    temporary = output.with_suffix('.zip.tmp')
    if shutil.disk_usage(output.parent).free < sum(i['bytes'] for i in manifest['files'].values()) + 1024**3:
        raise OSError('Insufficient space for laptop package')
    try:
        with ZipFile(temporary, 'w', allowZip64=True) as archive:
            for name, path in sorted(entries.items()):
                if path.suffix == '.command':
                    info = ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (0o100755 << 16)
                    archive.writestr(info, path.read_bytes(), compress_type=ZIP_DEFLATED)
                else:
                    archive.write(path, name, compress_type=ZIP_STORED if path.suffix in {'.gguf', '.gz'} else ZIP_DEFLATED)
            archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n', compress_type=ZIP_DEFLATED)
        count = verify_zip(temporary, expected_manifest=manifest)
        for name, path in entries.items():
            if digest(path) != manifest['files'][name]['sha256']:
                raise ValueError(f'Package source changed: {name}')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    output.with_suffix('.zip.sha256').write_text(digest(output) + '  ' + output.name + '\n', encoding='utf-8')
    print(f'Verified {count} files: {output} ({output.stat().st_size / 1024**3:.2f} GiB)')


if __name__ == '__main__':
    main()
