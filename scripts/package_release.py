"""Build a verified source-and-adapter ZIP; base weights stay separate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.release import RELEASE, RELEASE_ADAPTER
from scripts.run_release import verify_release


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main() -> None:
    verify_release()
    files = [ROOT / name for name in (
        'beacon-release.json', 'RELEASE_V1.md', 'SHOW_DAY.md', 'API.md', 'LICENSE',
        'requirements-inference.txt', 'run_trained_chat.bat', 'run_server.bat', 'show_day_ui.bat',
        'scripts/run_release.py', 'scripts/run_server.py', 'scripts/download_model.py', 'scripts/smoke_release.py',
        'scripts/utils/persona.py',
        'scripts/show_day_ui.py', 'scripts/show_day_backend.py', 'scripts/show_day_frontend.py',
        'scripts/smoke_show_day.py',
        'scripts/utils/show_day_control.py',
    )]
    files += [p for p in (ROOT / 'backend').rglob('*.py') if '__pycache__' not in p.parts]
    files += [ROOT / 'frontend' / name for name in ('index.html', 'app.js', 'stream.js', 'show-day.js', 'style.css', 'favicon.svg')]
    files += [RELEASE_ADAPTER / name for name in (
        'adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json',
        'tokenizer_config.json', 'chat_template.jinja',
    )]
    entries = {p.relative_to(ROOT).as_posix(): p for p in files}
    manifest = {'release': RELEASE, 'base_weights_included': False,
                'files': {name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                          for name, p in sorted(entries.items())}}
    output = ROOT / 'archives' / f"BEACON-v{RELEASE['version']}-seminar.zip"
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix('.zip.tmp')
    try:
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=1) as archive:
            for name, path in sorted(entries.items()):
                archive.write(path, name)
            archive.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
        with ZipFile(temporary) as archive:
            assert set(archive.namelist()) == set(entries) | {'MANIFEST.json'}
            for name, info in manifest['files'].items():
                with archive.open(name) as handle:
                    assert hashlib.file_digest(handle, 'sha256').hexdigest() == info['sha256'], name
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    checksum = sha256(output)
    output.with_suffix('.zip.sha256').write_text(f'{checksum}  {output.name}\n', encoding='utf-8')
    print(f'Verified {len(entries)} files: {output} ({output.stat().st_size / 1024**2:.1f} MiB)')


if __name__ == '__main__':
    main()
