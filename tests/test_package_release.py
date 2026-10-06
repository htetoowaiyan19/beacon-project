"""A transfer must detect corruption and exclude local environments and logs."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from scripts.package_release import package_files, RELEASE_ADAPTER
from scripts.verify_package import verify_zip, verify_directory


def fixture_manifest(data=b'weights'):
    return {'format_version': 1, 'files': {
        'models/test.bin': {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}}


def write_zip(path, data=b'weights', extra=False):
    with ZipFile(path, 'w') as archive:
        archive.writestr('MANIFEST.json', json.dumps(fixture_manifest()))
        archive.writestr('models/test.bin', data)
        if extra:
            archive.writestr('unexpected.txt', 'unexpected')


def test_zip_and_restored_directory_detect_corrupt_weights(tmp_path):
    path = tmp_path / 'transfer.zip'
    write_zip(path)
    assert verify_zip(path) == 1
    with ZipFile(path) as archive:
        archive.extractall(tmp_path / 'restored')
    assert verify_directory(tmp_path / 'restored') == 1
    (tmp_path / 'restored/models/test.bin').write_bytes(b'WEIGHTS')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_directory(tmp_path / 'restored')
    write_zip(path, data=b'WEIGHTS')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_zip(path)


def test_zip_rejects_unlisted_entry_and_unsafe_manifest(tmp_path):
    path = tmp_path / 'transfer.zip'
    write_zip(path, extra=True)
    with pytest.raises(ValueError, match='Unexpected'):
        verify_zip(path)
    manifest = fixture_manifest()
    manifest['files']['../outside'] = manifest['files'].pop('models/test.bin')
    (tmp_path / 'MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unsafe manifest path'):
        verify_directory(tmp_path)


def test_collection_includes_weights_and_data_but_no_machine_artifacts(tmp_path):
    names = ['README.md', 'TRANSFER.md', 'LICENSE', 'scripts/setup_device.py',
             'scripts/verify_package.py', 'backend/app.py', 'frontend/app.js',
             'datasets/active.json', 'datasets/freshes/team.jsonl',
             'models/qwen3-4b/model.safetensors',
             f'models/adapters/{RELEASE_ADAPTER.name}/adapter_model.safetensors',
             '.env', '.git/config', '.venv/library.py', 'outputs/show_day/visitor.json',
             'archives/old.zip', 'scripts/__pycache__/old.py', 'datasets/.secret.json']
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'data')
    entries = package_files(tmp_path)
    assert 'models/qwen3-4b/model.safetensors' in entries
    assert 'datasets/freshes/team.jsonl' in entries
    assert not any(name.startswith(('.venv', '.git/', 'outputs/', 'archives/')) for name in entries)
    assert '.env' not in entries
    assert 'scripts/__pycache__/old.py' not in entries
    assert 'datasets/.secret.json' not in entries
