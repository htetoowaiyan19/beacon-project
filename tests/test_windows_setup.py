"""Verify paired runtime extraction and rejection before any native execution."""
import hashlib
import json
from zipfile import ZipFile, ZipInfo
import pytest
from scripts.setup_windows_gpu import extract_runtimes


def bundle(root, overrides=None):
    runtime = root / 'runtime'
    runtime.mkdir()
    contents = {'windows-cuda.zip': {'llama-server.exe': b'CUDA', 'ggml-cuda.dll': b'CUDA backend'},
                'windows-cudart.zip': {'cudart64_12.dll': b'paired runtime'},
                'windows-cpu.zip': {'llama-server.exe': b'CPU'}}
    contents.update(overrides or {})
    files = []
    for name, entries in contents.items():
        path = runtime / name
        with ZipFile(path, 'w') as archive:
            for filename, value in entries.items():
                # ZipInfo's constructor normalizes Windows separators; overwrite to test a malformed ZIP.
                info = ZipInfo(filename)
                info.filename = filename
                archive.writestr(info, value)
        files.append({'file': name, 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (runtime / 'windows-provenance.json').write_text(json.dumps({'files': files}))
    return runtime


def test_paired_cuda_dlls_and_cpu_are_extracted_separately(tmp_path):
    runtime = bundle(tmp_path)
    binaries = extract_runtimes(tmp_path)
    assert binaries['cuda'].read_bytes() == b'CUDA'
    assert binaries['cpu'].read_bytes() == b'CPU'
    assert (runtime / 'llama-cuda/cudart64_12.dll').read_bytes() == b'paired runtime'
    assert not (runtime / 'llama-cpu/ggml-cuda.dll').exists()


def test_corrupted_archive_is_rejected_before_extracting(tmp_path):
    runtime = bundle(tmp_path)
    with (runtime / 'windows-cudart.zip').open('ab') as output:
        output.write(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        extract_runtimes(tmp_path)
    assert not (runtime / 'llama-cuda').exists()


@pytest.mark.parametrize('filename', ['../escape.dll', '/absolute.dll', 'C:/escape.dll', '..\\escape.dll'])
def test_unsafe_native_paths_are_rejected_before_extracting(tmp_path, filename):
    runtime = bundle(tmp_path, {'windows-cudart.zip': {filename: b'unsafe'}})
    with pytest.raises(ValueError, match='Unsafe'):
        extract_runtimes(tmp_path)
    assert not (runtime / 'llama-cuda').exists()


def test_conflicting_paired_dll_is_rejected_before_extracting(tmp_path):
    runtime = bundle(tmp_path, {'windows-cudart.zip': {'ggml-cuda.dll': b'conflicting'}})
    with pytest.raises(ValueError, match='Conflicting'):
        extract_runtimes(tmp_path)
    assert not (runtime / 'llama-cuda').exists()
