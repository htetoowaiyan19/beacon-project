"""Install the Windows x64 GGUF edition and its verified native runtimes."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import venv
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def extract_runtimes(root=ROOT):
    root = Path(root).resolve()
    runtime = root / 'runtime'
    provenance = json.loads((runtime / 'windows-provenance.json').read_text(encoding='utf-8'))
    destinations = {'windows-cuda.zip': 'llama-cuda', 'windows-cudart.zip': 'llama-cuda',
                    'windows-cpu.zip': 'llama-cpu'}
    if {item['file'] for item in provenance['files']} != set(destinations) or len(provenance['files']) != 3:
        raise ValueError('Invalid Windows runtime bundle manifest')
    entries = []
    duplicates = {}
    needed = 0
    # Validate all inputs before extracting or executing anything.
    for item in provenance['files']:
        package = runtime / item['file']
        if not package.resolve().is_relative_to(root):
            raise ValueError('Native archives must stay inside the project')
        with package.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if package.stat().st_size != item['bytes'] or digest != item['sha256']:
            raise ValueError('Native archive checksum mismatch: ' + item['file'])
        destination = (runtime / destinations[item['file']]).resolve()
        if not destination.is_relative_to(root):
            raise ValueError('Runtime destination must stay inside the project')
        with ZipFile(package) as archive:
            for info in archive.infolist():
                name = PurePosixPath(info.filename)
                target = destination / info.filename
                if (name.is_absolute() or '..' in name.parts or '\\' in info.filename
                        or ':' in info.filename or not target.resolve().is_relative_to(destination)
                        or stat.S_ISLNK(info.external_attr >> 16)):
                    raise ValueError('Unsafe runtime archive path: ' + info.filename)
                if not info.is_dir():
                    key = str(target)
                    if key in duplicates:
                        previous_package, previous_name = duplicates[key]
                        with ZipFile(previous_package) as previous, previous.open(previous_name) as handle:
                            previous_digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                        with archive.open(info) as handle:
                            current_digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                        if previous_digest != current_digest:
                            raise ValueError('Conflicting runtime archive file: ' + info.filename)
                    else:
                        # Paired official ZIPs are normally disjoint; compare any overlaps.
                        duplicates[key] = (package, info.filename)
                        entries.append((package, info.filename, target))
                        needed += info.file_size
    if shutil.disk_usage(root).free < needed + 512 * 1024**2:
        raise OSError('Not enough disk space to extract runtimes and create the environment')
    for package, name, target in entries:
        target.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(package) as archive, archive.open(name) as source, target.open('wb') as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
    binaries = {}
    for mode, folder in [('cuda', 'llama-cuda'), ('cpu', 'llama-cpu')]:
        matches = list((runtime / folder).rglob('llama-server.exe'))
        if len(matches) != 1:
            raise ValueError('Expected exactly one ' + mode + ' llama-server.exe')
        binaries[mode] = matches[0]
    return binaries


def check_devices(executable, cpu=False):
    result = subprocess.run([str(executable), '--list-devices'], capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    output = result.stdout + result.stderr
    if result.returncode or (not cpu and 'CUDA0:' not in output):
        raise RuntimeError('Native runtime check failed. Check NVIDIA driver / Visual C++ runtime.\n' + output)
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true', help='Check CPU fallback instead of requiring CUDA')
    args = parser.parse_args()
    if sys.version_info < (3, 11) or sys.platform != 'win32' or platform.machine().lower() not in {'amd64', 'x86_64'} or sys.maxsize <= 2**32:
        parser.exit(1, 'Use 64-bit Python 3.11 or newer on Windows x64.\n')
    from scripts.verify_package import verify_directory
    from scripts.utils.llama_runtime import verify_model
    if (ROOT / 'MANIFEST.json').exists():
        print(f'Verified {verify_directory(ROOT)} packaged files.', flush=True)
    verify_model(ROOT)
    binaries = extract_runtimes()
    check_devices(binaries['cpu' if args.cpu else 'cuda'], cpu=args.cpu)
    env = ROOT / '.venv'
    python = env / 'Scripts/python.exe'
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-laptop.txt')], check=True, cwd=ROOT)
    subprocess.run([str(python), str(ROOT / 'scripts/run_laptop.py'), '--check'], check=True, cwd=ROOT)
    print('Ready. Run start_windows_gpu.bat for the panel or chat_windows_gpu.bat for single chat.', flush=True)


if __name__ == '__main__':
    main()
