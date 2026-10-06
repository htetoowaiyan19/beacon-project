"""Install the small Python environment and unpack the bundled Apple runtime."""
import platform
from pathlib import Path
import subprocess
import sys
import tarfile
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.version_info < (3, 11):
        raise SystemExit('Install Python 3.11 or newer first.')
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        raise SystemExit('This setup package is for Apple Silicon macOS.')
    from scripts.verify_package import verify_directory
    if (ROOT / 'MANIFEST.json').exists():
        print(f'Verifying {verify_directory(ROOT)} packaged files...', flush=True)
    runtime = ROOT / 'runtime'
    package = runtime / 'macos-arm64.tar.gz'
    destination = runtime / 'llama'
    if not destination.resolve().is_relative_to(ROOT.resolve()):
        raise SystemExit('Runtime destination must stay inside the project.')
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package) as archive:
        archive.extractall(destination, filter='data')
    executable = next(destination.rglob('llama-server'))
    executable.chmod(0o755)
    env = ROOT / '.venv'
    venv.EnvBuilder(with_pip=True).create(env)
    python = env / 'bin/python'
    subprocess.run([str(python), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements-laptop.txt')], check=True)
    subprocess.run([str(python), str(ROOT / 'scripts/run_laptop.py'), '--check'], check=True)
    print('Ready. Run start_laptop.command for the show-day panel, or chat_laptop.command for single-page chat.')


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
