"""Create a fresh device-local environment; install the tested dependency versions."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', action='store_true', help='CPU PyTorch; inference will be slower')
    parser.add_argument('--training', action='store_true', help='Also install optional training dependencies')
    parser.add_argument('--evaluation', action='store_true', help='Also install optional evaluation dependencies')
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.exit(1, 'Python 3.11 or newer is required.\n')
    if sys.platform == 'darwin' and not args.cpu:
        parser.exit(1, 'Use --cpu on macOS; this project uses CUDA or CPU inference.\n')
    environment = ROOT / '.venv'
    python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    pip = [str(python), '-m', 'pip']
    subprocess.run(pip + ['install', '--upgrade', 'pip'], check=True, cwd=ROOT)
    # Install PyTorch first so the generic requirements keep the CUDA build.
    torch_args = ['install', 'torch==2.11.0']
    if sys.platform != 'darwin':
        torch_args += ['--index-url', 'https://download.pytorch.org/whl/' + ('cpu' if args.cpu else 'cu128')]
    subprocess.run(pip + torch_args, check=True, cwd=ROOT)
    for name in ['inference'] + (['training'] if args.training else []) + (['evaluation'] if args.evaluation else []):
        subprocess.run(pip + ['install', '-r', f'requirements-{name}.txt'], check=True, cwd=ROOT)
    subprocess.run([str(python), str(ROOT / 'scripts/run_release.py'), '--check'], check=True, cwd=ROOT)
    subprocess.run([str(python), '-c',
                    "import torch; print('PyTorch:', torch.__version__); "
                    "print('CUDA available:', torch.cuda.is_available())"], check=True, cwd=ROOT)
    print('Setup complete. Run show_day_ui.bat or run_trained_chat.bat on Windows.')


if __name__ == '__main__':
    main()
