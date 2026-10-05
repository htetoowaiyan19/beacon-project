"""Launch the Beacon Burmese AI model-only server."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# If uvicorn is missing in current Python, auto-delegate to project .venv
try:
    import uvicorn
except ModuleNotFoundError:
    if VENV_PYTHON.exists() and sys.executable.lower() != str(VENV_PYTHON).lower():
        try:
            sys.exit(subprocess.call([str(VENV_PYTHON), *sys.argv]))
        except KeyboardInterrupt:
            sys.exit(0)
    print("=" * 70)
    print("ERROR: Missing required packages (uvicorn, fastapi, etc.).")
    print("Please activate the virtual environment:")
    print(r"    .\.venv\Scripts\activate")
    print(r"Or run using the virtual environment python directly:")
    print(r"    .\.venv\Scripts\python.exe scripts/run_server.py")
    print("=" * 70)
    sys.exit(1)

sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import HOST, PORT


def main() -> None:
    parser = argparse.ArgumentParser(description="Beacon Burmese AI Server")
    parser.add_argument("--host", type=str, default=HOST, help=f"Server host (default: {HOST})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Server port (default: {PORT})")
    parser.add_argument("--reload", action="store_true", help="Enable automatic code reload")
    args = parser.parse_args()

    print("=" * 70)
    print("BEACON MYANMAR AI ASSISTANT - MODEL-ONLY SERVER")
    print("=" * 70)
    print(f"Web Interface:   http://localhost:{args.port}/")
    print(f"API Docs:        http://localhost:{args.port}/docs")
    print(f"Health Check:    http://localhost:{args.port}/api/health")
    print("=" * 70 + "\n")

    try:
        uvicorn.run("backend.app:app", host=args.host, port=args.port, reload=args.reload)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
