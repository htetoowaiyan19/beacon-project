"""Launch the frozen seminar release after verifying its adapter."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.release import RELEASE, RELEASE_ADAPTER


def verify_release() -> Path:
    weights = RELEASE_ADAPTER / 'adapter_model.safetensors'
    for path in (weights, RELEASE_ADAPTER / 'adapter_config.json'):
        if not path.is_file():
            raise ValueError(f'Missing release file: {path}. Restore the v1 release ZIP.')
    with weights.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != RELEASE['adapter_sha256']:
        raise ValueError('Release adapter checksum mismatch. Restore the v1 release ZIP.')
    base = ROOT / 'models' / 'qwen3-4b'
    if not (base / 'config.json').is_file() or not any(base.glob('*.safetensors')):
        raise ValueError('Base model is missing from models/qwen3-4b. See RELEASE_V1.md.')
    return RELEASE_ADAPTER


def main() -> None:
    parser = argparse.ArgumentParser(description='Run BEACON v1 seminar chat')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--check', action='store_true', help='Verify files without loading the model')
    args = parser.parse_args()
    try:
        adapter = verify_release()
    except ValueError as exc:
        parser.exit(1, f'{exc}\n')
    print(f"BEACON v{RELEASE['version']} | verified adapter | checkpoint {RELEASE['selected_checkpoint']}")
    if args.check:
        return
    # Set before importing backend.config; the release launcher always uses this snapshot.
    os.environ['BEACON_LORA_PATH'] = str(adapter)
    import uvicorn
    print(f'Chat: http://{args.host}:{args.port}/ | API: /docs | Ctrl+C stops the server')
    uvicorn.run('backend.app:app', host=args.host, port=args.port)


if __name__ == '__main__':
    main()
