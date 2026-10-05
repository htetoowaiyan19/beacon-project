"""Use the same release in desktop training, CLI training, evaluation and audits."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def active_release(root=ROOT):
    root = Path(root)
    descriptor = json.loads((root / 'datasets/active.json').read_text(encoding='utf-8'))
    release = (root / 'datasets' / descriptor['release_directory']).resolve()
    if not release.is_relative_to((root / 'datasets/releases').resolve()):
        raise ValueError('Active release must be inside datasets/releases')
    return release
