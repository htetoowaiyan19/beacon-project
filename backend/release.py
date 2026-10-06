"""Product metadata, separate from the existing API contract version."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RELEASE = json.loads((ROOT / 'beacon-release.json').read_text(encoding='utf-8'))
RELEASE_ADAPTER = (ROOT / RELEASE['adapter_directory']).resolve()
if not RELEASE_ADAPTER.is_relative_to((ROOT / 'models').resolve()):
    raise ValueError('Release adapter must be inside models/')
def public_release():
    return {key: RELEASE[key] for key in ('name','version','edition','released_on','model','limitations')}
