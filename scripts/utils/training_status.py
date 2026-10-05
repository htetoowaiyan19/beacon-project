"""Small, atomic status files shared by the trainer and desktop UI."""
import json
import os
import time
import tempfile
import warnings
from pathlib import Path


class StatusWriter:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.started = time.monotonic()
        self.data = {}
        self.warned = False

    def update(self, **values):
        self.data.update(values)
        self.data.update(updated_at=time.time(), elapsed_seconds=round(time.monotonic() - self.started, 1))
        if self.path:
            temporary = None
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                                 prefix=self.path.stem+'-', suffix='.tmp', delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump(self.data, stream, ensure_ascii=False, indent=2, default=str)
                for attempt in range(4):
                    try:
                        os.replace(temporary, self.path)
                        self.warned = False
                        return
                    except PermissionError:
                        if attempt == 3: raise
                        time.sleep(.025 * (attempt + 1))
            except OSError as exc:
                # Telemetry must never terminate training or prevent checkpoint saving.
                if not self.warned:
                    warnings.warn(f'Live status temporarily unavailable; training continues: {exc}')
                    self.warned = True
            finally:
                if temporary:
                    try: temporary.unlink(missing_ok=True)
                    except OSError: pass


def read_status(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
