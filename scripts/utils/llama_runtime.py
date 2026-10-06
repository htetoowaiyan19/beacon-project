"""Own one lightweight llama.cpp process; no PyTorch or training imports."""
from __future__ import annotations
import atexit
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def verify_model(root=ROOT, metadata_name=None):
    from backend.release import RELEASE
    directory = Path(root) / 'models/gguf'
    metadata_path = (directory / (metadata_name or os.getenv('BEACON_GGUF_METADATA', 'export.json'))).resolve()
    if not metadata_path.is_relative_to(directory.resolve()):
        raise ValueError('Export metadata must stay inside models/gguf')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    model = (directory / metadata['model_file']).resolve()
    if not model.is_relative_to(directory.resolve()) or not metadata.get('adapter_merged'):
        raise ValueError('Invalid trained GGUF export metadata')
    if metadata['source_adapter_sha256'] != RELEASE['adapter_sha256']:
        raise ValueError('GGUF was not exported from the frozen seminar adapter')
    if model.stat().st_size != metadata['bytes']:
        raise ValueError('GGUF file size differs from its export manifest')
    with model.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != metadata['sha256']:
            raise ValueError('GGUF checksum mismatch; restore the laptop package')
    return model, metadata


class LlamaRuntime:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.port = int(os.getenv('BEACON_LLAMA_PORT', '8090'))
        self.url = f'http://127.0.0.1:{self.port}'
        self.process = None
        self.buffers = {}
        self.ready = False
        self.key = secrets.token_urlsafe(32)
        atexit.register(self.close)

    def request(self, path, payload=None, timeout=180):
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8') if payload is not None else None
        request = Request(self.url + path, data=data, headers={'Content-Type': 'application/json',
                                                             'Authorization': 'Bearer ' + self.key})
        return urlopen(request, timeout=timeout)

    def json(self, path, payload=None, timeout=15):
        with self.request(path, payload, timeout) as response:
            return json.load(response)

    def start(self):
        if self.ready and self.process.poll() is None:
            return
        from scripts.utils.show_day_control import available_port
        available_port(self.port)
        model, self.metadata = verify_model(self.root)
        configured = os.getenv('BEACON_LLAMA_SERVER_PATH')
        binaries = list((self.root / 'runtime/llama').rglob('llama-server'))
        executable = Path(configured) if configured else binaries[0] if len(binaries) == 1 else None
        if executable is None or not executable.is_file():
            raise ValueError('llama-server is missing. Run setup_laptop.command first.')
        self.metal = sys.platform == 'darwin' and os.getenv('BEACON_GGUF_GPU_LAYERS', 'all') != '0'
        command = [str(executable.resolve()), '--model', str(model), '--host', '127.0.0.1',
                   '--port', str(self.port), '--ctx-size', '2048', '--parallel', '1',
                   '--batch-size', '128', '--ubatch-size', '128', '--threads', '4',
                   '--threads-batch', '4', '--n-gpu-layers', os.getenv('BEACON_GGUF_GPU_LAYERS', 'all'),
                   '--cache-ram', '0', '--jinja', '--api-key', self.key,
                   '--reasoning', 'off', '--no-ui']
        if self.metal:
            # The pinned Apple runtime names its first Metal device MTL0.
            command += ['--device', 'MTL0']
        self.process = subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        threading.Thread(target=self._logs, daemon=True).start()
        previous = {}
        if threading.current_thread() is threading.main_thread():
            def interrupt(signum, frame):
                self.close()
                raise KeyboardInterrupt
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous[signum] = signal.signal(signum, interrupt)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('llama.cpp exited while loading; check the backend log')
                try:
                    if self.json('/health', timeout=1).get('status') == 'ok':
                        self.ready = True
                        return
                except OSError:
                    pass
                time.sleep(.1)
            raise TimeoutError('llama.cpp did not become ready within 120 seconds')
        except BaseException:
            self.close()
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)

    def _logs(self):
        process = self.process
        for line in process.stdout:
            match = re.search(r'([\w_]+)\s+(model|KV|compute) buffer size\s*=\s*([\d.]+)\s*MiB', line)
            if match:
                self.buffers[(match[1], match[2])] = float(match[3])
            print('[llama.cpp] ' + line.rstrip(), flush=True)
        process.stdout.close()

    def close(self):
        process = self.process
        self.ready = False
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
