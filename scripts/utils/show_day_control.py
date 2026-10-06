"""Server ownership, logs and telemetry without importing model libraries."""
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import secrets
import socket
import subprocess
import sys
import threading
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def available_port(port):
    if not 1024 <= port <= 65535:
        raise ValueError('Choose a port between 1024 and 65535.')
    try:
        with socket.socket() as probe:
            if os.name == 'nt':
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(('127.0.0.1', port))
    except OSError:
        raise ValueError(f'Port {port} is already in use. Stop the old server or choose another port.') from None


class ManagedServer:
    def __init__(self, name, root=ROOT):
        self.name = name; self.root = Path(root)
        self.process = None; self.port = None; self.log_path = None
        self.events = queue.Queue(maxsize=1500)
        self.lock = threading.RLock()

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def emit(self, line):
        try:
            self.events.put_nowait(line)
        except queue.Full:
            pass

    def start(self, command, port, environment=None):
        with self.lock:
            if self.running:
                raise ValueError(f'{self.name} is already running.')
            available_port(port)
            directory = self.root / 'outputs/show_day' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            directory.mkdir(parents=True)
            self.port = port; self.log_path = directory / f'{self.name.lower()}.log'
            env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
            env.update(environment or {})
            self.process = subprocess.Popen(command, cwd=self.root, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            threading.Thread(target=self._collect, args=(self.process, self.log_path), daemon=True).start()

    def _collect(self, process, path):
        with path.open('w', encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line); log.flush(); self.emit(line)
            line = f'[{self.name} exited with code {process.wait()}]\n'
            log.write(line); self.emit(line)
        process.stdout.close()

    def stop(self):
        # Only stop this panel's child; never kill an unrelated process by its port.
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                return
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)


class ShowDayControl:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.backend = ManagedServer('Backend', root)
        self.frontend = ManagedServer('Frontend', root)
        self.token = None; self.backend_port = None; self.frontend_port = None
        python = self.root / '.venv/Scripts/python.exe'
        if not python.is_file():
            python = self.root / '.venv/bin/python'
        self.python = str(python) if python.is_file() else sys.executable

    def start_backend(self, port=8000, preview=False):
        if self.frontend.running and port != self.backend_port:
            raise ValueError('Stop the frontend before changing its backend port.')
        token = secrets.token_urlsafe(32)
        command = [self.python, '-u', str(self.root / 'scripts/show_day_backend.py'), '--port', str(port)]
        if preview:
            command.append('--preview')
        self.backend.start(command, port, {'BEACON_MONITOR_TOKEN': token, 'BEACON_SHOW_DAY': '1'})
        self.token = token; self.backend_port = port

    def start_frontend(self, port=8080, backend_port=8000):
        if port == backend_port:
            raise ValueError('Frontend and backend need different ports.')
        if self.backend.running and backend_port != self.backend_port:
            raise ValueError('Use the port of the running backend.')
        command = [self.python, '-u', str(self.root / 'scripts/show_day_frontend.py'),
                   '--port', str(port), '--backend-port', str(backend_port)]
        self.frontend.start(command, port)
        self.frontend_port = port; self.backend_port = backend_port

    def request(self, path, method='GET'):
        if not self.token or not self.backend.running:
            raise ValueError('Start the backend from this panel first.')
        request = Request(f'http://127.0.0.1:{self.backend_port}{path}', method=method,
                          headers={'Authorization': f'Bearer {self.token}'})
        with urlopen(request, timeout=12) as response:
            return json.load(response)

    def stop_backend(self):
        if self.backend.running:
            try:
                self.request('/api/operator/shutdown', 'POST')
                self.backend.process.wait(timeout=8)
            except Exception:
                self.backend.stop()

    def stop_all(self):
        self.frontend.stop(); self.stop_backend()


def gpu_telemetry():
    """Optional NVIDIA driver stats; unavailable fields stay unavailable."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode != 0:
            return {}
        values = result.stdout.splitlines()[0].split(',')
        return dict(utilization=float(values[0]), used_mb=float(values[1]),
                    total_mb=float(values[2]), temperature=float(values[3]))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return {}
