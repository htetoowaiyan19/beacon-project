"""Exercise laptop limits, cancellation and API compatibility without model libraries."""
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from backend.application import create_app
from backend.services.chat_context import ChatContextError
from backend.services.chat_service import ChatService
from backend.services.gguf_model_service import GGUFModelService


class FakeRuntime:
    def __init__(self):
        self.ready = True
        self.process = SimpleNamespace(poll=lambda: None)
        self.metal = True
        self.buffers = {('Metal', 'model'): 2400, ('Metal', 'KV'): 150}
        self.closed = False

    def start(self):
        pass

    def json(self, route, payload):
        if route == '/apply-template':
            assert payload['chat_template_kwargs']['enable_thinking'] is False
            self.conversation = payload['messages']
            return {'prompt': json.dumps(payload['messages'], ensure_ascii=False)}
        assert route == '/tokenize' and payload['parse_special']
        return {'tokens': list(range(len(payload['content'])))}

    def request(self, route, payload):
        assert route == '/completion'
        self.payload = payload
        parts = [{'content': 'မြန်မာ', 'stop': False}, {'content': 'စာ', 'stop': False},
                 {'content': '', 'stop': True, 'stop_type': 'eos', 'tokens_predicted': 2}]
        self.response = io.BytesIO(''.join('data: ' + json.dumps(p, ensure_ascii=False) + '\n\n'
                                         for p in parts).encode('utf-8'))
        return self.response

    def close(self):
        self.closed = True


def test_laptop_stream_caps_output_and_uses_existing_vercel_protocol():
    runtime = FakeRuntime()
    service = GGUFModelService(runtime)
    with TestClient(create_app(chat_service=ChatService(service))) as client:
        status = client.get('/api/health').json()['gpu']
        assert status['adapter_merged'] and status['shared_memory']
        assert status['vram_allocated_gb'] is None
        assert status['runtime_buffers_gb'] == 2.49
        response = client.post('/api/chat/stream', json={'message': 'Hi', 'max_new_tokens': 1024})
        assert response.headers['x-vercel-ai-ui-message-stream'] == 'v1'
        frames = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith('data: ') and '[DONE]' not in line]
        assert ''.join(p['delta'] for p in frames if p['type'] == 'text-delta') == 'မြန်မာစာ'
        metrics = next(p['data'] for p in frames if p['type'] == 'data-metrics')
        assert metrics['total_tokens'] == 2 and metrics['effective_max_new_tokens'] == 192
        assert runtime.payload['cache_prompt'] and runtime.payload['n_predict'] == 192
        assert 'English' in runtime.conversation[0]['content']
    assert runtime.closed


def test_laptop_context_keeps_recent_complete_turns_and_rejects_oversized_current():
    runtime = FakeRuntime()
    service = GGUFModelService(runtime)
    messages = []
    for i in range(7):
        messages.extend([{'role': 'user', 'content': str(i)}, {'role': 'assistant', 'content': 'answer'}])
    current = {'role': 'user', 'content': 'မင်္ဂလာပါ'}
    _, _, metrics = service.prepare(messages + [current], 'Be helpful')
    assert runtime.conversation[-1] == current
    assert runtime.conversation[1]['content'] == '5'
    assert metrics['history_turns_used'] == 2 and metrics['history_messages_dropped'] == 10
    with pytest.raises(ChatContextError, match='Shorten'):
        service.prepare([{'role': 'user', 'content': 'စာ' * 2000}], 'Be helpful')


def test_laptop_rejects_base_comparison_and_thinking_and_closes_aborted_stream():
    runtime = FakeRuntime()
    service = GGUFModelService(runtime)
    messages = [{'role': 'user', 'content': 'Hello'}]
    with pytest.raises(ChatContextError, match='Base comparison'):
        next(service.stream_chat(messages=messages, use_base_model=True))
    with pytest.raises(ChatContextError, match='Thinking mode'):
        next(service.stream_chat(messages=messages, think=True))
    generator = service.stream_chat(messages=messages)
    assert next(generator) == 'မြန်မာ'
    generator.close()
    assert runtime.response.closed
    assert service.lock.acquire(blocking=False)
    service.lock.release()


def test_laptop_imports_no_training_or_tensor_runtime():
    code = '''
import importlib.abc, os, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'peft', 'trl', 'datasets'}:
            raise AssertionError('Laptop imported ' + fullname)
sys.meta_path.insert(0, Block())
os.environ['BEACON_RUNTIME'] = 'gguf'
from backend.app import app
assert app.state.chat_service.model_service.max_output == 192
from scripts.run_laptop import main
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_m2_runtime_uses_metal_one_slot_small_cache_and_stops_its_child(monkeypatch, tmp_path):
    import scripts.utils.llama_runtime as module
    import scripts.utils.show_day_control as control
    executable = tmp_path / 'llama-server'
    executable.write_bytes(b'test')
    monkeypatch.setenv('BEACON_LLAMA_SERVER_PATH', str(executable))
    monkeypatch.delenv('BEACON_GGUF_GPU_LAYERS', raising=False)
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    monkeypatch.setattr(module, 'verify_model', lambda root: (tmp_path / 'model.gguf', {'quantization': 'Q4_K_M'}))
    monkeypatch.setattr(control, 'available_port', lambda port: None)
    commands = []
    class Child:
        stdout = io.StringIO('')
        stopped = False
        def poll(self): return 0 if self.stopped else None
        def terminate(self): self.stopped = True
        def wait(self, timeout): return 0
    child = Child()
    monkeypatch.setattr(module.subprocess, 'Popen', lambda command, **kw: commands.append(command) or child)
    runtime = module.LlamaRuntime(tmp_path)
    monkeypatch.setattr(runtime, 'json', lambda *args, **kw: {'status': 'ok'})
    runtime.start()
    command = commands[0]
    for flag, value in [('--device', 'MTL0'), ('--ctx-size', '2048'), ('--parallel', '1'),
                        ('--cache-ram', '0'), ('--threads', '4'), ('--reasoning', 'off')]:
        assert command[command.index(flag) + 1] == value
    assert command[command.index('--api-key') + 1] == runtime.key
    runtime.close()
    assert child.stopped and runtime.ready is False
