"""Live feed correctness, operator isolation, proxy streaming and server ownership."""
import json
import socket
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from fastapi.testclient import TestClient
from backend.application import create_app
from backend.dev_app import PreviewModel
from backend.services.chat_service import ChatService
from backend.services.show_day_monitor import ShowDayMonitor
from scripts.utils.show_day_control import ManagedServer, ShowDayControl


def app_with_monitor():
    monitor = ShowDayMonitor('test-operator-token')
    return create_app(chat_service=ChatService(PreviewModel()), monitor=monitor), monitor


def test_operator_feed_auth_and_recorded_chat():
    app, monitor = app_with_monitor()
    with TestClient(app) as client:
        assert client.get('/api/operator/snapshot').status_code == 401
        assert client.get('/api/operator/snapshot', headers={'Authorization': 'Bearer wrong'}).status_code == 401
        response = client.post('/api/chat/stream', json={'message': 'Hello', 'client_id': 'visitor-1', 'session_id': 'conversation-1'})
        assert response.headers['x-vercel-ai-ui-message-stream'] == 'v1'
        snapshot = client.get('/api/operator/snapshot', headers={'Authorization': 'Bearer test-operator-token'}).json()
        assert snapshot['counters'] == dict(total=1, completed=1, active=0, failed=0, cancelled=0)
        chat = snapshot['chats'][0]
        assert chat['message'] == 'Hello' and chat['client_id'] == 'visitor-1'
        assert chat['status'] == 'complete' and 'synthetic reply' in chat['answer']
        assert '[DONE]' in response.text
        assert client.post('/api/visitor/event', json={'event': 'typing', 'client_id': 'visitor-1', 'session_id': 'conversation-1'}).status_code == 200
        assert monitor.snapshot()['events'][-1]['type'] == 'typing'
        assert client.post('/api/visitor/event', json={'event': 'draft', 'client_id': 'visitor-1', 'session_id': 'conversation-1'}).status_code == 422
        assert client.post('/api/chat/stream', json={'message': 'Hi', 'session_id': '../private'}).status_code == 422


def test_normal_launch_disables_monitoring(monkeypatch):
    monkeypatch.delenv('BEACON_SHOW_DAY', raising=False)
    monkeypatch.delenv('BEACON_MONITOR_TOKEN', raising=False)
    with TestClient(create_app(chat_service=ChatService(PreviewModel()))) as client:
        assert client.get('/api/config').json()['show_day'] is False
        assert client.get('/api/operator/snapshot').status_code == 404


def test_live_partial_cancel_and_failure_are_distinct():
    monitor = ShowDayMonitor('token')
    closed = []
    def partial():
        try:
            yield 'data: {"type":"text-delta","delta":"partial"}\n\n'
            yield 'data: {"type":"finish","finishReason":"stop"}\n\n'
            yield 'data: [DONE]\n\n'
        finally:
            closed.append(True)
    stream = monitor.track(partial(), message='question')
    next(stream)
    assert monitor.snapshot()['chats'][0]['answer'] == 'partial'
    assert monitor.snapshot()['counters']['active'] == 1
    next(stream)  # A finish event without DONE is still an incomplete transport.
    stream.close()
    assert closed == [True]
    assert monitor.snapshot()['chats'][0]['status'] == 'cancelled'
    assert monitor.snapshot()['counters']['cancelled'] == 1
    failure = iter(['data: {"type":"error","errorText":"failed"}\n\n',
                    'data: {"type":"finish","finishReason":"error"}\n\n', 'data: [DONE]\n\n'])
    # Real sources are generators, with close() for cancellation.
    list(monitor.track((part for part in failure), message='bad request'))
    assert monitor.snapshot()['chats'][-1]['status'] == 'failed'
    assert monitor.snapshot()['counters']['failed'] == 1


def test_feed_is_bounded_and_event_cursor_filters():
    monitor = ShowDayMonitor('token', max_chats=2, max_events=3)
    for number in range(5):
        frames = (part for part in ('data: {"type":"finish","finishReason":"stop"}\n\n', 'data: [DONE]\n\n'))
        list(monitor.track(frames, message=str(number)))
    snapshot = monitor.snapshot()
    assert len(snapshot['chats']) == 2 and len(snapshot['events']) == 3
    assert snapshot['counters']['completed'] == 5
    assert monitor.snapshot(snapshot['sequence'])['events'] == []


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_until(check, timeout=20):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            result = check()
            if result: return result
        except Exception as exc:
            last = exc
        time.sleep(.05)
    raise AssertionError(f'Timed out: {last}')


def test_real_proxy_streams_and_blocks_operator_routes():
    control = ShowDayControl()
    backend, frontend = free_port(), free_port()
    try:
        control.start_backend(backend, preview=True)
        wait_until(lambda: control.request('/api/operator/snapshot'))
        control.start_frontend(frontend, backend)
        url = f'http://127.0.0.1:{frontend}'
        def page():
            with urlopen(url, timeout=2) as response: return response.read().decode()
        html = wait_until(page)
        assert 'class="visitor-mode"' in html and 'href="chat.html"' in html
        with urlopen(url + '/chat.html', timeout=2) as response:
            chat = response.read().decode()
        assert 'class="visitor-mode"' in chat and 'monitor-notice' in chat
        for path in ('/api/operator/snapshot', '/api/operator/shutdown', '/docs', '/README.md', '/%2e%2e/README.md'):
            with pytest.raises(HTTPError) as error:
                urlopen(url + path, timeout=2)
            assert error.value.code == 404
        request = Request(url + '/api/chat/stream', data=json.dumps({'message': 'Show day hello', 'client_id': 'test-visitor', 'session_id': 'test-session'}).encode(), headers={'Content-Type': 'application/json'})
        first_delta = None; finished_at = None; parts = []
        with urlopen(request, timeout=10) as response:
            assert response.headers['x-vercel-ai-ui-message-stream'] == 'v1'
            for line in response:
                if not line.startswith(b'data: '): continue
                if line.strip() == b'data: [DONE]': break
                part = json.loads(line[6:]); parts.append(part)
                if part['type'] == 'text-delta' and first_delta is None:
                    first_delta = time.monotonic()
                    snapshot = control.request('/api/operator/snapshot')
                    assert snapshot['counters']['active'] == 1
                    assert snapshot['chats'][0]['answer'] and snapshot['chats'][0]['status'] == 'replying'
                if part['type'] == 'finish': finished_at = time.monotonic()
        assert finished_at - first_delta > .1  # Not buffered until the entire answer completes.
        wait_until(lambda: control.request('/api/operator/snapshot')['counters']['completed'] == 1)
        assert 'synthetic reply' in ''.join(p.get('delta', '') for p in parts)
        cancelled = Request(url + '/api/chat/stream', data=json.dumps({'message': 'Cancel this response ' * 30, 'client_id': 'test-visitor', 'session_id': 'test-session'}).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(cancelled, timeout=5) as response:
            for line in response:
                if b'"text-delta"' in line:
                    break
        wait_until(lambda: control.request('/api/operator/snapshot')['counters']['cancelled'] == 1, timeout=8)
        old_token = control.token
        control.stop_backend()
        assert not control.backend.running
        with pytest.raises(HTTPError) as offline:
            urlopen(url + '/api/health', timeout=4)
        assert offline.value.code == 503
        control.start_backend(backend, preview=True)
        new_snapshot = wait_until(lambda: control.request('/api/operator/snapshot'))
        assert new_snapshot['counters']['total'] == 0 and control.token != old_token
        with pytest.raises(HTTPError) as old_auth:
            urlopen(Request(f'http://127.0.0.1:{backend}/api/operator/snapshot', headers={'Authorization': f'Bearer {old_token}'}), timeout=3)
        assert old_auth.value.code == 401
    finally:
        control.stop_all()
    assert not control.frontend.running and not control.backend.running
    assert 'Visitor page:' in control.frontend.log_path.read_text(encoding='utf-8')


def test_occupied_port_does_not_take_over_an_unowned_server(tmp_path):
    managed = ManagedServer('Test', tmp_path)
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0)); occupied.listen()
        port = occupied.getsockname()[1]
        with pytest.raises(ValueError, match='already in use'):
            managed.start([sys.executable, '-c', 'print("wrong")'], port)
        assert managed.process is None
        managed.stop()
        assert occupied.getsockname()[1] == port


def test_forced_server_stop_also_stops_owned_loading_child(tmp_path):
    import psutil
    server = ManagedServer('Loading', tmp_path)
    child = None
    command = [sys.executable, '-u', '-c',
               "import subprocess, sys, time; "
               "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
               "print(p.pid, flush=True); time.sleep(60)"]
    try:
        server.start(command, free_port())
        pid = wait_until(lambda: int(server.log_path.read_text().strip())
                         if server.log_path.exists() and server.log_path.read_text().strip().isdigit() else None)
        child = psutil.Process(pid)
        assert child.is_running()
        server.stop()
        assert not server.running
        wait_until(lambda: not child.is_running(), timeout=5)
    finally:
        server.stop()
        if child is not None and child.is_running():
            child.kill()


def test_desktop_renders_feed_and_closes_without_model_imports():
    import tkinter as tk
    from scripts.show_day_ui import ShowDayUI
    window = tk.Tk(); window.withdraw()
    ui = ShowDayUI(window, preview=True)
    app, monitor = app_with_monitor()
    with TestClient(app) as client:
        client.post('/api/chat/stream', json={'message': 'Hello members'})
        snapshot = client.get('/api/operator/snapshot', headers={'Authorization': 'Bearer test-operator-token'}).json()
    ui._snapshot(snapshot, {})
    assert 'Hello members' in ui.transcript.get('1.0', 'end')
    assert 'synthetic reply' in ui.transcript.get('1.0', 'end')
    assert ui.stat_labels['Model'].get() == 'PREVIEW · synthetic'
    window.update()
    ui.close()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            window.update()
            if not window.winfo_exists(): break
        except tk.TclError:
            break
        time.sleep(.02)
    else:
        window.destroy(); raise AssertionError('Dashboard did not close')


def test_preview_components_do_not_import_model_libraries():
    import subprocess
    subprocess.run([sys.executable, '-c',
        "import scripts.show_day_ui, scripts.show_day_frontend, backend.dev_app, sys; "
        "assert not {'torch','transformers','peft'} & sys.modules.keys()"], check=True)
