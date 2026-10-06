"""Run one real trained-model reply through the visitor proxy, then stop both servers."""
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.utils.show_day_control import ShowDayControl, gpu_telemetry


def ready(check, seconds=45):
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        try:
            result = check()
            if result: return result
        except Exception as exc:
            last = exc
        time.sleep(.1)
    raise RuntimeError(f'Server did not become ready: {last}')


def main():
    control = ShowDayControl()
    try:
        control.start_backend(8100)
        initial = ready(lambda: control.request('/api/operator/snapshot'))
        assert initial['gpu']['is_loaded'] and initial['gpu']['has_lora']
        control.start_frontend(8180, 8100)
        def page():
            with urlopen('http://127.0.0.1:8180/', timeout=5) as response:
                return response.read().decode('utf-8')
        assert 'visitor-mode' in ready(page)
        body = {'message': 'What is RAM? Answer briefly in English.', 'max_new_tokens': 128,
                'temperature': 0, 'client_id': 'staff-smoke-test', 'session_id': 'staff-demo'}
        request = Request('http://127.0.0.1:8180/api/chat/stream', data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        events = []; during = None; done = False
        with urlopen(request, timeout=180) as response:
            assert response.headers['x-vercel-ai-ui-message-stream'] == 'v1'
            for line in response:
                if not line.startswith(b'data: '): continue
                if line.strip() == b'data: [DONE]': done = True; break
                event = json.loads(line[6:]); events.append(event)
                if event['type'] == 'text-delta' and during is None:
                    during = control.request('/api/operator/snapshot')
        assert done and not any(e['type'] == 'error' for e in events)
        assert ''.join(e.get('delta', '') for e in events).strip()
        snapshot = ready(lambda: (s if (s := control.request('/api/operator/snapshot'))['counters']['completed'] == 1 else None))
        assert snapshot['gpu']['has_lora'] and snapshot['gpu']['adapter_name'] == 'beacon-v1.0.0'
        assert during and during['chats'][0]['answer']
        result = dict(passed=True, snapshot=snapshot, during_stream=during, driver=gpu_telemetry())
        output = ROOT / 'outputs/show_day/real-model-smoke.json'
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'PASS: trained adapter, visitor proxy and live operator feed. Report: {output}')
    finally:
        control.stop_all()


if __name__ == '__main__':
    main()
