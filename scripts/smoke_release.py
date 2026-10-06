"""Check a running local release with real English and Burmese streams."""
from __future__ import annotations
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://127.0.0.1:8000'


def main():
    checks = {}
    for path in ('/', '/chat.html', '/app.js', '/style.css', '/favicon.svg', '/docs', '/openapi.json', '/api/config', '/api/health'):
        with urlopen(BASE + path, timeout=30) as response:
            checks[path] = response.status
    replies = []
    for prompt in ('What is RAM? Answer in two short sentences.', 'TCP နဲ့ UDP ဘာကွာလဲ။ အတိုချုံးရှင်းပြပါ။'):
        request = Request(BASE + '/api/chat/stream',
                          data=json.dumps({'message': prompt, 'max_new_tokens': 256, 'temperature': 0.7}).encode(),
                          headers={'Content-Type': 'application/json'})
        events = []; terminated = False
        with urlopen(request, timeout=180) as response:
            assert response.headers['x-vercel-ai-ui-message-stream'] == 'v1'
            for raw in response:
                line = raw.decode('utf-8').strip()
                if not line.startswith('data: '):
                    continue
                value = line[6:]
                if value == '[DONE]':
                    terminated = True; break
                events.append(json.loads(value))
        assert terminated and events[-1]['type'] == 'finish'
        assert not any(e['type'] == 'error' for e in events), events
        answer = ''.join(e.get('delta', '') for e in events if e['type'] == 'text-delta')
        assert answer.strip()
        has_burmese = any('\u1000' <= char <= '\u109f' for char in answer)
        assert has_burmese == (not prompt.isascii()), 'Reply did not follow the prompt language'
        replies.append({'prompt': prompt, 'answer': answer,
                        'metrics': next(e['data'] for e in events if e['type'] == 'data-metrics')})
    with urlopen(BASE + '/api/health', timeout=30) as response:
        health = json.load(response)
    assert health['release']['version'] == '1.0.0'
    assert health['gpu']['has_lora'] and health['gpu']['adapter_name'] == 'beacon-v1.0.0'
    output = ROOT / 'outputs' / 'release_checks' / 'v1-smoke.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'endpoints': checks, 'health': health, 'replies': replies}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'PASS: {len(checks)} endpoints; two real trained-adapter streams. Report: {output}')


if __name__ == '__main__':
    main()
