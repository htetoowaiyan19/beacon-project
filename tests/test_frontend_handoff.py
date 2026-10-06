"""Frontend preview works without model imports or weights."""
import json
import subprocess
import sys
from fastapi.testclient import TestClient
from backend.dev_app import app


def test_preview_routes_and_stream():
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/chat.html').status_code == 200
        assert client.get('/stream.js').status_code == 200
        assert 'mock' in client.get('/api/health').json()['gpu']['device']
        response = client.post('/api/chat/stream', json={'message': 'Hello'})
        assert response.headers['x-vercel-ai-ui-message-stream'] == 'v1'
        frames = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
        assert frames[-1] == '[DONE]'
        events = [json.loads(frame) for frame in frames[:-1]]
        assert events[-1]['type'] == 'finish'
        assert 'synthetic reply' in ''.join(e.get('delta','') for e in events)
        assert client.post('/api/chat/stream', json={'message': ''}).status_code == 422


def test_preview_does_not_import_training_libraries():
    subprocess.run([sys.executable, '-c',
        "import backend.dev_app, sys; assert not {'torch','transformers','peft'} & sys.modules.keys()"], check=True)
