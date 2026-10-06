"""Release metadata and trained-model identity must remain explicit."""
from fastapi.testclient import TestClient
import pytest
from backend.dev_app import app
from backend.release import RELEASE, RELEASE_ADAPTER


def test_metadata_visible_and_snapshot_selected():
    with TestClient(app) as client:
        for route in ('/api/health', '/api/config'):
            metadata = client.get(route).json()['release']
            assert metadata['version'] == '1.0.0'
            assert 'adapter_directory' not in metadata
        assert client.get('/openapi.json').json()['info']['version'] == '3.0.0'
    assert RELEASE_ADAPTER.name == 'beacon-v1.0.0'
    assert len(RELEASE['adapter_sha256']) == 64


def test_cpu_status_and_missing_adapter_do_not_fake_trained_output(monkeypatch, tmp_path):
    import torch
    from backend.services.model_service import ModelService
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    service = ModelService(lora_checkpoint_path=tmp_path / 'missing')
    status = service.get_gpu_status()
    assert status['device'] == 'CPU'
    assert status['adapter_available'] is False
    assert status['is_loaded'] is False
    service._is_loaded = True
    service.base_model = object()
    with pytest.raises(RuntimeError, match='trained adapter is missing'):
        next(service.stream_chat(messages=[{'role': 'user', 'content': 'Hello'}]))
