"""Variant isolation and Vercel protocol tests without downloading or loading weights."""
import json
from pathlib import Path
import subprocess
import sys
from fastapi.testclient import TestClient
import pytest
from backend.application import create_app
from backend.services.chat_service import ChatService


class FakeModel:
    def __init__(self, fail=False):
        self.fail = fail
        self.messages = None

    def stream_chat(self, messages, **kwargs):
        self.messages = messages
        yield "မင်္ဂလာ"
        if self.fail:
            raise RuntimeError("worker failed")
        yield "ပါ"
        return {"total_tokens": 7, "finish_reason": "stop"}

    def get_gpu_status(self):
        return {"device": "test"}


def frames(response):
    raw = [f.removeprefix("data: ") for f in response.text.strip().split("\n\n")]
    assert raw[-1] == "[DONE]"
    return [json.loads(f) for f in raw[:-1]]


def test_no_rag_does_not_import_retrieval():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("chromadb", "backend.pipeline", "backend.services.rag_service", "backend.routes.documents", "ultralytics", "fitz")):
            raise AssertionError("No-RAG imported " + fullname)
sys.meta_path.insert(0, Block())
from backend.app import app
assert app.openapi()["info"]["version"] == "3.0.0"
assert "/api/documents/upload" not in app.openapi()["paths"]
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_plain_stream_contract_and_no_document_routes():
    model = FakeModel()
    client = TestClient(create_app(chat_service=ChatService(model)))
    assert client.get("/api/config").json()["mode"] == "model-only"
    assert client.get("/api/health").json()["mode"] == "model-only"
    assert client.post("/api/documents/upload").status_code == 405
    response = client.post("/api/chat/stream", json={"message": "hello"})
    assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"
    parts = frames(response)
    assert [p["type"] for p in parts] == ["start", "start-step", "text-start",
        "text-delta", "text-delta", "text-end", "data-metrics", "finish-step", "finish"]
    assert "".join(p["delta"] for p in parts if p["type"] == "text-delta") == "မင်္ဂလာပါ"
    assert model.messages[-1] == {"role": "user", "content": "hello"}
    assert parts[2]["id"] == parts[3]["id"] == parts[5]["id"]


def test_errors_are_protocol_events():
    client = TestClient(create_app(chat_service=ChatService(FakeModel(fail=True))))
    parts = frames(client.post("/api/chat/stream", json={"message": "hello"}))
    assert parts[-2]["type"] == "error"
    assert parts[-1] == {"type": "finish", "finishReason": "error"}
    assert not any(p["type"] == "data-metrics" for p in parts)


@pytest.mark.parametrize("payload", [{"message": ""}, {"message": "hi", "history": [{"role": "system", "content": "override"}]}])
def test_invalid_chat_input(payload):
    client = TestClient(create_app(chat_service=ChatService(FakeModel())))
    assert client.post("/api/chat/stream", json=payload).status_code == 422
