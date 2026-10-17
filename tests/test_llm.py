"""Checks the real HTTP client against a tiny local OpenAI-compatible server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from medverify import llm


class Handler(BaseHTTPRequestHandler):
    requests_seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests_seen.append(body)
        reply = {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": " 1 "}}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 1, "total_tokens": 43},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_port}/v1"
    monkeypatch.setitem(llm.PROVIDERS, "ollama", llm.Provider("ollama", url, None, 0.0))
    Handler.requests_seen = []
    yield
    httpd.shutdown()


def test_real_client_calls_caches_and_reads_usage(server, tmp_path):
    model = llm.make_model("llama", cache_path=tmp_path / "c.sqlite")
    msgs = [{"role": "user", "content": "hi"}]
    first = model.chat(msgs)
    second = model.chat(msgs)
    assert first.text == "1" and first.prompt_tokens == 42 and not first.cached
    assert second.cached and second.text == "1"
    assert len(Handler.requests_seen) == 1
    sent = Handler.requests_seen[0]
    assert sent["model"] == "llama3.1:8b" and sent["temperature"] == 0.3 and sent["top_p"] == 0.95


def test_resolve_model_and_missing_key(monkeypatch):
    assert llm.resolve_model("phi") == ("ollama", "phi4-mini")
    assert llm.DEFAULT_VERIFIERS == ("llama", "mistral", "phi")
    families = {llm.MODEL_FAMILY[name] for name in llm.DEFAULT_VERIFIERS}
    assert len(families) == 3  # one model family per verifier (condition 5)
    assert llm.resolve_model("gptoss") == ("groq", "openai/gpt-oss-120b")
    assert llm.resolve_model("groq:openai/gpt-oss-20b") == ("groq", "openai/gpt-oss-20b")
    with pytest.raises(ValueError):
        llm.resolve_model("unknown")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        llm.make_model("gptoss")
