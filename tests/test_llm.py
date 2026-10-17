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


def test_mistral_uses_free_plan_model_and_no_reasoning(monkeypatch, tmp_path):
    monkeypatch.setenv("MISTRAL_API_KEY", "test")
    m = llm.make_model("mistral", cache_path=tmp_path / "c.sqlite")
    assert m.model == "mistral-small-2603"
    assert m.extra_body == {"reasoning_effort": "none"}
    assert llm.PROVIDERS["mistral"].tokens_per_minute == 20_000


def test_reply_text_keeps_only_text_chunks():
    chunks = [{"type": "thinking", "thinking": [{"type": "text", "text": "hmm"}]}, {"type": "text", "text": " 1 "}]
    assert llm._reply_text(chunks) == "1"
    assert llm._reply_text(None) == ""
    assert llm._reply_text(" 0\n") == "0"


def test_token_budget_waits_when_minute_is_full(monkeypatch, tmp_path):
    monkeypatch.setenv("MISTRAL_API_KEY", "test")
    m = llm.make_model("mistral", cache_path=tmp_path / "c.sqlite")
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: (slept.append(s), m._recent.clear()))
    m._recent.append((llm.time.monotonic(), 16_000))
    m._wait_for_token_budget([{"role": "user", "content": "x" * 300}])
    assert slept and slept[0] > 50


class RejectsReasoning(Handler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        if "reasoning_effort" in body:
            Handler.requests_seen.append(body)
            data = json.dumps({"object": "error", "message": "Extra inputs are not permitted: reasoning_effort"}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.rfile = __import__("io").BytesIO(json.dumps(body).encode())
        self.headers.replace_header("Content-Length", str(len(json.dumps(body).encode())))
        super().do_POST()


def test_reasoning_effort_dropped_when_model_rejects_it(monkeypatch, tmp_path):
    httpd = HTTPServer(("127.0.0.1", 0), RejectsReasoning)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_port}/v1"
    monkeypatch.setitem(llm.PROVIDERS, "mistral", llm.Provider("mistral", url, "MISTRAL_API_KEY", 0.0, 20_000))
    monkeypatch.setenv("MISTRAL_API_KEY", "test")
    Handler.requests_seen = []
    try:
        model = llm.make_model("mistral", cache_path=tmp_path / "c.sqlite")
        res = model.chat([{"role": "user", "content": "hi"}])
    finally:
        httpd.shutdown()
    assert res.text == "1"
    assert [("reasoning_effort" in b) for b in Handler.requests_seen] == [True, False]


def test_pacing_follows_the_chosen_mistral_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MISTRAL_API_KEY", "test")
    m = llm.make_model("mistral:ministral-14b-2512", cache_path=tmp_path / "c.sqlite")
    assert m._tpm == 937_500 and abs(m._min_interval - 2.4) < 1e-9
    small = llm.make_model("mistral", cache_path=tmp_path / "c.sqlite")
    assert small._tpm == 20_000 and abs(small._min_interval - 1.2) < 1e-9
