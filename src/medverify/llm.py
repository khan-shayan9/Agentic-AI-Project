"""One client for all free model providers (they all speak the OpenAI chat format).

- ollama  : Llama 3.1 8B and Phi-4-mini running locally (Colab GPU or laptop), no limits
- mistral : Mistral Small on Mistral's free "Experiment" plan (about 1 request/second)
- groq    : optional; gpt-oss-120b on Groq's free plan, if your account works there

Every reply is cached in SQLite, so re-running an experiment never spends quota twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from .schemas import LLMResult

DEFAULT_CACHE = Path("data/cache/llm_cache.sqlite")


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: Optional[str]
    min_interval_s: float  # wait between calls to stay under the free rate limit


PROVIDERS = {
    "ollama": Provider("ollama", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), None, 0.0),
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", 2.1),
    "mistral": Provider("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", 1.2),
}

# Short names used on the command line -> (provider, model id).
MODEL_ALIASES = {
    "llama": ("ollama", "llama3.1:8b"),
    "mistral": ("mistral", "mistral-small-latest"),
    "phi": ("ollama", "phi4-mini"),
    "gptoss": ("groq", "openai/gpt-oss-120b"),  # optional, needs a working Groq account
}

# The three verifiers used by default: one model family each (Meta, Mistral AI, Microsoft).
DEFAULT_VERIFIERS = ("llama", "mistral", "phi")

# Model family of each alias, used for the independence analysis.
MODEL_FAMILY = {"llama": "Meta Llama", "mistral": "Mistral", "phi": "Microsoft Phi", "gptoss": "OpenAI gpt-oss"}


def resolve_model(spec: str) -> tuple[str, str]:
    """'llama' -> ('ollama', 'llama3.1:8b');  'groq:openai/gpt-oss-20b' -> ('groq', 'openai/gpt-oss-20b')."""
    if spec in MODEL_ALIASES:
        return MODEL_ALIASES[spec]
    if ":" in spec:
        provider, model = spec.split(":", 1)
        if provider in PROVIDERS:
            return provider, model
    raise ValueError(f"Unknown model '{spec}'. Use one of {sorted(MODEL_ALIASES)} or 'provider:model'.")


class ChatModel(Protocol):
    provider: str
    model: str

    def chat(self, messages: list[dict]) -> LLMResult: ...


class _Cache:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, result TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute("SELECT result FROM calls WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, result: dict) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO calls VALUES (?, ?)", (key, json.dumps(result)))
            self._conn.commit()


@dataclass
class OpenAICompatibleModel:
    """Calls an OpenAI-compatible endpoint with caching, pacing and retries."""

    provider: str
    model: str
    temperature: float = 0.3  # MedHallu authors used 0.3 (top_p 0.95)
    top_p: float = 0.95
    max_tokens: int = 512
    extra_body: dict = field(default_factory=dict)
    cache_path: Path = DEFAULT_CACHE
    max_retries: int = 6

    def __post_init__(self):
        from openai import OpenAI

        spec = PROVIDERS[self.provider]
        api_key = os.environ.get(spec.key_env, "") if spec.key_env else "ollama"
        if spec.key_env and not api_key:
            raise RuntimeError(f"{spec.key_env} is not set. Add it to your .env file.")
        self._client = OpenAI(api_key=api_key, base_url=spec.base_url, timeout=120)
        self._min_interval = spec.min_interval_s
        self._last_call = 0.0
        self._cache = _Cache(self.cache_path)

    def _key(self, messages: list[dict]) -> str:
        payload = dict(
            provider=self.provider,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def chat(self, messages: list[dict]) -> LLMResult:
        key = self._key(messages)
        hit = self._cache.get(key)
        if hit is not None:
            return LLMResult(**hit, cached=True)

        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        delay = 5.0
        for attempt in range(self.max_retries):
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            start = time.monotonic()
            try:
                self._last_call = start
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    extra_body=self.extra_body or None,
                )
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as err:
                if attempt == self.max_retries - 1:
                    raise
                print(f"[{self.provider}] {type(err).__name__}; retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            latency = time.monotonic() - start
            usage = resp.usage
            result = dict(
                text=(resp.choices[0].message.content or "").strip(),
                provider=self.provider,
                model=self.model,
                latency_s=round(latency, 3),
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
            self._cache.put(key, result)
            return LLMResult(**result)
        raise RuntimeError("unreachable")


def make_model(spec: str, **kwargs) -> OpenAICompatibleModel:
    provider, model = resolve_model(spec)
    if provider == "groq" and "gpt-oss" in model:
        # gpt-oss writes hidden reasoning that counts against Groq's 200K tokens/day;
        # low effort keeps each call small, and a higher cap stops the reasoning
        # from using up the whole reply before the verdict is written.
        kwargs.setdefault("extra_body", {"reasoning_effort": "low"})
        kwargs.setdefault("max_tokens", 1024)
    return OpenAICompatibleModel(provider=provider, model=model, **kwargs)


class FakeModel:
    """Deterministic stand-in used by the tests (no network)."""

    def __init__(self, name: str, reply_fn):
        self.provider = "fake"
        self.model = name
        self._reply_fn = reply_fn
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(text=self._reply_fn(messages), provider="fake", model=self.model, latency_s=0.0)
