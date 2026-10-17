"""One client for all free model providers (they all speak the OpenAI chat format).

- ollama  : Llama 3.1 8B and Phi-4-mini running locally (Colab GPU or laptop), no limits
- mistral : Mistral Small 4 (mistral-small-2603) on Mistral's free plan
            (1 request/second and 20,000 tokens/minute; both are paced below)
- mistral-local : Mistral 7B through Ollama, the fallback when the free Mistral API
  is rate-limited (same model family, no limits)
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
from collections import deque
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
    tokens_per_minute: Optional[int] = None  # free-plan token limit, if the provider has one


PROVIDERS = {
    "ollama": Provider("ollama", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), None, 0.0),
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", 2.1),
    # Free plan limits shown on Mistral's Limits page: 1 request/s, 20,000 tokens/minute.
    "mistral": Provider("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", 1.2, 20_000),
}

# Free-plan limits from Mistral's Limits page (Oct 2026): model -> (requests/second, tokens/minute).
# Pick another model with the MISTRAL_MODEL environment variable; the pacing follows it.
MISTRAL_FREE_LIMITS = {
    "mistral-small-2603": (1.0, 20_000),  # Mistral Small 4 (default)
    "ministral-14b-2512": (0.5, 937_500),
    "ministral-8b-2512": (3.13, 625_000),
    "mistral-large-2512": (1.0, 250_000),
}

# Short names used on the command line -> (provider, model id).
MODEL_ALIASES = {
    "llama": ("ollama", "llama3.1:8b"),
    # Exact id from the free plan's Limits page (Mistral Small 4); "latest" may point elsewhere.
    "mistral": ("mistral", os.environ.get("MISTRAL_MODEL", "mistral-small-2603")),
    "phi": ("ollama", "phi4-mini"),
    "mistral-local": ("ollama", "mistral:7b"),  # fallback when the Mistral API returns 429
    "gptoss": ("groq", "openai/gpt-oss-120b"),  # optional, needs a working Groq account
}

# The three verifiers used by default: one model family each (Meta, Mistral AI, Microsoft).
# Mistral runs locally (Mistral 7B) because the free Mistral API allowed 0 requests/minute on
# our account (Oct 2026); "mistral" (the API) is still supported for anyone whose key works.
DEFAULT_VERIFIERS = ("llama", "mistral-local", "phi")

# Model family of each alias, used for the independence analysis.
MODEL_FAMILY = {"llama": "Meta Llama", "mistral": "Mistral", "mistral-local": "Mistral", "phi": "Microsoft Phi", "gptoss": "OpenAI gpt-oss"}


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
        self._tpm = spec.tokens_per_minute
        if self.provider == "mistral" and self.model in MISTRAL_FREE_LIMITS:
            rps, tpm = MISTRAL_FREE_LIMITS[self.model]
            self._min_interval, self._tpm = 1.2 / rps, tpm  # 20% margin on the request rate
        self._recent: deque = deque()  # (time, tokens) of calls in the last minute
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

    def _wait_for_token_budget(self, messages: list[dict]) -> None:
        """Sleep until the next call fits under the provider's tokens-per-minute limit."""
        if not self._tpm:
            return
        budget = int(self._tpm * 0.8)  # keep a safety margin
        estimate = len(json.dumps(messages)) // 3 + 64  # rough prompt size + a short reply
        while True:
            now = time.monotonic()
            while self._recent and now - self._recent[0][0] > 60:
                self._recent.popleft()
            used = sum(t for _, t in self._recent)
            if not self._recent or used + estimate <= budget:
                return
            time.sleep(max(0.5, 60 - (now - self._recent[0][0])))

    def chat(self, messages: list[dict]) -> LLMResult:
        key = self._key(messages)
        hit = self._cache.get(key)
        if hit is not None:
            return LLMResult(**hit, cached=True)

        from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, RateLimitError

        delay = 5.0
        extra_body = dict(self.extra_body)
        for attempt in range(self.max_retries):
            self._wait_for_token_budget(messages)
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
                    extra_body=extra_body or None,
                )
            except BadRequestError as err:
                # Some models reject reasoning_effort; drop it once and try again.
                if "reasoning_effort" in extra_body and "reasoning" in str(err).lower():
                    print(f"[{self.provider}] model does not accept reasoning_effort; retrying without it")
                    extra_body.pop("reasoning_effort")
                    continue
                raise
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as err:
                if attempt == self.max_retries - 1:
                    raise
                print(f"[{self.provider}] {type(err).__name__}; retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            latency = time.monotonic() - start
            usage = resp.usage
            if self._tpm:
                spent = (getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0)
                self._recent.append((time.monotonic(), spent))
            result = dict(
                text=_reply_text(resp.choices[0].message.content),
                provider=self.provider,
                model=self.model,
                latency_s=round(latency, 3),
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
            self._cache.put(key, result)
            return LLMResult(**result)
        raise RuntimeError("unreachable")


def _reply_text(content) -> str:
    """Reply text; reasoning models may return a list of chunks (thinking + text), keep only the text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    parts = []
    for chunk in content:
        kind = chunk.get("type") if isinstance(chunk, dict) else getattr(chunk, "type", None)
        text = chunk.get("text") if isinstance(chunk, dict) else getattr(chunk, "text", None)
        if kind == "text" and text:
            parts.append(text)
    return "".join(parts).strip()


def make_model(spec: str, **kwargs) -> OpenAICompatibleModel:
    provider, model = resolve_model(spec)
    if provider == "mistral" and "small" in model:
        # Mistral Small 4 can reason step by step; "none" gives a short direct verdict
        # and keeps each call well under the free 20,000 tokens/minute.
        kwargs.setdefault("extra_body", {"reasoning_effort": "none"})
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
