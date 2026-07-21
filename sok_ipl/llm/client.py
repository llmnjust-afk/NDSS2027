"""Unified LLM client abstracting OpenAI-compatible and Anthropic backbones.

The framework talks to backbones through one interface so that a 4-backbone
comparison is a config change, not a code change. We intentionally do NOT depend
on the official SDKs at import time so the package imports cleanly in any
environment; backends are loaded lazily.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Protocol


class LLMBackend(Protocol):
    name: str

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str: ...


@dataclass
class _Usage:
    n_calls: int = 0
    n_tokens: int = 0
    latency_ms: float = 0.0


class OpenAICompatBackend:
    """Works with OpenAI, DeepSeek, vLLM, Ollama, any OpenAI-compatible endpoint."""

    def __init__(self, name: str, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.name = name
        self.model = model
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.usage = _Usage()
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        client = self._ensure_client()
        # Exponential backoff for rate limits (ChatAnywhere / OpenAI 429).
        import time as _time

        last_err = None
        for attempt in range(6):
            try:
                t0 = _time.time()
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self.usage.n_calls += 1
                self.usage.n_tokens += resp.usage.total_tokens if resp.usage else 0
                self.usage.latency_ms += (_time.time() - t0) * 1000
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # Retry on rate limit (429) and transient server errors (5xx).
                if "429" in msg or "rate" in msg or "timeout" in msg or "503" in msg or "529" in msg:
                    wait = min(2 ** attempt * 5, 120)  # 5,10,20,40,80,120s
                    _time.sleep(wait)
                    continue
                raise
        raise last_err  # type: ignore[misc]


class AnthropicBackend:
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.name = "anthropic"
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.usage = _Usage()
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic  # lazy import

            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        client = self._ensure_client()
        t0 = time.time()
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        self.usage.n_calls += 1
        self.usage.n_tokens += resp.usage.input_tokens + resp.usage.output_tokens
        self.usage.latency_ms += (time.time() - t0) * 1000
        return resp.content[0].text if resp.content else ""


class StubBackend:
    """Deterministic offline backend for tests and CI. Emits a fixed response."""

    def __init__(self, response: str = "STUB"):
        self.name = "stub"
        self.response = response
        self.usage = _Usage()

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.0) -> str:
        self.usage.n_calls += 1
        self.usage.n_tokens += len(prompt.split()) + len(self.response.split())
        return self.response


def build_backbone(spec: str) -> LLMBackend:
    """Parse a backbone spec like `openai:gpt-4o` or `anthropic:claude-3-5-sonnet`.

    `stub:` and `local:` (vLLM on localhost:8000) are also supported for tests
    and self-hosted inference on the single A100.
    """
    if ":" not in spec:
        raise ValueError(f"backbone spec must be `provider:model`, got {spec!r}")
    provider, model = spec.split(":", 1)
    if provider == "openai":
        return OpenAICompatBackend(name=model, model=model)
    if provider == "deepseek":
        return OpenAICompatBackend(
            name=model,
            model=model,
            base_url="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        )
    if provider == "local":
        return OpenAICompatBackend(name=model, model=model, base_url="http://localhost:8000/v1", api_key="x")
    if provider == "anthropic":
        return AnthropicBackend(model=model)
    if provider == "stub":
        return StubBackend(response=model)
    raise ValueError(f"unknown backbone provider: {provider!r}")
