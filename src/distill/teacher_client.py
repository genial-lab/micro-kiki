from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINTS = {
    "mistral-large": os.getenv("TEACHER_MISTRAL_URL", "http://192.168.0.120:8000/v1"),
    "qwen122": os.getenv("TEACHER_QWEN122_URL", "http://192.168.0.120:8001/v1"),
    "qwen35": os.getenv("TEACHER_QWEN35_URL", "http://kxkm-ai:8000/v1"),
    "devstral": os.getenv("TEACHER_DEVSTRAL_URL", "http://kxkm-ai:8001/v1"),
}


class TeacherClient:
    """OpenAI-compatible async teacher client with disk cache and retry."""

    def __init__(
        self,
        endpoints: dict[str, str] | None = None,
        cache_dir: str = "data/teacher_cache",
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        self._endpoints = endpoints or DEFAULT_ENDPOINTS
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_retries = max_retries
        self._timeout = timeout

    def _cache_key(self, prompt: str, model: str, **kwargs) -> str:
        raw = json.dumps({"prompt": prompt, "model": model, **kwargs}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> str | None:
        path = self._cache_path(key)
        if path.exists():
            return json.loads(path.read_text()).get("completion")
        return None

    def _write_cache(self, key: str, completion: str, model: str) -> None:
        self._cache_path(key).write_text(json.dumps({"model": model, "completion": completion}))

    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool | None = None,
    ) -> str:
        extra_params: dict[str, Any] = {}
        if enable_thinking is not None:
            extra_params["enable_thinking"] = enable_thinking

        cache_key = self._cache_key(prompt, model, temperature=temperature)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        base_url = self._endpoints.get(model)
        if not base_url:
            raise ValueError(f"Unknown model: {model}. Available: {list(self._endpoints.keys())}")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_params:
            payload["extra_body"] = extra_params

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{base_url}/chat/completions", json=payload, timeout=self._timeout,
                    )
                    response.raise_for_status()
                completion = response.json()["choices"][0]["message"]["content"]
                self._write_cache(cache_key, completion, model)
                return completion
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    logger.warning("Attempt %d/%d failed: %s", attempt, self._max_retries, e)
        raise last_error
