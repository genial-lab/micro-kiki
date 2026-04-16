"""vLLM server wrapper with dynamic LoRA and router sidecar."""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VLLMServerConfig:
    """Immutable configuration for the vLLM serving process."""

    model_path: str = "models/qwen3.5-4b-diffattn/"
    port: int = 8100
    max_loras: int = 4
    max_lora_rank: int = 16
    gpu_memory_utilization: float = 0.9


class VLLMServer:
    """Manages a vLLM subprocess with dynamic LoRA adapter support."""

    def __init__(self, config: VLLMServerConfig | None = None) -> None:
        self._config = config or VLLMServerConfig()
        self._process: subprocess.Popen | None = None
        self._base_url = f"http://localhost:{self._config.port}"

    @property
    def config(self) -> VLLMServerConfig:
        return self._config

    def _build_args(self) -> list[str]:
        return [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self._config.model_path,
            "--enable-lora",
            "--max-loras", str(self._config.max_loras),
            "--max-lora-rank", str(self._config.max_lora_rank),
            "--gpu-memory-utilization", str(self._config.gpu_memory_utilization),
            "--port", str(self._config.port),
            "--trust-remote-code",
        ]

    def start(self) -> None:
        """Launch the vLLM subprocess with runtime LoRA updating enabled."""
        env = {**os.environ, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
        cmd = self._build_args()
        logger.info("Starting vLLM: %s", " ".join(cmd))
        self._process = subprocess.Popen(cmd, env=env)

    def stop(self) -> None:
        """Terminate the vLLM subprocess if running."""
        if self._process is not None:
            self._process.terminate()
            self._process.wait()
            self._process = None
            logger.info("vLLM process stopped")

    def load_adapter(self, stack_id: str, adapter_path: str) -> bool:
        """Load a LoRA adapter via the vLLM API.

        Returns True on success, False on failure.
        """
        url = f"{self._base_url}/v1/load_lora_adapter"
        payload = {"lora_name": stack_id, "lora_path": adapter_path}
        try:
            resp = httpx.post(url, json=payload, timeout=30.0)
            resp.raise_for_status()
            logger.info("Loaded adapter %s from %s", stack_id, adapter_path)
            return True
        except httpx.HTTPError:
            logger.warning("Failed to load adapter %s", stack_id, exc_info=True)
            return False

    def unload_adapter(self, stack_id: str) -> bool:
        """Unload a LoRA adapter via the vLLM API.

        Returns True on success, False on failure.
        """
        url = f"{self._base_url}/v1/unload_lora_adapter"
        payload = {"lora_name": stack_id}
        try:
            resp = httpx.post(url, json=payload, timeout=30.0)
            resp.raise_for_status()
            logger.info("Unloaded adapter %s", stack_id)
            return True
        except httpx.HTTPError:
            logger.warning("Failed to unload adapter %s", stack_id, exc_info=True)
            return False

    def health(self) -> dict:
        """Check vLLM server health.

        Returns a dict with at least ``{"status": "ok"|"error"}``.
        """
        url = f"{self._base_url}/health"
        try:
            resp = httpx.get(url, timeout=5.0)
            resp.raise_for_status()
            return {"status": "ok"}
        except httpx.HTTPError:
            logger.warning("Health check failed", exc_info=True)
            return {"status": "error"}

    def generate(
        self,
        prompt: str,
        model: str,
        adapter: str | None = None,
    ) -> str:
        """Send a chat completion request to the vLLM server.

        Args:
            prompt: The user message.
            model: The base model name to target.
            adapter: Optional LoRA adapter name.

        Returns:
            The generated text content.
        """
        url = f"{self._base_url}/v1/chat/completions"
        payload: dict = {
            "model": adapter if adapter else model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            resp = httpx.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError) as exc:
            logger.error("Generate failed: %s", exc, exc_info=True)
            return ""
