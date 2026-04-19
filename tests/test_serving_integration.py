"""Story-36: Integration test for the serving pipeline.

Tests the full serving chain: MLX server config -> AeonServingHook -> response.
Mocks actual MLX model (no GPU). Verifies aeon_hook pre/post inference,
and tests switchable runtime (mlx vs vllm selection).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.memory.aeon import AeonPalace
from src.serving.aeon_hook import AeonServingHook, MEMORY_BUDGET
from src.serving.mlx_server import MLXServer, MLXServerConfig
from src.serving.switchable import SwitchableModel, MAX_ACTIVE_STACKS
from src.serving.vllm_server import VLLMServer, VLLMServerConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_embed_fn(text: str) -> np.ndarray:
    """Deterministic mock embedding function (dim=64)."""
    return np.random.RandomState(hash(text) % 2**31).randn(64).astype(np.float32)


@pytest.fixture
def palace() -> AeonPalace:
    """AeonPalace with mock embeddings (dim=64)."""
    return AeonPalace(dim=64, embed_fn=_mock_embed_fn)


@pytest.fixture
def aeon_hook(palace: AeonPalace) -> AeonServingHook:
    """AeonServingHook backed by mock palace."""
    return AeonServingHook(palace=palace)


@pytest.fixture
def mlx_config(tmp_path: Path) -> MLXServerConfig:
    """MLXServerConfig pointing to a temp directory."""
    return MLXServerConfig(
        model_path=str(tmp_path / "fake-model"),
        adapter_dir=str(tmp_path / "adapters"),
        port=18200,
    )


@pytest.fixture
def vllm_config() -> VLLMServerConfig:
    """VLLMServerConfig with test values."""
    return VLLMServerConfig(
        model_path="test-model",
        port=18100,
    )


@pytest.fixture
def switchable(tmp_path: Path) -> SwitchableModel:
    """SwitchableModel with no real model, temp stacks dir."""
    stacks_dir = tmp_path / "stacks"
    stacks_dir.mkdir()
    # Create fake adapter directories
    for name in ["code", "chat-fr", "math"]:
        (stacks_dir / name).mkdir()
    return SwitchableModel(base_model=None, tokenizer=None, stacks_dir=str(stacks_dir))


# ---------------------------------------------------------------------------
# Tests: MLXServerConfig
# ---------------------------------------------------------------------------


class TestMLXServerConfig:
    """Test MLXServerConfig loading and defaults."""

    def test_default_config(self) -> None:
        config = MLXServerConfig()
        assert config.port == 8200
        assert config.max_active_adapters == 4
        assert "qwen" in config.model_path

    def test_from_json(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "model_path": "models/test",
            "port": 9999,
        }))
        config = MLXServerConfig.from_json(cfg_file)
        assert config.port == 9999
        assert config.model_path == "models/test"

    def test_frozen(self) -> None:
        config = MLXServerConfig()
        with pytest.raises(AttributeError):
            config.port = 1234  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: MLXServer (mocked subprocess)
# ---------------------------------------------------------------------------


class TestMLXServer:
    """Test MLXServer with mocked subprocess."""

    def test_base_url(self, mlx_config: MLXServerConfig) -> None:
        server = MLXServer(config=mlx_config)
        assert server.base_url == f"http://127.0.0.1:{mlx_config.port}"

    def test_not_running_initially(self, mlx_config: MLXServerConfig) -> None:
        server = MLXServer(config=mlx_config)
        assert server.is_running is False
        assert server.active_adapter is None

    @patch("subprocess.Popen")
    def test_start_launches_process(self, mock_popen: MagicMock, mlx_config: MLXServerConfig) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        server = MLXServer(config=mlx_config)
        server.start(adapter="code")

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "--model" in cmd
        assert "--adapter-path" in cmd
        assert server.active_adapter == "code"

    @patch("subprocess.Popen")
    def test_switch_adapter_restarts(self, mock_popen: MagicMock, mlx_config: MLXServerConfig) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        server = MLXServer(config=mlx_config)
        server.start(adapter="code")
        switched = server.switch_adapter("math")
        assert switched is True
        assert server.active_adapter == "math"

    @patch("subprocess.Popen")
    def test_switch_same_adapter_noop(self, mock_popen: MagicMock, mlx_config: MLXServerConfig) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        server = MLXServer(config=mlx_config)
        server.start(adapter="code")
        switched = server.switch_adapter("code")
        assert switched is False

    @patch("subprocess.Popen")
    def test_stop(self, mock_popen: MagicMock, mlx_config: MLXServerConfig) -> None:
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        server = MLXServer(config=mlx_config)
        server.start()
        server.stop()
        mock_proc.terminate.assert_called_once()
        assert server.active_adapter is None


# ---------------------------------------------------------------------------
# Tests: AeonServingHook — pre_inference injects memories
# ---------------------------------------------------------------------------


class TestAeonHookPreInference:
    """Verify that aeon_hook.pre_inference injects memories."""

    def test_pre_inference_no_memories_returns_original(self, aeon_hook: AeonServingHook) -> None:
        """With an empty palace, pre_inference returns the original prompt."""
        prompt = "What is LoRA?"
        result = aeon_hook.pre_inference(prompt)
        assert result == prompt

    def test_pre_inference_injects_context(self, aeon_hook: AeonServingHook, palace: AeonPalace) -> None:
        """After writing memories, pre_inference should prepend context."""
        # Write some memories
        palace.write(content="LoRA is Low-Rank Adaptation for LLMs.", domain="ml", source="t1")
        palace.write(content="QLoRA adds quantization to LoRA.", domain="ml", source="t2")

        prompt = "What is LoRA?"
        result = aeon_hook.pre_inference(prompt)

        # Should contain the memory context header
        assert "Previous conversation context" in result
        # Should end with the original prompt
        assert result.endswith(prompt)
        # Should contain recalled memory content
        assert "LoRA" in result

    def test_pre_inference_respects_budget(self, aeon_hook: AeonServingHook, palace: AeonPalace) -> None:
        """Memory injection should not exceed MEMORY_BUDGET chars."""
        # Write many long memories
        for i in range(20):
            palace.write(
                content=f"Memory entry {i}: " + "x" * 500,
                domain="test",
                source=f"t{i}",
            )

        prompt = "recall everything"
        result = aeon_hook.pre_inference(prompt, top_k=10)
        # The injected context (before the prompt) should be bounded
        context_part = result[: result.index("### Current question:")]
        assert len(context_part) <= MEMORY_BUDGET + 200  # header overhead


# ---------------------------------------------------------------------------
# Tests: AeonServingHook — post_inference writes memories
# ---------------------------------------------------------------------------


class TestAeonHookPostInference:
    """Verify that aeon_hook.post_inference writes memories."""

    def test_post_inference_writes_to_palace(self, aeon_hook: AeonServingHook, palace: AeonPalace) -> None:
        """post_inference should write the interaction to Aeon memory."""
        assert palace.stats["episodes"] == 0

        aeon_hook.post_inference(
            prompt="What is attention?",
            response="Attention is a mechanism for weighing input tokens.",
            domain="ml",
            turn_id="turn-001",
        )

        assert palace.stats["episodes"] == 1

    def test_post_inference_content_format(self, aeon_hook: AeonServingHook, palace: AeonPalace) -> None:
        """Written memory should contain both user prompt and assistant response."""
        aeon_hook.post_inference(
            prompt="Hello",
            response="Hi there!",
            domain="chat",
            turn_id="turn-002",
        )

        # Recall to verify content
        episodes = palace.recall("Hello", top_k=1)
        assert len(episodes) == 1
        assert "User: Hello" in episodes[0].content
        assert "Assistant: Hi there!" in episodes[0].content

    def test_post_then_pre_roundtrip(self, aeon_hook: AeonServingHook, palace: AeonPalace) -> None:
        """Memory written by post_inference should be recallable by pre_inference."""
        aeon_hook.post_inference(
            prompt="Explain transformers",
            response="Transformers use self-attention for sequence modeling.",
            domain="ml",
            turn_id="turn-003",
        )

        result = aeon_hook.pre_inference("Tell me about transformers")
        # The recalled context should include the previously written memory
        assert "Previous conversation context" in result
        assert "transformers" in result.lower()


# ---------------------------------------------------------------------------
# Tests: SwitchableModel runtime selection
# ---------------------------------------------------------------------------


class TestSwitchableRuntime:
    """Test switchable runtime for MLX vs vLLM selection."""

    def test_list_available_stacks(self, switchable: SwitchableModel) -> None:
        available = switchable.list_available()
        assert "code" in available
        assert "chat-fr" in available
        assert "math" in available

    def test_apply_stacks(self, switchable: SwitchableModel) -> None:
        """Applying stacks should update active_stacks."""
        switchable.apply_stacks(["code", "math"])
        assert set(switchable.active_stacks) == {"code", "math"}

    def test_apply_stacks_caching(self, switchable: SwitchableModel) -> None:
        """Applying the same stacks twice should use cache (no error)."""
        switchable.apply_stacks(["code"])
        switchable.apply_stacks(["code"])  # Should not raise
        assert switchable.active_stacks == ["code"]

    def test_apply_stacks_max_limit(self, switchable: SwitchableModel) -> None:
        """Cannot apply more than MAX_ACTIVE_STACKS."""
        with pytest.raises(ValueError, match="Cannot activate"):
            switchable.apply_stacks(["a", "b", "c", "d", "e"])

    def test_clear_stacks(self, switchable: SwitchableModel) -> None:
        switchable.apply_stacks(["code"])
        switchable.clear_stacks()
        assert switchable.active_stacks == []

    def test_no_base_model(self, switchable: SwitchableModel) -> None:
        """With no base model, apply_stacks should still track stack names."""
        switchable.apply_stacks(["chat-fr"])
        assert switchable.model is None
        assert switchable.active_stacks == ["chat-fr"]


# ---------------------------------------------------------------------------
# Tests: Runtime selection (mlx vs vllm)
# ---------------------------------------------------------------------------


class TestRuntimeSelection:
    """Test choosing between MLX and vLLM backends."""

    def test_mlx_server_config_differs_from_vllm(self) -> None:
        """MLX and vLLM configs should have different default ports."""
        mlx = MLXServerConfig()
        vllm = VLLMServerConfig()
        assert mlx.port != vllm.port

    def test_vllm_server_build_args(self, vllm_config: VLLMServerConfig) -> None:
        """VLLMServer should build correct launch arguments."""
        server = VLLMServer(config=vllm_config)
        args = server._build_args()
        assert "--model" in args
        assert "--enable-lora" in args
        assert str(vllm_config.port) in args

    def test_mlx_server_build_command(self, mlx_config: MLXServerConfig) -> None:
        """MLXServer should build correct launch command."""
        server = MLXServer(config=mlx_config)
        cmd = server._build_command(adapter="code")
        assert "mlx_lm.server" in " ".join(cmd)
        assert "--adapter-path" in cmd

    def test_switchable_with_mlx_config(self, tmp_path: Path) -> None:
        """SwitchableModel can be used alongside MLXServerConfig."""
        mlx_cfg = MLXServerConfig(
            adapter_dir=str(tmp_path / "stacks"),
            port=18200,
        )
        (tmp_path / "stacks").mkdir()
        (tmp_path / "stacks" / "code").mkdir()

        switch = SwitchableModel(stacks_dir=str(tmp_path / "stacks"))
        assert "code" in switch.list_available()

    def test_switchable_with_vllm_config(self, tmp_path: Path) -> None:
        """SwitchableModel can be used alongside VLLMServerConfig."""
        vllm_cfg = VLLMServerConfig(port=18100)
        (tmp_path / "stacks").mkdir()
        (tmp_path / "stacks" / "math").mkdir()

        switch = SwitchableModel(stacks_dir=str(tmp_path / "stacks"))
        assert "math" in switch.list_available()


# ---------------------------------------------------------------------------
# Tests: Full pipeline integration (mock end-to-end)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end mock test: config -> hook -> serve -> memory."""

    def test_full_chain_mlx(
        self, mlx_config: MLXServerConfig, aeon_hook: AeonServingHook, palace: AeonPalace,
    ) -> None:
        """Simulate the full MLX serving chain with aeon hook."""
        # 1. Create server (don't actually start it)
        server = MLXServer(config=mlx_config)
        assert server.is_running is False

        # 2. Pre-inference: inject memories (empty initially)
        prompt = "Explain MoE routing"
        enriched = aeon_hook.pre_inference(prompt)
        assert enriched == prompt  # No memories yet

        # 3. Simulate model response (mocked)
        mock_response = "MoE uses a router to dispatch tokens to specialized experts."

        # 4. Post-inference: write memory
        aeon_hook.post_inference(
            prompt=prompt,
            response=mock_response,
            domain="ml",
            turn_id="turn-100",
        )

        # 5. Verify memory was stored
        assert palace.stats["episodes"] == 1

        # 6. Next turn: pre_inference should now inject the memory
        next_prompt = "How does MoE work?"
        enriched2 = aeon_hook.pre_inference(next_prompt)
        assert "Previous conversation context" in enriched2
        assert enriched2.endswith(next_prompt)

    def test_full_chain_with_switchable(
        self, switchable: SwitchableModel, aeon_hook: AeonServingHook, palace: AeonPalace,
    ) -> None:
        """Simulate the full chain with switchable runtime."""
        # 1. Select stacks
        switchable.apply_stacks(["code"])
        assert switchable.active_stacks == ["code"]

        # 2. Pre-inference
        prompt = "Write a Python function"
        enriched = aeon_hook.pre_inference(prompt)

        # 3. Mock response
        mock_response = "def hello(): return 'world'"

        # 4. Post-inference
        aeon_hook.post_inference(
            prompt=prompt,
            response=mock_response,
            domain="code",
            turn_id="turn-200",
        )

        # 5. Switch to different stack
        switchable.apply_stacks(["math"])
        assert switchable.active_stacks == ["math"]

        # 6. Verify memory persists across stack switches
        episodes = palace.recall("Python function", top_k=5)
        assert len(episodes) >= 1

    def test_aeon_hook_error_resilience(self, palace: AeonPalace) -> None:
        """AeonServingHook should gracefully handle palace errors."""
        # Create a hook with a broken palace (mock recall to raise)
        hook = AeonServingHook(palace=palace)

        with patch.object(palace, "recall", side_effect=RuntimeError("simulated failure")):
            # pre_inference should return original prompt on error
            result = hook.pre_inference("test prompt")
            assert result == "test prompt"

        # post_inference should not raise on write failure
        with patch.object(palace, "write", side_effect=RuntimeError("write failure")):
            hook.post_inference(
                prompt="test",
                response="response",
                domain="test",
                turn_id="turn-err",
            )
            # No exception means the test passes
