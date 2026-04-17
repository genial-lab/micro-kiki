# tests/test_eval_backend.py
"""Test MLXBackend can be instantiated and has required interface."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_mlx_backend_has_required_methods():
    """MLXBackend must implement all AdapterBackend methods."""
    from eval_v2_v3 import MLXBackend
    backend = MLXBackend.__new__(MLXBackend)
    assert hasattr(backend, "load_base_model")
    assert hasattr(backend, "load_adapter")
    assert hasattr(backend, "unload_adapter")
    assert hasattr(backend, "generate")
    assert hasattr(backend, "compute_perplexity")


def test_mlx_backend_instantiation():
    """MLXBackend should instantiate without errors."""
    from eval_v2_v3 import MLXBackend
    backend = MLXBackend()
    assert backend is not None


@pytest.mark.skipif(
    not Path("/Users/clems/KIKI-Mac_tunner/models/Qwen3.5-4B").exists(),
    reason="Base model not available locally",
)
def test_mlx_backend_load_model():
    """MLXBackend can load the base model (Studio only)."""
    from eval_v2_v3 import MLXBackend
    backend = MLXBackend()
    backend.load_base_model("/Users/clems/KIKI-Mac_tunner/models/Qwen3.5-4B")
    assert backend._model is not None
    assert backend._tokenizer is not None
