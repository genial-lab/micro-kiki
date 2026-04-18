"""Inference-time embedder — text → 128-d vector via trained student."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from src.routing.text_jepa.encoder import StudentEncoder


class TextJEPAEmbedder:
    """Produce a single vector per text by mean-pooling student token embeddings."""

    def __init__(
        self,
        student: StudentEncoder,
        token_embed_fn: Callable[[str], torch.Tensor],
        latent_dim: int = 128,
    ) -> None:
        """
        Args:
            student: trained StudentEncoder.
            token_embed_fn: Callable `str -> (seq_len, input_dim)` tensor.
                E.g. wraps a frozen sentence-transformers backbone.
            latent_dim: Output dim (for shape assertions).
        """
        self.student = student.eval()
        self.token_embed_fn = token_embed_fn
        self.latent_dim = latent_dim

    def embed(self, text: str) -> np.ndarray:
        """Return a single (latent_dim,) numpy vector."""
        with torch.no_grad():
            tokens = self.token_embed_fn(text)  # (S, input_dim)
            if tokens.dim() != 2:
                raise ValueError(f"token_embed_fn must return 2-D tensor, got {tokens.shape}")
            latent = self.student(tokens.unsqueeze(0))  # (1, S, latent_dim)
            pooled = latent.mean(dim=1).squeeze(0)  # (latent_dim,)
        assert pooled.shape == (self.latent_dim,), f"bad shape {pooled.shape}"
        return pooled.cpu().numpy()
