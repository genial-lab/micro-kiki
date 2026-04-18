"""Text-JEPA training loop — student + teacher EMA + predictor + L1 masked loss."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn, optim

from src.routing.text_jepa.collapse import CollapseMonitor, embedding_std
from src.routing.text_jepa.encoder import StudentEncoder, TeacherEncoder
from src.routing.text_jepa.loss import masked_l1_loss
from src.routing.text_jepa.masking import span_mask
from src.routing.text_jepa.predictor import Predictor


class TextJEPATrainer:
    """One-step JEPA trainer. Call :meth:`step` repeatedly."""

    def __init__(
        self,
        input_dim: int = 384,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        predictor_hidden: int = 32,
        lr: float = 1e-3,
        ema_momentum: float = 0.99,
        mask_ratio: float = 0.4,
        min_span: int = 3,
        max_span: int = 5,
        collapse_floor: float = 0.01,
        collapse_patience: int = 2,
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        self.student = StudentEncoder(input_dim, hidden_dim, latent_dim)
        self.teacher = TeacherEncoder(self.student)
        self.predictor = Predictor(latent_dim, predictor_hidden)

        params = list(self.student.parameters()) + list(self.predictor.parameters())
        self.optimizer = optim.AdamW(params, lr=lr)

        self.ema_momentum = ema_momentum
        self.mask_ratio = mask_ratio
        self.min_span = min_span
        self.max_span = max_span
        self._rng = np.random.default_rng(seed)

        self.monitor = CollapseMonitor(floor=collapse_floor, patience=collapse_patience)
        self.collapsed = False

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        """One training step.

        Args:
            tokens: (batch, seq_len, input_dim) — frozen-backbone token embeddings.

        Returns:
            Scalar loss tensor.
        """
        batch, seq_len, _ = tokens.shape

        # Shared span mask for the batch (V-JEPA 2: same mask within a batch row is fine;
        # we share across batch for simplicity in this PoC)
        mask_np = span_mask(
            seq_len=seq_len,
            mask_ratio=self.mask_ratio,
            min_span=self.min_span,
            max_span=self.max_span,
            rng=self._rng,
        )
        mask = torch.from_numpy(mask_np).unsqueeze(0).expand(batch, -1).contiguous()

        # Student sees masked input: zero-out masked positions
        mask_f = mask.unsqueeze(-1).to(tokens.dtype)
        masked_tokens = tokens * (1.0 - mask_f)

        student_latent = self.student(masked_tokens)  # (B, S, D)
        pred = self.predictor(student_latent)  # (B, S, D)

        with torch.no_grad():
            target = self.teacher(tokens).detach()

        loss = masked_l1_loss(pred, target, mask)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # EMA update
        self.teacher.update(self.student, momentum=self.ema_momentum)

        # Collapse check on student latent
        std = embedding_std(student_latent.detach())
        if self.monitor.check(std):
            self.collapsed = True

        return loss.detach()
