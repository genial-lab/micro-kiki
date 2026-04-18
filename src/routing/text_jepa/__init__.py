"""Text-JEPA: JEPA-style masked prediction for VQC router embeddings.

V-JEPA 2 adapted for text — frozen sentence-transformer backbone, trainable
student/EMA-teacher MLP projectors, tiny predictor, L1 masked loss.
"""
from __future__ import annotations

__all__ = [
    "span_mask",
    "StudentEncoder",
    "TeacherEncoder",
    "Predictor",
    "masked_l1_loss",
    "CollapseMonitor",
    "TextJEPAEmbedder",
]
