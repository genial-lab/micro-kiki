"""AeonPredictor — JEPA-inspired latent predictor on top of AeonSleep.

Adds a small numpy MLP that learns h_t -> h_{t+1} from the temporal
edges of TraceGraph. No torch, no sklearn — same pattern as
ForgettingGate so this runs on GrosMac M5 / 16 GB and on CI.

Public API:
    AeonPredictor(palace, config)
        .ingest_latent(turn_id, h, ts, stack_id=None)
        .predict_next(h_t, horizon=1, stack_id=None) -> np.ndarray
        .recall(query_vec, top_k=10)          # delegates to palace
        .fit_on_buffer(lr=1e-3, epochs=1, batch_size=32)
        .ready -> bool

    LatentMLP(dim, hidden, n_stacks)
        .forward(x, stack_onehot) -> h_hat
        .backward_cosine(x, stack_onehot, target) -> float  # returns loss

    PredictorConfig(frozen dataclass)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.memory.aeonsleep import AeonSleep


@dataclass(frozen=True)
class PredictorConfig:
    """Immutable predictor config."""

    dim: int
    hidden: int = 256
    horizon: int = 1
    n_stacks: int = 16
    cold_start_threshold: int = 500
    seed: int = 0
    use_centering: bool = False
    centering_momentum: float = 0.9
    per_stack_centering: bool = False  # if True, maintain n_stacks+1 running means


def detect_collapse(
    h_t: np.ndarray, h_hat: np.ndarray, threshold: float = 0.1
) -> tuple[bool, float]:
    """Flag predictor collapse when std(h_hat) << std(h_t).

    Returns (flagged, ratio). ratio = std(h_hat) / std(h_t) averaged
    across feature dims. Flagged is True iff ratio < threshold.
    """
    if h_t.shape != h_hat.shape:
        raise ValueError(f"shape mismatch {h_t.shape} vs {h_hat.shape}")
    std_t = float(np.std(h_t))
    std_hat = float(np.std(h_hat))
    if std_t < 1e-9:
        return False, 1.0
    ratio = std_hat / std_t
    return bool(ratio < threshold), float(ratio)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class LatentMLP:
    """2-layer numpy MLP with skip connection.

    Input: concat(x[dim], stack_onehot[n_stacks]) of size dim+n_stacks
    Hidden: linear(hidden) -> ReLU -> linear(hidden) -> ReLU
    Output: linear(dim) + x   (residual / skip on the embedding path)
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        n_stacks: int,
        seed: int = 0,
        use_centering: bool = False,
        centering_momentum: float = 0.9,
        per_stack_centering: bool = False,
    ) -> None:
        self.dim = dim
        self.hidden = hidden
        self.n_stacks = n_stacks
        rng = np.random.default_rng(seed)
        in_dim = dim + n_stacks
        scale1 = np.sqrt(2.0 / in_dim)
        scale2 = np.sqrt(2.0 / hidden)
        scale3 = np.sqrt(2.0 / hidden) * 0.1  # small init so skip dominates at t=0
        self.w1 = (rng.standard_normal((in_dim, hidden)) * scale1).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = (rng.standard_normal((hidden, hidden)) * scale2).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.w3 = (rng.standard_normal((hidden, dim)) * scale3).astype(np.float32)
        self.b3 = np.zeros(dim, dtype=np.float32)
        self._cache: dict = {}
        # Centering state
        self.use_centering = bool(use_centering)
        self.centering_momentum = float(centering_momentum)
        self._running_mean = np.zeros(dim, dtype=np.float32)
        # When per_stack_centering, maintain one running mean per stack + a shared fallback at index 0.
        # Index layout: running_means[0] = shared fallback (for stack_id=-1 or unknown),
        #               running_means[i+1] = per-stack mean for stack_id=i (i in 0..n_stacks-1)
        self.per_stack_centering = bool(per_stack_centering)
        if self.per_stack_centering:
            self._running_means_per_stack = np.zeros((n_stacks + 1, dim), dtype=np.float32)

    def forward(self, x: np.ndarray, stack_onehot: np.ndarray, stack_ids: np.ndarray | None = None) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError(f"x must be (batch, {self.dim}), got {x.shape}")
        if stack_onehot.shape != (x.shape[0], self.n_stacks):
            raise ValueError(
                f"stack_onehot must be (batch, {self.n_stacks}), got {stack_onehot.shape}"
            )
        inp = np.concatenate([x, stack_onehot], axis=1).astype(np.float32)
        z1 = np.clip(inp @ self.w1 + self.b1, -30.0, 30.0)
        h1 = _relu(z1)
        z2 = np.clip(h1 @ self.w2 + self.b2, -30.0, 30.0)
        h2 = _relu(z2)
        delta = h2 @ self.w3 + self.b3
        out = (x + delta).astype(np.float32)
        # DinoV3-style centering (anti-collapse by distribution shift).
        if self.use_centering:
            if self.per_stack_centering and stack_ids is not None:
                # Per-stack centering: maintain separate running means for each stack
                for sid in np.unique(stack_ids):
                    slot = int(sid) + 1  # -1 -> 0, 0 -> 1, ..., n-1 -> n
                    mask_idx = (stack_ids == sid)
                    group_mean = out[mask_idx].mean(axis=0)
                    self._running_means_per_stack[slot] = (
                        self.centering_momentum * self._running_means_per_stack[slot]
                        + (1.0 - self.centering_momentum) * group_mean
                    ).astype(np.float32)
                    out[mask_idx] = (out[mask_idx] - self._running_means_per_stack[slot]).astype(np.float32)
            else:
                # Shared centering (existing behavior)
                batch_mean = out.mean(axis=0)
                self._running_mean = (
                    self.centering_momentum * self._running_mean
                    + (1.0 - self.centering_momentum) * batch_mean
                ).astype(np.float32)
                out = (out - self._running_mean).astype(np.float32)
        # Cache for backward.
        self._cache = {"inp": inp, "z1": z1, "h1": h1, "z2": z2, "h2": h2, "x": x}
        return out

    def backward_cosine(self, target: np.ndarray, lr: float = 1e-3) -> float:
        """One SGD step with cosine-similarity loss.

        loss = 1 - mean(cos(h_hat, target)). Returns the scalar loss
        BEFORE the update (for logging / convergence checks).
        """
        cache = self._cache
        x = cache["x"]
        inp = cache["inp"]
        h1 = cache["h1"]
        h2 = cache["h2"]
        z1 = cache["z1"]
        z2 = cache["z2"]
        batch, dim = x.shape
        if target.shape != (batch, dim):
            raise ValueError(f"target shape {target.shape} != {(batch, dim)}")

        # Recompute h_hat from the same x + delta path so grad lines up.
        delta = h2 @ self.w3 + self.b3
        h_hat = (x + delta).astype(np.float32)

        eps = 1e-8
        n_hat = np.linalg.norm(h_hat, axis=1, keepdims=True) + eps
        n_tgt = np.linalg.norm(target, axis=1, keepdims=True) + eps
        cos = np.sum(h_hat * target, axis=1, keepdims=True) / (n_hat * n_tgt)
        loss = float(1.0 - cos.mean())

        # d loss / d h_hat = -(1/batch) * [target/(|h_hat|*|target|)
        #                   - cos * h_hat / (|h_hat|^2)]
        d_h_hat = -(
            target / (n_hat * n_tgt)
            - cos * h_hat / (n_hat * n_hat)
        ) / batch

        # d_h_hat flows into delta and into the skip (skip grad on x is
        # not used — we don't update x, it is input data).
        d_delta = d_h_hat  # shape (batch, dim)
        d_w3 = h2.T @ d_delta
        d_b3 = d_delta.sum(axis=0)
        d_h2 = d_delta @ self.w3.T
        d_z2 = d_h2 * (z2 > 0)
        d_w2 = h1.T @ d_z2
        d_b2 = d_z2.sum(axis=0)
        d_h1 = d_z2 @ self.w2.T
        d_z1 = d_h1 * (z1 > 0)
        d_w1 = inp.T @ d_z1
        d_b1 = d_z1.sum(axis=0)

        # Gradient clip — mirrors ForgettingGate pattern.
        clip = 5.0
        for g in (d_w1, d_b1, d_w2, d_b2, d_w3, d_b3):
            np.clip(g, -clip, clip, out=g)

        self.w1 -= lr * d_w1
        self.b1 -= lr * d_b1
        self.w2 -= lr * d_w2
        self.b2 -= lr * d_b2
        self.w3 -= lr * d_w3
        self.b3 -= lr * d_b3
        return loss



@dataclass
class _PairSample:
    turn_id: str
    h: np.ndarray            # shape (dim,)
    ts: datetime
    stack_id: int            # -1 means "unknown / default"

class AeonPredictor:
    """Facade wrapping AeonSleep with a JEPA-style latent predictor."""

    def __init__(self, palace: "AeonSleep", config: PredictorConfig) -> None:
        if config.dim != palace.dim:
            raise ValueError(
                f"config.dim={config.dim} != palace.dim={palace.dim}"
            )
        self.palace = palace
        self.config = config
        self.mlp = LatentMLP(
            dim=config.dim,
            hidden=config.hidden,
            n_stacks=config.n_stacks,
            seed=config.seed,
            use_centering=config.use_centering,
            centering_momentum=config.centering_momentum,
            per_stack_centering=config.per_stack_centering,
        )
        self._buffer: list[_PairSample] = []
        self._trained_once = False

    # ---------------------------------------------------------- buffer mgmt

    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def ready(self) -> bool:
        return (
            self._trained_once
            and len(self._buffer) >= self.config.cold_start_threshold
        )

    # ---------------------------------------------------------- ingest

    def ingest_latent(
        self,
        turn_id: str,
        h: np.ndarray,
        ts: datetime,
        stack_id: int | None = None,
    ) -> None:
        """Append a latent sample to the buffer and persist to the palace.

        Writes an Episode to AeonSleep (atlas + graph) so existing recall
        paths still work. Adds a temporal edge from the previous sample
        (if any) to this one.
        """
        if h.shape != (self.config.dim,):
            raise ValueError(
                f"h.shape={h.shape} != expected ({self.config.dim},) dim"
            )
        from src.memory.aeonsleep import Episode

        ep = Episode(
            id=turn_id,
            text="",
            embedding=h.astype(np.float32).tolist(),
            ts=ts,
            topic="__predictor__",
            payload={"stack_id": -1 if stack_id is None else int(stack_id)},
        )
        self.palace.write(ep)

        prev = self._buffer[-1] if self._buffer else None
        if prev is not None and self.palace.graph.has_node(prev.turn_id):
            # Add explicit temporal edge even across different topics.
            self.palace.graph.add_typed_edge(
                prev.turn_id, turn_id, "temporal"
            )

        self._buffer.append(
            _PairSample(
                turn_id=turn_id,
                h=h.astype(np.float32).copy(),
                ts=ts,
                stack_id=-1 if stack_id is None else int(stack_id),
            )
        )

    def pairs_for_training(self) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Build `(h_t, h_{t+1}, stack_id)` triples from the ordered buffer."""
        out: list[tuple[np.ndarray, np.ndarray, int]] = []
        for a, b in zip(self._buffer, self._buffer[1:]):
            out.append((a.h, b.h, a.stack_id))
        return out


    # ---------------------------------------------------------- training

    def fit_on_buffer(
        self,
        *,
        lr: float = 1e-3,
        epochs: int = 1,
        batch_size: int = 32,
    ) -> list[float]:
        """One (or more) SGD passes over the current pair buffer.

        Returns per-epoch mean loss. Sets `_trained_once=True` iff at
        least one pair was seen.
        """
        triples = self.pairs_for_training()
        if not triples:
            return []
        n = len(triples)
        rng = np.random.default_rng(self.config.seed)
        history: list[float] = []
        for _ in range(max(1, epochs)):
            order = rng.permutation(n)
            losses: list[float] = []
            for start in range(0, n, batch_size):
                batch = [triples[i] for i in order[start:start + batch_size]]
                x = np.stack([t[0] for t in batch]).astype(np.float32)
                tgt = np.stack([t[1] for t in batch]).astype(np.float32)
                stack = self._stack_onehot([t[2] for t in batch])
                stack_ids = np.array([t[2] for t in batch], dtype=np.int64)
                self.mlp.forward(x, stack, stack_ids=stack_ids)
                loss = self.mlp.backward_cosine(tgt, lr=lr)
                losses.append(loss)
            history.append(float(np.mean(losses)))
        self._trained_once = True
        return history

    # ---------------------------------------------------------- predict

    def predict_next(
        self,
        h_t: np.ndarray,
        horizon: int = 1,
        stack_id: int | None = None,
    ) -> np.ndarray:
        """Predict h_{t+horizon} by iterating the MLP `horizon` times.

        Cold-start: if `not self.ready`, return `h_t` unchanged so the
        serving path stays on pure-retrieval mode.
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if h_t.shape != (self.config.dim,):
            raise ValueError(
                f"h_t.shape={h_t.shape} != ({self.config.dim},)"
            )
        if not self.ready:
            return h_t.astype(np.float32).copy()
        x = h_t.reshape(1, -1).astype(np.float32)
        stack = self._stack_onehot([-1 if stack_id is None else int(stack_id)])
        stack_ids = np.array([-1 if stack_id is None else int(stack_id)], dtype=np.int64)
        current = x
        for _ in range(horizon):
            current = self.mlp.forward(current, stack, stack_ids=stack_ids)
        return current[0].astype(np.float32)

    # ---------------------------------------------------------- recall

    def recall(self, query_vec: np.ndarray, top_k: int = 10):
        """Delegate to the underlying AeonSleep — backward-compat path."""
        if query_vec.shape != (self.config.dim,):
            raise ValueError(
                f"query_vec.shape={query_vec.shape} != ({self.config.dim},)"
            )
        return self.palace.recall(query_vec.tolist(), k=top_k)

    # ---------------------------------------------------------- helpers

    def _stack_onehot(self, stack_ids: list[int]) -> np.ndarray:
        n = len(stack_ids)
        out = np.zeros((n, self.config.n_stacks), dtype=np.float32)
        for i, sid in enumerate(stack_ids):
            if sid < 0:
                continue  # unknown stack -> all zeros (null condition)
            idx = sid % self.config.n_stacks
            out[i, idx] = 1.0
        # Amplify the stack signal so it isn't washed out by a high-dim x.
        # Ratio calibrated so stack column L2 ~ sqrt(dim)/sqrt(n_stacks).
        scale = float(np.sqrt(self.config.dim / max(self.config.n_stacks, 1)))
        return (out * scale).astype(np.float32)
