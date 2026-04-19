"""Pure-torch VQC forward pass matching PennyLane's default.qubit.

Replaces PennyLane for our specific 6-qubit, 6-layer StronglyEntanglingLayers
circuit. Enables MPS/CUDA execution + autograd-based gradients (vs parameter
shift) + batched inference.

Conventions (matched to PennyLane default.qubit):
  - Wire 0 = most significant bit in |q_0 q_1 ... q_{N-1}⟩.
  - Basis index: idx = sum_i q_i · 2^{N-1-i}.
  - Rot(φ, θ, ω) = RZ(ω) · RY(θ) · RZ(φ).
  - StronglyEntanglingLayers ranges: r_l = l % (N-1) + 1.
"""
from __future__ import annotations

import torch


def _ry(theta: torch.Tensor, cdtype: torch.dtype) -> torch.Tensor:
    """RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]. θ scalar → (2, 2)."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    zero = torch.zeros_like(c)
    row0 = torch.stack([torch.complex(c, zero), torch.complex(-s, zero)])
    row1 = torch.stack([torch.complex(s, zero), torch.complex(c, zero)])
    return torch.stack([row0, row1]).to(cdtype)


def _rz(theta: torch.Tensor, cdtype: torch.dtype) -> torch.Tensor:
    """RZ(θ) = diag(exp(-iθ/2), exp(iθ/2)). θ scalar → (2, 2)."""
    half = theta / 2
    cos_h = torch.cos(half)
    sin_h = torch.sin(half)
    zero_c = torch.complex(torch.zeros_like(cos_h), torch.zeros_like(cos_h))
    e_minus = torch.complex(cos_h, -sin_h)  # exp(-iθ/2)
    e_plus = torch.complex(cos_h, sin_h)    # exp(+iθ/2)
    row0 = torch.stack([e_minus, zero_c])
    row1 = torch.stack([zero_c, e_plus])
    return torch.stack([row0, row1]).to(cdtype)


def _rot(phi: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor, cdtype: torch.dtype) -> torch.Tensor:
    """PennyLane convention: Rot(φ, θ, ω) = RZ(ω) @ RY(θ) @ RZ(φ)."""
    return _rz(omega, cdtype) @ _ry(theta, cdtype) @ _rz(phi, cdtype)


def _ry_batched(theta: torch.Tensor, cdtype: torch.dtype) -> torch.Tensor:
    """RY for a batch of angles. theta shape (B,) → (B, 2, 2)."""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    zero = torch.zeros_like(c)
    row0 = torch.stack([torch.complex(c, zero), torch.complex(-s, zero)], dim=-1)
    row1 = torch.stack([torch.complex(s, zero), torch.complex(c, zero)], dim=-1)
    return torch.stack([row0, row1], dim=-2).to(cdtype)


def _rx_batched(theta: torch.Tensor, cdtype: torch.dtype) -> torch.Tensor:
    """RX for a batch of angles. theta shape (B,) → (B, 2, 2).

    RX(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]].
    PennyLane AngleEmbedding default is RX.
    """
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    zero = torch.zeros_like(c)
    row0 = torch.stack([torch.complex(c, zero), torch.complex(zero, -s)], dim=-1)
    row1 = torch.stack([torch.complex(zero, -s), torch.complex(c, zero)], dim=-1)
    return torch.stack([row0, row1], dim=-2).to(cdtype)


def _apply_1q(state: torch.Tensor, U: torch.Tensor, q: int, n: int) -> torch.Tensor:
    """Apply a single-qubit gate U (shape (2,2)) on wire q. state: (B, 2^n)."""
    B = state.shape[0]
    s = state.reshape(B, 2**q, 2, 2**(n - q - 1))
    s = torch.einsum("ij,xajk->xaik", U, s)
    return s.reshape(B, 2**n)


def _apply_1q_batched(state: torch.Tensor, U_batch: torch.Tensor, q: int, n: int) -> torch.Tensor:
    """Apply a batched single-qubit gate. U_batch: (B, 2, 2), state: (B, 2^n)."""
    B = state.shape[0]
    s = state.reshape(B, 2**q, 2, 2**(n - q - 1))
    s = torch.einsum("xij,xajk->xaik", U_batch, s)
    return s.reshape(B, 2**n)


def _apply_cnot(state: torch.Tensor, ctrl: int, targ: int, n: int) -> torch.Tensor:
    """Apply CNOT(ctrl, targ) via index permutation. CNOT is an involution."""
    idx = torch.arange(2**n, device=state.device)
    q_c = (idx >> (n - 1 - ctrl)) & 1
    new_idx = idx ^ (q_c * (1 << (n - 1 - targ)))
    return state.index_select(-1, new_idx)


def torch_vqc_forward(
    features: torch.Tensor,
    weights: torch.Tensor,
    n_qubits: int = 6,
    n_layers: int = 6,
) -> torch.Tensor:
    """Forward pass of the VQC matching PennyLane default.qubit output.

    Args:
        features: Shape (>=n_qubits,) single sample, or (B, >=n_qubits) batched.
            Only the first n_qubits entries are used as RY embedding angles.
        weights: Shape (n_layers, n_qubits, 3). Same weights applied across batch.
        n_qubits: Number of wires.
        n_layers: Number of StronglyEntanglingLayers layers.

    Returns:
        Tensor of expectation values <Z_i>, shape (n_qubits,) if unbatched input
        or (B, n_qubits) if batched input. Values in [-1, 1].
    """
    batched = features.dim() == 2
    if not batched:
        features = features.unsqueeze(0)

    B = features.shape[0]
    device = features.device
    cdtype = torch.complex128 if features.dtype == torch.float64 else torch.complex64

    state = torch.zeros(B, 2**n_qubits, dtype=cdtype, device=device)
    state[:, 0] = 1.0

    # AngleEmbedding = RX(features[i]) on wire i (PennyLane default is 'X')
    for i in range(n_qubits):
        theta = features[..., i]  # (B,)
        U_batch = _rx_batched(theta, cdtype)
        state = _apply_1q_batched(state, U_batch, i, n_qubits)

    # StronglyEntanglingLayers
    for l in range(n_layers):
        for q in range(n_qubits):
            phi = weights[l, q, 0]
            theta = weights[l, q, 1]
            omega = weights[l, q, 2]
            U = _rot(phi, theta, omega, cdtype)
            state = _apply_1q(state, U, q, n_qubits)
        r = l % (n_qubits - 1) + 1
        for q in range(n_qubits):
            state = _apply_cnot(state, q, (q + r) % n_qubits, n_qubits)

    # Measure <Z_i> on each wire
    probs = (state.conj() * state).real  # (B, 2^n)
    idx = torch.arange(2**n_qubits, device=device)
    real_dtype = torch.float64 if cdtype == torch.complex128 else torch.float32
    z_expvals = torch.zeros(B, n_qubits, dtype=real_dtype, device=device)
    for i in range(n_qubits):
        bit = (idx >> (n_qubits - 1 - i)) & 1
        sign = 1.0 - 2.0 * bit.to(real_dtype)
        z_expvals[:, i] = (probs * sign).sum(dim=-1)

    if not batched:
        return z_expvals.squeeze(0)
    return z_expvals
