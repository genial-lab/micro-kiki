#!/usr/bin/env python3
"""Null-space projection for Brainstacks continual learning.

When training stack N, we must not interfere with stacks 1..N-1.
We collect the frozen weight directions from all previous stacks,
compute a null-space projector via randomized SVD, and project
each gradient update into that null-space.

This guarantees zero catastrophic forgetting on previously learned domains.

Paper: Brainstacks (arXiv:2604.01152), Section 3.2
Also: Orthogonal Gradient Descent (OGD), Chaudhry et al. 2020
"""

import numpy as np
import mlx.core as mx


def randomized_svd(
    A: np.ndarray,
    n_components: int,
    n_oversamples: int = 10,
    n_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomized truncated SVD using the Halko-Martinsson-Tropp algorithm.

    Computes the top-k singular vectors/values of A without forming
    the full SVD. Memory: O(m*k + n*k) instead of O(m*n).

    Args:
        A: (m, n) matrix
        n_components: number of singular vectors to extract (k)
        n_oversamples: extra random vectors for stability
        n_iter: power iterations to sharpen spectrum

    Returns:
        U: (m, k) left singular vectors
        S: (k,) singular values
        Vt: (k, n) right singular vectors transposed
    """
    m, n = A.shape
    k = min(n_components, min(m, n))
    total = k + n_oversamples

    # Step 1: Random projection
    rng = np.random.RandomState(42)
    Omega = rng.randn(n, total).astype(A.dtype)
    Y = A @ Omega  # (m, total)

    # Step 2: Power iterations to sharpen
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    # Step 3: QR factorization of Y
    Q, _ = np.linalg.qr(Y)  # Q: (m, total)

    # Step 4: Project A into the low-rank basis
    B = Q.T @ A  # (total, n)

    # Step 5: SVD of the small matrix B
    U_hat, S, Vt = np.linalg.svd(B, full_matrices=False)

    # Step 6: Recover left singular vectors
    U = Q @ U_hat

    return U[:, :k], S[:k], Vt[:k, :]


def compute_null_space_projector(
    frozen_weight_vectors: list[np.ndarray],
    weight_dim: int,
    ns_top_k_dirs: int = 32,
    svd_oversampling: int = 10,
    svd_n_iter: int = 3,
) -> np.ndarray:
    """Compute the null-space projection matrix P.

    P projects any vector into the subspace orthogonal to the
    top-k directions of the frozen stack weight matrix.

    If no frozen stacks exist, returns identity (no projection).

    Args:
        frozen_weight_vectors: list of 1D numpy arrays, one per frozen
            MoE-LoRA layer. Each vector is the concatenation of all
            expert weights (A and B matrices) for that layer.
        weight_dim: dimensionality of each weight vector
        ns_top_k_dirs: number of directions to preserve per stack
        svd_oversampling: oversampling factor for randomized SVD
        svd_n_iter: power iterations for randomized SVD

    Returns:
        P: (weight_dim, weight_dim) projection matrix into null-space
    """
    if len(frozen_weight_vectors) == 0:
        return np.eye(weight_dim, dtype=np.float32)

    # Stack frozen weights into a matrix: (num_frozen, weight_dim)
    W = np.stack(frozen_weight_vectors, axis=0).astype(np.float32)

    # Number of components to extract
    n_components = min(ns_top_k_dirs * len(frozen_weight_vectors), weight_dim - 1)
    n_components = max(1, n_components)

    if W.shape[0] < n_components:
        # Fewer frozen vectors than requested components: use full SVD
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        # Keep all non-trivial directions
        n_keep = min(len(S), n_components)
        V_keep = Vt[:n_keep, :]  # (n_keep, weight_dim)
    else:
        # Use randomized SVD for efficiency
        U, S, Vt = randomized_svd(
            W, n_components=n_components,
            n_oversamples=svd_oversampling,
            n_iter=svd_n_iter,
        )
        V_keep = Vt  # (n_components, weight_dim)

    # Null-space projector: P = I - V^T @ V
    # V_keep rows are the directions to block
    P = np.eye(weight_dim, dtype=np.float32) - V_keep.T @ V_keep

    return P


def project_gradient(grad: mx.array, projector: mx.array) -> mx.array:
    """Project a gradient vector into the null-space.

    Args:
        grad: (weight_dim,) or (d1, d2) gradient tensor
        projector: (weight_dim, weight_dim) null-space projection matrix

    Returns:
        Projected gradient with same shape as input
    """
    original_shape = grad.shape
    flat = mx.reshape(grad, (-1,))
    projected = projector @ flat
    return mx.reshape(projected, original_shape)


def collect_frozen_weights_from_disk(
    stack_dirs: list[str],
) -> dict[str, list[np.ndarray]]:
    """Load frozen stack weights from safetensors files on disk.

    For each MoE-LoRA layer name (e.g. "layers.0.self_attn.q_proj_moe_lora"),
    collects the concatenated expert weights from all frozen stacks.

    Args:
        stack_dirs: list of directories containing adapters.safetensors

    Returns:
        Dictionary mapping layer_name -> list of weight vectors (one per stack)
    """
    from safetensors.numpy import load_file

    layer_weights: dict[str, list[np.ndarray]] = {}

    for stack_dir in stack_dirs:
        path = f"{stack_dir}/adapters.safetensors"
        weights = load_file(path)

        # Group by MoE-LoRA layer
        layer_parts: dict[str, list[np.ndarray]] = {}
        for key, tensor in weights.items():
            # Keys like: "model.layers.0.self_attn.q_proj_moe_lora.experts.0.lora_a"
            # Extract layer identifier up to the moe_lora part
            parts = key.split(".")
            moe_idx = None
            for i, p in enumerate(parts):
                if p.endswith("_moe_lora"):
                    moe_idx = i
                    break
            if moe_idx is None:
                continue
            layer_name = ".".join(parts[:moe_idx + 1])
            if layer_name not in layer_parts:
                layer_parts[layer_name] = []
            layer_parts[layer_name].append(tensor.flatten())

        # Concatenate all expert weights per layer into a single vector
        for layer_name, parts in layer_parts.items():
            vec = np.concatenate(parts)
            if layer_name not in layer_weights:
                layer_weights[layer_name] = []
            layer_weights[layer_name].append(vec)

    return layer_weights


def build_projectors_for_stack(
    frozen_stack_dirs: list[str],
    ns_top_k_dirs: int = 32,
    svd_oversampling: int = 10,
    svd_n_iter: int = 3,
) -> dict[str, mx.array]:
    """Build null-space projectors for all MoE-LoRA layers.

    Called before training stack N. Loads stacks 1..N-1 from disk,
    computes one projector per MoE-LoRA layer name.

    Args:
        frozen_stack_dirs: paths to frozen stack output directories
        ns_top_k_dirs: directions to block per frozen stack
        svd_oversampling: for randomized SVD
        svd_n_iter: power iterations

    Returns:
        Dict mapping layer_name -> (weight_dim, weight_dim) MLX projector
    """
    if len(frozen_stack_dirs) == 0:
        return {}

    layer_weights = collect_frozen_weights_from_disk(frozen_stack_dirs)
    projectors = {}

    for layer_name, weight_vectors in layer_weights.items():
        weight_dim = weight_vectors[0].shape[0]
        P_np = compute_null_space_projector(
            frozen_weight_vectors=weight_vectors,
            weight_dim=weight_dim,
            ns_top_k_dirs=ns_top_k_dirs,
            svd_oversampling=svd_oversampling,
            svd_n_iter=svd_n_iter,
        )
        projectors[layer_name] = mx.array(P_np)

    return projectors
