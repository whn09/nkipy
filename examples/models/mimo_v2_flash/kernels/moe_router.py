"""MoE Router kernel for MiMo-V2-Flash

MiMo-V2-Flash uses sigmoid-based routing with noaux_tc (no auxiliary loss with
token choice). This is different from standard softmax-based routing.

Key features:
- Sigmoid activation (not softmax) for expert scores
- Top-8 expert selection out of 256 experts
- noaux_tc: Token choice without auxiliary load balancing loss
- Normalized routing weights for selected experts
"""

import numpy as np


def topk_numpy(x, k, axis=-1, is_ascend=False):
    """
    Pure numpy implementation of top-k.

    Args:
        x: Input array
        k: Number of top elements
        axis: Axis along which to find top-k
        is_ascend: If True, return smallest k values; if False, return largest

    Returns:
        values: Top-k values
        indices: Indices of top-k values
    """
    if axis < 0:
        axis = x.ndim + axis

    if is_ascend:
        indices = np.argpartition(x, k - 1, axis=axis)
        indices = np.take(indices, range(k), axis=axis)
        values = np.take_along_axis(x, indices, axis=axis)
        sort_indices = np.argsort(values, axis=axis)
        indices = np.take_along_axis(indices, sort_indices, axis=axis)
        values = np.take_along_axis(values, sort_indices, axis=axis)
    else:
        indices = np.argpartition(-x, k - 1, axis=axis)
        indices = np.take(indices, range(k), axis=axis)
        values = np.take_along_axis(x, indices, axis=axis)
        sort_indices = np.argsort(-values, axis=axis)
        indices = np.take_along_axis(indices, sort_indices, axis=axis)
        values = np.take_along_axis(values, sort_indices, axis=axis)

    return values, indices.astype(np.int32)


def sigmoid(x):
    """Numerically stable sigmoid function."""
    x = x.astype(np.float32)
    # Use the stable formula: sigmoid(x) = 1 / (1 + exp(-x))
    # For negative x: sigmoid(x) = exp(x) / (1 + exp(x)) to avoid overflow
    pos_mask = x >= 0
    neg_mask = ~pos_mask

    result = np.zeros_like(x)

    # For positive x
    result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))

    # For negative x (more stable)
    exp_x = np.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1.0 + exp_x)

    return result


def moe_router_sigmoid(
    hidden_states,
    router_weight,
    num_experts: int,
    top_k: int,
):
    """
    MoE router using sigmoid activation (noaux_tc).

    Args:
        hidden_states: [batch_size * seq_len, hidden_size]
        router_weight: [hidden_size, num_experts]
        num_experts: Total number of experts (256)
        top_k: Number of experts to select per token (8)

    Returns:
        topk_indices: [batch_size * seq_len, top_k] - Selected expert indices
        topk_weights: [batch_size * seq_len, top_k] - Normalized routing weights
        router_logits: [batch_size * seq_len, num_experts] - Raw router scores
    """
    original_dtype = hidden_states.dtype

    # 1. Compute router logits
    router_logits = hidden_states @ router_weight  # [B*S, 256]
    router_logits = router_logits.astype(np.float32)

    # 2. Apply sigmoid (not softmax!) - noaux_tc uses sigmoid
    router_probs = sigmoid(router_logits)  # [B*S, 256]

    # 3. Select top-k experts
    topk_values, topk_indices = topk_numpy(
        router_probs,
        k=top_k,
        axis=-1,
        is_ascend=False,  # We want largest values
    )  # Both: [B*S, 8]

    # 4. Normalize top-k weights to sum to 1
    topk_weights = topk_values / np.sum(topk_values, axis=-1, keepdims=True)
    topk_weights = topk_weights.astype(original_dtype)

    # Convert indices to int32 for indexing
    topk_indices = topk_indices.astype(np.int32)

    return topk_indices, topk_weights, router_logits


def compute_expert_capacity(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    capacity_factor: float = 1.25,
) -> int:
    """
    Compute expert capacity for load-balanced MoE.

    Each expert processes at most `capacity` tokens to prevent imbalance.

    Args:
        num_tokens: Total number of tokens (batch_size * seq_len)
        num_experts: Number of experts
        top_k: Experts selected per token
        capacity_factor: Multiplier for capacity headroom (default 1.25)

    Returns:
        capacity: Maximum tokens per expert
    """
    # Average tokens per expert = (num_tokens * top_k) / num_experts
    avg_tokens_per_expert = (num_tokens * top_k) / num_experts

    # Add headroom for load imbalance
    capacity = int(avg_tokens_per_expert * capacity_factor)

    # Ensure at least 1
    return max(capacity, 1)


def compute_expert_assignment(
    topk_indices,
    num_experts: int,
    num_devices: int,
):
    """
    Compute which device each token should be sent to based on expert assignment.

    For Expert Parallelism (EP), experts are distributed across devices:
    - Device 0: Experts 0-7
    - Device 1: Experts 8-15
    - ...
    - Device 31: Experts 248-255

    Args:
        topk_indices: [batch_size * seq_len, top_k] - Selected expert indices
        num_experts: Total number of experts (256)
        num_devices: Number of devices (32)

    Returns:
        device_ids: [batch_size * seq_len, top_k] - Device ID for each expert
        local_expert_ids: [batch_size * seq_len, top_k] - Local expert index on device
    """
    experts_per_device = num_experts // num_devices

    # Compute device assignment
    device_ids = topk_indices // experts_per_device

    # Compute local expert index within device
    local_expert_ids = topk_indices % experts_per_device

    return device_ids.astype(np.int32), local_expert_ids.astype(np.int32)
