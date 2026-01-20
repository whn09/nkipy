"""MoE Expert FFN kernel for MiMo-V2-Flash

Each expert is a gated feed-forward network with SiLU activation.
MiMo-V2-Flash uses relatively small experts (intermediate_size=2048) but
many of them (256 experts, 8 active per token).

Expert FFN structure:
- gate_proj: [hidden_size, moe_intermediate_size] = [4096, 2048]
- up_proj: [hidden_size, moe_intermediate_size] = [4096, 2048]
- down_proj: [moe_intermediate_size, hidden_size] = [2048, 4096]
- Activation: SiLU (Swish)
"""

import numpy as np


def silu_kernel(x):
    """
    SiLU (Swish) activation function: x * sigmoid(x)

    Args:
        x: Input tensor

    Returns:
        SiLU activated tensor
    """
    x = x.astype(np.float32)
    return x * (1.0 / (1.0 + np.exp(-x)))


def moe_expert_ffn(
    x,
    gate_weight,
    up_weight,
    down_weight,
):
    """
    Single expert feed-forward network with SiLU gating.

    Args:
        x: Input tensor [num_tokens, hidden_size]
        gate_weight: Gate projection [hidden_size, intermediate_size]
        up_weight: Up projection [hidden_size, intermediate_size]
        down_weight: Down projection [intermediate_size, hidden_size]

    Returns:
        output: [num_tokens, hidden_size]
    """
    original_dtype = x.dtype

    # Gate projection + SiLU activation
    gate = x @ gate_weight
    gate = silu_kernel(gate)

    # Up projection
    up = x @ up_weight

    # Element-wise multiplication (gating)
    hidden = gate * up

    # Down projection
    output = hidden.astype(np.float32) @ down_weight.astype(np.float32)

    return output.astype(original_dtype)


def moe_expert_ffn_fused(
    x,
    gate_up_weight,
    down_weight,
):
    """
    Single expert FFN with fused gate+up projection.

    Args:
        x: Input tensor [num_tokens, hidden_size]
        gate_up_weight: Fused gate and up projection [hidden_size, 2 * intermediate_size]
        down_weight: Down projection [intermediate_size, hidden_size]

    Returns:
        output: [num_tokens, hidden_size]
    """
    original_dtype = x.dtype

    # Fused gate + up projection
    gate_up = x @ gate_up_weight  # [N, 2 * intermediate]

    # Split into gate and up
    gate, up = np.split(gate_up, 2, axis=-1)

    # Gate with SiLU activation
    gate = silu_kernel(gate)

    # Gating
    hidden = gate * up

    # Down projection
    output = hidden.astype(np.float32) @ down_weight.astype(np.float32)

    return output.astype(original_dtype)


def apply_selected_experts(
    hidden_states,
    topk_indices,
    topk_weights,
    expert_gate_weights,
    expert_up_weights,
    expert_down_weights,
):
    """
    Apply selected experts to input tokens and combine outputs.

    This is a simple sequential implementation for correctness verification.
    Production would use parallel expert computation or EP.

    Args:
        hidden_states: [batch_size * seq_len, hidden_size]
        topk_indices: [batch_size * seq_len, top_k] - Selected expert indices
        topk_weights: [batch_size * seq_len, top_k] - Routing weights
        expert_gate_weights: List of [hidden_size, intermediate_size] per expert
        expert_up_weights: List of [hidden_size, intermediate_size] per expert
        expert_down_weights: List of [intermediate_size, hidden_size] per expert

    Returns:
        output: [batch_size * seq_len, hidden_size] - Combined expert outputs
    """
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    top_k = topk_indices.shape[1]
    original_dtype = hidden_states.dtype

    # Initialize output
    output = np.zeros((num_tokens, hidden_size), dtype=np.float32)

    # Process each token
    for token_idx in range(num_tokens):
        token_hidden = hidden_states[token_idx : token_idx + 1]  # [1, hidden_size]

        # Apply each selected expert
        for k in range(top_k):
            expert_idx = topk_indices[token_idx, k]
            weight = topk_weights[token_idx, k]

            # Get expert weights
            gate_w = expert_gate_weights[expert_idx]
            up_w = expert_up_weights[expert_idx]
            down_w = expert_down_weights[expert_idx]

            # Apply expert FFN
            expert_out = moe_expert_ffn(token_hidden, gate_w, up_w, down_w)

            # Weighted accumulation
            output[token_idx] += weight * expert_out[0]

    return output.astype(original_dtype)


def apply_selected_experts_batched(
    hidden_states,
    topk_indices,
    topk_weights,
    all_gate_weights,
    all_up_weights,
    all_down_weights,
    num_experts: int,
):
    """
    Batched expert application - processes all tokens through each expert.

    More efficient than per-token processing when experts are small.

    Args:
        hidden_states: [batch_size * seq_len, hidden_size]
        topk_indices: [batch_size * seq_len, top_k]
        topk_weights: [batch_size * seq_len, top_k]
        all_gate_weights: [num_experts, hidden_size, intermediate_size]
        all_up_weights: [num_experts, hidden_size, intermediate_size]
        all_down_weights: [num_experts, intermediate_size, hidden_size]
        num_experts: Number of experts

    Returns:
        output: [batch_size * seq_len, hidden_size]
    """
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    top_k = topk_indices.shape[1]
    original_dtype = hidden_states.dtype

    # Initialize output
    output = np.zeros((num_tokens, hidden_size), dtype=np.float32)

    # Process each expert
    for expert_idx in range(num_experts):
        # Find which tokens use this expert
        # Create mask: [num_tokens, top_k] - True where expert_idx matches
        expert_mask = topk_indices == expert_idx

        # Get tokens that route to this expert
        token_indices = np.where(np.any(expert_mask, axis=1))[0]

        if len(token_indices) == 0:
            continue

        # Get hidden states for these tokens
        expert_input = hidden_states[token_indices]  # [N_expert, hidden_size]

        # Apply expert FFN
        expert_output = moe_expert_ffn(
            expert_input,
            all_gate_weights[expert_idx],
            all_up_weights[expert_idx],
            all_down_weights[expert_idx],
        )

        # Get weights for these tokens (for this expert)
        for i, token_idx in enumerate(token_indices):
            # Find which top-k position has this expert
            k_positions = np.where(expert_mask[token_idx])[0]
            for k in k_positions:
                weight = topk_weights[token_idx, k]
                output[token_idx] += weight * expert_output[i]

    return output.astype(original_dtype)
