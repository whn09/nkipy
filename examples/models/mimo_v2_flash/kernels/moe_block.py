"""MoE Block kernel for MiMo-V2-Flash

This implements the complete MoE block which replaces the standard FFN
in MiMo-V2-Flash transformer layers.

MoE Block structure:
1. Post-attention layer norm
2. Router: sigmoid-based top-8 selection from 256 experts
3. Expert computation: each expert is a gated FFN
4. Weighted combination of expert outputs
5. Residual connection

For Expert Parallelism (EP) with 32+ devices:
- 256 experts distributed: 8 experts per device (for 32 devices)
- Uses all_to_all to send tokens to their assigned devices
- Local expert computation on each device
- Uses all_to_all to gather results back
"""

import numpy as np
from .rmsnorm import rmsnorm
from .moe_router import moe_router_sigmoid, compute_expert_assignment
from .moe_expert import moe_expert_ffn, apply_selected_experts_batched


def moe_block(
    hidden_states,
    post_attn_layernorm_weight,
    router_weight,
    expert_gate_weights,
    expert_up_weights,
    expert_down_weights,
    config,
):
    """
    Complete MoE block for a single layer.

    This is the single-device implementation. For multi-device EP,
    use moe_block_ep.

    自动检测可用专家数量，支持专家子集测试：
    - 如果只加载了 8 个专家，会用 modulo 映射 256 个路由选择到 8 个专家

    Args:
        hidden_states: [batch_size * seq_len, hidden_size]
        post_attn_layernorm_weight: [hidden_size]
        router_weight: [hidden_size, num_experts]
        expert_gate_weights: [num_available_experts, hidden_size, intermediate_size]
        expert_up_weights: [num_available_experts, hidden_size, intermediate_size]
        expert_down_weights: [num_available_experts, intermediate_size, hidden_size]
        config: MiMoConfig

    Returns:
        output: [batch_size * seq_len, hidden_size]
    """
    original_dtype = hidden_states.dtype

    # 检测实际可用的专家数量
    num_available_experts = expert_gate_weights.shape[0]
    num_router_experts = config.num_routed_experts

    # Store for residual
    residual = hidden_states

    # 1. Post-attention layer normalization
    hidden_states = rmsnorm(hidden_states, post_attn_layernorm_weight, config.rms_norm_eps)

    # 2. Router: select top-k experts
    topk_indices, topk_weights, _ = moe_router_sigmoid(
        hidden_states,
        router_weight,
        num_experts=num_router_experts,
        top_k=config.experts_per_tok,
    )

    # 3. 如果只有专家子集，用 modulo 映射
    # 注意: modulo 操作在 nkipy tracing 中不支持，只在 CPU 模式下使用
    if num_available_experts < num_router_experts:
        # 将 0-255 的专家索引映射到 0-(num_available_experts-1)
        # 使用 floor division 和 subtraction 代替 modulo: x % n = x - (x // n) * n
        topk_indices = topk_indices - (topk_indices // num_available_experts) * num_available_experts

    # 4. Apply selected experts and combine
    expert_output = apply_selected_experts_batched(
        hidden_states,
        topk_indices,
        topk_weights,
        expert_gate_weights,
        expert_up_weights,
        expert_down_weights,
        num_experts=num_available_experts,
    )

    # 5. Residual connection
    output = expert_output + residual

    return output.astype(original_dtype)


def moe_block_with_subset(
    hidden_states,
    post_attn_layernorm_weight,
    router_weight,
    expert_gate_weights,
    expert_up_weights,
    expert_down_weights,
    config,
    active_expert_indices,
):
    """
    MoE block with a subset of experts for testing/debugging.

    Useful for:
    - Single device testing with limited memory
    - Debugging expert routing behavior
    - Phase 1 development (8-expert subset)

    Args:
        hidden_states: [batch_size * seq_len, hidden_size]
        post_attn_layernorm_weight: [hidden_size]
        router_weight: [hidden_size, num_experts] - Full router weight
        expert_gate_weights: [num_active, hidden_size, intermediate_size]
        expert_up_weights: [num_active, hidden_size, intermediate_size]
        expert_down_weights: [num_active, intermediate_size, hidden_size]
        config: MiMoConfig
        active_expert_indices: List of expert indices that are available

    Returns:
        output: [batch_size * seq_len, hidden_size]
    """
    original_dtype = hidden_states.dtype
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]

    # Store for residual
    residual = hidden_states

    # 1. Post-attention layer normalization
    hidden_states = rmsnorm(hidden_states, post_attn_layernorm_weight, config.rms_norm_eps)

    # 2. Router: compute scores for all experts
    topk_indices, topk_weights, router_logits = moe_router_sigmoid(
        hidden_states,
        router_weight,
        num_experts=config.num_routed_experts,
        top_k=config.experts_per_tok,
    )

    # 3. Map global expert indices to local indices
    # Create mapping: global_idx -> local_idx (or -1 if not available)
    active_set = set(active_expert_indices)
    global_to_local = {
        global_idx: local_idx
        for local_idx, global_idx in enumerate(active_expert_indices)
    }

    # Filter to only available experts
    valid_mask = np.isin(topk_indices, active_expert_indices)

    # Initialize output
    output = np.zeros((num_tokens, hidden_size), dtype=np.float32)

    # 4. Apply available experts
    for token_idx in range(num_tokens):
        token_hidden = hidden_states[token_idx : token_idx + 1]
        weight_sum = 0.0

        for k in range(config.experts_per_tok):
            if not valid_mask[token_idx, k]:
                continue

            global_expert_idx = topk_indices[token_idx, k]
            local_expert_idx = global_to_local[global_expert_idx]
            weight = topk_weights[token_idx, k]

            # Apply expert
            expert_out = moe_expert_ffn(
                token_hidden,
                expert_gate_weights[local_expert_idx],
                expert_up_weights[local_expert_idx],
                expert_down_weights[local_expert_idx],
            )

            output[token_idx] += weight * expert_out[0]
            weight_sum += weight

        # Re-normalize weights if some experts were dropped
        if weight_sum > 0 and weight_sum < 1.0:
            output[token_idx] /= weight_sum

    # 5. Residual connection
    output = output + residual.astype(np.float32)

    return output.astype(original_dtype)


def moe_block_ep(
    hidden_states,
    post_attn_layernorm_weight,
    router_weight,
    local_expert_gate_weights,
    local_expert_up_weights,
    local_expert_down_weights,
    config,
    device_id: int,
    replica_groups,
):
    """
    MoE block with Expert Parallelism (EP) for distributed execution.

    This uses all_to_all collectives to distribute tokens to their
    assigned expert devices and gather results.

    Workflow:
    1. All devices compute routing (replicated)
    2. all_to_all: send tokens to devices owning their experts
    3. Each device processes tokens with its local experts
    4. all_to_all: gather results back to original devices
    5. Weighted combination on original device

    Args:
        hidden_states: [batch_size * seq_len, hidden_size]
        post_attn_layernorm_weight: [hidden_size]
        router_weight: [hidden_size, num_experts]
        local_expert_gate_weights: [experts_per_device, hidden_size, intermediate]
        local_expert_up_weights: [experts_per_device, hidden_size, intermediate]
        local_expert_down_weights: [experts_per_device, intermediate, hidden_size]
        config: MiMoConfig
        device_id: Current device ID (0 to num_devices-1)
        replica_groups: Replica groups for collectives

    Returns:
        output: [batch_size * seq_len, hidden_size]
    """
    from nkipy.core.ops.collectives import all_to_all

    original_dtype = hidden_states.dtype
    num_tokens = hidden_states.shape[0]

    # Store for residual
    residual = hidden_states

    # 1. Post-attention layer normalization
    hidden_states = rmsnorm(hidden_states, post_attn_layernorm_weight, config.rms_norm_eps)

    # 2. Router (replicated on all devices)
    topk_indices, topk_weights, _ = moe_router_sigmoid(
        hidden_states,
        router_weight,
        num_experts=config.num_routed_experts,
        top_k=config.experts_per_tok,
    )

    # 3. Compute device assignments
    device_ids, local_expert_ids = compute_expert_assignment(
        topk_indices,
        num_experts=config.num_routed_experts,
        num_devices=config.num_devices,
    )

    # 4. Prepare tokens for all_to_all
    # Group tokens by destination device
    # For each top-k selection, the token needs to go to the expert's device
    # This is complex because each token may go to multiple devices

    # Simplified approach: flatten to [num_tokens * top_k, hidden_size]
    # and send each (token, expert) pair to the right device
    expanded_hidden = np.repeat(
        hidden_states[:, np.newaxis, :], config.experts_per_tok, axis=1
    )
    expanded_hidden = expanded_hidden.reshape(-1, config.hidden_size)

    # 5. all_to_all to send tokens to expert devices
    # This redistributes tokens based on device_ids
    dispatched = all_to_all(
        expanded_hidden,
        split_dimension=0,
        concat_dimension=0,
        replica_groups=replica_groups,
    )

    # 6. Local expert computation
    # Process tokens that arrived at this device
    local_output = np.zeros_like(dispatched)

    for local_expert_idx in range(config.experts_per_device):
        # Find which tokens are for this local expert
        global_expert_idx = device_id * config.experts_per_device + local_expert_idx
        expert_mask = (topk_indices == global_expert_idx).any(axis=1)

        if not expert_mask.any():
            continue

        # Get tokens for this expert
        expert_tokens = hidden_states[expert_mask]

        # Apply expert FFN
        expert_output = moe_expert_ffn(
            expert_tokens,
            local_expert_gate_weights[local_expert_idx],
            local_expert_up_weights[local_expert_idx],
            local_expert_down_weights[local_expert_idx],
        )

        # Store results
        local_output[expert_mask] = expert_output

    # 7. all_to_all to gather results
    gathered = all_to_all(
        local_output,
        split_dimension=0,
        concat_dimension=0,
        replica_groups=replica_groups,
    )

    # 8. Weighted combination
    gathered = gathered.reshape(num_tokens, config.experts_per_tok, -1)
    output = np.sum(
        gathered * topk_weights[:, :, np.newaxis],
        axis=1,
    )

    # 9. Residual connection
    output = output + residual.astype(np.float32)

    return output.astype(original_dtype)
