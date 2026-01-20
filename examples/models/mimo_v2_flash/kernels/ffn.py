"""Dense FFN kernel for MiMo-V2-Flash (Layer 0)

Layer 0 of MiMo-V2-Flash uses a standard dense FFN instead of MoE.
This is a gated FFN with SiLU activation, similar to LLaMA/Qwen.
"""

import numpy as np


def silu_kernel(x):
    """SiLU (Swish) activation function: x * sigmoid(x)

    使用数值稳定的实现，避免 exp 溢出。
    """
    x = x.astype(np.float32)
    # Clip to avoid overflow in exp (use maximum/minimum instead of clip for nkipy compatibility)
    x_clipped = np.maximum(np.minimum(x, 88.0), -88.0)
    sigmoid = 1.0 / (1.0 + np.exp(-x_clipped))
    return x * sigmoid


def dense_ffn(x, gate_weight, up_weight, down_weight):
    """
    Dense (non-MoE) feed-forward network with SiLU gating.

    Used in Layer 0 of MiMo-V2-Flash.

    Args:
        x: Input tensor [batch_size * seq_len, hidden_size]
        gate_weight: Gate projection [hidden_size, intermediate_size]
        up_weight: Up projection [hidden_size, intermediate_size]
        down_weight: Down projection [intermediate_size, hidden_size]

    Returns:
        output: [batch_size * seq_len, hidden_size]
    """
    original_dtype = x.dtype

    # Gate projection + SiLU
    gate = x @ gate_weight
    gate = silu_kernel(gate)

    # Up projection
    up = x @ up_weight

    # Gating (element-wise multiply)
    hidden = gate * up

    # Down projection
    output = hidden.astype(np.float32) @ down_weight.astype(np.float32)

    return output.astype(original_dtype)


def dense_ffn_block(
    hidden_states,
    post_attn_layernorm_weight,
    gate_weight,
    up_weight,
    down_weight,
    config,
):
    """
    Dense FFN block with layer norm and residual connection.

    Args:
        hidden_states: [batch_size * seq_len, hidden_size]
        post_attn_layernorm_weight: [hidden_size]
        gate_weight: [hidden_size, intermediate_size]
        up_weight: [hidden_size, intermediate_size]
        down_weight: [intermediate_size, hidden_size]
        config: MiMoConfig

    Returns:
        output: [batch_size * seq_len, hidden_size]
    """
    from .rmsnorm import rmsnorm

    original_dtype = hidden_states.dtype

    # Store for residual
    residual = hidden_states

    # Layer norm
    hidden_states = rmsnorm(hidden_states, post_attn_layernorm_weight, config.rms_norm_eps)

    # FFN
    hidden_states = dense_ffn(hidden_states, gate_weight, up_weight, down_weight)

    # Residual connection
    output = hidden_states + residual

    return output.astype(original_dtype)
