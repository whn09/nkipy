"""Full attention kernel for MiMo-V2-Flash

This implements global (full) attention used in certain layers of MiMo-V2-Flash.
Key features:
- Heterogeneous head dimensions: Q/K use head_dim=192, V uses v_head_dim=128
- Partial RoPE: Only 64 dimensions (33.4%) get rotary embeddings
- Grouped Query Attention: 64 Q heads, 4 KV heads (16x repetition)
- Separate Q, K, V projections (cannot fuse due to different V dimension)
"""

import numpy as np
from .rmsnorm import rmsnorm
from .rope_partial import rope_partial
from .softmax import softmax


def mimo_attention_full(
    hidden_states,
    input_layernorm_weight,
    q_weight,
    k_weight,
    v_weight,
    o_weight,
    cos,
    sin,
    config,
    compute_dtype,
):
    """
    Full attention (global attention) for MiMo-V2-Flash.

    Shapes:
    - hidden_states: [batch_size * seq_len, hidden_size=4096]
    - Q projection: [4096, 64 * 192] = [4096, 12288]
    - K projection: [4096, 4 * 192] = [4096, 768]
    - V projection: [4096, 4 * 128] = [4096, 512]
    - O projection: [64 * 128, 4096] = [8192, 4096]

    Args:
        hidden_states: [batch_size * seq_len, hidden_size]
        input_layernorm_weight: [hidden_size]
        q_weight: [hidden_size, num_heads * head_dim]
        k_weight: [hidden_size, num_kv_heads * head_dim]
        v_weight: [hidden_size, num_kv_heads * v_head_dim]
        o_weight: [num_heads * v_head_dim, hidden_size]
        cos: [max_model_len, rotary_dim // 2]
        sin: [max_model_len, rotary_dim // 2]
        config: MiMoConfig
        compute_dtype: Computation dtype

    Returns:
        output: [batch_size * seq_len, hidden_size]
    """
    original_dtype = hidden_states.dtype
    hidden_states = hidden_states.astype(compute_dtype)
    q_weight = q_weight.astype(compute_dtype)
    k_weight = k_weight.astype(compute_dtype)
    v_weight = v_weight.astype(compute_dtype)
    o_weight = o_weight.astype(compute_dtype)

    # Store for residual connection
    residual = hidden_states

    # 1. Input layer normalization
    hidden_states = rmsnorm(hidden_states, input_layernorm_weight, config.rms_norm_eps)

    # Infer dimensions
    total_tokens = hidden_states.shape[0]
    batch_size = 1  # Assume batch_size=1 for now
    seq_len = total_tokens // batch_size

    # Reshape to [batch, seq_len, hidden_size]
    hidden_states = hidden_states.reshape(batch_size, seq_len, config.hidden_size)

    # 2. Separate Q, K, V projections (no fused QKV due to different V dimension)
    q = hidden_states @ q_weight  # [B, S, 64 * 192]
    k = hidden_states @ k_weight  # [B, S, 4 * 192]
    v = hidden_states @ v_weight  # [B, S, 4 * 128]

    # 3. Reshape for attention
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_kv_heads_full
    head_dim = config.head_dim
    v_head_dim = config.v_head_dim

    q = q.reshape(batch_size, seq_len, n_heads, head_dim)
    k = k.reshape(batch_size, seq_len, n_kv_heads, head_dim)
    v = v.reshape(batch_size, seq_len, n_kv_heads, v_head_dim)

    # 4. Apply partial RoPE (only first rotary_dim dimensions)
    q, k = rope_partial(q, k, cos, sin, config.rotary_dim)

    # 5. Repeat K, V for GQA (64 / 4 = 16x repetition)
    n_rep = config.gqa_ratio_full
    if n_rep > 1:
        k = np.repeat(k, n_rep, axis=2)  # [B, S, 64, 192]
        v = np.repeat(v, n_rep, axis=2)  # [B, S, 64, 128]

    # 6. Transpose for attention: [batch, n_heads, seq_len, head_dim]
    q = q.transpose(0, 2, 1, 3)  # [B, 64, S, 192]
    k = k.transpose(0, 2, 1, 3)  # [B, 64, S, 192]
    v = v.transpose(0, 2, 1, 3)  # [B, 64, S, 128]

    # 7. Compute attention scores: Q @ K^T / sqrt(head_dim)
    # Score shape: [B, 64, S, S]
    scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    scores = scores.astype(compute_dtype)

    # 8. Apply causal mask (full attention - attend to all previous positions)
    causal_mask = np.triu(np.ones((seq_len, seq_len)) * -10000.0, k=1).astype(
        compute_dtype
    )
    scores = scores + causal_mask[None, None, :, :]

    # 9. Softmax
    attn_weights = softmax(scores)

    # 10. Apply attention to values
    # attn_output: [B, 64, S, 128]
    attn_output = (attn_weights @ v).astype(compute_dtype)

    # 11. Transpose back and reshape
    attn_output = attn_output.transpose(0, 2, 1, 3)  # [B, S, 64, 128]
    attn_output = attn_output.reshape(batch_size, seq_len, n_heads * v_head_dim)

    # 12. Output projection
    output = attn_output @ o_weight  # [B, S, 4096]

    # 13. Reshape to original format
    output = output.reshape(batch_size * seq_len, config.hidden_size)

    # 14. Add residual connection
    output = output + residual

    return output.astype(original_dtype)
