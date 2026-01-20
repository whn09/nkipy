"""Partial Rotary Position Embedding (RoPE) kernel for MiMo-V2-Flash

MiMo-V2-Flash uses partial RoPE where only ~33.4% of the head dimensions
have rotary embeddings applied. This is different from standard transformers
where RoPE is applied to all dimensions.

Key parameters:
- head_dim: 192 (Q/K head dimension)
- partial_rotary_factor: 0.334
- rotary_dim: 64 (floor(192 * 0.334) rounded to even)

The first 64 dimensions get RoPE, the remaining 128 dimensions are passed through unchanged.
"""

import numpy as np


def compute_cos_sin_partial(
    max_model_len: int,
    head_dim: int,
    rotary_dim: int,
    theta: float = 1000000.0,
):
    """
    Compute cos/sin tables for partial RoPE.

    Args:
        max_model_len: Maximum sequence length
        head_dim: Full head dimension (192 for MiMo)
        rotary_dim: Number of dimensions to apply RoPE to (64 for MiMo)
        theta: RoPE base frequency

    Returns:
        cos, sin: Precomputed tables [seq_len, rotary_dim // 2]
    """
    # Only compute for the rotary dimensions
    half_rotary_dim = rotary_dim // 2

    # Compute inverse frequencies
    inv_freq = 1.0 / (
        theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
    )

    # Compute position indices
    t = np.arange(max_model_len, dtype=np.float32)

    # Compute frequencies for each position
    freqs = np.outer(t, inv_freq)  # [seq_len, rotary_dim // 2]

    # Compute cos and sin
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)

    return cos, sin


def rope_partial(xq, xk, freqs_cos, freqs_sin, rotary_dim: int):
    """
    Apply partial rotary position embedding to query and key tensors.

    Only the first `rotary_dim` dimensions of Q and K are rotated.
    The remaining dimensions pass through unchanged.

    Args:
        xq: Query tensor [batch, seq_len, n_heads, head_dim]
        xk: Key tensor [batch, seq_len, n_kv_heads, head_dim]
        freqs_cos: Cosine frequencies [seq_len, rotary_dim // 2]
        freqs_sin: Sine frequencies [seq_len, rotary_dim // 2]
        rotary_dim: Number of dimensions to rotate (e.g., 64)

    Returns:
        xq_out, xk_out: Rotated tensors with same shape as inputs
    """
    # Split into rotary and pass-through parts
    # xq: [batch, seq_len, n_heads, head_dim] -> rotary [batch, seq_len, n_heads, rotary_dim]
    xq_rot = xq[:, :, :, :rotary_dim]
    xq_pass = xq[:, :, :, rotary_dim:]

    xk_rot = xk[:, :, :, :rotary_dim]
    xk_pass = xk[:, :, :, rotary_dim:]

    # Reshape freqs for broadcasting: [1, seq_len, 1, rotary_dim // 2]
    freqs_cos = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin = np.expand_dims(freqs_sin, axis=(0, 2))

    # Split rotary part into two halves for rotation
    half_rotary = rotary_dim // 2
    xq_rot_0 = xq_rot[:, :, :, :half_rotary]
    xq_rot_1 = xq_rot[:, :, :, half_rotary:]

    xk_rot_0 = xk_rot[:, :, :, :half_rotary]
    xk_rot_1 = xk_rot[:, :, :, half_rotary:]

    # Apply rotation: (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
    xq_out_0 = xq_rot_0 * freqs_cos - xq_rot_1 * freqs_sin
    xq_out_1 = xq_rot_0 * freqs_sin + xq_rot_1 * freqs_cos

    xk_out_0 = xk_rot_0 * freqs_cos - xk_rot_1 * freqs_sin
    xk_out_1 = xk_rot_0 * freqs_sin + xk_rot_1 * freqs_cos

    # Concatenate rotated parts back together
    xq_rot_out = np.concatenate([xq_out_0, xq_out_1], axis=-1)
    xk_rot_out = np.concatenate([xk_out_0, xk_out_1], axis=-1)

    # Concatenate with pass-through parts
    xq_out = np.concatenate([xq_rot_out, xq_pass], axis=-1)
    xk_out = np.concatenate([xk_rot_out, xk_pass], axis=-1)

    return xq_out, xk_out
