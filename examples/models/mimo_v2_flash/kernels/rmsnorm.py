"""RMSNorm kernel for MiMo-V2-Flash"""

import numpy as np


def rmsnorm(x, weight, eps: float):
    """
    Root Mean Square Layer Normalization.

    Args:
        x: Input tensor of any shape [..., hidden_size]
        weight: Learnable scale parameter [hidden_size]
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    # Use float32 for numerical stability
    compute_dtype = np.float32
    original_dtype = x.dtype

    x = x.astype(compute_dtype)
    weight = weight.astype(compute_dtype)

    # Compute RMS: sqrt(mean(x^2) + eps)
    z = np.square(x)
    z = np.mean(z, axis=-1, keepdims=True)
    z = x / np.sqrt(z + eps)

    # Scale with learnable weight
    res = z * weight
    res = res.astype(original_dtype)

    return res
