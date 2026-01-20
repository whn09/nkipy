"""Softmax kernel for MiMo-V2-Flash"""

import numpy as np


def softmax(x):
    """
    Numerically stable softmax function.

    Args:
        x: Input tensor of any shape, softmax applied along last axis

    Returns:
        Softmax output with same shape as input
    """
    original_dtype = x.dtype
    x = x.astype(np.float32)

    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))

    return (exp_x / np.sum(exp_x, axis=-1, keepdims=True)).astype(original_dtype)
