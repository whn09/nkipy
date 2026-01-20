"""MiMo-V2-Flash model implementation for nkipy/Trainium2

This module implements the MiMo-V2-Flash model (309B total parameters, 15B active)
for execution on AWS Trainium2 accelerators using the nkipy framework.

Key features:
- 48 transformer layers
- Mixed attention: Full attention + Sliding Window Attention (SWA)
- 256 experts with top-8 selection per token
- Heterogeneous head dimensions: Q/K=192, V=128
- Partial RoPE: 33.4% of dimensions use rotary embeddings
- Sigmoid-based MoE routing (noaux_tc)

Supported configurations:
- Full model: 256 experts with Expert Parallelism (32+ Trainium2 devices)
- Subset model: Limited experts for single-device development/testing

Usage:
    from mimo_v2_flash import MiMoConfig, MiMoV2FlashModel
    from mimo_v2_flash.prepare_weights import load_weights

    config = MiMoConfig()
    weights = load_weights("path/to/weights.safetensors", config)
    model = MiMoV2FlashModel(weights, config)

    output = model.forward(input_ids)
"""

from .config import MiMoConfig, get_build_dir
from .model import MiMoV2FlashModel
from .layer import MiMoLayer, MiMoLayerSubset

__all__ = [
    "MiMoConfig",
    "MiMoV2FlashModel",
    "MiMoLayer",
    "MiMoLayerSubset",
    "get_build_dir",
]

__version__ = "0.1.0"
