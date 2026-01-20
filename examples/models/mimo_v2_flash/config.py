"""MiMo-V2-Flash configuration for nkipy/Trainium2"""

import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
from neuronxcc.nki.language import bfloat16

# Build directory for compiled kernels
_BUILD_DIR = "/tmp/build"


def get_build_dir():
    """Get the build directory for compiled kernels"""
    return _BUILD_DIR


# Default constants
DEFAULT_MODEL_NAME = "XiaomiMiMo/MiMo-V2-Flash"
DEFAULT_WEIGHTS_DIR = "tmp_mimo_weights"
DEFAULT_WEIGHTS_FILENAME = "mimo_weights.safetensors"


@dataclass
class MiMoConfig:
    """
    Configuration for MiMo-V2-Flash model (309B total, 15B active).

    Key architecture features:
    - Heterogeneous attention: Q/K use head_dim=192, V uses v_head_dim=128
    - Partial RoPE: Only 33.4% of head dimensions use rotary embeddings
    - Mixed attention: Full attention + Sliding Window Attention (SWA)
    - Large-scale MoE: 256 experts with 8 selected per token
    - Sigmoid routing with noaux_tc load balancing
    """

    # Model architecture - core dimensions
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    vocab_size: int = 151669

    # Attention architecture - heterogeneous head dimensions
    num_attention_heads: int = 64  # Number of Q heads
    num_kv_heads_full: int = 4  # KV heads for full attention layers
    num_kv_heads_swa: int = 8  # KV heads for SWA layers
    head_dim: int = 192  # Q/K head dimension
    v_head_dim: int = 128  # V head dimension (different from Q/K!)

    # RoPE parameters - partial rotary embedding
    partial_rotary_factor: float = 0.334  # Only ~33.4% of head_dim uses RoPE
    rope_theta_full: float = 5000000.0  # RoPE theta for full attention
    rope_theta_swa: float = 10000.0  # RoPE theta for SWA

    # Sliding Window Attention
    sliding_window_size: int = 128
    sink_tokens: int = 4  # Sink tokens for SWA (always attend to first N tokens)

    # MoE (Mixture of Experts) configuration
    num_routed_experts: int = 256  # Total number of experts
    experts_per_tok: int = 8  # Number of experts selected per token
    moe_intermediate_size: int = 2048  # Expert FFN intermediate size
    router_aux_loss_coef: float = 0.0  # noaux_tc uses 0 auxiliary loss

    # Normalization
    rms_norm_eps: float = 1e-6

    # Runtime configuration
    max_batch_size: int = 1
    max_model_len: int = 128
    max_position_embeddings: int = 32768
    dtype: np.dtype = bfloat16

    # Model source
    model_name: str = DEFAULT_MODEL_NAME
    weights_path: str = None

    # Expert parallelism configuration
    num_devices: int = 32  # Number of Trainium2 devices
    experts_per_device: int = 8  # 256 / 32 = 8 experts per device

    # Full attention layer indices (layer 0 + every 6th layer approximately)
    # This pattern follows the typical MiMo architecture
    full_attention_layers: List[int] = field(default_factory=lambda: [0, 6, 12, 18, 24, 30, 36, 42])

    def __post_init__(self):
        # Validate heterogeneous GQA setup for full attention
        assert self.num_attention_heads % self.num_kv_heads_full == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible "
            f"by num_kv_heads_full ({self.num_kv_heads_full})"
        )

        # Validate heterogeneous GQA setup for SWA
        assert self.num_attention_heads % self.num_kv_heads_swa == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible "
            f"by num_kv_heads_swa ({self.num_kv_heads_swa})"
        )

        # Validate expert parallelism
        assert self.num_routed_experts % self.num_devices == 0, (
            f"num_routed_experts ({self.num_routed_experts}) must be divisible "
            f"by num_devices ({self.num_devices})"
        )
        self.experts_per_device = self.num_routed_experts // self.num_devices

        # Compute rotary dimension (floor of partial_rotary_factor * head_dim, must be even)
        self.rotary_dim = int(self.partial_rotary_factor * self.head_dim)
        self.rotary_dim = (self.rotary_dim // 2) * 2  # Ensure even
        # Should be ~64 for head_dim=192 and factor=0.334

        # Set default weights path if not provided
        if self.weights_path is None:
            self.weights_path = os.path.join(
                DEFAULT_WEIGHTS_DIR, DEFAULT_WEIGHTS_FILENAME
            )

    @property
    def q_proj_size(self) -> int:
        """Total Q projection size: num_heads * head_dim"""
        return self.num_attention_heads * self.head_dim

    @property
    def k_proj_size_full(self) -> int:
        """K projection size for full attention: num_kv_heads_full * head_dim"""
        return self.num_kv_heads_full * self.head_dim

    @property
    def k_proj_size_swa(self) -> int:
        """K projection size for SWA: num_kv_heads_swa * head_dim"""
        return self.num_kv_heads_swa * self.head_dim

    @property
    def v_proj_size_full(self) -> int:
        """V projection size for full attention: num_kv_heads_full * v_head_dim"""
        return self.num_kv_heads_full * self.v_head_dim

    @property
    def v_proj_size_swa(self) -> int:
        """V projection size for SWA: num_kv_heads_swa * v_head_dim"""
        return self.num_kv_heads_swa * self.v_head_dim

    @property
    def o_proj_size(self) -> int:
        """Output projection input size: num_heads * v_head_dim"""
        return self.num_attention_heads * self.v_head_dim

    @property
    def gqa_ratio_full(self) -> int:
        """GQA repetition ratio for full attention"""
        return self.num_attention_heads // self.num_kv_heads_full

    @property
    def gqa_ratio_swa(self) -> int:
        """GQA repetition ratio for SWA"""
        return self.num_attention_heads // self.num_kv_heads_swa

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """Check if a layer uses full attention (vs SWA)"""
        return layer_idx in self.full_attention_layers

    def get_num_kv_heads(self, layer_idx: int) -> int:
        """Get the number of KV heads for a specific layer"""
        if self.is_full_attention_layer(layer_idx):
            return self.num_kv_heads_full
        return self.num_kv_heads_swa

    def get_rope_theta(self, layer_idx: int) -> float:
        """Get the RoPE theta for a specific layer"""
        if self.is_full_attention_layer(layer_idx):
            return self.rope_theta_full
        return self.rope_theta_swa
