"""MiMo-V2-Flash Transformer Layer with Mixed Attention and MoE

Each layer consists of:
1. Attention block (Full or SWA based on layer index)
2. MoE block (256 experts, top-8 selection)

Key features:
- Mixed attention: Full attention for certain layers, SWA for others
- Heterogeneous head dimensions: Q/K=192, V=128
- Partial RoPE: Only 33.4% of dimensions use rotary embeddings
- Large-scale MoE: 256 experts with sigmoid routing
"""

import numpy as np
from config import MiMoConfig
from nkipy.runtime import DeviceKernel, DeviceTensor


class MiMoLayer:
    """
    Single transformer layer for MiMo-V2-Flash.

    Uses shared kernels (compiled once at model level) with layer-specific weights.
    Dynamically selects between full attention and SWA based on layer index.
    """

    def __init__(
        self,
        layer_id: int,
        config: MiMoConfig,
        # Attention weights (separate Q, K, V due to heterogeneous dimensions)
        q_weight: DeviceTensor,
        k_weight: DeviceTensor,
        v_weight: DeviceTensor,
        o_weight: DeviceTensor,
        input_layernorm_weight: DeviceTensor,
        # MoE weights
        post_attention_layernorm_weight: DeviceTensor,
        router_weight: DeviceTensor,
        expert_gate_weights: DeviceTensor,  # [num_experts, hidden, intermediate]
        expert_up_weights: DeviceTensor,  # [num_experts, hidden, intermediate]
        expert_down_weights: DeviceTensor,  # [num_experts, intermediate, hidden]
        # RoPE tables (different for full vs SWA)
        cos: DeviceTensor,
        sin: DeviceTensor,
        # Shared kernels
        shared_attention_full_kernel: DeviceKernel,
        shared_attention_swa_kernel: DeviceKernel,
        shared_moe_kernel: DeviceKernel,
    ):
        self.layer_id = layer_id
        self.config = config
        self.is_full_attention = config.is_full_attention_layer(layer_id)

        # Store attention weights
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.o_weight = o_weight
        self.input_layernorm_weight = input_layernorm_weight

        # Store MoE weights
        self.post_attention_layernorm_weight = post_attention_layernorm_weight
        self.router_weight = router_weight
        self.expert_gate_weights = expert_gate_weights
        self.expert_up_weights = expert_up_weights
        self.expert_down_weights = expert_down_weights

        # Store RoPE tables
        self.cos = cos
        self.sin = sin

        # Select appropriate attention kernel
        if self.is_full_attention:
            self.attention_kernel = shared_attention_full_kernel
        else:
            self.attention_kernel = shared_attention_swa_kernel

        self.moe_kernel = shared_moe_kernel

    def forward(self, hidden_states: DeviceTensor) -> DeviceTensor:
        """
        Forward pass through the layer.

        Args:
            hidden_states: [batch_size * seq_len, hidden_size]

        Returns:
            hidden_states: [batch_size * seq_len, hidden_size]
        """
        from model import OUTPUT_PREFIX

        # 1. Attention block (includes input norm and residual)
        attn_output = DeviceTensor.from_numpy(
            np.empty_like(hidden_states.numpy()), f"attn_output_L{self.layer_id}"
        )

        self.attention_kernel(
            inputs={
                "hidden_states": hidden_states,
                "input_layernorm_weight": self.input_layernorm_weight,
                "q_weight": self.q_weight,
                "k_weight": self.k_weight,
                "v_weight": self.v_weight,
                "o_weight": self.o_weight,
                "cos": self.cos,
                "sin": self.sin,
            },
            outputs={f"{OUTPUT_PREFIX}0": attn_output},
        )

        hidden_states = attn_output

        # 2. MoE block (includes post-attention norm and residual)
        moe_output = DeviceTensor.from_numpy(
            np.empty_like(hidden_states.numpy()), f"moe_output_L{self.layer_id}"
        )

        self.moe_kernel(
            inputs={
                "hidden_states": hidden_states,
                "post_attn_layernorm_weight": self.post_attention_layernorm_weight,
                "router_weight": self.router_weight,
                "expert_gate_weights": self.expert_gate_weights,
                "expert_up_weights": self.expert_up_weights,
                "expert_down_weights": self.expert_down_weights,
            },
            outputs={f"{OUTPUT_PREFIX}0": moe_output},
        )

        return moe_output


class MiMoLayerSubset:
    """
    MiMo layer variant for development/testing with expert subset.

    Uses only a subset of experts (e.g., 8 out of 256) to fit on a single device.
    Useful for Phase 1 development before implementing Expert Parallelism.
    """

    def __init__(
        self,
        layer_id: int,
        config: MiMoConfig,
        # Attention weights
        q_weight: DeviceTensor,
        k_weight: DeviceTensor,
        v_weight: DeviceTensor,
        o_weight: DeviceTensor,
        input_layernorm_weight: DeviceTensor,
        # MoE weights (subset)
        post_attention_layernorm_weight: DeviceTensor,
        router_weight: DeviceTensor,  # Full router weight for all 256 experts
        expert_gate_weights: DeviceTensor,  # [num_active_experts, hidden, intermediate]
        expert_up_weights: DeviceTensor,
        expert_down_weights: DeviceTensor,
        active_expert_indices: list,  # Which expert indices are loaded
        # RoPE tables
        cos: DeviceTensor,
        sin: DeviceTensor,
        # Shared kernels
        shared_attention_full_kernel: DeviceKernel,
        shared_attention_swa_kernel: DeviceKernel,
        shared_moe_subset_kernel: DeviceKernel,
    ):
        self.layer_id = layer_id
        self.config = config
        self.is_full_attention = config.is_full_attention_layer(layer_id)
        self.active_expert_indices = active_expert_indices

        # Store weights
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.o_weight = o_weight
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight
        self.router_weight = router_weight
        self.expert_gate_weights = expert_gate_weights
        self.expert_up_weights = expert_up_weights
        self.expert_down_weights = expert_down_weights
        self.cos = cos
        self.sin = sin

        # Select kernels
        if self.is_full_attention:
            self.attention_kernel = shared_attention_full_kernel
        else:
            self.attention_kernel = shared_attention_swa_kernel

        self.moe_kernel = shared_moe_subset_kernel

    def forward(self, hidden_states: DeviceTensor) -> DeviceTensor:
        """Forward pass with expert subset."""
        from model import OUTPUT_PREFIX

        # 1. Attention block
        attn_output = DeviceTensor.from_numpy(
            np.empty_like(hidden_states.numpy()), f"attn_output_L{self.layer_id}"
        )

        self.attention_kernel(
            inputs={
                "hidden_states": hidden_states,
                "input_layernorm_weight": self.input_layernorm_weight,
                "q_weight": self.q_weight,
                "k_weight": self.k_weight,
                "v_weight": self.v_weight,
                "o_weight": self.o_weight,
                "cos": self.cos,
                "sin": self.sin,
            },
            outputs={f"{OUTPUT_PREFIX}0": attn_output},
        )

        hidden_states = attn_output

        # 2. MoE block with subset
        moe_output = DeviceTensor.from_numpy(
            np.empty_like(hidden_states.numpy()), f"moe_output_L{self.layer_id}"
        )

        self.moe_kernel(
            inputs={
                "hidden_states": hidden_states,
                "post_attn_layernorm_weight": self.post_attention_layernorm_weight,
                "router_weight": self.router_weight,
                "expert_gate_weights": self.expert_gate_weights,
                "expert_up_weights": self.expert_up_weights,
                "expert_down_weights": self.expert_down_weights,
                "active_expert_indices": np.array(self.active_expert_indices, dtype=np.int32),
            },
            outputs={f"{OUTPUT_PREFIX}0": moe_output},
        )

        return moe_output
