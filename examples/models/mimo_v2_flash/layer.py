"""MiMo-V2-Flash Transformer Layer with Mixed Attention and MoE/Dense FFN

Each layer consists of:
1. Attention block (Full or SWA based on layer index)
2. FFN block (Dense for Layer 0, MoE for Layer 1-47)

Key features:
- Mixed attention: Full attention for certain layers, SWA for others
- Heterogeneous head dimensions: Q/K=192, V=128
- Partial RoPE: Only 33.4% of dimensions use rotary embeddings
- Layer 0: Dense FFN
- Layer 1-47: MoE with 256 experts (or subset)
"""

import numpy as np
from config import MiMoConfig
from nkipy.runtime import DeviceKernel, DeviceTensor


class MiMoLayer:
    """
    Single transformer layer for MiMo-V2-Flash.

    Supports both:
    - Dense FFN (Layer 0)
    - MoE FFN (Layer 1-47)
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
        post_attention_layernorm_weight: DeviceTensor,
        # RoPE tables (different for full vs SWA)
        cos: DeviceTensor,
        sin: DeviceTensor,
        # Shared kernels
        shared_attention_full_kernel: DeviceKernel,
        shared_attention_swa_kernel: DeviceKernel,
        # FFN weights - either dense or MoE
        # Dense FFN weights (Layer 0)
        gate_weight: DeviceTensor = None,
        up_weight: DeviceTensor = None,
        down_weight: DeviceTensor = None,
        shared_dense_ffn_kernel: DeviceKernel = None,
        # MoE weights (Layer 1-47)
        router_weight: DeviceTensor = None,
        expert_gate_weights: DeviceTensor = None,
        expert_up_weights: DeviceTensor = None,
        expert_down_weights: DeviceTensor = None,
        shared_moe_kernel: DeviceKernel = None,
    ):
        self.layer_id = layer_id
        self.config = config
        self.is_full_attention = config.is_full_attention_layer(layer_id)
        self.is_dense = config.is_dense_layer(layer_id)

        # Store attention weights
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.o_weight = o_weight
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight

        # Store RoPE tables
        self.cos = cos
        self.sin = sin

        # Select appropriate attention kernel
        if self.is_full_attention:
            self.attention_kernel = shared_attention_full_kernel
        else:
            self.attention_kernel = shared_attention_swa_kernel

        # Store FFN weights and kernel based on layer type
        if self.is_dense:
            self.gate_weight = gate_weight
            self.up_weight = up_weight
            self.down_weight = down_weight
            self.ffn_kernel = shared_dense_ffn_kernel
        else:
            self.router_weight = router_weight
            self.expert_gate_weights = expert_gate_weights
            self.expert_up_weights = expert_up_weights
            self.expert_down_weights = expert_down_weights
            self.ffn_kernel = shared_moe_kernel

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

        # 2. FFN block (Dense or MoE)
        ffn_output = DeviceTensor.from_numpy(
            np.empty_like(hidden_states.numpy()), f"ffn_output_L{self.layer_id}"
        )

        if self.is_dense:
            # Dense FFN (Layer 0)
            self.ffn_kernel(
                inputs={
                    "hidden_states": hidden_states,
                    "post_attn_layernorm_weight": self.post_attention_layernorm_weight,
                    "gate_weight": self.gate_weight,
                    "up_weight": self.up_weight,
                    "down_weight": self.down_weight,
                },
                outputs={f"{OUTPUT_PREFIX}0": ffn_output},
            )
        else:
            # MoE FFN (Layer 1-47)
            self.ffn_kernel(
                inputs={
                    "hidden_states": hidden_states,
                    "post_attn_layernorm_weight": self.post_attention_layernorm_weight,
                    "router_weight": self.router_weight,
                    "expert_gate_weights": self.expert_gate_weights,
                    "expert_up_weights": self.expert_up_weights,
                    "expert_down_weights": self.expert_down_weights,
                },
                outputs={f"{OUTPUT_PREFIX}0": ffn_output},
            )

        return ffn_output


class MiMoLayerCPU:
    """
    CPU-only MiMo layer for testing without Trainium.

    Runs kernels directly in numpy without DeviceKernel compilation.
    """

    def __init__(
        self,
        layer_id: int,
        config: MiMoConfig,
        weights: dict,
        cos: np.ndarray,
        sin: np.ndarray,
    ):
        self.layer_id = layer_id
        self.config = config
        self.weights = weights
        self.cos = cos
        self.sin = sin
        self.is_full_attention = config.is_full_attention_layer(layer_id)
        self.is_dense = config.is_dense_layer(layer_id)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """Forward pass using numpy kernels."""
        from kernels.attention_full import mimo_attention_full
        from kernels.attention_swa import mimo_attention_swa
        from kernels.ffn import dense_ffn_block
        from kernels.moe_block import moe_block

        # 1. Attention
        if self.is_full_attention:
            hidden_states = mimo_attention_full(
                hidden_states,
                self.weights["input_layernorm_weight"],
                self.weights["q_weight"],
                self.weights["k_weight"],
                self.weights["v_weight"],
                self.weights["o_weight"],
                self.cos,
                self.sin,
                self.config,
                np.float32,
            )
        else:
            hidden_states = mimo_attention_swa(
                hidden_states,
                self.weights["input_layernorm_weight"],
                self.weights["q_weight"],
                self.weights["k_weight"],
                self.weights["v_weight"],
                self.weights["o_weight"],
                self.cos,
                self.sin,
                self.config,
                np.float32,
            )

        # 2. FFN
        if self.is_dense:
            hidden_states = dense_ffn_block(
                hidden_states,
                self.weights["post_attention_layernorm_weight"],
                self.weights["gate_weight"],
                self.weights["up_weight"],
                self.weights["down_weight"],
                self.config,
            )
        else:
            hidden_states = moe_block(
                hidden_states,
                self.weights["post_attention_layernorm_weight"],
                self.weights["router_weight"],
                self.weights["expert_gate_weights"],
                self.weights["expert_up_weights"],
                self.weights["expert_down_weights"],
                self.config,
            )

        return hidden_states
