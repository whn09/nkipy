"""MiMo-V2-Flash Model for nkipy/Trainium2

Main model class that:
1. Compiles shared kernels once for all layers
2. Manages layer instantiation with layer-specific weights
3. Handles forward pass through all layers

Supports two modes:
- Full model: 256 experts with Expert Parallelism (32+ devices)
- Subset model: Limited experts for single-device development
"""

from typing import Optional, List
import numpy as np

from config import MiMoConfig
from layer import MiMoLayer, MiMoLayerSubset
from kernels.attention_full import mimo_attention_full
from kernels.attention_swa import mimo_attention_swa
from kernels.moe_block import moe_block, moe_block_with_subset
from kernels.rmsnorm import rmsnorm
from kernels.rope_partial import compute_cos_sin_partial
from kernels.token_embedding import token_embedding

from nkipy.core.trace import NKIPyKernel
from nkipy.runtime import DeviceKernel, DeviceTensor

# Compiler arguments for Trainium2
additional_compiler_args = (
    " --lnc 1 --model-type transformer"
    " --tensorizer-options='--enable-ccop-compute-overlap"
    " --cc-pipeline-tiling-factor=2 --vectorize-strided-dma'"
    " --enable-mixed-precision-accumulation"
)

BACKEND = "hlo"
if BACKEND == "hlo":
    OUTPUT_PREFIX = "output"
else:
    raise ValueError(f"Unknown backend: {BACKEND}")


def get_logger():
    """Simple logger for now."""
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("mimo_v2_flash")


logger = get_logger()


class MiMoV2FlashModel:
    """
    MiMo-V2-Flash model (309B total, 15B active).

    Key architecture:
    - 48 transformer layers
    - Mixed attention: Full + Sliding Window
    - 256 experts with top-8 selection
    - Heterogeneous head dims: Q/K=192, V=128
    - Partial RoPE: 33.4% of dimensions
    """

    def __init__(
        self,
        weights: dict,
        config: MiMoConfig,
        expert_subset: Optional[List[int]] = None,
    ):
        """
        Initialize MiMo-V2-Flash model.

        Args:
            weights: Dictionary of model weights
            config: Model configuration
            expert_subset: Optional list of expert indices to load (for single-device testing)
        """
        self.config = config
        self.expert_subset = expert_subset
        self.use_subset = expert_subset is not None

        # Extract global weights
        self.tok_embedding_device = DeviceTensor.from_torch(
            weights.pop("tok_embedding"), "tok_embedding"
        )
        self.norm_weight_device = DeviceTensor.from_torch(
            weights.pop("norm_weight"), "norm_weight"
        )

        # Precompute RoPE cos/sin tables (separate for full and SWA)
        self._compute_rope_tables()

        # Compile shared kernels
        self._compile_shared_kernels()

        # Create layers
        self.layers = self._create_layers(weights)

        logger.info(
            f"Initialized MiMo-V2-Flash with {len(self.layers)} layers, "
            f"{'subset' if self.use_subset else 'full'} expert mode"
        )

    def _compute_rope_tables(self):
        """Precompute RoPE cos/sin tables for both attention types."""
        # Full attention RoPE (theta=5000000)
        cos_full, sin_full = compute_cos_sin_partial(
            max_model_len=self.config.max_model_len,
            head_dim=self.config.head_dim,
            rotary_dim=self.config.rotary_dim,
            theta=self.config.rope_theta_full,
        )
        self.cos_full = DeviceTensor.from_numpy(cos_full, "cos_full")
        self.sin_full = DeviceTensor.from_numpy(sin_full, "sin_full")

        # SWA RoPE (theta=10000)
        cos_swa, sin_swa = compute_cos_sin_partial(
            max_model_len=self.config.max_model_len,
            head_dim=self.config.head_dim,
            rotary_dim=self.config.rotary_dim,
            theta=self.config.rope_theta_swa,
        )
        self.cos_swa = DeviceTensor.from_numpy(cos_swa, "cos_swa")
        self.sin_swa = DeviceTensor.from_numpy(sin_swa, "sin_swa")

    def _compile_shared_kernels(self):
        """Compile kernels shared across all layers."""
        logger.info("Compiling shared kernels...")

        batch_seq_tokens = self.config.max_batch_size * self.config.max_model_len
        hidden_states = np.empty(
            (batch_seq_tokens, self.config.hidden_size), dtype=self.config.dtype
        )

        # 1. Token embedding kernel
        logger.info("Compiling token embedding kernel...")
        self.token_embedding_kernel = DeviceKernel.compile_and_load(
            NKIPyKernel.trace(token_embedding, backend=BACKEND),
            name="mimo_token_embedding",
            tok_embedding=np.empty(
                (self.config.vocab_size, self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            token_ids=np.zeros(
                (self.config.max_batch_size, self.config.max_model_len),
                dtype=np.uint32,
            ),
            additional_compiler_args=additional_compiler_args,
        )

        # 2. Full attention kernel
        logger.info("Compiling full attention kernel...")
        self.shared_attention_full_kernel = DeviceKernel.compile_and_load(
            kernel=NKIPyKernel.trace(mimo_attention_full, backend=BACKEND),
            hidden_states=hidden_states,
            input_layernorm_weight=np.empty(
                self.config.hidden_size, dtype=self.config.dtype
            ),
            q_weight=np.empty(
                (self.config.hidden_size, self.config.q_proj_size),
                dtype=self.config.dtype,
            ),
            k_weight=np.empty(
                (self.config.hidden_size, self.config.k_proj_size_full),
                dtype=self.config.dtype,
            ),
            v_weight=np.empty(
                (self.config.hidden_size, self.config.v_proj_size_full),
                dtype=self.config.dtype,
            ),
            o_weight=np.empty(
                (self.config.o_proj_size, self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            cos=self.cos_full,
            sin=self.sin_full,
            config=self.config,
            compute_dtype=self.config.dtype,
            name="mimo_attention_full_shared",
            additional_compiler_args=additional_compiler_args,
        )

        # 3. SWA kernel
        logger.info("Compiling SWA kernel...")
        self.shared_attention_swa_kernel = DeviceKernel.compile_and_load(
            kernel=NKIPyKernel.trace(mimo_attention_swa, backend=BACKEND),
            hidden_states=hidden_states,
            input_layernorm_weight=np.empty(
                self.config.hidden_size, dtype=self.config.dtype
            ),
            q_weight=np.empty(
                (self.config.hidden_size, self.config.q_proj_size),
                dtype=self.config.dtype,
            ),
            k_weight=np.empty(
                (self.config.hidden_size, self.config.k_proj_size_swa),
                dtype=self.config.dtype,
            ),
            v_weight=np.empty(
                (self.config.hidden_size, self.config.v_proj_size_swa),
                dtype=self.config.dtype,
            ),
            o_weight=np.empty(
                (self.config.o_proj_size, self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            cos=self.cos_swa,
            sin=self.sin_swa,
            config=self.config,
            compute_dtype=self.config.dtype,
            name="mimo_attention_swa_shared",
            additional_compiler_args=additional_compiler_args,
        )

        # 4. MoE kernel
        if self.use_subset:
            num_experts = len(self.expert_subset)
            logger.info(f"Compiling MoE subset kernel ({num_experts} experts)...")
            self.shared_moe_kernel = DeviceKernel.compile_and_load(
                kernel=NKIPyKernel.trace(moe_block_with_subset, backend=BACKEND),
                hidden_states=hidden_states,
                post_attn_layernorm_weight=np.empty(
                    self.config.hidden_size, dtype=self.config.dtype
                ),
                router_weight=np.empty(
                    (self.config.hidden_size, self.config.num_routed_experts),
                    dtype=self.config.dtype,
                ),
                expert_gate_weights=np.empty(
                    (num_experts, self.config.hidden_size, self.config.moe_intermediate_size),
                    dtype=self.config.dtype,
                ),
                expert_up_weights=np.empty(
                    (num_experts, self.config.hidden_size, self.config.moe_intermediate_size),
                    dtype=self.config.dtype,
                ),
                expert_down_weights=np.empty(
                    (num_experts, self.config.moe_intermediate_size, self.config.hidden_size),
                    dtype=self.config.dtype,
                ),
                config=self.config,
                active_expert_indices=np.array(self.expert_subset, dtype=np.int32),
                name="mimo_moe_subset_shared",
                additional_compiler_args=additional_compiler_args,
            )
        else:
            logger.info("Compiling full MoE kernel (256 experts)...")
            self.shared_moe_kernel = DeviceKernel.compile_and_load(
                kernel=NKIPyKernel.trace(moe_block, backend=BACKEND),
                hidden_states=hidden_states,
                post_attn_layernorm_weight=np.empty(
                    self.config.hidden_size, dtype=self.config.dtype
                ),
                router_weight=np.empty(
                    (self.config.hidden_size, self.config.num_routed_experts),
                    dtype=self.config.dtype,
                ),
                expert_gate_weights=np.empty(
                    (
                        self.config.num_routed_experts,
                        self.config.hidden_size,
                        self.config.moe_intermediate_size,
                    ),
                    dtype=self.config.dtype,
                ),
                expert_up_weights=np.empty(
                    (
                        self.config.num_routed_experts,
                        self.config.hidden_size,
                        self.config.moe_intermediate_size,
                    ),
                    dtype=self.config.dtype,
                ),
                expert_down_weights=np.empty(
                    (
                        self.config.num_routed_experts,
                        self.config.moe_intermediate_size,
                        self.config.hidden_size,
                    ),
                    dtype=self.config.dtype,
                ),
                config=self.config,
                name="mimo_moe_full_shared",
                additional_compiler_args=additional_compiler_args,
            )

        # 5. Final layer norm
        logger.info("Compiling final norm kernel...")
        self.final_norm_kernel = DeviceKernel.compile_and_load(
            NKIPyKernel.trace(rmsnorm, backend=BACKEND),
            x=hidden_states,
            weight=self.norm_weight_device,
            eps=self.config.rms_norm_eps,
            name="mimo_final_norm",
            additional_compiler_args=additional_compiler_args,
        )

        logger.info("All shared kernels compiled successfully!")

    def _create_layers(self, weights: dict) -> list:
        """Create all transformer layers with shared kernels."""
        layers = []

        for layer_id in range(self.config.num_hidden_layers):
            layer_prefix = f"layers.{layer_id}"
            is_full_attn = self.config.is_full_attention_layer(layer_id)

            # Determine K/V projection sizes based on attention type
            if is_full_attn:
                num_kv_heads = self.config.num_kv_heads_full
                k_proj_size = self.config.k_proj_size_full
                v_proj_size = self.config.v_proj_size_full
                cos = self.cos_full
                sin = self.sin_full
            else:
                num_kv_heads = self.config.num_kv_heads_swa
                k_proj_size = self.config.k_proj_size_swa
                v_proj_size = self.config.v_proj_size_swa
                cos = self.cos_swa
                sin = self.sin_swa

            # Extract attention weights
            attention_weights = {
                "q_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.q_weight"), f"q_weight_L{layer_id}"
                ),
                "k_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.k_weight"), f"k_weight_L{layer_id}"
                ),
                "v_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.v_weight"), f"v_weight_L{layer_id}"
                ),
                "o_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.o_weight"), f"o_weight_L{layer_id}"
                ),
                "input_layernorm_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.input_layernorm_weight"),
                    f"input_layernorm_weight_L{layer_id}",
                ),
            }

            # Extract MoE weights
            moe_weights = {
                "post_attention_layernorm_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.post_attention_layernorm_weight"),
                    f"post_attn_ln_weight_L{layer_id}",
                ),
                "router_weight": DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.router_weight"),
                    f"router_weight_L{layer_id}",
                ),
            }

            # Extract expert weights
            if self.use_subset:
                # Load only subset of experts
                expert_gate = []
                expert_up = []
                expert_down = []
                for expert_idx in self.expert_subset:
                    expert_gate.append(
                        weights.pop(f"{layer_prefix}.expert.{expert_idx}.gate_weight")
                    )
                    expert_up.append(
                        weights.pop(f"{layer_prefix}.expert.{expert_idx}.up_weight")
                    )
                    expert_down.append(
                        weights.pop(f"{layer_prefix}.expert.{expert_idx}.down_weight")
                    )

                moe_weights["expert_gate_weights"] = DeviceTensor.from_numpy(
                    np.stack([w.numpy() for w in expert_gate]),
                    f"expert_gate_weights_L{layer_id}",
                )
                moe_weights["expert_up_weights"] = DeviceTensor.from_numpy(
                    np.stack([w.numpy() for w in expert_up]),
                    f"expert_up_weights_L{layer_id}",
                )
                moe_weights["expert_down_weights"] = DeviceTensor.from_numpy(
                    np.stack([w.numpy() for w in expert_down]),
                    f"expert_down_weights_L{layer_id}",
                )
            else:
                # Load all experts
                moe_weights["expert_gate_weights"] = DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.expert_gate_weights"),
                    f"expert_gate_weights_L{layer_id}",
                )
                moe_weights["expert_up_weights"] = DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.expert_up_weights"),
                    f"expert_up_weights_L{layer_id}",
                )
                moe_weights["expert_down_weights"] = DeviceTensor.from_torch(
                    weights.pop(f"{layer_prefix}.expert_down_weights"),
                    f"expert_down_weights_L{layer_id}",
                )

            # Create layer
            if self.use_subset:
                layer = MiMoLayerSubset(
                    layer_id=layer_id,
                    config=self.config,
                    cos=cos,
                    sin=sin,
                    active_expert_indices=self.expert_subset,
                    shared_attention_full_kernel=self.shared_attention_full_kernel,
                    shared_attention_swa_kernel=self.shared_attention_swa_kernel,
                    shared_moe_subset_kernel=self.shared_moe_kernel,
                    **attention_weights,
                    **moe_weights,
                )
            else:
                layer = MiMoLayer(
                    layer_id=layer_id,
                    config=self.config,
                    cos=cos,
                    sin=sin,
                    shared_attention_full_kernel=self.shared_attention_full_kernel,
                    shared_attention_swa_kernel=self.shared_attention_swa_kernel,
                    shared_moe_kernel=self.shared_moe_kernel,
                    **attention_weights,
                    **moe_weights,
                )

            layers.append(layer)

        return layers

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Truncate if needed
        if seq_len > self.config.max_model_len:
            input_ids = input_ids[:, : self.config.max_model_len]
            seq_len = self.config.max_model_len

        # Pad to max_model_len for fixed kernel shapes
        if seq_len < self.config.max_model_len:
            padded_input_ids = np.zeros(
                (batch_size, self.config.max_model_len), dtype=np.uint32
            )
            padded_input_ids[:, :seq_len] = input_ids
            input_ids = padded_input_ids

        # 1. Token embedding
        input_ids_device = DeviceTensor.from_numpy(input_ids, "input_ids")
        hidden_states_3d = DeviceTensor.from_numpy(
            np.zeros(
                (batch_size, self.config.max_model_len, self.config.hidden_size),
                dtype=self.config.dtype,
            ),
            "hidden_states_3d",
        )

        self.token_embedding_kernel(
            inputs={
                "tok_embedding": self.tok_embedding_device,
                "token_ids": input_ids_device,
            },
            outputs={f"{OUTPUT_PREFIX}0": hidden_states_3d},
        )

        # Reshape to 2D: [batch * seq, hidden]
        hidden_states_2d = DeviceTensor.from_numpy(
            hidden_states_3d.numpy().reshape(
                batch_size * self.config.max_model_len, self.config.hidden_size
            ),
            "hidden_states_2d",
        )

        # 2. Pass through all layers
        for layer in self.layers:
            hidden_states_2d = layer.forward(hidden_states_2d)

        # 3. Final layer norm
        normed_hidden = DeviceTensor.from_numpy(
            np.empty_like(hidden_states_2d.numpy()), "final_normed"
        )

        self.final_norm_kernel(
            inputs={"x": hidden_states_2d, "weight": self.norm_weight_device},
            outputs={f"{OUTPUT_PREFIX}0": normed_hidden},
        )

        # 4. Reshape back to 3D
        final_hidden = normed_hidden.numpy().reshape(
            batch_size, self.config.max_model_len, self.config.hidden_size
        )

        # Return hidden states (for generation, you'd apply lm_head here)
        return final_hidden

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> np.ndarray:
        """
        Generate tokens autoregressively.

        Note: This is a simple implementation. Production would use KV cache.

        Args:
            input_ids: Initial token IDs [batch_size, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            generated_ids: [batch_size, seq_len + max_new_tokens]
        """
        batch_size = input_ids.shape[0]
        generated = input_ids.copy()

        for _ in range(max_new_tokens):
            # Get model output
            hidden = self.forward(generated)

            # Get logits for last position (need lm_head projection)
            # For now, just return hidden states
            # In production: logits = hidden[:, -1, :] @ lm_head_weight

            # Placeholder: sample from vocabulary
            # next_token = sample_from_logits(logits, temperature, top_p)

            # For now, just append a placeholder
            # next_token = np.zeros((batch_size, 1), dtype=np.uint32)
            # generated = np.concatenate([generated, next_token], axis=1)

            break  # Remove when implementing proper generation

        return generated
