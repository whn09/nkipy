"""Weight preparation and conversion for MiMo-V2-Flash

This module handles:
1. Loading weights from HuggingFace format
2. Converting to nkipy-compatible format
3. Saving/loading in safetensors format
4. Expert weight handling for EP (Expert Parallelism)

HuggingFace MiMo-V2-Flash weight naming convention:
- model.embed_tokens.weight
- model.layers.{i}.self_attn.q_proj.weight
- model.layers.{i}.self_attn.k_proj.weight
- model.layers.{i}.self_attn.v_proj.weight
- model.layers.{i}.self_attn.o_proj.weight
- model.layers.{i}.input_layernorm.weight
- model.layers.{i}.post_attention_layernorm.weight
- model.layers.{i}.mlp.gate.weight (router)
- model.layers.{i}.mlp.experts.{j}.gate_proj.weight
- model.layers.{i}.mlp.experts.{j}.up_proj.weight
- model.layers.{i}.mlp.experts.{j}.down_proj.weight
- model.norm.weight
- lm_head.weight
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    from safetensors.torch import load_file, save_file
    from safetensors import safe_open
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

from config import MiMoConfig, DEFAULT_WEIGHTS_DIR, DEFAULT_WEIGHTS_FILENAME


def download_and_convert_weights(
    model_name: str = "XiaomiMiMo/MiMo-V2-Flash",
    output_dir: str = DEFAULT_WEIGHTS_DIR,
    output_filename: str = DEFAULT_WEIGHTS_FILENAME,
    expert_subset: Optional[List[int]] = None,
    dtype: str = "bfloat16",
) -> str:
    """
    Download MiMo-V2-Flash weights from HuggingFace and convert to nkipy format.

    Args:
        model_name: HuggingFace model name
        output_dir: Directory to save converted weights
        output_filename: Filename for converted weights
        expert_subset: Optional list of expert indices to include (for memory-limited systems)
        dtype: Data type for weights ("bfloat16" or "float32")

    Returns:
        Path to saved weights file
    """
    if not HAS_TORCH:
        raise ImportError("torch and safetensors are required for weight conversion")
    if not HAS_HF_HUB:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    print(f"Downloading weights from {model_name}...")

    # List safetensor files in the repo
    try:
        files = list_repo_files(model_name)
        safetensor_files = [f for f in files if f.endswith('.safetensors')]
        print(f"Found {len(safetensor_files)} safetensor files")
    except Exception as e:
        print(f"Error listing files: {e}")
        raise

    # Download safetensor files
    print("Downloading safetensor files (this may take a while)...")
    cache_dir = snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.bin", "*.pt"],
    )
    print(f"Downloaded to: {cache_dir}")

    # Load and convert weights
    print("Loading and converting weights...")
    state_dict = {}

    for sf_file in safetensor_files:
        file_path = os.path.join(cache_dir, sf_file)
        if os.path.exists(file_path):
            print(f"  Loading {sf_file}...")
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

    print(f"Loaded {len(state_dict)} tensors")

    # Convert weights
    print("Converting to nkipy format...")
    converted_weights = convert_hf_weights(
        state_dict,
        expert_subset=expert_subset,
    )

    # Make all tensors contiguous before saving
    print("Making tensors contiguous...")
    for key in converted_weights:
        if not converted_weights[key].is_contiguous():
            converted_weights[key] = converted_weights[key].contiguous()

    # Save as safetensors
    print(f"Saving to {output_path}...")
    save_file(converted_weights, output_path)

    print(f"Weights saved successfully!")
    return output_path


def convert_hf_weights(
    state_dict: Dict[str, "torch.Tensor"],
    expert_subset: Optional[List[int]] = None,
) -> Dict[str, "torch.Tensor"]:
    """
    Convert HuggingFace MiMo-V2-Flash state dict to nkipy format.

    MiMo-V2-Flash structure:
    - Layer 0: Dense layer (no MoE, uses mlp.gate_proj/up_proj/down_proj)
    - Layer 1-47: MoE layers (256 experts per layer)

    Args:
        state_dict: HuggingFace model state dict
        expert_subset: Optional list of expert indices to include

    Returns:
        Converted state dict for nkipy
    """
    converted = {}

    # Token embedding
    converted["tok_embedding"] = state_dict["model.embed_tokens.weight"]

    # Final layer norm
    converted["norm_weight"] = state_dict["model.norm.weight"]

    # LM head (optional, for generation)
    if "lm_head.weight" in state_dict:
        converted["lm_head_weight"] = state_dict["lm_head.weight"]

    # Count layers
    layer_count = 0
    while f"model.layers.{layer_count}.self_attn.q_proj.weight" in state_dict:
        layer_count += 1

    print(f"Found {layer_count} layers")

    # Convert each layer
    for layer_idx in range(layer_count):
        hf_prefix = f"model.layers.{layer_idx}"
        nki_prefix = f"layers.{layer_idx}"

        print(f"  Converting layer {layer_idx}...", end=" ", flush=True)

        # Attention weights (separate Q, K, V for MiMo due to heterogeneous dims)
        converted[f"{nki_prefix}.q_weight"] = state_dict[
            f"{hf_prefix}.self_attn.q_proj.weight"
        ].T  # Transpose for matmul

        converted[f"{nki_prefix}.k_weight"] = state_dict[
            f"{hf_prefix}.self_attn.k_proj.weight"
        ].T

        converted[f"{nki_prefix}.v_weight"] = state_dict[
            f"{hf_prefix}.self_attn.v_proj.weight"
        ].T

        converted[f"{nki_prefix}.o_weight"] = state_dict[
            f"{hf_prefix}.self_attn.o_proj.weight"
        ].T

        # Layer norms
        converted[f"{nki_prefix}.input_layernorm_weight"] = state_dict[
            f"{hf_prefix}.input_layernorm.weight"
        ]

        converted[f"{nki_prefix}.post_attention_layernorm_weight"] = state_dict[
            f"{hf_prefix}.post_attention_layernorm.weight"
        ]

        # Check if this is a dense layer (layer 0) or MoE layer (layer 1+)
        is_moe_layer = f"{hf_prefix}.mlp.gate.weight" in state_dict

        if is_moe_layer:
            # MoE layer (layers 1-47)
            # Router weight: mlp.gate.weight [256, 4096] -> [4096, 256]
            converted[f"{nki_prefix}.router_weight"] = state_dict[
                f"{hf_prefix}.mlp.gate.weight"
            ].T

            # Optional: e_score_correction_bias
            if f"{hf_prefix}.mlp.gate.e_score_correction_bias" in state_dict:
                converted[f"{nki_prefix}.e_score_correction_bias"] = state_dict[
                    f"{hf_prefix}.mlp.gate.e_score_correction_bias"
                ]

            # Expert weights
            if expert_subset is not None:
                # Only include specified experts
                for expert_idx in expert_subset:
                    gate_key = f"{hf_prefix}.mlp.experts.{expert_idx}.gate_proj.weight"
                    up_key = f"{hf_prefix}.mlp.experts.{expert_idx}.up_proj.weight"
                    down_key = f"{hf_prefix}.mlp.experts.{expert_idx}.down_proj.weight"

                    if gate_key in state_dict:
                        converted[f"{nki_prefix}.expert.{expert_idx}.gate_weight"] = state_dict[gate_key].T
                        converted[f"{nki_prefix}.expert.{expert_idx}.up_weight"] = state_dict[up_key].T
                        converted[f"{nki_prefix}.expert.{expert_idx}.down_weight"] = state_dict[down_key].T
            else:
                # Include all experts - stack into single tensors
                num_experts = 256

                gate_weights = []
                up_weights = []
                down_weights = []

                for expert_idx in range(num_experts):
                    gate_key = f"{hf_prefix}.mlp.experts.{expert_idx}.gate_proj.weight"
                    up_key = f"{hf_prefix}.mlp.experts.{expert_idx}.up_proj.weight"
                    down_key = f"{hf_prefix}.mlp.experts.{expert_idx}.down_proj.weight"

                    if gate_key in state_dict:
                        gate_weights.append(state_dict[gate_key].T)
                        up_weights.append(state_dict[up_key].T)
                        down_weights.append(state_dict[down_key].T)

                if gate_weights:
                    converted[f"{nki_prefix}.expert_gate_weights"] = torch.stack(gate_weights)
                    converted[f"{nki_prefix}.expert_up_weights"] = torch.stack(up_weights)
                    converted[f"{nki_prefix}.expert_down_weights"] = torch.stack(down_weights)

            print(f"MoE ({len(gate_weights) if not expert_subset else len(expert_subset)} experts)")
        else:
            # Dense layer (layer 0)
            # Standard FFN: gate_proj, up_proj, down_proj
            converted[f"{nki_prefix}.gate_weight"] = state_dict[
                f"{hf_prefix}.mlp.gate_proj.weight"
            ].T

            converted[f"{nki_prefix}.up_weight"] = state_dict[
                f"{hf_prefix}.mlp.up_proj.weight"
            ].T

            converted[f"{nki_prefix}.down_weight"] = state_dict[
                f"{hf_prefix}.mlp.down_proj.weight"
            ].T

            print("Dense")

    return converted


def load_weights(
    weights_path: str,
    config: MiMoConfig,
) -> Dict[str, "torch.Tensor"]:
    """
    Load converted weights from safetensors file.

    Args:
        weights_path: Path to safetensors file
        config: Model configuration

    Returns:
        Dictionary of weight tensors
    """
    if not HAS_TORCH:
        raise ImportError("torch and safetensors are required")

    print(f"Loading weights from {weights_path}...")
    weights = load_file(weights_path)
    print(f"Loaded {len(weights)} weight tensors")

    return weights


def shard_experts_for_ep(
    weights: Dict[str, "torch.Tensor"],
    num_devices: int,
    device_id: int,
    config: MiMoConfig,
) -> Dict[str, "torch.Tensor"]:
    """
    Shard expert weights for Expert Parallelism.

    Each device gets a subset of experts:
    - 32 devices: 8 experts per device
    - 64 devices: 4 experts per device

    Args:
        weights: Full model weights
        num_devices: Number of devices for EP
        device_id: This device's ID (0 to num_devices-1)
        config: Model configuration

    Returns:
        Weights with sharded experts for this device
    """
    experts_per_device = config.num_routed_experts // num_devices
    start_expert = device_id * experts_per_device
    end_expert = start_expert + experts_per_device

    print(f"Device {device_id}: experts {start_expert} to {end_expert-1}")

    sharded = {}

    for key, value in weights.items():
        if ".expert_gate_weights" in key or ".expert_up_weights" in key or ".expert_down_weights" in key:
            # Shard expert weights: [num_experts, ...] -> [experts_per_device, ...]
            sharded[key.replace("expert_", "local_expert_")] = value[start_expert:end_expert]
        else:
            # Keep other weights as-is
            sharded[key] = value

    return sharded


def verify_weights(
    weights: Dict[str, "torch.Tensor"],
    config: MiMoConfig,
) -> bool:
    """
    Verify that loaded weights match expected shapes.

    Args:
        weights: Weight dictionary
        config: Model configuration

    Returns:
        True if all shapes match
    """
    expected_shapes = {
        "tok_embedding": (config.vocab_size, config.hidden_size),
        "norm_weight": (config.hidden_size,),
    }

    # Add per-layer expected shapes
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        is_full = config.is_full_attention_layer(layer_idx)

        if is_full:
            k_size = config.k_proj_size_full
            v_size = config.v_proj_size_full
        else:
            k_size = config.k_proj_size_swa
            v_size = config.v_proj_size_swa

        expected_shapes.update({
            f"{prefix}.q_weight": (config.hidden_size, config.q_proj_size),
            f"{prefix}.k_weight": (config.hidden_size, k_size),
            f"{prefix}.v_weight": (config.hidden_size, v_size),
            f"{prefix}.o_weight": (config.o_proj_size, config.hidden_size),
            f"{prefix}.input_layernorm_weight": (config.hidden_size,),
            f"{prefix}.post_attention_layernorm_weight": (config.hidden_size,),
            f"{prefix}.router_weight": (config.hidden_size, config.num_routed_experts),
        })

    # Check shapes
    all_match = True
    for key, expected_shape in expected_shapes.items():
        if key not in weights:
            print(f"Missing weight: {key}")
            all_match = False
            continue

        actual_shape = tuple(weights[key].shape)
        if actual_shape != expected_shape:
            print(f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")
            all_match = False

    return all_match


def create_test_weights(config: MiMoConfig, expert_subset: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
    """
    Create random test weights for development/testing.

    Args:
        config: Model configuration
        expert_subset: Optional subset of experts

    Returns:
        Dictionary of random weight arrays
    """
    import torch
    from neuronxcc.nki.language import bfloat16

    dtype = config.dtype
    weights = {}

    # Token embedding
    weights["tok_embedding"] = torch.randn(
        config.vocab_size, config.hidden_size
    ).to(torch.bfloat16)

    # Final norm
    weights["norm_weight"] = torch.ones(config.hidden_size).to(torch.bfloat16)

    # Per-layer weights
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"layers.{layer_idx}"
        is_full = config.is_full_attention_layer(layer_idx)

        if is_full:
            k_size = config.k_proj_size_full
            v_size = config.v_proj_size_full
        else:
            k_size = config.k_proj_size_swa
            v_size = config.v_proj_size_swa

        # Attention weights
        weights[f"{prefix}.q_weight"] = torch.randn(
            config.hidden_size, config.q_proj_size
        ).to(torch.bfloat16) * 0.02

        weights[f"{prefix}.k_weight"] = torch.randn(
            config.hidden_size, k_size
        ).to(torch.bfloat16) * 0.02

        weights[f"{prefix}.v_weight"] = torch.randn(
            config.hidden_size, v_size
        ).to(torch.bfloat16) * 0.02

        weights[f"{prefix}.o_weight"] = torch.randn(
            config.o_proj_size, config.hidden_size
        ).to(torch.bfloat16) * 0.02

        # Layer norms
        weights[f"{prefix}.input_layernorm_weight"] = torch.ones(
            config.hidden_size
        ).to(torch.bfloat16)

        weights[f"{prefix}.post_attention_layernorm_weight"] = torch.ones(
            config.hidden_size
        ).to(torch.bfloat16)

        # Router weight
        weights[f"{prefix}.router_weight"] = torch.randn(
            config.hidden_size, config.num_routed_experts
        ).to(torch.bfloat16) * 0.02

        # Expert weights
        if expert_subset is not None:
            for expert_idx in expert_subset:
                weights[f"{prefix}.expert.{expert_idx}.gate_weight"] = torch.randn(
                    config.hidden_size, config.moe_intermediate_size
                ).to(torch.bfloat16) * 0.02

                weights[f"{prefix}.expert.{expert_idx}.up_weight"] = torch.randn(
                    config.hidden_size, config.moe_intermediate_size
                ).to(torch.bfloat16) * 0.02

                weights[f"{prefix}.expert.{expert_idx}.down_weight"] = torch.randn(
                    config.moe_intermediate_size, config.hidden_size
                ).to(torch.bfloat16) * 0.02
        else:
            # All experts stacked
            weights[f"{prefix}.expert_gate_weights"] = torch.randn(
                config.num_routed_experts, config.hidden_size, config.moe_intermediate_size
            ).to(torch.bfloat16) * 0.02

            weights[f"{prefix}.expert_up_weights"] = torch.randn(
                config.num_routed_experts, config.hidden_size, config.moe_intermediate_size
            ).to(torch.bfloat16) * 0.02

            weights[f"{prefix}.expert_down_weights"] = torch.randn(
                config.num_routed_experts, config.moe_intermediate_size, config.hidden_size
            ).to(torch.bfloat16) * 0.02

    return weights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare MiMo-V2-Flash weights")
    parser.add_argument(
        "--model-name",
        default="XiaomiMiMo/MiMo-V2-Flash",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_WEIGHTS_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--expert-subset",
        type=int,
        nargs="+",
        default=None,
        help="Expert indices to include (for testing)",
    )
    parser.add_argument(
        "--test-weights",
        action="store_true",
        help="Create random test weights instead of downloading",
    )

    args = parser.parse_args()

    if args.test_weights:
        print("Creating test weights...")
        config = MiMoConfig()
        weights = create_test_weights(config, args.expert_subset)
        output_path = os.path.join(args.output_dir, "test_weights.safetensors")
        os.makedirs(args.output_dir, exist_ok=True)
        save_file(weights, output_path)
        print(f"Test weights saved to {output_path}")
    else:
        download_and_convert_weights(
            model_name=args.model_name,
            output_dir=args.output_dir,
            expert_subset=args.expert_subset,
        )
