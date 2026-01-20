#!/usr/bin/env python3
"""
使用真实权重运行 MiMo-V2-Flash (CPU 模式)

这个脚本加载转换后的权重并在 CPU 上运行推理。
用于验证权重加载和模型结构的正确性。

用法:
    python run_with_weights.py --weights-path tmp_mimo_weights/mimo_weights.safetensors --num-layers 2
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import MiMoConfig
from layer import MiMoLayerCPU
from kernels.rmsnorm import rmsnorm
from kernels.rope_partial import compute_cos_sin_partial
from kernels.token_embedding import token_embedding


def load_weights_safetensors(weights_path: str):
    """加载 safetensors 权重文件

    使用 PyTorch 加载以支持 FP8 等特殊数据类型，然后转换为 numpy float32。
    """
    from safetensors import safe_open
    import torch

    print(f"Loading weights from {weights_path}...")
    weights = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # 将所有数据类型转换为 float32 (支持 FP8, BF16 等)
            if tensor.dtype in [torch.bfloat16, torch.float16]:
                weights[key] = tensor.float().numpy()
            elif hasattr(torch, 'float8_e4m3fn') and tensor.dtype == torch.float8_e4m3fn:
                weights[key] = tensor.float().numpy()
            elif hasattr(torch, 'float8_e5m2') and tensor.dtype == torch.float8_e5m2:
                weights[key] = tensor.float().numpy()
            else:
                # 其他类型尝试直接转换
                try:
                    weights[key] = tensor.float().numpy()
                except:
                    weights[key] = tensor.numpy()

    print(f"Loaded {len(weights)} tensors")
    return weights


def extract_layer_weights(all_weights: dict, layer_idx: int, config: MiMoConfig):
    """提取单层的权重"""
    prefix = f"layers.{layer_idx}"
    is_dense = config.is_dense_layer(layer_idx)

    layer_weights = {
        "q_weight": all_weights[f"{prefix}.q_weight"],
        "k_weight": all_weights[f"{prefix}.k_weight"],
        "v_weight": all_weights[f"{prefix}.v_weight"],
        "o_weight": all_weights[f"{prefix}.o_weight"],
        "input_layernorm_weight": all_weights[f"{prefix}.input_layernorm_weight"],
        "post_attention_layernorm_weight": all_weights[f"{prefix}.post_attention_layernorm_weight"],
    }

    if is_dense:
        # Dense layer (Layer 0)
        layer_weights["gate_weight"] = all_weights[f"{prefix}.gate_weight"]
        layer_weights["up_weight"] = all_weights[f"{prefix}.up_weight"]
        layer_weights["down_weight"] = all_weights[f"{prefix}.down_weight"]
    else:
        # MoE layer
        layer_weights["router_weight"] = all_weights[f"{prefix}.router_weight"]

        # 检查专家权重格式 (stacked 或 individual)
        if f"{prefix}.expert_gate_weights" in all_weights:
            # Stacked format
            layer_weights["expert_gate_weights"] = all_weights[f"{prefix}.expert_gate_weights"]
            layer_weights["expert_up_weights"] = all_weights[f"{prefix}.expert_up_weights"]
            layer_weights["expert_down_weights"] = all_weights[f"{prefix}.expert_down_weights"]
        else:
            # Individual expert format - need to stack
            gate_list = []
            up_list = []
            down_list = []

            expert_idx = 0
            while f"{prefix}.expert.{expert_idx}.gate_weight" in all_weights:
                gate_list.append(all_weights[f"{prefix}.expert.{expert_idx}.gate_weight"])
                up_list.append(all_weights[f"{prefix}.expert.{expert_idx}.up_weight"])
                down_list.append(all_weights[f"{prefix}.expert.{expert_idx}.down_weight"])
                expert_idx += 1

            if gate_list:
                layer_weights["expert_gate_weights"] = np.stack(gate_list)
                layer_weights["expert_up_weights"] = np.stack(up_list)
                layer_weights["expert_down_weights"] = np.stack(down_list)

    return layer_weights


def main():
    parser = argparse.ArgumentParser(description="使用真实权重运行 MiMo-V2-Flash")
    parser.add_argument(
        "--weights-path",
        type=str,
        default="tmp_mimo_weights/mimo_weights.safetensors",
        help="权重文件路径",
    )
    parser.add_argument("--num-layers", type=int, default=2, help="运行的层数 (默认 2)")
    parser.add_argument("--seq-len", type=int, default=16, help="序列长度 (默认 16)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (默认 1)")

    args = parser.parse_args()

    # 检查权重文件
    if not os.path.exists(args.weights_path):
        print(f"Error: 权重文件不存在: {args.weights_path}")
        print("请先运行: python run_example.py --mode download --expert-subset 0 1 2 3 4 5 6 7")
        return

    # 加载权重
    all_weights = load_weights_safetensors(args.weights_path)

    # 检测专家数量
    # 检查 layer 1 的专家权重
    if "layers.1.expert_gate_weights" in all_weights:
        num_experts = all_weights["layers.1.expert_gate_weights"].shape[0]
    else:
        num_experts = 0
        while f"layers.1.expert.{num_experts}.gate_weight" in all_weights:
            num_experts += 1

    print(f"Detected {num_experts} experts per layer")

    # 创建配置
    config = MiMoConfig(
        num_hidden_layers=args.num_layers,
        max_model_len=args.seq_len,
        max_batch_size=args.batch_size,
        num_routed_experts=256,  # 保持原始配置用于路由器
    )

    # 更新配置以匹配实际专家数（用于 MoE 计算）
    actual_experts_per_layer = num_experts

    print("=" * 70)
    print("MiMo-V2-Flash 真实权重测试")
    print("=" * 70)

    print(f"\n配置:")
    print(f"  - Layers:           {config.num_hidden_layers}")
    print(f"  - Seq len:          {args.seq_len}")
    print(f"  - Batch size:       {args.batch_size}")
    print(f"  - Experts loaded:   {num_experts}")
    print(f"  - Hidden size:      {config.hidden_size}")

    # 预计算 RoPE
    print("\n预计算 RoPE tables...")
    cos_full, sin_full = compute_cos_sin_partial(
        args.seq_len, config.head_dim, config.rotary_dim, config.rope_theta_full
    )
    cos_swa, sin_swa = compute_cos_sin_partial(
        args.seq_len, config.head_dim, config.rotary_dim, config.rope_theta_swa
    )

    # Token embedding
    print("加载 token embedding...")
    tok_embedding = all_weights["tok_embedding"]
    print(f"  Embedding shape: {tok_embedding.shape}")

    # 创建测试输入
    print("\n创建测试输入...")
    # 使用真实 token IDs (这里用随机的进行测试)
    input_ids = np.random.randint(0, 1000, (args.batch_size, args.seq_len), dtype=np.uint32)
    print(f"  Input IDs shape: {input_ids.shape}")

    # Token embedding lookup
    hidden_states = token_embedding(tok_embedding, input_ids)
    hidden_states = hidden_states.reshape(args.batch_size * args.seq_len, config.hidden_size)
    print(f"  Hidden states shape: {hidden_states.shape}")

    # 运行层
    print("\n" + "=" * 70)
    print("开始逐层推理...")
    print("=" * 70)

    total_start = time.time()

    for layer_idx in range(args.num_layers):
        is_full = config.is_full_attention_layer(layer_idx)
        is_dense = config.is_dense_layer(layer_idx)
        attn_type = "Full" if is_full else "SWA"
        ffn_type = "Dense" if is_dense else f"MoE({num_experts})"

        print(f"\n[Layer {layer_idx:2d}] {attn_type} + {ffn_type}", flush=True)

        layer_start = time.time()

        # 提取层权重
        print("    加载权重...", end=" ", flush=True)
        layer_weights = extract_layer_weights(all_weights, layer_idx, config)
        print("done", flush=True)

        # 选择 RoPE
        cos = cos_full if is_full else cos_swa
        sin = sin_full if is_full else sin_swa

        # 创建层
        layer = MiMoLayerCPU(layer_idx, config, layer_weights, cos, sin)

        # 运行 forward
        print("    运行 forward...", end=" ", flush=True)
        hidden_states = layer.forward(hidden_states.astype(np.float32))
        print("done", flush=True)

        layer_time = time.time() - layer_start
        print(f"    完成: {layer_time:.1f}s, output: {hidden_states.shape}, "
              f"range: [{hidden_states.min():.3f}, {hidden_states.max():.3f}]")

    total_time = time.time() - total_start

    # Final LayerNorm
    print("\n应用 Final LayerNorm...")
    norm_weight = all_weights["norm_weight"]
    output = rmsnorm(hidden_states, norm_weight, config.rms_norm_eps)

    print("\n" + "=" * 70)
    print("推理完成!")
    print("=" * 70)
    print(f"\n结果:")
    print(f"  - 输出 shape: {output.shape}")
    print(f"  - 输出范围:   [{output.min():.4f}, {output.max():.4f}]")
    print(f"  - 输出均值:   {output.mean():.4f}")
    print(f"  - 输出标准差: {output.std():.4f}")
    print(f"  - 总耗时:     {total_time:.1f}s")


if __name__ == "__main__":
    main()
