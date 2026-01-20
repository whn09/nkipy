#!/usr/bin/env python3
"""
MiMo-V2-Flash 完整模拟

完整模拟 MiMo-V2-Flash 的架构：
- 48 层 Transformer
- 256 专家，每 token 选择 8 个
- 混合注意力：Full + SWA
- 异构头维度：Q/K=192, V=128
- 部分 RoPE (33.4%)

注意：使用随机权重，仅验证架构和计算流程
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import MiMoConfig
from kernels.rmsnorm import rmsnorm
from kernels.rope_partial import compute_cos_sin_partial, rope_partial
from kernels.attention_full import mimo_attention_full
from kernels.attention_swa import mimo_attention_swa
from kernels.moe_block import moe_block
from kernels.token_embedding import token_embedding


def create_layer_weights(config: MiMoConfig, layer_idx: int, dtype=np.float32, verbose=False):
    """为单层创建随机权重"""
    is_full = config.is_full_attention_layer(layer_idx)

    if is_full:
        k_size = config.k_proj_size_full
        v_size = config.v_proj_size_full
    else:
        k_size = config.k_proj_size_swa
        v_size = config.v_proj_size_swa

    if verbose:
        print("    创建 attention 权重...", end=" ", flush=True)

    weights = {
        # Attention weights
        "q_weight": np.random.randn(config.hidden_size, config.q_proj_size).astype(dtype) * 0.02,
        "k_weight": np.random.randn(config.hidden_size, k_size).astype(dtype) * 0.02,
        "v_weight": np.random.randn(config.hidden_size, v_size).astype(dtype) * 0.02,
        "o_weight": np.random.randn(config.o_proj_size, config.hidden_size).astype(dtype) * 0.02,
        "input_ln_weight": np.ones(config.hidden_size, dtype=dtype),
        "post_attn_ln_weight": np.ones(config.hidden_size, dtype=dtype),
        "router_weight": np.random.randn(config.hidden_size, config.num_routed_experts).astype(dtype) * 0.02,
    }

    if verbose:
        print("done", flush=True)
        print(f"    创建 {config.num_routed_experts} 专家权重...", end=" ", flush=True)

    # MoE weights - 使用更高效的方式
    weights["expert_gate"] = (np.random.randn(
        config.num_routed_experts, config.hidden_size, config.moe_intermediate_size
    ) * 0.02).astype(dtype)

    weights["expert_up"] = (np.random.randn(
        config.num_routed_experts, config.hidden_size, config.moe_intermediate_size
    ) * 0.02).astype(dtype)

    weights["expert_down"] = (np.random.randn(
        config.num_routed_experts, config.moe_intermediate_size, config.hidden_size
    ) * 0.02).astype(dtype)

    if verbose:
        print("done", flush=True)

    return weights


def run_layer(hidden_states, weights, cos, sin, config, layer_idx):
    """运行单层 Transformer"""
    is_full = config.is_full_attention_layer(layer_idx)

    # 1. Attention
    if is_full:
        hidden_states = mimo_attention_full(
            hidden_states,
            weights["input_ln_weight"],
            weights["q_weight"],
            weights["k_weight"],
            weights["v_weight"],
            weights["o_weight"],
            cos, sin,
            config,
            np.float32,
        )
    else:
        hidden_states = mimo_attention_swa(
            hidden_states,
            weights["input_ln_weight"],
            weights["q_weight"],
            weights["k_weight"],
            weights["v_weight"],
            weights["o_weight"],
            cos, sin,
            config,
            np.float32,
        )

    # 2. MoE
    hidden_states = moe_block(
        hidden_states,
        weights["post_attn_ln_weight"],
        weights["router_weight"],
        weights["expert_gate"],
        weights["expert_up"],
        weights["expert_down"],
        config,
    )

    return hidden_states


def estimate_memory(config: MiMoConfig):
    """估算内存需求"""
    # 单层权重大小 (bytes, float32)
    attn_size = (
        config.hidden_size * config.q_proj_size +  # Q
        config.hidden_size * config.k_proj_size_full +  # K (取较大的)
        config.hidden_size * config.v_proj_size_full +  # V
        config.o_proj_size * config.hidden_size +  # O
        config.hidden_size * 2  # LayerNorms
    ) * 4  # float32

    moe_size = (
        config.hidden_size * config.num_routed_experts +  # Router
        config.num_routed_experts * config.hidden_size * config.moe_intermediate_size * 3  # Gate, Up, Down
    ) * 4

    layer_size = attn_size + moe_size
    total_size = layer_size * config.num_hidden_layers

    return {
        "attention_per_layer_gb": attn_size / 1e9,
        "moe_per_layer_gb": moe_size / 1e9,
        "layer_total_gb": layer_size / 1e9,
        "model_total_gb": total_size / 1e9,
    }


def main():
    parser = argparse.ArgumentParser(description="MiMo-V2-Flash 完整模拟")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--num-layers", type=int, default=48, help="层数 (默认 48)")
    parser.add_argument("--num-experts", type=int, default=256, help="专家数 (默认 256)")
    parser.add_argument("--layer-by-layer", action="store_true", help="逐层处理以节省内存")
    parser.add_argument("--estimate-only", action="store_true", help="仅估算内存需求")

    args = parser.parse_args()

    # 创建配置
    config = MiMoConfig(
        num_hidden_layers=args.num_layers,
        num_routed_experts=args.num_experts,
        max_model_len=args.seq_len,
        max_batch_size=args.batch_size,
    )

    print("=" * 70)
    print("MiMo-V2-Flash 完整模拟")
    print("=" * 70)

    print(f"\n模型配置:")
    print(f"  - Hidden size:        {config.hidden_size}")
    print(f"  - Layers:             {config.num_hidden_layers}")
    print(f"  - Attention heads:    {config.num_attention_heads}")
    print(f"  - KV heads (full):    {config.num_kv_heads_full}")
    print(f"  - KV heads (swa):     {config.num_kv_heads_swa}")
    print(f"  - Head dim (Q/K):     {config.head_dim}")
    print(f"  - V head dim:         {config.v_head_dim}")
    print(f"  - Rotary dim:         {config.rotary_dim} ({config.partial_rotary_factor*100:.1f}%)")
    print(f"  - Experts:            {config.num_routed_experts}")
    print(f"  - Experts per token:  {config.experts_per_tok}")
    print(f"  - Expert intermediate: {config.moe_intermediate_size}")
    print(f"  - Sliding window:     {config.sliding_window_size}")

    # Full attention 层
    full_attn_layers = [i for i in range(config.num_hidden_layers) if config.is_full_attention_layer(i)]
    print(f"  - Full attention layers: {full_attn_layers}")

    # 内存估算
    mem = estimate_memory(config)
    print(f"\n内存估算 (float32):")
    print(f"  - Attention/layer:    {mem['attention_per_layer_gb']:.2f} GB")
    print(f"  - MoE/layer:          {mem['moe_per_layer_gb']:.2f} GB")
    print(f"  - Total/layer:        {mem['layer_total_gb']:.2f} GB")
    print(f"  - 全模型 ({args.num_layers} 层): {mem['model_total_gb']:.1f} GB")

    if args.estimate_only:
        print("\n[仅估算模式，不运行模拟]")
        return

    print(f"\n输入配置:")
    print(f"  - Batch size:         {args.batch_size}")
    print(f"  - Sequence length:    {args.seq_len}")

    # 预计算 RoPE
    print("\n预计算 RoPE tables...")
    cos_full, sin_full = compute_cos_sin_partial(
        args.seq_len, config.head_dim, config.rotary_dim, config.rope_theta_full
    )
    cos_swa, sin_swa = compute_cos_sin_partial(
        args.seq_len, config.head_dim, config.rotary_dim, config.rope_theta_swa
    )

    # 初始化输入
    print("初始化输入...")
    hidden_states = np.random.randn(
        args.batch_size * args.seq_len, config.hidden_size
    ).astype(np.float32)
    print(f"  输入 shape: {hidden_states.shape}")

    # 运行模拟
    print("\n" + "=" * 70)
    print("开始逐层模拟...")
    print("=" * 70)

    total_start = time.time()

    for layer_idx in range(config.num_hidden_layers):
        is_full = config.is_full_attention_layer(layer_idx)
        attn_type = "Full" if is_full else "SWA"
        cos = cos_full if is_full else cos_swa
        sin = sin_full if is_full else sin_swa

        print(f"\n[Layer {layer_idx:2d}/{config.num_hidden_layers}] {attn_type}", flush=True)

        layer_start = time.time()

        # 创建权重 (逐层释放以节省内存)
        weights = create_layer_weights(config, layer_idx, verbose=True)

        # 运行层
        print("    运行 forward...", end=" ", flush=True)
        hidden_states = run_layer(hidden_states, weights, cos, sin, config, layer_idx)
        print("done", flush=True)

        # 释放权重
        del weights
        import gc
        gc.collect()

        layer_time = time.time() - layer_start
        print(f"    完成: {layer_time:.1f}s, output shape: {hidden_states.shape}", flush=True)

    total_time = time.time() - total_start

    # Final LayerNorm
    print("\n应用 Final LayerNorm...")
    final_ln_weight = np.ones(config.hidden_size, dtype=np.float32)
    output = rmsnorm(hidden_states, final_ln_weight, config.rms_norm_eps)

    print("\n" + "=" * 70)
    print("模拟完成!")
    print("=" * 70)
    print(f"\n结果:")
    print(f"  - 输出 shape:  {output.shape}")
    print(f"  - 输出 dtype:  {output.dtype}")
    print(f"  - 输出范围:    [{output.min():.4f}, {output.max():.4f}]")
    print(f"  - 总耗时:      {total_time:.1f}s")
    print(f"  - 平均每层:    {total_time/config.num_hidden_layers:.2f}s")


if __name__ == "__main__":
    main()
