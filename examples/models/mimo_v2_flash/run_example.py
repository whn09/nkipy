#!/usr/bin/env python3
"""
MiMo-V2-Flash 运行示例

使用方法:

1. 仅测试 kernels (无需 Trainium):
   python run_example.py --mode test

2. 使用随机权重测试 (单设备，8专家子集):
   python run_example.py --mode random --num-experts 8

3. 下载并转换 HuggingFace 权重:
   python run_example.py --mode download --expert-subset 0 1 2 3 4 5 6 7

4. 使用真实权重运行:
   python run_example.py --mode inference --weights-path tmp_mimo_weights/mimo_weights.safetensors
"""

import argparse
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def run_kernel_tests():
    """运行 kernel 单元测试"""
    print("=" * 60)
    print("运行 Kernel 单元测试")
    print("=" * 60)

    from test_kernels import run_all_tests
    run_all_tests()


def run_with_random_weights(num_experts: int = 8, num_layers: int = 2):
    """使用随机权重测试模型结构"""
    print("=" * 60)
    print(f"使用随机权重测试 ({num_experts} 专家, {num_layers} 层)")
    print("=" * 60)

    from config import MiMoConfig
    from kernels.rmsnorm import rmsnorm
    from kernels.rope_partial import compute_cos_sin_partial, rope_partial
    from kernels.attention_full import mimo_attention_full
    from kernels.attention_swa import mimo_attention_swa
    from kernels.moe_block import moe_block_with_subset

    # 创建简化配置 (减少层数用于测试)
    config = MiMoConfig(
        num_hidden_layers=num_layers,
        max_model_len=32,
        max_batch_size=1,
    )

    print(f"\n模型配置:")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Layers: {config.num_hidden_layers}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - KV heads (full/swa): {config.num_kv_heads_full}/{config.num_kv_heads_swa}")
    print(f"  - Head dim (Q/K): {config.head_dim}")
    print(f"  - V head dim: {config.v_head_dim}")
    print(f"  - Rotary dim: {config.rotary_dim}")
    print(f"  - Experts: {config.num_routed_experts} total, {config.experts_per_tok} active")
    print(f"  - Testing with: {num_experts} expert subset")

    # 创建测试输入
    batch_size = 1
    seq_len = config.max_model_len
    hidden_size = config.hidden_size

    # 随机隐藏状态
    hidden_states = np.random.randn(batch_size * seq_len, hidden_size).astype(np.float32)
    print(f"\n输入 shape: {hidden_states.shape}")

    # 测试 RMSNorm
    print("\n测试 RMSNorm...")
    norm_weight = np.ones(hidden_size, dtype=np.float32)
    normed = rmsnorm(hidden_states, norm_weight, config.rms_norm_eps)
    print(f"  输出 shape: {normed.shape}")

    # 测试 Full Attention
    print("\n测试 Full Attention (Layer 0)...")
    cos_full, sin_full = compute_cos_sin_partial(
        seq_len, config.head_dim, config.rotary_dim, config.rope_theta_full
    )

    q_weight = np.random.randn(hidden_size, config.q_proj_size).astype(np.float32) * 0.02
    k_weight = np.random.randn(hidden_size, config.k_proj_size_full).astype(np.float32) * 0.02
    v_weight = np.random.randn(hidden_size, config.v_proj_size_full).astype(np.float32) * 0.02
    o_weight = np.random.randn(config.o_proj_size, hidden_size).astype(np.float32) * 0.02

    attn_out = mimo_attention_full(
        hidden_states, norm_weight,
        q_weight, k_weight, v_weight, o_weight,
        cos_full, sin_full, config, np.float32
    )
    print(f"  输出 shape: {attn_out.shape}")

    # 测试 SWA
    print("\n测试 Sliding Window Attention (Layer 1)...")
    cos_swa, sin_swa = compute_cos_sin_partial(
        seq_len, config.head_dim, config.rotary_dim, config.rope_theta_swa
    )

    k_weight_swa = np.random.randn(hidden_size, config.k_proj_size_swa).astype(np.float32) * 0.02
    v_weight_swa = np.random.randn(hidden_size, config.v_proj_size_swa).astype(np.float32) * 0.02

    swa_out = mimo_attention_swa(
        hidden_states, norm_weight,
        q_weight, k_weight_swa, v_weight_swa, o_weight,
        cos_swa, sin_swa, config, np.float32
    )
    print(f"  输出 shape: {swa_out.shape}")

    # 测试 MoE Block (子集)
    print(f"\n测试 MoE Block ({num_experts} 专家子集)...")
    expert_indices = list(range(num_experts))

    router_weight = np.random.randn(hidden_size, config.num_routed_experts).astype(np.float32) * 0.02
    expert_gate = np.random.randn(num_experts, hidden_size, config.moe_intermediate_size).astype(np.float32) * 0.02
    expert_up = np.random.randn(num_experts, hidden_size, config.moe_intermediate_size).astype(np.float32) * 0.02
    expert_down = np.random.randn(num_experts, config.moe_intermediate_size, hidden_size).astype(np.float32) * 0.02

    moe_out = moe_block_with_subset(
        attn_out, norm_weight, router_weight,
        expert_gate, expert_up, expert_down,
        config, expert_indices
    )
    print(f"  输出 shape: {moe_out.shape}")

    print("\n" + "=" * 60)
    print("测试完成! 所有组件工作正常。")
    print("=" * 60)


def download_weights(expert_subset=None):
    """下载并转换 HuggingFace 权重"""
    print("=" * 60)
    print("下载并转换 MiMo-V2-Flash 权重")
    print("=" * 60)

    try:
        from prepare_weights import download_and_convert_weights

        download_and_convert_weights(
            model_name="XiaomiMiMo/MiMo-V2-Flash",
            expert_subset=expert_subset,
        )
    except ImportError as e:
        print(f"错误: 需要安装依赖: {e}")
        print("\n请运行:")
        print("  pip install torch transformers safetensors")


def run_inference(weights_path: str):
    """使用真实权重运行推理"""
    print("=" * 60)
    print("运行推理")
    print("=" * 60)

    print(f"\n注意: 完整模型推理需要:")
    print("  - 32+ Trainium2 设备 (Expert Parallelism)")
    print("  - ~600GB 内存 (完整 309B 参数)")
    print("\n当前仅展示 API 使用示例...")

    print("""
# 示例代码 (需要在 Trainium2 环境中运行):

from config import MiMoConfig
from model import MiMoV2FlashModel
from prepare_weights import load_weights

# 1. 加载配置
config = MiMoConfig(
    num_devices=32,  # 32 Trainium2 设备
)

# 2. 加载权重
weights = load_weights(weights_path, config)

# 3. 初始化模型
model = MiMoV2FlashModel(weights, config)

# 4. 运行推理
input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.uint32)
output = model.forward(input_ids)
print(f"Output shape: {output.shape}")
""")


def main():
    parser = argparse.ArgumentParser(
        description="MiMo-V2-Flash 运行示例",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_example.py --mode test          # 运行 kernel 测试
  python run_example.py --mode random        # 使用随机权重测试
  python run_example.py --mode download      # 下载权重
        """
    )

    parser.add_argument(
        "--mode",
        choices=["test", "random", "download", "inference"],
        default="test",
        help="运行模式"
    )

    parser.add_argument(
        "--num-experts",
        type=int,
        default=8,
        help="测试时使用的专家数量 (默认: 8)"
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="测试时使用的层数 (默认: 2)"
    )

    parser.add_argument(
        "--expert-subset",
        type=int,
        nargs="+",
        default=None,
        help="下载时包含的专家索引"
    )

    parser.add_argument(
        "--weights-path",
        type=str,
        default="tmp_mimo_weights/mimo_weights.safetensors",
        help="权重文件路径"
    )

    args = parser.parse_args()

    if args.mode == "test":
        run_kernel_tests()
    elif args.mode == "random":
        run_with_random_weights(args.num_experts, args.num_layers)
    elif args.mode == "download":
        download_weights(args.expert_subset)
    elif args.mode == "inference":
        run_inference(args.weights_path)


if __name__ == "__main__":
    main()
