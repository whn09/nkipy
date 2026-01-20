#!/usr/bin/env python3
"""
MiMo-V2-Flash Trainium2 验证脚本

逐步验证各个kernel在Trainium2上的编译和运行。

用法:
    python verify_trainium.py --test all
    python verify_trainium.py --test rmsnorm
    python verify_trainium.py --test attention
    python verify_trainium.py --test moe
    python verify_trainium.py --test layer
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from config import MiMoConfig, get_build_dir
from nkipy.core.trace import NKIPyKernel
from nkipy.runtime import DeviceKernel, DeviceTensor

# Compiler arguments for Trainium2
COMPILER_ARGS = (
    " --lnc 1 --model-type transformer"
    " --tensorizer-options='--enable-ccop-compute-overlap"
    " --cc-pipeline-tiling-factor=2 --vectorize-strided-dma'"
    " --enable-mixed-precision-accumulation"
)

BACKEND = "hlo"
OUTPUT_PREFIX = "output"


def test_rmsnorm():
    """测试 RMSNorm kernel"""
    from kernels.rmsnorm import rmsnorm

    print("\n" + "=" * 60)
    print("测试 RMSNorm Kernel")
    print("=" * 60)

    config = MiMoConfig()
    batch_seq = 32
    hidden_size = config.hidden_size

    # 创建测试数据
    x = np.random.randn(batch_seq, hidden_size).astype(np.float32)
    weight = np.ones(hidden_size, dtype=np.float32)
    eps = config.rms_norm_eps

    # CPU 参考结果
    print("计算 CPU 参考结果...")
    cpu_result = rmsnorm(x, weight, eps)

    # 编译 kernel
    print("编译 RMSNorm kernel...")
    start = time.time()
    kernel = DeviceKernel.compile_and_load(
        NKIPyKernel.trace(rmsnorm, backend=BACKEND),
        name="mimo_rmsnorm_test",
        x=x,
        weight=weight,
        eps=eps,
        additional_compiler_args=COMPILER_ARGS,
    )
    compile_time = time.time() - start
    print(f"编译完成: {compile_time:.1f}s")

    # 运行 kernel
    print("运行 kernel...")
    x_device = DeviceTensor.from_numpy(x, "x")
    weight_device = DeviceTensor.from_numpy(weight, "weight")
    output_device = DeviceTensor.from_numpy(np.zeros_like(x), "output")

    start = time.time()
    kernel(
        inputs={"x": x_device, "weight": weight_device},
        outputs={f"{OUTPUT_PREFIX}0": output_device},
    )
    run_time = time.time() - start
    print(f"运行完成: {run_time*1000:.2f}ms")

    # 验证结果
    device_result = output_device.numpy()
    max_diff = np.max(np.abs(cpu_result - device_result))
    print(f"最大差异: {max_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ RMSNorm 验证通过!")
        return True
    else:
        print("✗ RMSNorm 验证失败!")
        return False


def test_token_embedding():
    """测试 Token Embedding kernel"""
    from kernels.token_embedding import token_embedding

    print("\n" + "=" * 60)
    print("测试 Token Embedding Kernel")
    print("=" * 60)

    config = MiMoConfig()
    batch_size = 1
    seq_len = 32
    vocab_size = 1000  # 使用较小的vocab进行测试
    hidden_size = config.hidden_size

    # 创建测试数据
    tok_embedding = np.random.randn(vocab_size, hidden_size).astype(np.float32)
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len), dtype=np.uint32)

    # CPU 参考结果
    print("计算 CPU 参考结果...")
    cpu_result = token_embedding(tok_embedding, token_ids)

    # 编译 kernel
    print("编译 Token Embedding kernel...")
    start = time.time()
    kernel = DeviceKernel.compile_and_load(
        NKIPyKernel.trace(token_embedding, backend=BACKEND),
        name="mimo_token_embedding_test",
        tok_embedding=tok_embedding,
        token_ids=token_ids,
        additional_compiler_args=COMPILER_ARGS,
    )
    compile_time = time.time() - start
    print(f"编译完成: {compile_time:.1f}s")

    # 运行 kernel
    print("运行 kernel...")
    tok_emb_device = DeviceTensor.from_numpy(tok_embedding, "tok_embedding")
    token_ids_device = DeviceTensor.from_numpy(token_ids, "token_ids")
    output_device = DeviceTensor.from_numpy(np.zeros_like(cpu_result), "output")

    start = time.time()
    kernel(
        inputs={"tok_embedding": tok_emb_device, "token_ids": token_ids_device},
        outputs={f"{OUTPUT_PREFIX}0": output_device},
    )
    run_time = time.time() - start
    print(f"运行完成: {run_time*1000:.2f}ms")

    # 验证结果
    device_result = output_device.numpy()
    max_diff = np.max(np.abs(cpu_result - device_result))
    print(f"最大差异: {max_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ Token Embedding 验证通过!")
        return True
    else:
        print("✗ Token Embedding 验证失败!")
        return False


def test_attention_full():
    """测试 Full Attention kernel"""
    from kernels.attention_full import mimo_attention_full
    from kernels.rope_partial import compute_cos_sin_partial

    print("\n" + "=" * 60)
    print("测试 Full Attention Kernel")
    print("=" * 60)

    config = MiMoConfig()
    batch_seq = 32
    hidden_size = config.hidden_size

    # 创建测试数据
    hidden_states = np.random.randn(batch_seq, hidden_size).astype(np.float32) * 0.1
    input_layernorm_weight = np.ones(hidden_size, dtype=np.float32)
    q_weight = np.random.randn(hidden_size, config.q_proj_size).astype(np.float32) * 0.02
    k_weight = np.random.randn(hidden_size, config.k_proj_size_full).astype(np.float32) * 0.02
    v_weight = np.random.randn(hidden_size, config.v_proj_size_full).astype(np.float32) * 0.02
    o_weight = np.random.randn(config.o_proj_size, hidden_size).astype(np.float32) * 0.02

    cos, sin = compute_cos_sin_partial(batch_seq, config.head_dim, config.rotary_dim, config.rope_theta_full)

    # CPU 参考结果
    print("计算 CPU 参考结果...")
    cpu_result = mimo_attention_full(
        hidden_states, input_layernorm_weight,
        q_weight, k_weight, v_weight, o_weight,
        cos, sin, config, np.float32
    )

    # 编译 kernel
    print("编译 Full Attention kernel...")
    start = time.time()
    kernel = DeviceKernel.compile_and_load(
        NKIPyKernel.trace(mimo_attention_full, backend=BACKEND),
        name="mimo_attention_full_test",
        hidden_states=hidden_states,
        input_layernorm_weight=input_layernorm_weight,
        q_weight=q_weight,
        k_weight=k_weight,
        v_weight=v_weight,
        o_weight=o_weight,
        cos=cos,
        sin=sin,
        config=config,
        compute_dtype=np.float32,
        additional_compiler_args=COMPILER_ARGS,
    )
    compile_time = time.time() - start
    print(f"编译完成: {compile_time:.1f}s")

    # 运行 kernel
    print("运行 kernel...")
    inputs = {
        "hidden_states": DeviceTensor.from_numpy(hidden_states, "hidden_states"),
        "input_layernorm_weight": DeviceTensor.from_numpy(input_layernorm_weight, "input_layernorm_weight"),
        "q_weight": DeviceTensor.from_numpy(q_weight, "q_weight"),
        "k_weight": DeviceTensor.from_numpy(k_weight, "k_weight"),
        "v_weight": DeviceTensor.from_numpy(v_weight, "v_weight"),
        "o_weight": DeviceTensor.from_numpy(o_weight, "o_weight"),
        "cos": DeviceTensor.from_numpy(cos, "cos"),
        "sin": DeviceTensor.from_numpy(sin, "sin"),
    }
    output_device = DeviceTensor.from_numpy(np.zeros_like(cpu_result), "output")

    start = time.time()
    kernel(inputs=inputs, outputs={f"{OUTPUT_PREFIX}0": output_device})
    run_time = time.time() - start
    print(f"运行完成: {run_time*1000:.2f}ms")

    # 验证结果
    device_result = output_device.numpy()
    max_diff = np.max(np.abs(cpu_result - device_result))
    mean_diff = np.mean(np.abs(cpu_result - device_result))
    print(f"最大差异: {max_diff:.6f}, 平均差异: {mean_diff:.6f}")

    if max_diff < 0.1:  # Attention 有更大的数值误差
        print("✓ Full Attention 验证通过!")
        return True
    else:
        print("✗ Full Attention 验证失败!")
        return False


def test_dense_ffn():
    """测试 Dense FFN kernel (层0)"""
    from kernels.ffn import dense_ffn_block

    print("\n" + "=" * 60)
    print("测试 Dense FFN Kernel")
    print("=" * 60)

    config = MiMoConfig()
    batch_seq = 32
    hidden_size = config.hidden_size
    intermediate_size = 11008  # 层0的intermediate size

    # 创建测试数据
    hidden_states = np.random.randn(batch_seq, hidden_size).astype(np.float32) * 0.1
    post_attn_layernorm_weight = np.ones(hidden_size, dtype=np.float32)
    gate_weight = np.random.randn(hidden_size, intermediate_size).astype(np.float32) * 0.02
    up_weight = np.random.randn(hidden_size, intermediate_size).astype(np.float32) * 0.02
    down_weight = np.random.randn(intermediate_size, hidden_size).astype(np.float32) * 0.02

    # CPU 参考结果
    print("计算 CPU 参考结果...")
    cpu_result = dense_ffn_block(
        hidden_states, post_attn_layernorm_weight,
        gate_weight, up_weight, down_weight, config
    )

    # 编译 kernel
    print("编译 Dense FFN kernel...")
    start = time.time()
    kernel = DeviceKernel.compile_and_load(
        NKIPyKernel.trace(dense_ffn_block, backend=BACKEND),
        name="mimo_dense_ffn_test",
        hidden_states=hidden_states,
        post_attn_layernorm_weight=post_attn_layernorm_weight,
        gate_weight=gate_weight,
        up_weight=up_weight,
        down_weight=down_weight,
        config=config,
        additional_compiler_args=COMPILER_ARGS,
    )
    compile_time = time.time() - start
    print(f"编译完成: {compile_time:.1f}s")

    # 运行 kernel
    print("运行 kernel...")
    inputs = {
        "hidden_states": DeviceTensor.from_numpy(hidden_states, "hidden_states"),
        "post_attn_layernorm_weight": DeviceTensor.from_numpy(post_attn_layernorm_weight, "post_attn_ln"),
        "gate_weight": DeviceTensor.from_numpy(gate_weight, "gate_weight"),
        "up_weight": DeviceTensor.from_numpy(up_weight, "up_weight"),
        "down_weight": DeviceTensor.from_numpy(down_weight, "down_weight"),
    }
    output_device = DeviceTensor.from_numpy(np.zeros_like(cpu_result), "output")

    start = time.time()
    kernel(inputs=inputs, outputs={f"{OUTPUT_PREFIX}0": output_device})
    run_time = time.time() - start
    print(f"运行完成: {run_time*1000:.2f}ms")

    # 验证结果
    device_result = output_device.numpy()
    max_diff = np.max(np.abs(cpu_result - device_result))
    print(f"最大差异: {max_diff:.6f}")

    if max_diff < 0.1:
        print("✓ Dense FFN 验证通过!")
        return True
    else:
        print("✗ Dense FFN 验证失败!")
        return False


def test_moe_block():
    """测试 MoE Block kernel"""
    from kernels.moe_block import moe_block

    print("\n" + "=" * 60)
    print("测试 MoE Block Kernel")
    print("=" * 60)

    # 使用小型配置，专家数量匹配以避免需要modulo映射
    num_experts = 8
    config = MiMoConfig(num_routed_experts=num_experts, num_devices=num_experts)  # 使config匹配实际专家数
    batch_seq = 32
    hidden_size = config.hidden_size

    # 创建测试数据
    hidden_states = np.random.randn(batch_seq, hidden_size).astype(np.float32) * 0.1
    post_attn_layernorm_weight = np.ones(hidden_size, dtype=np.float32)
    router_weight = np.random.randn(hidden_size, num_experts).astype(np.float32) * 0.02
    expert_gate_weights = np.random.randn(num_experts, hidden_size, config.moe_intermediate_size).astype(np.float32) * 0.02
    expert_up_weights = np.random.randn(num_experts, hidden_size, config.moe_intermediate_size).astype(np.float32) * 0.02
    expert_down_weights = np.random.randn(num_experts, config.moe_intermediate_size, hidden_size).astype(np.float32) * 0.02

    # CPU 参考结果
    print("计算 CPU 参考结果...")
    cpu_result = moe_block(
        hidden_states, post_attn_layernorm_weight, router_weight,
        expert_gate_weights, expert_up_weights, expert_down_weights, config
    )

    # 编译 kernel
    print("编译 MoE Block kernel...")
    start = time.time()
    kernel = DeviceKernel.compile_and_load(
        NKIPyKernel.trace(moe_block, backend=BACKEND),
        name="mimo_moe_block_test",
        hidden_states=hidden_states,
        post_attn_layernorm_weight=post_attn_layernorm_weight,
        router_weight=router_weight,
        expert_gate_weights=expert_gate_weights,
        expert_up_weights=expert_up_weights,
        expert_down_weights=expert_down_weights,
        config=config,
        additional_compiler_args=COMPILER_ARGS,
    )
    compile_time = time.time() - start
    print(f"编译完成: {compile_time:.1f}s")

    # 运行 kernel
    print("运行 kernel...")
    inputs = {
        "hidden_states": DeviceTensor.from_numpy(hidden_states, "hidden_states"),
        "post_attn_layernorm_weight": DeviceTensor.from_numpy(post_attn_layernorm_weight, "post_attn_ln"),
        "router_weight": DeviceTensor.from_numpy(router_weight, "router_weight"),
        "expert_gate_weights": DeviceTensor.from_numpy(expert_gate_weights, "expert_gate"),
        "expert_up_weights": DeviceTensor.from_numpy(expert_up_weights, "expert_up"),
        "expert_down_weights": DeviceTensor.from_numpy(expert_down_weights, "expert_down"),
    }
    output_device = DeviceTensor.from_numpy(np.zeros_like(cpu_result), "output")

    start = time.time()
    kernel(inputs=inputs, outputs={f"{OUTPUT_PREFIX}0": output_device})
    run_time = time.time() - start
    print(f"运行完成: {run_time*1000:.2f}ms")

    # 验证结果
    device_result = output_device.numpy()
    max_diff = np.max(np.abs(cpu_result - device_result))
    print(f"最大差异: {max_diff:.6f}")

    if max_diff < 0.5:  # MoE 有更大的数值误差
        print("✓ MoE Block 验证通过!")
        return True
    else:
        print("✗ MoE Block 验证失败!")
        return False


def main():
    parser = argparse.ArgumentParser(description="MiMo-V2-Flash Trainium2 验证")
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "rmsnorm", "embedding", "attention", "ffn", "moe"],
        help="要测试的 kernel",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("MiMo-V2-Flash Trainium2 验证")
    print("=" * 60)

    # 检查 Neuron 设备
    os.system("neuron-ls 2>/dev/null | head -5")

    results = {}

    if args.test in ["all", "rmsnorm"]:
        results["rmsnorm"] = test_rmsnorm()

    if args.test in ["all", "embedding"]:
        results["embedding"] = test_token_embedding()

    if args.test in ["all", "attention"]:
        results["attention"] = test_attention_full()

    if args.test in ["all", "ffn"]:
        results["ffn"] = test_dense_ffn()

    if args.test in ["all", "moe"]:
        results["moe"] = test_moe_block()

    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n所有测试通过!")
    else:
        print("\n部分测试失败!")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
