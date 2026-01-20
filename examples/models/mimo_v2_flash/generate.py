#!/usr/bin/env python3
"""
MiMo-V2-Flash 文本生成 (CPU 模式)

使用真实权重和 tokenizer 生成文本。

用法:
    python generate.py --prompt "Hello, world!" --max-tokens 20
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
    """加载 safetensors 权重文件"""
    from safetensors import safe_open
    import torch

    print(f"Loading weights from {weights_path}...")
    weights = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if tensor.dtype in [torch.bfloat16, torch.float16]:
                weights[key] = tensor.float().numpy()
            elif hasattr(torch, 'float8_e4m3fn') and tensor.dtype == torch.float8_e4m3fn:
                weights[key] = tensor.float().numpy()
            elif hasattr(torch, 'float8_e5m2') and tensor.dtype == torch.float8_e5m2:
                weights[key] = tensor.float().numpy()
            else:
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
        layer_weights["gate_weight"] = all_weights[f"{prefix}.gate_weight"]
        layer_weights["up_weight"] = all_weights[f"{prefix}.up_weight"]
        layer_weights["down_weight"] = all_weights[f"{prefix}.down_weight"]
    else:
        layer_weights["router_weight"] = all_weights[f"{prefix}.router_weight"]
        if f"{prefix}.expert_gate_weights" in all_weights:
            layer_weights["expert_gate_weights"] = all_weights[f"{prefix}.expert_gate_weights"]
            layer_weights["expert_up_weights"] = all_weights[f"{prefix}.expert_up_weights"]
            layer_weights["expert_down_weights"] = all_weights[f"{prefix}.expert_down_weights"]
        else:
            gate_list, up_list, down_list = [], [], []
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


def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sample_token(logits, temperature=1.0, top_k=50):
    """Sample a token from logits"""
    if temperature == 0:
        return np.argmax(logits)

    logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = -np.inf

    probs = softmax(logits)
    return np.random.choice(len(probs), p=probs)


def main():
    parser = argparse.ArgumentParser(description="MiMo-V2-Flash 文本生成")
    parser.add_argument(
        "--weights-path",
        type=str,
        default="tmp_mimo_weights/mimo_weights.safetensors",
        help="权重文件路径",
    )
    parser.add_argument("--prompt", type=str, default="Hello", help="输入提示词")
    parser.add_argument("--max-tokens", type=int, default=20, help="最大生成 token 数")
    parser.add_argument("--num-layers", type=int, default=None, help="使用的层数 (默认全部48层)")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k 采样")

    args = parser.parse_args()

    # 检查权重文件
    if not os.path.exists(args.weights_path):
        print(f"Error: 权重文件不存在: {args.weights_path}")
        return

    # 加载 tokenizer
    print("加载 tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("XiaomiMiMo/MiMo-V2-Flash", trust_remote_code=True)
    except Exception as e:
        print(f"无法加载 tokenizer: {e}")
        print("请确保安装了 transformers: uv pip install transformers")
        return

    # 加载权重
    all_weights = load_weights_safetensors(args.weights_path)

    # 检测专家数量
    if "layers.1.expert_gate_weights" in all_weights:
        num_experts = all_weights["layers.1.expert_gate_weights"].shape[0]
    else:
        num_experts = 0
        while f"layers.1.expert.{num_experts}.gate_weight" in all_weights:
            num_experts += 1

    # 检测层数
    actual_num_layers = 0
    while f"layers.{actual_num_layers}.q_weight" in all_weights:
        actual_num_layers += 1

    num_layers = args.num_layers if args.num_layers else actual_num_layers
    num_layers = min(num_layers, actual_num_layers)

    print(f"模型配置: {num_layers} 层, {num_experts} 专家/层")

    # 创建配置
    config = MiMoConfig(
        num_hidden_layers=num_layers,
        max_model_len=512,
        max_batch_size=1,
        num_routed_experts=256,
    )

    # 预计算 RoPE
    max_seq_len = 512
    cos_full, sin_full = compute_cos_sin_partial(
        max_seq_len, config.head_dim, config.rotary_dim, config.rope_theta_full
    )
    cos_swa, sin_swa = compute_cos_sin_partial(
        max_seq_len, config.head_dim, config.rotary_dim, config.rope_theta_swa
    )

    # Token embedding 和 LM head
    tok_embedding = all_weights["tok_embedding"]
    lm_head_weight = all_weights["lm_head_weight"]
    norm_weight = all_weights["norm_weight"]

    # Tokenize 输入
    print(f"\n输入: {args.prompt}")
    input_ids = tokenizer.encode(args.prompt, return_tensors="np")[0]
    print(f"Token IDs: {input_ids.tolist()}")

    # 预创建层
    print("\n预加载层权重...")
    layers = []
    for layer_idx in range(num_layers):
        layer_weights = extract_layer_weights(all_weights, layer_idx, config)
        is_full = config.is_full_attention_layer(layer_idx)
        cos = cos_full if is_full else cos_swa
        sin = sin_full if is_full else sin_swa
        layer = MiMoLayerCPU(layer_idx, config, layer_weights, cos, sin)
        layers.append(layer)
    print(f"已加载 {len(layers)} 层")

    # 生成循环
    print("\n" + "=" * 60)
    print("开始生成...")
    print("=" * 60)

    generated_tokens = []
    current_ids = input_ids.copy()

    start_time = time.time()

    for step in range(args.max_tokens):
        seq_len = len(current_ids)

        # Token embedding
        hidden_states = token_embedding(tok_embedding, current_ids.reshape(1, -1))
        hidden_states = hidden_states.reshape(seq_len, config.hidden_size)

        # Forward through all layers
        for layer in layers:
            hidden_states = layer.forward(hidden_states.astype(np.float32))

        # Final LayerNorm
        hidden_states = rmsnorm(hidden_states, norm_weight, config.rms_norm_eps)

        # 只取最后一个 token 的 hidden state
        last_hidden = hidden_states[-1]  # [hidden_size]

        # LM head
        logits = last_hidden @ lm_head_weight.T  # [vocab_size]

        # 采样下一个 token
        next_token = sample_token(logits, args.temperature, args.top_k)
        generated_tokens.append(next_token)
        current_ids = np.append(current_ids, next_token)

        # 解码并打印
        decoded = tokenizer.decode([next_token])
        print(decoded, end="", flush=True)

        # 检查 EOS
        if next_token == tokenizer.eos_token_id:
            break

    elapsed = time.time() - start_time
    print("\n")
    print("=" * 60)
    print(f"生成完成!")
    print("=" * 60)
    print(f"\n完整输出: {tokenizer.decode(current_ids)}")
    print(f"生成 {len(generated_tokens)} tokens, 耗时 {elapsed:.1f}s")
    print(f"速度: {len(generated_tokens) / elapsed:.2f} tokens/s")


if __name__ == "__main__":
    main()
