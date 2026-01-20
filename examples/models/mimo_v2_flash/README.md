# MiMo-V2-Flash for nkipy/Trainium2

MiMo-V2-Flash (309B 总参数, 15B 活跃参数) MoE 模型在 AWS Trainium2 上的实现。

## 模型架构

| 参数 | 值 |
|------|-----|
| hidden_size | 4096 |
| num_layers | 48 |
| num_attention_heads | 64 |
| num_kv_heads (full/swa) | 4 / 8 |
| head_dim (Q/K) | 192 |
| v_head_dim (V) | 128 |
| num_experts | 256 |
| experts_per_token | 8 |
| moe_intermediate_size | 2048 |
| sliding_window | 128 |
| partial_rotary_factor | 33.4% |

### 关键特性

- **混合注意力**: 全局注意力 (层 0, 5, 11, 17, 23, 29, 35, 41, 47) + 滑动窗口注意力 (SWA)
- **异构头维度**: Q/K 使用 192 维, V 使用 128 维
- **部分 RoPE**: 仅 33.4% 的维度使用旋转位置编码 (64/192)
- **Dense + MoE**: 层 0 使用 Dense FFN, 层 1-47 使用 MoE
- **Sigmoid 路由**: 使用 sigmoid (非 softmax) 进行专家选择
- **Expert Parallelism**: 支持 32+ 设备分布式专家计算

## 文件结构

```
mimo_v2_flash/
├── __init__.py                 # 模块导出
├── config.py                   # 模型配置 (MiMoConfig)
├── model.py                    # 主模型类
├── layer.py                    # Transformer 层 (MiMoLayer, MiMoLayerCPU)
├── prepare_weights.py          # HuggingFace 权重转换
├── run_example.py              # 运行示例
├── run_with_weights.py         # 使用真实权重测试 (CPU)
├── generate.py                 # 文本生成
├── test_kernels.py             # 单元测试
└── kernels/
    ├── rmsnorm.py              # RMSNorm
    ├── softmax.py              # Softmax
    ├── rope_partial.py         # 部分 RoPE (33.4%)
    ├── token_embedding.py      # Token 嵌入
    ├── attention_full.py       # 全局注意力
    ├── attention_swa.py        # 滑动窗口注意力
    ├── ffn.py                  # Dense FFN (层 0)
    ├── moe_router.py           # Sigmoid 路由 + Top-8
    ├── moe_expert.py           # 专家 FFN
    └── moe_block.py            # MoE 块 (含 EP 支持)
```

## 快速开始

### 1. 安装依赖

```bash
uv pip install torch transformers safetensors huggingface_hub
```

### 2. 下载权重 (专家子集)

```bash
cd examples/models/mimo_v2_flash

# 下载 8 专家子集 (~18GB)
python run_example.py --mode download --expert-subset 0 1 2 3 4 5 6 7
```

### 3. 运行推理测试

```bash
# 使用真实权重测试 48 层
python run_with_weights.py --weights-path tmp_mimo_weights/mimo_weights.safetensors --num-layers 48

# 快速测试 2 层
python run_with_weights.py --num-layers 2 --seq-len 16
```

### 4. 文本生成

```bash
# 使用 4 层进行快速测试 (非真实输出)
python generate.py --prompt "Hello" --max-tokens 10 --num-layers 4

# 使用更多层 (更慢但更真实)
python generate.py --prompt "What is AI?" --max-tokens 5 --num-layers 12
```

## 详细用法

### 运行单元测试

```bash
# 运行 kernel 测试 (无需权重)
python run_example.py --mode test
```

### 使用随机权重测试

```bash
# 测试 8 专家子集，2 层
python run_example.py --mode random --num-experts 8 --num-layers 2

# 完整 256 专家模拟 (慢)
python run_example.py --mode random --num-experts 256 --num-layers 2
```

### Python API

```python
from config import MiMoConfig
from layer import MiMoLayerCPU
from kernels.rope_partial import compute_cos_sin_partial
from kernels.token_embedding import token_embedding
from kernels.rmsnorm import rmsnorm
import numpy as np

# 创建配置
config = MiMoConfig(num_hidden_layers=2)

# 预计算 RoPE
cos, sin = compute_cos_sin_partial(128, config.head_dim, config.rotary_dim, config.rope_theta_full)

# 加载权重...
# layer_weights = extract_layer_weights(all_weights, 0, config)

# 创建层
# layer = MiMoLayerCPU(0, config, layer_weights, cos, sin)

# 推理
# hidden_states = layer.forward(hidden_states)
```

## 开发阶段

### Phase 1: 核心实现 (已完成)

- [x] 所有 kernel 实现
- [x] CPU 测试 (MiMoLayerCPU)
- [x] HuggingFace 权重转换
- [x] 专家子集支持
- [x] 48 层完整推理验证

### Phase 2: Trainium2 验证 (进行中)

**已验证 Kernel**:
| Kernel | 状态 | 最大差异 | 运行时间 |
|--------|------|---------|---------|
| RMSNorm | ✓ 通过 | 0.000024 | 0.46ms |
| Token Embedding | ✓ 通过 | 0.000000 | 0.59ms |
| Full Attention | ✓ 通过 | 0.000010 | 2.96ms |
| Dense FFN | ✓ 通过 | 0.000010 | 4.06ms |
| MoE Block | ✗ 需重构 | - | - |

**待完成**:
- [ ] MoE Block 重构 (替换 `np.where`, `np.any` 等为 nkipy 支持的操作)
- [ ] SWA Attention 验证
- [ ] 完整层验证

### Phase 3: 完整模型

- [ ] KV Cache 实现
- [ ] 增量解码优化

### Phase 4: Expert Parallelism

- [ ] 256 专家分布在 32+ 设备
- [ ] `all_to_all` token 分发
- [ ] 负载均衡优化

### Phase 5: 生产优化

- [ ] FP8 量化
- [ ] Kernel 融合
- [ ] 性能 profiling

## 硬件要求

| 配置 | 设备数 | 每设备专家数 | 内存需求 |
|------|--------|-------------|----------|
| 开发测试 | 1 | 8 (子集) | ~20GB |
| 完整模型 | 32 | 8 | ~600GB |
| 优化配置 | 64 | 4 | ~600GB |

## 性能参考

在 CPU 上测试 (不代表 Trainium2 性能):

| 配置 | 层数 | 序列长度 | 耗时 |
|------|------|----------|------|
| 8 专家 | 2 | 16 | ~1.3s |
| 8 专家 | 12 | 16 | ~9.5s |
| 8 专家 | 48 | 16 | ~37s |

## 注意事项

1. **专家子集**: 使用 modulo 映射将 256 专家路由映射到可用专家数，输出不代表真实模型行为
2. **数值稳定性**: 中间值可能很大，但 RMSNorm 会归一化最终输出
3. **无 KV Cache**: 当前实现每次重新计算整个序列，生成速度慢
4. **权重格式**: 支持 FP8/BF16/FP16 自动转换为 FP32

## 参考

- [MiMo-V2-Flash HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)
- [nkipy 文档](https://github.com/your-repo/nkipy)

## License

同 nkipy 项目许可证。
