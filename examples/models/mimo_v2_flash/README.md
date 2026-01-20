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

- **混合注意力**: 全局注意力 + 滑动窗口注意力 (SWA)
- **异构头维度**: Q/K 使用 192 维, V 使用 128 维
- **部分 RoPE**: 仅 33.4% 的维度使用旋转位置编码
- **Sigmoid 路由**: 使用 sigmoid (非 softmax) 进行专家选择
- **Expert Parallelism**: 支持 32+ 设备分布式专家计算

## 文件结构

```
mimo_v2_flash/
├── __init__.py                 # 模块导出
├── config.py                   # 模型配置 (MiMoConfig)
├── model.py                    # 主模型类
├── layer.py                    # Transformer 层
├── prepare_weights.py          # HuggingFace 权重转换
├── run_example.py              # 运行示例
├── test_kernels.py             # 单元测试
└── kernels/
    ├── rmsnorm.py              # RMSNorm
    ├── softmax.py              # Softmax
    ├── rope_partial.py         # 部分 RoPE (33.4%)
    ├── token_embedding.py      # Token 嵌入
    ├── attention_full.py       # 全局注意力
    ├── attention_swa.py        # 滑动窗口注意力
    ├── moe_router.py           # Sigmoid 路由 + Top-8
    ├── moe_expert.py           # 专家 FFN
    └── moe_block.py            # MoE 块 (含 EP 支持)
```

## 快速开始

### 1. 运行单元测试

```bash
cd examples/models/mimo_v2_flash

# 运行 kernel 测试 (无需 Trainium)
python run_example.py --mode test
```

### 2. 使用随机权重测试

```bash
# 测试 8 专家子集，2 层
python run_example.py --mode random --num-experts 8 --num-layers 2
```

### 3. 下载 HuggingFace 权重

```bash
# 安装依赖
pip install torch transformers safetensors accelerate

# 下载专家子集 (用于测试)
python run_example.py --mode download --expert-subset 0 1 2 3 4 5 6 7

# 下载完整权重 (需要大量内存)
python run_example.py --mode download
```

### 4. 运行推理

```python
from config import MiMoConfig
from model import MiMoV2FlashModel
from prepare_weights import load_weights

# 加载配置和权重
config = MiMoConfig()
weights = load_weights("tmp_mimo_weights/mimo_weights.safetensors", config)

# 初始化模型 (使用 8 专家子集)
model = MiMoV2FlashModel(weights, config, expert_subset=list(range(8)))

# 推理
import numpy as np
input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.uint32)
output = model.forward(input_ids)
```

## 开发阶段

### Phase 1: 单设备测试 (当前)

- 8 专家子集
- 验证各 kernel 正确性
- CPU/单 Trainium2 测试

```bash
python run_example.py --mode random --num-experts 8
```

### Phase 2: 完整模型 (专家子集)

- 48 层完整 Transformer
- 使用专家子集减少内存

### Phase 3: Expert Parallelism

- 256 专家分布在 32+ 设备
- 使用 `all_to_all` 进行 token 分发

```python
config = MiMoConfig(num_devices=32)  # 每设备 8 专家
```

### Phase 4: 优化

- KV Cache 实现
- FP8 量化 (可选)
- Kernel 融合优化

## 硬件要求

| 配置 | 设备数 | 每设备专家数 |
|------|--------|-------------|
| 开发测试 | 1 | 8 (子集) |
| 完整模型 | 32 | 8 |
| 优化配置 | 64 | 4 |

## 注意事项

1. **内存**: 完整 256 专家需要约 600GB 显存
2. **权重下载**: HuggingFace 模型约 600GB
3. **编译时间**: 首次 kernel 编译可能需要数分钟

## 参考

- [MiMo-V2-Flash HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)
- [nkipy 文档](https://github.com/your-repo/nkipy)

## License

同 nkipy 项目许可证。
