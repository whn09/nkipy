"""Unit tests for MiMo-V2-Flash kernels

These tests verify individual kernel correctness by comparing with
reference implementations or expected behavior.
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from config import MiMoConfig
from kernels.rmsnorm import rmsnorm
from kernels.softmax import softmax
from kernels.rope_partial import compute_cos_sin_partial, rope_partial
from kernels.token_embedding import token_embedding
from kernels.moe_router import moe_router_sigmoid, sigmoid
from kernels.moe_expert import moe_expert_ffn, silu_kernel


def test_rmsnorm():
    """Test RMSNorm kernel."""
    print("Testing RMSNorm...")

    batch_size, seq_len, hidden_size = 2, 8, 4096
    x = np.random.randn(batch_size * seq_len, hidden_size).astype(np.float32)
    weight = np.ones(hidden_size, dtype=np.float32)
    eps = 1e-6

    output = rmsnorm(x, weight, eps)

    # Check output shape
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"

    # Check that output is normalized (variance should be close to 1)
    variance = np.mean(np.square(output), axis=-1)
    assert np.allclose(variance, 1.0, atol=0.1), f"Variance not normalized: {variance}"

    print("  RMSNorm: PASSED")


def test_softmax():
    """Test softmax kernel."""
    print("Testing Softmax...")

    x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    output = softmax(x)

    # Check that output sums to 1
    sums = np.sum(output, axis=-1)
    assert np.allclose(sums, 1.0), f"Softmax doesn't sum to 1: {sums}"

    # Check that output is non-negative
    assert np.all(output >= 0), "Softmax has negative values"

    # Check expected values for uniform input
    expected_uniform = np.array([1/3, 1/3, 1/3], dtype=np.float32)
    assert np.allclose(output[1], expected_uniform, atol=1e-5)

    print("  Softmax: PASSED")


def test_partial_rope():
    """Test partial RoPE kernel."""
    print("Testing Partial RoPE...")

    config = MiMoConfig()
    max_len = 32
    head_dim = config.head_dim  # 192
    rotary_dim = config.rotary_dim  # 64

    # Compute cos/sin tables
    cos, sin = compute_cos_sin_partial(max_len, head_dim, rotary_dim, theta=10000.0)

    assert cos.shape == (max_len, rotary_dim // 2)
    assert sin.shape == (max_len, rotary_dim // 2)

    # Test RoPE application
    batch_size = 1
    seq_len = 8
    n_heads = 64
    n_kv_heads = 4

    xq = np.random.randn(batch_size, seq_len, n_heads, head_dim).astype(np.float32)
    xk = np.random.randn(batch_size, seq_len, n_kv_heads, head_dim).astype(np.float32)

    # Only use cos/sin for the sequence length we need
    cos_seq = cos[:seq_len]
    sin_seq = sin[:seq_len]

    xq_out, xk_out = rope_partial(xq, xk, cos_seq, sin_seq, rotary_dim)

    # Check shapes are preserved
    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape

    # Check that non-rotary dimensions are unchanged
    assert np.allclose(xq_out[:, :, :, rotary_dim:], xq[:, :, :, rotary_dim:])
    assert np.allclose(xk_out[:, :, :, rotary_dim:], xk[:, :, :, rotary_dim:])

    # Check that rotary dimensions are modified
    assert not np.allclose(xq_out[:, :, :, :rotary_dim], xq[:, :, :, :rotary_dim])

    print("  Partial RoPE: PASSED")


def test_token_embedding():
    """Test token embedding kernel."""
    print("Testing Token Embedding...")

    vocab_size = 1000
    hidden_size = 4096
    batch_size = 2
    seq_len = 8

    embedding_table = np.random.randn(vocab_size, hidden_size).astype(np.float32)
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.uint32)

    output = token_embedding(embedding_table, token_ids)

    assert output.shape == (batch_size, seq_len, hidden_size)

    # Verify specific lookups
    for b in range(batch_size):
        for s in range(seq_len):
            token_id = token_ids[b, s]
            expected = embedding_table[token_id]
            actual = output[b, s]
            assert np.allclose(expected, actual)

    print("  Token Embedding: PASSED")


def test_sigmoid():
    """Test sigmoid function."""
    print("Testing Sigmoid...")

    x = np.array([-100, -1, 0, 1, 100], dtype=np.float32)
    output = sigmoid(x)

    expected = np.array([0.0, 0.2689, 0.5, 0.7311, 1.0], dtype=np.float32)
    assert np.allclose(output, expected, atol=1e-3)

    print("  Sigmoid: PASSED")


def test_moe_router():
    """Test MoE router kernel."""
    print("Testing MoE Router...")

    batch_seq = 4
    hidden_size = 4096
    num_experts = 256
    top_k = 8

    hidden_states = np.random.randn(batch_seq, hidden_size).astype(np.float32)
    router_weight = np.random.randn(hidden_size, num_experts).astype(np.float32) * 0.02

    topk_indices, topk_weights, router_logits = moe_router_sigmoid(
        hidden_states, router_weight, num_experts, top_k
    )

    # Check shapes
    assert topk_indices.shape == (batch_seq, top_k)
    assert topk_weights.shape == (batch_seq, top_k)
    assert router_logits.shape == (batch_seq, num_experts)

    # Check that indices are valid
    assert np.all(topk_indices >= 0)
    assert np.all(topk_indices < num_experts)

    # Check that weights sum to 1
    weight_sums = np.sum(topk_weights, axis=-1)
    assert np.allclose(weight_sums, 1.0)

    # Check that indices are unique per token
    for i in range(batch_seq):
        assert len(set(topk_indices[i])) == top_k

    print("  MoE Router: PASSED")


def test_silu():
    """Test SiLU activation."""
    print("Testing SiLU...")

    x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    output = silu_kernel(x)

    # SiLU(x) = x * sigmoid(x)
    expected = x * (1.0 / (1.0 + np.exp(-x)))
    assert np.allclose(output, expected, atol=1e-5)

    # Check that SiLU(0) = 0
    assert np.isclose(output[2], 0.0)

    print("  SiLU: PASSED")


def test_expert_ffn():
    """Test expert FFN kernel."""
    print("Testing Expert FFN...")

    num_tokens = 4
    hidden_size = 4096
    intermediate_size = 2048

    x = np.random.randn(num_tokens, hidden_size).astype(np.float32)
    gate_weight = np.random.randn(hidden_size, intermediate_size).astype(np.float32) * 0.02
    up_weight = np.random.randn(hidden_size, intermediate_size).astype(np.float32) * 0.02
    down_weight = np.random.randn(intermediate_size, hidden_size).astype(np.float32) * 0.02

    output = moe_expert_ffn(x, gate_weight, up_weight, down_weight)

    # Check shape
    assert output.shape == (num_tokens, hidden_size)

    # Manual computation for verification
    gate = silu_kernel(x @ gate_weight)
    up = x @ up_weight
    expected = (gate * up) @ down_weight
    assert np.allclose(output, expected, atol=1e-4)

    print("  Expert FFN: PASSED")


def test_config():
    """Test MiMoConfig."""
    print("Testing Config...")

    config = MiMoConfig()

    # Check key parameters
    assert config.hidden_size == 4096
    assert config.num_hidden_layers == 48
    assert config.num_attention_heads == 64
    assert config.num_kv_heads_full == 4
    assert config.num_kv_heads_swa == 8
    assert config.head_dim == 192
    assert config.v_head_dim == 128
    assert config.num_routed_experts == 256
    assert config.experts_per_tok == 8

    # Check computed properties
    assert config.rotary_dim == 64  # floor(192 * 0.334) rounded to even
    assert config.q_proj_size == 64 * 192
    assert config.k_proj_size_full == 4 * 192
    assert config.v_proj_size_full == 4 * 128
    assert config.o_proj_size == 64 * 128
    assert config.gqa_ratio_full == 16
    assert config.gqa_ratio_swa == 8

    # Check full attention layer detection
    # Actual pattern: [0, 5, 11, 17, 23, 29, 35, 41, 47]
    assert config.is_full_attention_layer(0) == True
    assert config.is_full_attention_layer(1) == False
    assert config.is_full_attention_layer(5) == True
    assert config.is_full_attention_layer(6) == False

    print("  Config: PASSED")


def run_all_tests():
    """Run all kernel tests."""
    print("=" * 60)
    print("MiMo-V2-Flash Kernel Tests")
    print("=" * 60)

    test_config()
    test_rmsnorm()
    test_softmax()
    test_sigmoid()
    test_silu()
    test_partial_rope()
    test_token_embedding()
    test_moe_router()
    test_expert_ffn()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
