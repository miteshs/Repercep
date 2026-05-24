"""Tests for the NVIDIA attention ops.  Most are skipped without CUDA.

Sibling of ``test_attention.py`` (which exercises the AMD ops); follows the
same pattern of unconditional structural checks plus GPU-gated functional
checks.
"""

from __future__ import annotations

import importlib.util

import pytest

from mirage.attention import AttentionShape, select_attention_op
from mirage.attention.fp8_hopper_triton import FP8HopperTritonAttention
from mirage.attention.hopper_flash import HopperFlashAttention
from mirage.attention.naive import NaiveAttention
from mirage.attention.protocol import AttentionOp
from mirage.attention.transformer_engine import TransformerEngineAttention
from mirage.backend.cuda import CUDABackend
from mirage.hardware import H100, DType

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def test_hopper_flash_satisfies_protocol() -> None:
    assert isinstance(HopperFlashAttention(), AttentionOp)


def test_fp8_hopper_triton_satisfies_protocol() -> None:
    assert isinstance(FP8HopperTritonAttention(), AttentionOp)


def test_transformer_engine_satisfies_protocol() -> None:
    assert isinstance(TransformerEngineAttention(), AttentionOp)


def test_hopper_flash_op_name() -> None:
    # Stable name for diagnostic surfaces (mirage info, registry traces).
    assert HopperFlashAttention().name == "nvidia-flash"


def test_fp8_hopper_triton_op_name() -> None:
    assert FP8HopperTritonAttention().name == "fp8-hopper-triton-flash"


def test_transformer_engine_op_name() -> None:
    assert TransformerEngineAttention().name == "transformer-engine-fp8"


def test_select_hopper_returns_attention_op() -> None:
    # Without flash-attn, FP8 env, or TE installed, selection falls to naive.
    shape = AttentionShape(batch=1, heads=16, seq_len_q=1024, seq_len_kv=1024, head_dim=128)
    op = select_attention_op(H100, shape, DType.BF16)
    assert isinstance(op, AttentionOp)
    # Names that can appear when the FP8 env var is unset and flash-attn is
    # not built: TE is also unconditional in the FP8 branch when env is on,
    # so its name should not appear here.
    assert op.name in ("nvidia-flash", "naive-sdpa")


def test_fp8_hopper_triton_supports_requires_min_seqlen() -> None:
    # Even when the kernel itself is unavailable, supports() should be False
    # without it.  When importable, the min-seqlen gate applies.
    op = FP8HopperTritonAttention()
    short_shape = AttentionShape(batch=1, heads=8, seq_len_q=64, seq_len_kv=64, head_dim=64)
    assert op.supports(short_shape, DType.BF16) is False


def test_fp8_hopper_triton_supports_cross_attention_false() -> None:
    op = FP8HopperTritonAttention()
    cross = AttentionShape(batch=1, heads=8, seq_len_q=4096, seq_len_kv=2048, head_dim=128)
    # Self-attention only for now.
    assert op.supports(cross, DType.BF16) is False


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_naive_runs_on_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU on host")
    if not (torch.version.cuda is not None and not torch.version.hip):
        pytest.skip("not a CUDA host")
    dev = torch.device("cuda", 0)
    b, h, s, d = 2, 8, 256, 64
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
    out = NaiveAttention()(q, k, v, causal=True)
    assert out.shape == (b, h, s, d)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not CUDABackend().is_available(), reason="no CUDA GPU on host")
def test_hopper_flash_runs_when_built() -> None:
    # Skipped unless flash-attn is installed; structural gate on availability.
    op = HopperFlashAttention()
    if not op.available:
        pytest.skip("flash-attn not installed (see docs/COSMOS_ON_H100.md)")
    import torch

    dev = CUDABackend().torch_device(0)
    b, h, s, d = 1, 16, 512, 128
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = op(q, k, v)
    assert tuple(out.shape) == (b, h, s, d)
    assert torch.isfinite(out).all()
