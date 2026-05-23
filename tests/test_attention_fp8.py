"""Tests for FP8 attention paths (Triton flash + scaled_mm)."""

from __future__ import annotations

import importlib.util

import pytest

from mirage.attention import AttentionKind, AttentionShape
from mirage.attention.fp8_scaled_mm import FP8ScaledMMAttention
from mirage.attention.fp8_triton import FP8TritonAttention
from mirage.attention.naive import NaiveAttention
from mirage.attention.protocol import AttentionOp
from mirage.hardware import DType

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_HAS_TRITON = importlib.util.find_spec("triton") is not None


def _gpu_or_skip() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no GPU on host")


def test_fp8_scaled_mm_satisfies_protocol() -> None:
    op = FP8ScaledMMAttention()
    assert isinstance(op, AttentionOp)
    assert op.name == "fp8-scaled-mm"


def test_fp8_triton_satisfies_protocol() -> None:
    op = FP8TritonAttention()
    assert isinstance(op, AttentionOp)
    assert op.name == "fp8-triton-flash"


def test_fp8_scaled_mm_rejects_unsupported_kinds() -> None:
    op = FP8ScaledMMAttention()
    if not op.available:
        pytest.skip("torch._scaled_mm + fp8_e4m3fnuz unavailable")
    # NEIGHBORHOOD requires a mask in the scores matrix — not supported today.
    shape = AttentionShape(
        batch=1,
        heads=8,
        seq_len_q=512,
        seq_len_kv=512,
        head_dim=128,
        kind=AttentionKind.NEIGHBORHOOD,
    )
    assert not op.supports(shape, DType.BF16)


def test_fp8_triton_requires_min_seq_len() -> None:
    op = FP8TritonAttention()
    if not op.available:
        pytest.skip("triton FP8 kernel unavailable")
    # Below the min length the supports() must decline.
    small = AttentionShape(batch=1, heads=2, seq_len_q=64, seq_len_kv=64, head_dim=64)
    assert not op.supports(small, DType.BF16)
    ok = AttentionShape(batch=1, heads=2, seq_len_q=256, seq_len_kv=256, head_dim=64)
    assert op.supports(ok, DType.BF16)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_fp8_scaled_mm_matches_sdpa_within_fp8_tolerance() -> None:
    import torch

    _gpu_or_skip()
    op = FP8ScaledMMAttention()
    if not op.available:
        pytest.skip("torch._scaled_mm + fp8_e4m3fnuz unavailable")

    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    b, h, s, d = 1, 4, 512, 128  # full set above _MIN_DIM and a multiple of 16
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    k = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    v = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)

    ref = NaiveAttention()(q, k, v)
    out = op(q, k, v)
    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    # FP8 tolerance: max abs diff is dominated by the e4m3 quantization
    # step; we allow up to ~10x typical magnitude as the per-element ceiling.
    ref_scale = ref.float().abs().mean().item() + 1e-6
    diff = (out.float() - ref.float()).abs()
    assert diff.mean().item() / ref_scale < 0.30, (
        f"FP8 scaled_mm mean rel error {diff.mean().item() / ref_scale:.3f} too large"
    )


@pytest.mark.skipif(not _HAS_TORCH or not _HAS_TRITON, reason="torch+triton required")
def test_fp8_triton_matches_sdpa_within_fp8_tolerance() -> None:
    import torch

    _gpu_or_skip()
    op = FP8TritonAttention()
    if not op.available:
        pytest.skip("triton FP8 flash kernel unavailable")

    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    b, h, s, d = 1, 4, 256, 64
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    k = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    v = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)

    ref = NaiveAttention()(q, k, v)
    out = op(q, k, v)
    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    ref_scale = ref.float().abs().mean().item() + 1e-6
    diff = (out.float() - ref.float()).abs()
    # FP8 fused attention: per-tile quantization keeps the rel error in the
    # 1-5% range across typical magnitudes; we use a generous 15% cap for CI
    # stability.
    assert diff.mean().item() / ref_scale < 0.15


@pytest.mark.skipif(not _HAS_TORCH or not _HAS_TRITON, reason="torch+triton required")
def test_fp8_triton_causal_matches_sdpa() -> None:
    """Causal mask must match SDPA causal output within FP8 noise."""
    import torch

    _gpu_or_skip()
    op = FP8TritonAttention()
    if not op.available:
        pytest.skip("triton FP8 flash kernel unavailable")

    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    b, h, s, d = 1, 4, 256, 64
    q = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    k = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16) / 8.0
    v = torch.randn(b, h, s, d, device=dev, dtype=torch.bfloat16)

    ref = NaiveAttention()(q, k, v, causal=True)
    out = op(q, k, v, causal=True)
    ref_scale = ref.float().abs().mean().item() + 1e-6
    diff = (out.float() - ref.float()).abs()
    assert diff.mean().item() / ref_scale < 0.15
