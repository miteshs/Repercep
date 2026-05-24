"""Tests for ``mirage.runtime.quantize``."""

from __future__ import annotations

import pytest


def test_quantize_linear_basic() -> None:
    """Quantize a small weight, dequant, and assert within INT8 rounding tolerance."""
    import torch

    from mirage.runtime.quantize import dequantize_linear, quantize_linear_symmetric

    torch.manual_seed(0)
    weight = torch.randn(64, 128) * 2.0  # not normalised; tests scale calc
    q = quantize_linear_symmetric(weight)
    # Layout invariants.
    assert q.qweight.shape == weight.shape
    assert q.qweight.dtype is torch.int8
    assert q.scale.shape == (weight.shape[0],)
    assert q.scale.dtype is torch.float32
    assert q.bias is None
    # Dequant should reconstruct within ~scale/2 absolute error per element.
    reconstructed = dequantize_linear(q)
    err = (reconstructed - weight).abs()
    per_row_scale = q.scale.unsqueeze(1).expand_as(err)
    # INT8 symmetric rounding guarantees max error <= scale/2 per element.
    assert (err <= per_row_scale).all(), f"max err / scale = {(err / per_row_scale).max().item():.3f}"


def test_quantize_linear_preserves_bias() -> None:
    """Bias passes through unchanged (we don't quantize it — see docstring)."""
    import torch

    from mirage.runtime.quantize import quantize_linear_symmetric

    weight = torch.randn(8, 16)
    bias = torch.randn(8, dtype=torch.bfloat16)
    q = quantize_linear_symmetric(weight, bias)
    assert q.bias is not None
    assert torch.equal(q.bias, bias)
    assert q.bias.dtype is torch.bfloat16


def test_quantize_linear_handles_zero_row() -> None:
    """All-zero row gets a clamped scale, not a NaN."""
    import torch

    from mirage.runtime.quantize import quantize_linear_symmetric

    weight = torch.zeros(4, 8)
    weight[1] = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125])
    q = quantize_linear_symmetric(weight)
    assert torch.isfinite(q.scale).all()
    assert torch.isfinite(q.qweight.to(torch.float32)).all()
    # Row 1 had non-zero values; its scale must be sensible (>0).
    assert q.scale[1] > 0
    # Rows 0, 2, 3 are zero; their qweight rows must be all zero.
    assert (q.qweight[0] == 0).all()
    assert (q.qweight[2] == 0).all()
    assert (q.qweight[3] == 0).all()


def test_quantize_linear_rejects_wrong_dim() -> None:
    """Only 2-D weights are supported."""
    import torch

    from mirage.runtime.quantize import quantize_linear_symmetric

    with pytest.raises(ValueError, match="expected 2-D"):
        quantize_linear_symmetric(torch.randn(8))


def test_quantize_module_linears_filters_by_name() -> None:
    """``name_filter`` selects a substring match against dotted module names."""
    import torch

    from mirage.runtime.quantize import quantize_module_linears

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dit = torch.nn.Sequential(
                torch.nn.Linear(8, 8),
                torch.nn.Linear(8, 8),
            )
            self.vae = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
            )

    m = Toy()
    # Without a filter, every Linear is quantized.
    all_q = quantize_module_linears(m)
    assert set(all_q.keys()) == {"dit.0", "dit.1", "vae.0"}
    # With ``name_filter="dit"``, only DiT linears.
    dit_q = quantize_module_linears(m, name_filter="dit")
    assert set(dit_q.keys()) == {"dit.0", "dit.1"}


def test_quantize_linear_bfloat16_input() -> None:
    """Quant + dequant of a BF16 weight stays within BF16-then-quant tolerance."""
    import torch

    from mirage.runtime.quantize import dequantize_linear, quantize_linear_symmetric

    torch.manual_seed(1)
    weight = (torch.randn(16, 32) * 1.5).to(torch.bfloat16)
    q = quantize_linear_symmetric(weight)
    # Reconstructed in BF16 to match weight dtype for comparison.
    reconstructed = dequantize_linear(q, dtype=torch.bfloat16)
    # Combined error: BF16 truncation + symmetric INT8 rounding.  Relaxed
    # tolerance (BF16 mantissa is 7 bits; INT8 symmetric is roughly 1/256
    # of dynamic range) — together ~4 % rel for typical input distributions.
    err = (reconstructed.to(torch.float32) - weight.to(torch.float32)).abs()
    rel = err / weight.to(torch.float32).abs().clamp(min=1e-3)
    assert rel.median().item() < 0.04
