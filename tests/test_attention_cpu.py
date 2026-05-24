"""Tests for the CPU attention ops.

Mirrors ``tests/test_attention_cuda.py``.  GPU-specific tests skip on
CPU-only hosts; CPU tests run on every host.
"""

from __future__ import annotations

import platform

import pytest

from mirage.attention.amx_sdpa import AMXSDPAAttention
from mirage.attention.types import AttentionKind, AttentionShape
from mirage.hardware import SAPPHIRE_RAPIDS, DType, Vendor

linux_only = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="AMX detection via /proc/cpuinfo is Linux-only",
)


# --- AMXSDPAAttention -------------------------------------------------------


def test_amx_sdpa_available() -> None:
    """SDPA on CPU is the always-available floor when torch is importable."""
    op = AMXSDPAAttention()
    assert op.available is True
    assert op.name == "amx-sdpa"


def test_amx_sdpa_supports_expected_dtypes() -> None:
    """BF16 / FP16 / FP32 / INT8 all supported on the full kind set."""
    op = AMXSDPAAttention()
    full = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64, kind=AttentionKind.FULL
    )
    causal = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64, kind=AttentionKind.CAUSAL
    )
    for dtype in (DType.FP32, DType.FP16, DType.BF16, DType.INT8):
        assert op.supports(full, dtype) is True
        assert op.supports(causal, dtype) is True


def test_amx_sdpa_rejects_unsupported_dtype() -> None:
    """FP8 has no CPU ISA today, so the op disqualifies itself."""
    op = AMXSDPAAttention()
    shape = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64, kind=AttentionKind.FULL
    )
    assert op.supports(shape, DType.FP8_E4M3) is False
    assert op.supports(shape, DType.FP8_E5M2) is False


def test_amx_sdpa_matches_torch_reference() -> None:
    """Output equals ``torch.nn.functional.scaled_dot_product_attention``."""
    import torch

    op = AMXSDPAAttention()
    torch.manual_seed(0)
    q = torch.randn(2, 4, 64, 32, dtype=torch.float32)
    k = torch.randn(2, 4, 64, 32, dtype=torch.float32)
    v = torch.randn(2, 4, 64, 32, dtype=torch.float32)
    out = op(q, k, v, causal=False, scale=None)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_amx_sdpa_causal_matches_reference() -> None:
    """Causal mask propagates through correctly."""
    import torch

    op = AMXSDPAAttention()
    torch.manual_seed(1)
    q = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    k = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    v = torch.randn(1, 2, 16, 32, dtype=torch.float32)
    out = op(q, k, v, causal=True, scale=None)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_amx_sdpa_bf16_roundtrip() -> None:
    """BF16 path runs end-to-end (oneDNN auto-dispatches to AMX on SPR+)."""
    import torch

    op = AMXSDPAAttention()
    torch.manual_seed(2)
    q = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    k = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    out = op(q, k, v, causal=False, scale=None)
    assert out.dtype == torch.bfloat16
    assert out.shape == q.shape


# --- AMX flash attention (custom kernel) -------------------------------------


@linux_only
def test_amx_flash_unavailable_without_kernel_module() -> None:
    """When the C++ extension is not built, ``available`` is False cleanly."""
    from mirage.attention.amx_flash import AMXFlashAttention

    op = AMXFlashAttention()
    # If the kernel _native module isn't built (the common case in CI),
    # the op must declare itself unavailable rather than throwing on import.
    if not op.available:
        assert op._import_error is not None
    # If the kernel IS built (only on a dev host that ran `make kernels-cpu`),
    # the op is callable and produces correct output for a small shape.
    else:
        import torch

        torch.manual_seed(3)
        q = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16)
        k = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16)
        out = op(q, k, v, causal=False, scale=None)
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        # AMX BF16 has lower precision than the reference SDPA path; assert
        # a relaxed bound matching typical FA-2 vs SDPA tolerances.
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


# --- IPEX flash attention ---------------------------------------------------


def test_ipex_flash_handles_missing_install() -> None:
    """When IPEX is not installed, ``available`` is False — no crash."""
    from mirage.attention.ipex_flash import IPEXFlashAttention

    op = IPEXFlashAttention()
    # The op must construct cleanly whether IPEX is present or not.  We do
    # not assert one way or the other on the result of ``available``; it's
    # environment-dependent.
    assert isinstance(op.available, bool)


# --- Registry routing -------------------------------------------------------


def test_registry_intel_branch_returns_supported_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """On Intel arch with AMX env unset, SDPA is the selected op."""
    monkeypatch.delenv("MIRAGE_AMX_ATTENTION", raising=False)
    from mirage.attention.registry import select_attention_op

    shape = AttentionShape(
        batch=1, heads=4, seq_len_q=128, seq_len_kv=128, head_dim=64, kind=AttentionKind.FULL
    )
    op = select_attention_op(SAPPHIRE_RAPIDS, shape, DType.BF16)
    # When the AMX flash kernel isn't built, the floor wins.  We accept
    # either "amx-sdpa" or the custom kernel name — both are valid INTEL
    # branch outcomes.
    assert op.name in ("amx-sdpa", "amx-bf16-flash", "ipex-flash")


def test_registry_unknown_intel_arch_still_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Generic AVX-512 host still gets a valid Intel-branch op."""
    monkeypatch.delenv("MIRAGE_AMX_ATTENTION", raising=False)
    from mirage.attention.registry import select_attention_op
    from mirage.hardware import DeviceArch

    skylake_avx512 = DeviceArch(Vendor.INTEL, "avx512", "Generic AVX-512")
    shape = AttentionShape(
        batch=1, heads=4, seq_len_q=64, seq_len_kv=64, head_dim=64, kind=AttentionKind.FULL
    )
    op = select_attention_op(skylake_avx512, shape, DType.BF16)
    assert op.name in ("amx-sdpa", "naive-sdpa")
