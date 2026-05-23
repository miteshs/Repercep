"""Attention op selection."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from mirage.attention.fp8_scaled_mm import FP8ScaledMMAttention
from mirage.attention.fp8_triton import FP8TritonAttention
from mirage.attention.naive import NaiveAttention
from mirage.attention.rocm_flash import ROCmFlashAttention
from mirage.hardware import DeviceArch, DType, Vendor

if TYPE_CHECKING:
    from mirage.attention.protocol import AttentionOp
    from mirage.attention.types import AttentionShape


# FP8 is opt-in via env var until the perf bench across all Cosmos shapes
# bears out the universal win.  At long S (~32k+) the Triton FP8 flash
# kernel beats SDPA→aotriton; at short S the dispatch + quantization overhead
# wins.  See docs/OPTIMIZATION.md and docs/BUILD_LOG.md for the measured
# crossover.  Set MIRAGE_FP8_ATTENTION=1 to enable, =triton to force the
# fused path, =scaled_mm to force the unfused path.
_FP8_ENV = os.environ.get("MIRAGE_FP8_ATTENTION", "").lower()
_FP8_ENABLED = _FP8_ENV in ("1", "true", "on", "triton", "scaled_mm")


def select_attention_op(arch: DeviceArch, shape: AttentionShape, dtype: DType) -> AttentionOp:
    """Pick the fastest attention op that supports ``(shape, dtype)`` on ``arch``.

    Order: FP8 fused (when enabled and shape qualifies) → vendor flash → naive.
    Selection is by problem shape, so a model can use the FP8 kernel for its
    big DiT blocks, the flash kernel for medium ones, and the naive floor for
    an unusual head_dim — no caller-side branching.
    """
    candidates: list[AttentionOp] = []
    if arch.vendor is Vendor.AMD:
        if _FP8_ENABLED and _FP8_ENV != "scaled_mm":
            # The fused Triton kernel — preferred when it qualifies.  Its
            # supports() already gates on seq_len >= 128 etc.
            candidates.append(FP8TritonAttention())
        if _FP8_ENABLED and _FP8_ENV != "triton":
            candidates.append(FP8ScaledMMAttention())
        candidates.append(ROCmFlashAttention())
    candidates.append(NaiveAttention())

    for op in candidates:
        if op.supports(shape, dtype):
            return op
    return NaiveAttention()
