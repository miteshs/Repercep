"""Attention op selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mirage.attention.naive import NaiveAttention
from mirage.attention.rocm_flash import ROCmFlashAttention
from mirage.hardware import DeviceArch, DType, Vendor

if TYPE_CHECKING:
    from mirage.attention.protocol import AttentionOp
    from mirage.attention.types import AttentionShape


def select_attention_op(arch: DeviceArch, shape: AttentionShape, dtype: DType) -> AttentionOp:
    """Pick the fastest attention op that supports ``(shape, dtype)`` on ``arch``.

    Order: the vendor flash kernel first, then ``NaiveAttention`` as the
    always-correct floor.  Selection is by problem shape, so a model can use
    the flash kernel for its DiT blocks and fall back to naive for an unusual
    head_dim without any caller-side branching.
    """
    candidates: list[AttentionOp] = []
    if arch.vendor is Vendor.AMD:
        candidates.append(ROCmFlashAttention())
    candidates.append(NaiveAttention())

    for op in candidates:
        if op.supports(shape, dtype):
            return op
    return NaiveAttention()
