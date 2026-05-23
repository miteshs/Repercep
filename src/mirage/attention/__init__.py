"""Attention primitives for Mirage.

World models lean hard on attention: DiT blocks use full bidirectional spatial
attention, temporal layers use causal attention, and video models often use
NATTEN-style neighborhood attention.  Each op implements the ``AttentionOp``
Protocol; ``select_attention_op`` picks the fastest one that supports a given
problem shape on a given architecture.

On MI300X the Hopper-only FlashAttention-3 kernels do not apply (no wgmma/TMA);
the equivalent is the Composable-Kernel ``flash-attn`` build or aotriton-backed
SDPA.  See ADR-0002.
"""

from __future__ import annotations

from mirage.attention.fp8_scaled_mm import FP8ScaledMMAttention
from mirage.attention.fp8_triton import FP8TritonAttention
from mirage.attention.protocol import AttentionOp
from mirage.attention.registry import select_attention_op
from mirage.attention.types import AttentionKind, AttentionShape

__all__ = [
    "AttentionKind",
    "AttentionOp",
    "AttentionShape",
    "FP8ScaledMMAttention",
    "FP8TritonAttention",
    "select_attention_op",
]
