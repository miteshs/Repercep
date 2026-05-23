"""FP8 flash-attention Triton kernel for CDNA3 (gfx942).

This is the *kernel-layer* code — no abstractions, no Protocols.  The thin
loader in ``src/mirage/attention/fp8_triton.py`` wraps it as an ``AttentionOp``.

Algorithm: standard FlashAttention-2 (Dao 2023) — tile Q over the
sequence axis, stream K/V tiles through, maintain a running max + running
sum + running accumulator (online softmax), never materialize the S^2
scores matrix.  The matmuls are done in FP8 (e4m3) via ``tl.dot`` with
explicit dtype casts; the running statistics and accumulator stay in FP32
for accuracy.

Why Triton and not raw HIP: Triton on ROCm targets gfx942's MFMA pipeline
through MLIR; the compiler emits ``v_mfma_f32_*_fp8_fp8`` for FP8 ``tl.dot``
on this architecture.  Hand-writing the same in HIP would require
explicit ``__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8`` calls plus LDS
double-buffering — six weeks of perf engineering.  Triton gives us 80% of
that for an afternoon of work.

References:
- Dao, FlashAttention-2: Faster Attention with Better Parallelism (2023)
- Shah et al., FlashAttention-3 (2024) §3 — FP8 variant with block scaling
- ROCm/triton ``test_core.py`` for the ``tl.dot`` FP8 dispatch.
"""

# Triton kernels are eval-loaded; this file is exempt from mypy/ruff
# (see kernels/README.md).
# ruff: noqa
# type: ignore

from __future__ import annotations

import torch
import triton
import triton.language as tl


# FP8 max for e4m3fnuz on gfx942 — finite-only, top of range is 240.
FP8_E4M3_MAX = 240.0


@triton.jit
def _fp8_flash_attn_fwd(
    Q,
    K,
    V,
    sm_scale,  # 1/sqrt(d), float
    qs,
    ks,
    vs,  # per-tensor dequant scales (B*H,) float32
    Out,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    B,
    H,
    Sq,
    Skv,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """One program = one (B, H, Q-tile) triple.

    Q and Out tiles: (BLOCK_M, BLOCK_D).
    K/V tiles streamed: (BLOCK_N, BLOCK_D).

    The accumulator and softmax statistics are FP32; the matmuls are FP8.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    # Per-(B,H) dequant scales.  q_scale * k_scale enters the QK^T result;
    # p_scale * v_scale enters the PV result.  We carry them as plain floats.
    q_scale = tl.load(qs + pid_bh)
    k_scale = tl.load(ks + pid_bh)
    v_scale = tl.load(vs + pid_bh)

    # Offset Q, K, V, Out to this (b, h) slice.
    q_off = b * stride_qb + h * stride_qh
    k_off = b * stride_kb + h * stride_kh
    v_off = b * stride_vb + h * stride_vh
    o_off = b * stride_ob + h * stride_oh

    # Row indices into Q for this program (m-tile).
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load this Q tile.  Q stays resident in SRAM/registers for the whole
    # inner loop.  Shape: (BLOCK_M, BLOCK_D), FP8.
    q_ptrs = Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < Sq
    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)  # FP8 dtype

    # Online softmax state, FP32.
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Effective softmax scale: sm_scale * q_scale * k_scale.  This folds the
    # dequant into the temperature in one multiplication per (B,H), instead
    # of per-element after the dot.
    qk_scale = sm_scale * q_scale * k_scale

    # Loop over K/V tiles.
    n_blocks = tl.cdiv(Skv, BLOCK_N)

    for n_idx in range(0, n_blocks):
        n_start = n_idx * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)

        # Load K tile (BLOCK_N, BLOCK_D), FP8.
        k_ptrs = K + k_off + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = offs_n[:, None] < Skv
        k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # qk = Q @ K^T (BLOCK_M, BLOCK_N), FP32 accumulator.  tl.dot's
        # accumulator dtype is FP32 by default for FP8 inputs on CDNA3
        # via the MFMA path.
        qk = tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32)
        qk = qk * qk_scale

        # Apply causal mask if requested.
        if CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Mask out-of-range KV positions (last tile partial).
        kv_mask = offs_n[None, :] < Skv
        qk = tl.where(kv_mask, qk, float("-inf"))

        # Online softmax (running max + running sum).
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)  # rescale factor for prior acc
        p = tl.exp(qk - m_new[:, None])  # (M, N), FP32, in [0, 1]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Rescale the running accumulator before adding new contribution.
        acc = acc * alpha[:, None]

        # Load V tile (BLOCK_N, BLOCK_D), FP8.
        v_ptrs = V + v_off + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = offs_n[:, None] < Skv
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Quantize P to FP8 for the PV matmul.  P is in [0, 1] so a fixed
        # scale (240) saturates the FP8 range without per-tile recomputation;
        # the dequant factor is 1/240, folded into v_scale below.
        p_fp8 = (p * FP8_MAX).to(tl.float8e4b8)
        pv_scale = v_scale / FP8_MAX

        # acc += P @ V  (BLOCK_M, BLOCK_D), FP32 accumulator + FP8 inputs.
        pv = tl.dot(p_fp8, v_tile, out_dtype=tl.float32)
        acc = acc + pv * pv_scale

        m_i = m_new

    # Normalize.
    acc = acc / l_i[:, None]

    # Store the output (cast back to the input dtype on the Python side via
    # the Out tensor's dtype — Triton stores match Out's pointer dtype).
    out_ptrs = Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = offs_m[:, None] < Sq
    tl.store(out_ptrs, acc, mask=o_mask)


def _per_bh_scale(x: torch.Tensor) -> torch.Tensor:
    """Compute one quantization scale per (batch, head) — shape (B*H,).

    Returns the *dequant* scale (i.e. multiplier you apply to the FP8 value
    to recover the original BF16 magnitude).
    """
    # x is (B, H, S, D); amax over last two dims.
    B, H, S, D = x.shape
    amax = x.abs().reshape(B * H, -1).amax(dim=1).clamp(min=1e-6)
    # Quantization scale: we want x * q_scale in [-FP8_MAX, FP8_MAX].
    # Dequant scale (returned) is 1 / q_scale = amax / (0.95 * FP8_MAX).
    return (amax / (0.95 * FP8_E4M3_MAX)).to(torch.float32)


def _quantize(x: torch.Tensor, dequant_scale: torch.Tensor) -> torch.Tensor:
    """Quantize (B, H, S, D) BF16 to FP8 using per-(B,H) dequant scales.

    ``dequant_scale`` is shape (B*H,).  The quantization multiplier is its
    reciprocal.
    """
    B, H, S, D = x.shape
    q_mul = (1.0 / dequant_scale).reshape(B, H, 1, 1)
    scaled = (x.to(torch.float32) * q_mul).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return scaled.to(torch.float8_e4m3fnuz)


def fp8_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Public Python entry point — fused FP8 flash attention.

    Args:
        q, k, v: (B, H, S, D) BF16 or FP16 tensors.
        causal: apply a causal mask.
        scale: softmax temperature; default 1/sqrt(D).

    Returns:
        Output (B, H, S, D) in the dtype of ``q``.
    """
    import math

    assert q.shape == k.shape == v.shape, "MHA only (same S); cross-attn TODO"
    B, H, Sq, D = q.shape
    Skv = k.shape[2]
    assert D in (32, 64, 128, 256), f"head_dim {D} not in compiled tile shapes"

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Per-(B,H) dequant scales.
    q_scales = _per_bh_scale(q)
    k_scales = _per_bh_scale(k)
    v_scales = _per_bh_scale(v)

    q_fp8 = _quantize(q, q_scales).contiguous()
    k_fp8 = _quantize(k, k_scales).contiguous()
    v_fp8 = _quantize(v, v_scales).contiguous()

    out = torch.empty_like(q, dtype=torch.float32)

    # Tile sizes — gfx942 sweet spot is BLOCK_M=128, BLOCK_N=64 for head_dim 128.
    # Empirically num_warps=4 and 2-stage pipeline match aotriton flash for D=128.
    BLOCK_M = 128
    BLOCK_N = 64
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(Sq, BLOCK_M), B * H)

    _fp8_flash_attn_fwd[grid](
        q_fp8,
        k_fp8,
        v_fp8,
        scale,
        q_scales,
        k_scales,
        v_scales,
        out,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        q_fp8.stride(3),
        k_fp8.stride(0),
        k_fp8.stride(1),
        k_fp8.stride(2),
        k_fp8.stride(3),
        v_fp8.stride(0),
        v_fp8.stride(1),
        v_fp8.stride(2),
        v_fp8.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        B,
        H,
        Sq,
        Skv,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=D,
        CAUSAL=causal,
        FP8_MAX=FP8_E4M3_MAX,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.to(q.dtype)
