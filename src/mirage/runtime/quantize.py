"""Weight quantization helpers — per-channel symmetric INT8.

The motivation for landing this on the CPU side first is the AMX_INT8
matmul lever: TDPBSSD on Sapphire Rapids does INT8 · INT8 -> INT32 at
twice the throughput of TDPBF16PS (BF16 · BF16 -> FP32) — *if* the
weights are pre-quantized.  This module owns the BF16/FP16 -> INT8
quantization path with per-output-channel scales, mirroring the
per-channel symmetric scheme used in oneDNN / IPEX's smooth_quant.

GPU sibling: ``torchao``-driven INT8 quant of the DiT linears.  We
deliberately implement this here rather than calling ``torchao`` because
(a) ``torchao`` requires a CUDA wheel that does not co-install cleanly
with our CPU torch wheel on the same host, and (b) the AMX kernels need
the quantized tensors in a specific tile-friendly layout that ``torchao``
doesn't produce.

The quantization is *static* (offline) — weights are quantized once at
load time, scales are stored alongside.  Activations are quantized
per-token on the fly inside the AMX kernel (per F23 the dynamic activation
quant cost is well below the matmul savings at the Cosmos DiT shape).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class QuantizedLinear:
    """A 2-D weight matrix quantized per output-channel to INT8 (symmetric).

    The original ``weight: (out_features, in_features)`` is decomposed into
    ``qweight: (out_features, in_features) int8`` and ``scale:
    (out_features,) float32``: ``weight ≈ qweight.to(float) * scale[:,
    None]``.  Bias (when present) stays in BF16/FP32 — quantizing bias to
    INT8 isn't worth the 0.001 % memory savings.

    Per-channel-symmetric is the right choice here for the same reason it
    is on GPU: each output channel of a Linear has its own scale, so a
    layer's dynamic range doesn't get collapsed into one outlier-driven
    scale.  Asymmetric would buy us nothing on these distributions
    (post-LayerNorm activations are already centered).
    """

    qweight: torch.Tensor  # (out, in) int8
    scale: torch.Tensor    # (out,) float32
    bias: torch.Tensor | None  # (out,) original dtype or None


def quantize_linear_symmetric(
    weight: torch.Tensor, bias: torch.Tensor | None = None
) -> QuantizedLinear:
    """Per-output-channel symmetric INT8 quantization of a 2-D weight.

    ``weight`` is expected ``(out_features, in_features)``, any float dtype.
    Returns a :class:`QuantizedLinear` with qweight + scale tensors live on
    the same device as ``weight``.

    Algorithm:
        for each output channel i:
            scale_i = max(|weight[i]|) / 127.0
            qweight[i] = round(weight[i] / scale_i).clamp(-128, 127).to(int8)

    Numerical caveat: weights with a single outlier per row produce a
    too-coarse scale for the rest of the row.  Mitigation (Session 16+):
    pre-process with ``smooth_quant``-style activation/weight rebalancing.
    """
    import torch

    if weight.dim() != 2:
        raise ValueError(f"expected 2-D weight; got shape {tuple(weight.shape)}")
    # Work in float32 for the scale computation regardless of input dtype —
    # at BF16 the per-row max can lose precision when the row's max
    # magnitude has lots of significant bits.
    w_fp32 = weight.detach().to(torch.float32)
    # Per-row max-abs.  Use ``unbiased=False`` semantics implicitly via the
    # max + abs ops; clamp to a small floor to avoid div-by-zero on
    # all-zero rows (which are rare but show up in pruned Linears).
    row_max = w_fp32.abs().amax(dim=1).clamp(min=1e-8)
    scale = (row_max / 127.0).to(torch.float32)
    qweight = (w_fp32 / scale.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)
    return QuantizedLinear(qweight=qweight, scale=scale, bias=bias)


def dequantize_linear(q: QuantizedLinear, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Reconstruct the weight matrix for reference/correctness checks.

    Useful in tests and in the kernel-fallback path when the AMX_INT8
    kernel isn't available (e.g. on pre-SPR hardware).  Defaults to
    float32 for fidelity; pass ``dtype=torch.bfloat16`` when comparing
    against an AMX BF16 reference.
    """
    import torch

    out = q.qweight.to(torch.float32) * q.scale.unsqueeze(1)
    if dtype is not None:
        out = out.to(dtype)
    return out


def quantize_module_linears(
    module: torch.nn.Module,
    *,
    name_filter: str | None = None,
) -> dict[str, QuantizedLinear]:
    """Walk ``module`` and quantize every ``nn.Linear`` whose name matches.

    Returns a dict mapping the dotted module path to its
    :class:`QuantizedLinear`.  Does NOT modify ``module`` — the caller is
    expected to wrap the linears with an INT8-aware forward (e.g. via a
    ``QuantizedLinearModule`` from a future PR).  Keeping quantization
    pure here lets the test suite assert exact values without side
    effects.

    The ``name_filter`` is a substring match against dotted names; pass
    ``"dit"`` to quantize only the diffusion transformer blocks, leaving
    the VAE in BF16 (the right default for Cosmos — the VAE is
    quantization-sensitive).
    """
    import torch

    out: dict[str, QuantizedLinear] = {}
    for name, sub in module.named_modules():
        if not isinstance(sub, torch.nn.Linear):
            continue
        if name_filter is not None and name_filter not in name:
            continue
        out[name] = quantize_linear_symmetric(sub.weight, sub.bias)
    return out
