"""Bridge Mirage's FP8 attention into diffusers' attention dispatcher.

The Cosmos diffusers pipeline routes attention through
``diffusers.models.attention_dispatch.dispatch_attention_fn``, which only
consults its own ``_AttentionBackendRegistry`` — Mirage's
``select_attention_op`` is never called from that path. This module registers
a new backend ``"mirage_fp8"`` with the diffusers dispatcher; the backend
function dispatches to :class:`mirage.attention.FP8TritonAttention` when the
problem shape qualifies (head_dim in the kernel's compiled set, long enough
sequence, self-attention) and falls back to native SDPA otherwise.

Registration happens on import; the dispatcher still uses ``"native"`` unless
the user explicitly selects ``"mirage_fp8"`` (either via the
``DIFFUSERS_ATTN_BACKEND`` env var, the ``attention_backend(...)`` context
manager, or by calling :func:`activate_mirage_fp8_backend` from a Mirage
engine's ``load()``). Importing this module is therefore zero-cost in the
default code path.

See ``docs/BUILD_LOG.md`` F19 / Phase 2.5 — the kernel itself is unchanged
(``kernels/triton_kernels/fp8_flash_attn.py``); only the seam is new.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


logger = logging.getLogger(__name__)


#: The string handle for the Mirage FP8 backend in diffusers' registry.
MIRAGE_FP8_BACKEND_NAME = "mirage_fp8"

#: Sequence-length crossover. Below this we fall back to native SDPA — the
#: FP8 kernel pays a dispatch + quantization tax that only amortizes at long
#: S. The crossover was measured in Session 9 (``scripts/bench_fp8.py``):
#: S=4096 -> 0.53x (FP8 loses), S=8192 -> 1.92x (FP8 wins). 4096 is the
#: practical threshold for routing — at S=4k SDPA→aotriton is faster.
_FP8_MIN_SEQ_LEN = 4096

#: head_dim values the Triton kernel is compiled for. Mirrors
#: ``FP8TritonAttention._SUPPORTED_HEAD_DIMS`` — duplicated here to avoid an
#: import of the op when only routing logic is needed.
_FP8_SUPPORTED_HEAD_DIMS = frozenset({32, 64, 128, 256})


def _route_should_use_fp8(query: Any, key: Any, value: Any) -> bool:
    """Decide whether to dispatch a single call to the FP8 kernel.

    Diffusers passes ``(batch, seq_len, num_heads, head_dim)``. The FP8 kernel
    operates on ``(batch, num_heads, seq_len, head_dim)`` and requires:

    - head_dim in the compiled tile set,
    - ``query.shape == key.shape == value.shape`` (self-attention; the kernel
      asserts this — cross-attention falls back),
    - seq_len above the empirical crossover (~4 k tokens),
    - BF16/FP16 dtype (the kernel casts inputs to FP8 internally).

    Returning ``False`` is the safe default — the caller falls back to native
    SDPA.
    """
    import torch

    if query.dtype not in (torch.bfloat16, torch.float16):
        return False
    # Diffusers layout: (B, S, H, D). The kernel requires Q/K/V same shape;
    # cross-attention has seq_len_kv != seq_len_q so it falls out here.
    if query.shape != key.shape or query.shape != value.shape:
        return False
    head_dim = query.shape[-1]
    if head_dim not in _FP8_SUPPORTED_HEAD_DIMS:
        return False
    seq_len = query.shape[-3]
    return bool(seq_len >= _FP8_MIN_SEQ_LEN)


def _native_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> torch.Tensor:
    """Replicate diffusers' ``_native_attention`` (BHSD-permuted SDPA).

    Diffusers' ``dispatch_attention_fn`` always passes ``(B, S, H, D)`` to
    backends; SDPA wants ``(B, H, S, D)``. The native path permutes in/out;
    we do the same here for the fallback so the dispatch table semantics are
    preserved regardless of which branch we take.
    """
    import torch

    if (
        attn_mask is not None
        and attn_mask.ndim == 2
        and attn_mask.shape[0] == query.shape[0]
        and attn_mask.shape[1] == key.shape[1]
    ):
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)

    q_bhsd = query.permute(0, 2, 1, 3)
    k_bhsd = key.permute(0, 2, 1, 3)
    v_bhsd = value.permute(0, 2, 1, 3)
    out = torch.nn.functional.scaled_dot_product_attention(
        query=q_bhsd,
        key=k_bhsd,
        value=v_bhsd,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    return out.permute(0, 2, 1, 3)


def _mirage_fp8_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Any = None,
) -> torch.Tensor:
    """The diffusers-registered ``"mirage_fp8"`` backend function.

    Routes to the Triton FP8 fused FA-2 kernel when the call qualifies (long
    self-attention at a compiled head_dim, BF16/FP16, no mask), and to a
    native-SDPA fallback otherwise. The signature mirrors
    ``_native_attention`` — every kwarg ``dispatch_attention_fn`` passes is
    accepted (some ignored, with documented rationale).

    Layout convention follows diffusers: ``(B, S, H, D)`` in, ``(B, S, H, D)``
    out. The FP8 kernel internally expects ``(B, H, S, D)``; this function
    transposes on entry and on exit.
    """
    import torch

    if return_lse:
        raise ValueError("mirage_fp8 backend does not support return_lse=True.")
    if _parallel_config is not None:
        # Context parallelism is not in scope for Phase 2.5; fall back so we
        # never silently break the multi-GPU path (Mirage is single-GPU today
        # but the backend should still degrade gracefully).
        return _native_fallback(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    # Routing decision: use FP8 only when the shape clears the crossover and
    # the kernel can accept it. Otherwise SDPA wins.
    if (
        attn_mask is not None
        or dropout_p != 0.0
        or enable_gqa
        or not _route_should_use_fp8(query, key, value)
    ):
        return _native_fallback(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    # Lazy import — the op holds a Triton compile bag and shouldn't cost
    # anything when the backend isn't active. Cache on the function attribute
    # so repeated calls don't re-instantiate.
    op = getattr(_mirage_fp8_attention, "_op", None)
    if op is None:
        from mirage.attention.fp8_triton import FP8TritonAttention

        op = FP8TritonAttention()
        _mirage_fp8_attention._op = op  # type: ignore[attr-defined]
    if not op.available:
        # The op imported but its kernel isn't usable (e.g. triton broken).
        # Fall through to SDPA — better silent perf regression than a crash.
        return _native_fallback(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    # Diffusers (B, S, H, D) -> kernel (B, H, S, D), then back out.
    q_bhsd = query.permute(0, 2, 1, 3).contiguous()
    k_bhsd = key.permute(0, 2, 1, 3).contiguous()
    v_bhsd = value.permute(0, 2, 1, 3).contiguous()
    out_bhsd = op(q_bhsd, k_bhsd, v_bhsd, causal=is_causal, scale=scale)
    # Numerical guardrail: if FP8 quantization tripped (NaN/Inf from a
    # degenerate amax or an unstable softmax), fall back to SDPA for *this*
    # call rather than poisoning the whole denoising step.
    if not torch.isfinite(out_bhsd).all():
        logger.warning(
            "mirage_fp8: non-finite output at shape %s; falling back to native SDPA",
            tuple(query.shape),
        )
        return _native_fallback(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    return out_bhsd.permute(0, 2, 1, 3).type_as(query)


def _ensure_backend_name_registered() -> Any:
    """Make ``"mirage_fp8"`` resolvable through ``AttentionBackendName(...)``.

    The diffusers enum is sealed (``str, Enum``), but its dispatch only ever
    reads from ``_member_map_`` / ``_value2member_map_``. Injecting a new
    member into those maps lets ``AttentionBackendName("mirage_fp8")`` resolve
    cleanly without subclassing the enum (which would require shadowing every
    downstream import). This is the same pattern diffusers' own
    ``attention_backend(...)`` context manager assumes — it just constructs
    the enum from a string and indexes the registry.
    """
    from diffusers.models.attention_dispatch import AttentionBackendName

    enum_value = MIRAGE_FP8_BACKEND_NAME
    enum_name = "MIRAGE_FP8"
    if enum_value in AttentionBackendName._value2member_map_:
        return AttentionBackendName._value2member_map_[enum_value]
    # New str-Enum member. Constructing via ``str.__new__`` bypasses Enum's
    # frozen-after-creation guard; we then patch the maps the lookup uses.
    new_member = str.__new__(AttentionBackendName, enum_value)
    new_member._name_ = enum_name
    new_member._value_ = enum_value
    AttentionBackendName._member_map_[enum_name] = new_member
    AttentionBackendName._value2member_map_[enum_value] = new_member
    if enum_name not in AttentionBackendName._member_names_:
        AttentionBackendName._member_names_.append(enum_name)
    return new_member


def register_mirage_fp8_backend() -> Any:
    """Register the ``"mirage_fp8"`` backend with diffusers' dispatcher.

    Idempotent — calling it twice is a no-op. Returns the new
    ``AttentionBackendName`` member so callers can pass it to the
    ``attention_backend(...)`` context manager directly.
    """
    import inspect

    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    member = _ensure_backend_name_registered()
    if member in _AttentionBackendRegistry._backends:
        return member

    _AttentionBackendRegistry._backends[member] = _mirage_fp8_attention
    _AttentionBackendRegistry._constraints[member] = []
    # Mirror the same "supported arg names" introspection the decorator does.
    _AttentionBackendRegistry._supported_arg_names[member] = set(
        inspect.signature(_mirage_fp8_attention).parameters.keys()
    )
    return member


def activate_mirage_fp8_backend() -> bool:
    """Make ``"mirage_fp8"`` the active backend for the diffusers dispatcher.

    Returns ``True`` if the activation happened (and a prior backend was
    replaced), ``False`` if the backend was already active (idempotent
    no-op).

    Callers are expected to gate this on ``MIRAGE_FP8_ATTENTION``; see
    :func:`maybe_activate_from_env`. The function deliberately mutates global
    state — the diffusers dispatcher is a process-wide singleton, so there is
    no per-call alternative short of wrapping every diffusers call in
    ``attention_backend("mirage_fp8")``.
    """
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    member = register_mirage_fp8_backend()
    if _AttentionBackendRegistry._active_backend == member:
        return False
    _AttentionBackendRegistry.set_active_backend(member)
    return True


def maybe_activate_from_env() -> bool:
    """Activate the Mirage FP8 backend iff ``MIRAGE_FP8_ATTENTION`` is set.

    Bridges Mirage's user-facing env var into diffusers' dispatcher. Accepts
    the same truthy values as ``mirage.attention.registry``
    (``1/true/on/triton/scaled_mm``). Returns whether activation happened.

    This is called from ``CosmosEngine.load()`` so the env var has the same
    surface in Cosmos as it does in ``select_attention_op`` — flipping a
    single variable now affects both the explicit-Mirage path (registry) and
    the diffusers pipeline path (dispatcher).
    """
    fp8_env = os.environ.get("MIRAGE_FP8_ATTENTION", "").lower()
    if fp8_env not in ("1", "true", "on", "triton", "scaled_mm"):
        return False
    # The diffusers dispatcher only has the fused-Triton path — the
    # ``scaled_mm`` variant lives only in ``mirage.attention.registry``. We
    # still honour the env var here so callers don't see asymmetric routing,
    # but log if the user picked a backend the bridge can't deliver.
    if fp8_env == "scaled_mm":
        logger.info(
            "mirage_fp8: MIRAGE_FP8_ATTENTION=scaled_mm — the diffusers bridge "
            "only exposes the fused Triton path; routing through Triton."
        )
    return activate_mirage_fp8_backend()


# Register on import so the backend handle is available the moment someone
# selects it (via env var or ``attention_backend``). The dispatcher remains
# on NATIVE unless explicitly switched, so this is free in the default path.
try:
    register_mirage_fp8_backend()
except Exception as exc:  # pragma: no cover - environment dependent
    # If diffusers isn't installed or its internals changed shape, log and
    # carry on — Mirage core does not depend on this bridge.
    logger.debug("mirage_fp8: registration deferred (%s: %s)", type(exc).__name__, exc)


__all__ = [
    "MIRAGE_FP8_BACKEND_NAME",
    "activate_mirage_fp8_backend",
    "maybe_activate_from_env",
    "register_mirage_fp8_backend",
]
