"""Regression tests for ``mirage.runtime.denoise``.

These are structural tests — they don't need a GPU or model — but they pin the
invariants that recently caused observable runtime OOMs.
"""

from __future__ import annotations

import inspect

from mirage.runtime import denoise


def test_denoise_cosmos_video_runs_under_no_grad_gate() -> None:
    """Without ``torch.inference_mode()`` (or ``no_grad``) wrapping the body,
    the autograd graph holds activations across all diffusion steps. At 17f /
    8 steps that fits in HBM (~28 GiB peak observed); at 121f / 36 steps it
    OOMs at ~189 GiB on a 192 GiB MI300X, which is the regression this guard
    prevents. Diffusers' own ``CosmosTextToWorldPipeline.__call__`` has the
    same decorator — this is parity, not a workaround.
    """
    src = inspect.getsource(denoise.denoise_cosmos_video)
    assert "inference_mode" in src or "no_grad" in src, (
        "denoise_cosmos_video must wrap its body in torch.inference_mode() or "
        "torch.no_grad(); without it activation memory grows ~linearly in "
        "step count and 121f / 36 steps OOMs."
    )
