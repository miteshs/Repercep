#!/usr/bin/env python3
"""Run one Cosmos-Predict-7B Text2World generation on the Mirage runtime (MI300X).

This is the end-to-end "does it actually run" script: it loads Cosmos-Predict-7B
through Mirage's CosmosEngine, generates a clip, writes an mp4, and prints a
JSON result line.

    .venv/bin/python scripts/run_cosmos.py --frames 17 --steps 8     # fast smoke
    .venv/bin/python scripts/run_cosmos.py                           # full 121-frame
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Cosmos-Predict-7B on Mirage / MI300X")
    parser.add_argument(
        "--prompt",
        default=(
            "A sleek autonomous delivery robot rolls along a sunlit city "
            "sidewalk past glass storefronts, smooth forward motion, "
            "photorealistic, high detail."
        ),
    )
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmark-results/cosmos_sample.mp4")
    parser.add_argument(
        "--guardrail", action="store_true", help="enable the Cosmos safety guardrail"
    )
    parser.add_argument(
        "--native-loop",
        action="store_true",
        help="use Mirage's native denoising loop (CFG batching)",
    )
    parser.add_argument(
        "--cache-skip-every",
        type=int,
        default=0,
        help="step-skip caching: full DiT forward only every Nth step (native loop only)",
    )
    args = parser.parse_args()

    import torch

    from mirage.backend.registry import select_backend
    from mirage.models.cosmos import CosmosConfig, CosmosEngine, GuardrailError
    from mirage.runtime.types import GenerationParams, GenerationRequest

    backend = select_backend()
    device = backend.devices()[0]
    print(
        f"[mirage] backend={backend.name}  device={device.name}  "
        f"{device.total_memory_gib:.0f} GiB  arch={device.arch}",
        flush=True,
    )

    engine = CosmosEngine(
        backend,
        CosmosConfig(
            enable_guardrail=args.guardrail,
            use_native_loop=args.native_loop,
            cache_skip_every=args.cache_skip_every,
        ),
    )
    print(
        f"[mirage] loading Cosmos-Predict-7B (~38 GB), guardrail={args.guardrail} ...",
        flush=True,
    )
    t0 = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - t0
    print(f"[mirage] model loaded in {load_s:.1f}s", flush=True)

    request = GenerationRequest(
        prompt=args.prompt,
        params=GenerationParams(
            num_frames=args.frames,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance,
            seed=args.seed,
        ),
    )
    print(
        f"[mirage] generating {args.frames} frames @ {args.width}x{args.height}, "
        f"{args.steps} steps, seed {args.seed} ...",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    try:
        frames = list(engine.generate(request))
    except GuardrailError as exc:
        print(f"[mirage] GUARDRAIL BLOCKED: {exc}", flush=True)
        return 0
    gen_s = time.perf_counter() - t1
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = _save_video([f.pixels for f in frames], out, fps=24)

    summary = {
        "model": engine.model_name,
        "device": device.name,
        "frames": len(frames),
        "resolution": f"{args.width}x{args.height}",
        "steps": args.steps,
        "load_seconds": round(load_s, 1),
        "generate_seconds": round(gen_s, 1),
        "frames_per_second": round(len(frames) / gen_s, 3) if gen_s else 0.0,
        "seconds_per_step": round(gen_s / args.steps, 2) if args.steps else 0.0,
        "peak_hbm_gib": round(peak_gib, 1),
        "output": str(saved),
    }
    print("[mirage] RESULT " + json.dumps(summary), flush=True)
    return 0


def _save_video(frame_tensors: list, path: Path, fps: int = 24) -> Path:
    """Write frames to mp4 via imageio; fall back to PNG frames via PIL."""
    import numpy as np

    stacked = np.stack([t.numpy() for t in frame_tensors])  # (T, H, W, 3) uint8
    try:
        import imageio.v3 as iio

        iio.imwrite(path, stacked, fps=fps, codec="libx264")
        return path
    except Exception as exc:
        print(f"[mirage] mp4 encode unavailable ({exc}); writing PNG frames", flush=True)
        from PIL import Image

        frame_dir = path.with_suffix("")
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(stacked):
            Image.fromarray(frame).save(frame_dir / f"frame_{i:04d}.png")
        return frame_dir


if __name__ == "__main__":
    raise SystemExit(main())
