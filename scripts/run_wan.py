#!/usr/bin/env python3
"""Run one Wan-2.2 T2V generation on the Mirage runtime (MI300X).

This is the Wan analogue of ``scripts/run_cosmos.py`` — it loads the Wan-2.2
MoE A14B text-to-video pipeline through Mirage's :class:`WanEngine`, generates
a clip, writes an mp4, and prints a JSON RESULT line.

    .venv/bin/python scripts/run_wan.py --frames 17 --steps 8       # fast smoke
    .venv/bin/python scripts/run_wan.py --frames 81 --steps 40      # reference

The default Wan-2.2 T2V-A14B reference config is 81 frames at 1280x720 with 40
inference steps (5 s at 16 FPS). The smaller TI2V-5B variant is available via
``--small`` for VRAM-constrained smoke tests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Wan-2.2 T2V on Mirage / MI300X")
    parser.add_argument(
        "--prompt",
        default=(
            "A sleek autonomous delivery robot rolls along a sunlit city "
            "sidewalk past glass storefronts, smooth forward motion, "
            "photorealistic, high detail."
        ),
    )
    parser.add_argument(
        "--negative-prompt",
        default=(
            "blurry, low quality, distorted, overexposed, static, "
            "subtitles, watermark, deformed limbs"
        ),
    )
    # Wan-2.2 T2V-A14B reference defaults: 81 frames at 1280x720, 40 steps.
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument(
        "--guidance-2",
        type=float,
        default=3.0,
        help="MoE second-stage guidance scale (A14B only)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="benchmark-results/wan_sample.mp4")
    parser.add_argument(
        "--small",
        action="store_true",
        help="use the Wan-2.2 TI2V-5B variant (~10 GiB BF16) instead of T2V-A14B",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the DiT transformer(s) — slow first run, faster steady-state",
    )
    args = parser.parse_args()

    import torch

    from mirage.backend.registry import select_backend
    from mirage.models.wan import NATIVE_FPS, SMALL_REPO, WanConfig, WanEngine
    from mirage.runtime.types import GenerationParams, GenerationRequest

    backend = select_backend()
    device = backend.devices()[0]
    print(
        f"[mirage] backend={backend.name}  device={device.name}  "
        f"{device.total_memory_gib:.0f} GiB  arch={device.arch}",
        flush=True,
    )

    config = WanConfig(
        guidance_scale_2=args.guidance_2,
        compile_transformer=args.compile,
    )
    if args.small:
        config.repo_id = SMALL_REPO
    engine = WanEngine(backend, config)

    size_hint = "~10 GiB BF16" if args.small else "~52 GiB BF16"
    print(f"[mirage] loading {config.repo_id} ({size_hint}) ...", flush=True)
    t0 = time.perf_counter()
    engine.load()
    load_s = time.perf_counter() - t0
    print(f"[mirage] model loaded in {load_s:.1f}s", flush=True)

    request = GenerationRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        params=GenerationParams(
            num_frames=args.frames,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance,
            fps=NATIVE_FPS,  # Wan was trained at 16 FPS.
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
    frames = list(engine.generate(request))
    gen_s = time.perf_counter() - t1
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = _save_video([f.pixels for f in frames], out, fps=NATIVE_FPS)

    summary = {
        "model": engine.model_name,
        "repo": config.repo_id,
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


def _save_video(frame_tensors: list, path: Path, fps: int = 16) -> Path:
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
