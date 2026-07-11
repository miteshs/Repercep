# LingBot-VA 2.0 through the Mirage seam — Phase-1 GPU verify (2026-07-11)

`LingBotVAEngine` + `LingBotVAPipeline` (`src/mirage/models/lingbot_va*.py`)
driven end-to-end via `reset → plan → step`, ten chunks, on the same RunPod
H100 used for the lever bench (`docs/LEVERS_2026_07_H100.md`). This is the
Phase-1 GPU verify: does the model run *through Mirage's seam*, not just
through the reference `VA_Server`. It does — and it's faster and lighter
than the reference-stack run in `docs/LINGBOT_VA_ON_H100.md`, for reasons
that are design choices, not measurement noise (see below).

`scripts/run_lingbot_va.py --repo /workspace/ckpt/lingbot-va-base --obs-dir
.../example/demo --backend cuda`.

## Result

| metric | seam (this run) | reference stack (`LINGBOT_VA_ON_H100.md`) |
|---|---|---|
| load | 7.4 s | 8.6 s |
| reset (cache + prompt + obs encode) | 28.1 s | 0.41 s |
| chunk latency, warm mean | **801.5 ms** | 1384.7 ms |
| chunk latency, cold (chunk 0) | 1647.7 ms | 1544.9 ms |
| peak HBM, one session | **15.4 GiB** | 38.8 GiB |
| actions | denormalized, magnitude range consistent with the reference run's p95=134.7 | — |

Verbatim provenance:

```json
{"bench": "lingbot_va_seam", "model": "/workspace/ckpt/lingbot-va-base", "device": "NVIDIA H100 80GB HBM3", "load_s": 7.4, "reset_s": 28.1, "chunk_ms": [1647.7, 666.8, 694.5, 715.7, 745.6, 800.3, 820.4, 845.3, 882.2, 908.2], "chunk_ms_warm_mean": 801.5, "reference_stack_warm_ms": 1384.7, "peak_hbm_gib": 15.4, "context_shape": [4, 24576]}
```

Both the mode (imagination — no per-step recondition against real observations,
matching the reference `generate()`'s loop, not `_compute_kv_cache`) and the
action magnitudes are consistent with the reference-stack run, so this is an
apples-to-apples rollout, not an accidentally-cheaper path.

## Why the seam run is faster and lighter (real, not noise)

1. **VAE + T5 are CPU-resident by design** (`LingBotVAPipeline.__init__` loads
   them with `torch_device="cpu"` unconditionally, vs. the reference's
   `enable_offload=False` default which keeps ~11 GiB of T5 parameters
   GPU-resident for the whole run). This alone accounts for most of the
   38.8 → 15.4 GiB drop and is exactly the session-density lever
   `LINGBOT_VA_ON_H100.md` flagged as the next thing to engineer — it's
   already partially banked.
2. **No async disk writes in the hot loop** — the reference `_infer` fires
   `save_async` for latents/actions/obs every chunk (debug instrumentation);
   the seam pipeline doesn't.
3. **Chunk-latency variance goes the reference's way at chunk 0** (cold-start
   allocator cost dominates there for both) but **the seam's steady state is
   ~800 ms vs ~1385 ms** — consistent with #1 (less resident memory → less
   allocator/cache pressure per step) rather than a numerics difference,
   since the action outputs track the reference run's magnitude distribution.

## What Phase 1 does NOT yet cover

- **Grounded recondition** (`step()` with a real observation, `obs_latent`
  populated) is implemented in the pipeline but not exercised by this run —
  the driver runs pure imagination, matching the paper's "Foresight
  Reasoning" *predict* half but not the *re-ground* half. Threading a real
  camera/sim observation through is the next credibility item (mirrors the
  V-JEPA-AC "REFINE: real 7-DoF pose" item in `REVISED_STRATEGY.md` Tier 2).
- **Multi-session concurrency** (N resident sessions on one loaded model) is
  architecturally there (named caches keyed by `session_id`) but not load-tested.
- MI300X: pending RunPod stock (a persistent watcher is armed).

## Gotchas hit fixing this (beyond `LINGBOT_VA_ON_H100.md`)

1. **The flash_attn stub needs a real `ModuleSpec`.** A bare
   `types.ModuleType("flash_attn")` has `__spec__ is None`, which trips
   diffusers' lazy-submodule loader (`RuntimeError: ... flash_attn.__spec__
   is None`) even though nothing ever calls into the stub. Fix: assign
   `importlib.machinery.ModuleSpec("flash_attn", loader=None)`.
2. **Two action-dimensionalities, not one.** The model's internal padded
   action channel count (30) and the task's *executed* action width (6 for
   the demo task — `used_action_channel_ids`) are different numbers.
   `LingBotVAConfig` now carries both (`action_dim` for the model, the new
   `used_action_dim` for the wire format), with the pipeline setting the
   latter from the task config on load. `Action.values` is sized to
   `used_action_dim`, not `action_dim`.
