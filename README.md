# Mirage Runtime

A **world-model-native inference engine**. Lead workload: **Cosmos-Predict-7B**.
Lead hardware: **AMD Instinct MI300X** (`gfx942`, CDNA3).

The LLM-era serving stack (vLLM, TensorRT-LLM, Diffusers, `torch.compile`) is
built for autoregressive token decode. World models violate those assumptions:
diffusion-temporal denoising over latent volumes, bidirectional attention,
action conditioning, closed-loop latency. That gap is the 30–60% of silicon
performance Mirage is built to recover.

## Why MI300X first

The implementation plan leads with H100 and treats MI300X as a Phase-5
fast-follow. **This repo deliberately inverts that** — see
[ADR-0001](docs/adr/0001-mi300x-first-hardware-target.md).

Short version: on NVIDIA silicon you compete with bundled, subsidized
NIM/TensorRT-LLM; on MI300X there is *no* production-grade world-model serving
stack at all. The plan's own competitive analysis names the non-NVIDIA wedge as
the defensible position. The codebase stays vendor-neutral (Protocol-based
backends, [ADR-0003](docs/adr/0003-vendor-neutral-backend-protocol.md)) so an
NVIDIA backend is a zero-rewrite fast-follow.

## Status

**Pre-alpha, and running on three silicon targets.** Cosmos-Predict-7B,
Wan-2.2-T2V-A14B, and V-JEPA 2 all run end-to-end through the Mirage runtime on
**AMD MI300X**, **NVIDIA H100**, and **Intel CPU (AMX)** — one engine behind one
vendor-neutral backend Protocol (ADR-0003).

Headline numbers (system-vs-system against each vendor's *published* reference —
read [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the honest apples-to-apples
accounting before trusting any of these):

- **Cosmos H100 — 99.6 s, 3.81× NVIDIA's published reference** (5 prompts × 5
  seeds). Decomposes honestly as adaptive cache 2.75× × FA-3 1.39×.
- **Cosmos MI300X — 142 s, 2.68×** — to our knowledge the first publicly
  reported Cosmos benchmark on any AMD GPU.
- **Wan-2.2-A14B** runs both 14B MoE experts resident in the MI300X's 192 GiB —
  a config an 80 GiB H100 cannot hold without offload or VAE tiling.

Landed: the vendor-neutral backend layer (ROCm/CUDA/CPU), the attention
abstraction + autotuned FP8 Triton flash kernels (gfx942/Hopper/Ada) + CPU AMX
kernels, the Mirage-native denoise loop with TeaCache-style adaptive caching, the
Cosmos and Wan engines (+ the Cosmos guardrail), a Rust+PyO3 core
(cache/scheduler/router), and a FastAPI frame-streaming serving API (v1 sync +
v2 router path).

**The control regime (Session 24→25):** an independent strategic assessment
([`docs/STRATEGIC_ASSESSMENT.md`](docs/STRATEGIC_ASSESSMENT.md)) argued the
defensible moat is the **interactive, action-conditioned, closed-loop** regime —
the energy-based / JEPA world model — not the (public) cache. That seam exists:
an `InteractiveWorldModel` Protocol, a V-JEPA 2-AC engine with energy-MPC
planning, and a `/v2/world/session` WebSocket
([ADR-0008](docs/adr/0008-interactive-world-model-seam.md)). **The real V-JEPA
2-AC weights are GPU-verified** (H100, `docs/LEVERS_2026_07_H100.md`) — CEM
planning went from a 68.8 s sequential-fp32 baseline to **9.42 s batched-bf16
(7.3×)**, with a further KV/latent-reuse structural lever designed and
GPU-verified against real weights ([ADR-0009](docs/adr/0009-kv-latent-reuse.md)).
A second engine, **LingBot-VA 2.0** (policy-regime, video-action), now runs on
the same seam at **754.6 ms/chunk warm, 11 resident sessions/GPU** measured
through the shared harness (`docs/CONTROL_LOOP_BENCH.md`) — the **Control-Loop
Serving Benchmark v0**, the leaderboard for closed-loop world-model serving
that this project publishes first.

**👉 Picking up the project?** Read
[`docs/SESSION_25_HANDOFF.md`](docs/SESSION_25_HANDOFF.md) first (latest state),
then [`docs/HANDOFF.md`](docs/HANDOFF.md) for the deeper orientation. New to how
inference actually works here? Start the line-by-line tour at
[`docs/walkthroughs/`](docs/walkthroughs/).

Other docs:
- [`docs/CONTROL_LOOP_BENCH.md`](docs/CONTROL_LOOP_BENCH.md) — the control-loop serving leaderboard (closed-loop step latency, planning-decisions/sec, energy-evals/sec, resident sessions/GPU)
- [`docs/LEVERS_2026_07_H100.md`](docs/LEVERS_2026_07_H100.md) — V-JEPA 2-AC latency levers (batching + bf16 + warm-start), GPU-verified
- [`docs/adr/0009-kv-latent-reuse.md`](docs/adr/0009-kv-latent-reuse.md) — KV/latent-reuse design + GPU-verify findings (growing-window exact; sliding-window needs an explicit approximation)
- [`docs/STRATEGIC_ASSESSMENT.md`](docs/STRATEGIC_ASSESSMENT.md) — moat analysis, competitive scan, next-steps fork
- [`docs/POSITIONING.md`](docs/POSITIONING.md) — what's defensible to claim, and what isn't
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — how the numbers are measured (read before trusting a benchmark)
- [`docs/walkthroughs/`](docs/walkthroughs/) — line-by-line tour of Cosmos inference, outermost → kernel
- [`docs/walkthroughs-vjepa2-ac/`](docs/walkthroughs-vjepa2-ac/) — the same, for the interactive / energy-based path (V-JEPA 2-AC)
- [`docs/walkthroughs-avid/`](docs/walkthroughs-avid/) — **design** walkthrough for the AVID pixel world-model path (ADR-0008 Phase 3; not yet built)
- [`docs/architecture.md`](docs/architecture.md) — the component map
- [`docs/COSMOS_ON_MI300X.md`](docs/COSMOS_ON_MI300X.md) / [`docs/COSMOS_ON_H100.md`](docs/COSMOS_ON_H100.md) — measured writeups
- [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) — strategy + measured ledger
- [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) — chronological log of work, decisions, findings (F1–F47)
- [`docs/adr/`](docs/adr/) — architecture decision records (0001–0009)

## Requirements

- An AMD GPU host with **ROCm 7.x** installed (`/dev/kfd` readable by your user).
- **Python 3.11+**.
- ~30 GB free disk for weights once the model loader lands.

## Setup

`torch` must come from the ROCm wheel index that matches the system ROCm — the
default PyPI `torch` is CUDA-only and will not work.

```bash
# 1. environment (uv manages the venv; python3-venv is not required)
pip install --user uv
uv venv --python 3.12 .venv

# 2. ROCm PyTorch — match the index to your system ROCm (here: 7.2)
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/rocm7.2

# 3. Mirage + extras
make install        # == uv pip install --python .venv -e ".[models,serving,dev]"

# 4. verify the GPU is visible
make check-gpu
make info
```

If `check-gpu` reports no GPU, the device nodes are likely not readable:

```bash
sudo usermod -aG render,video "$USER"   # then log out / back in
```

## Developing

```bash
make lint        # ruff
make typecheck   # mypy --strict
make test        # pytest (GPU tests skip cleanly without a GPU)
```

The non-kernel codebase is held to `mypy --strict`; strong typing is a
deliberate force-multiplier for AI-assisted development. The kernel layer, when
it lands, is a separate codebase with different review standards.

## Layout

```
src/mirage/
  hardware.py      framework-agnostic device/dtype domain types
  config.py        runtime configuration (MIRAGE_* env vars)
  backend/         vendor-neutral compute backends (rocm / cuda / cpu)
  attention/       attention ops behind the AttentionOp Protocol (+ FP8/AMX bridges)
  runtime/         denoise loop + adaptive cache, scheduler, router, types,
                   and the interactive (action-conditioned) seam
  models/          Cosmos-Predict-7B, Wan-2.2, and the V-JEPA 2-AC engine
  serving/         frame-streaming HTTP (v1/v2) + the /v2/world/session WebSocket
  bench/           benchmark harness + per-stage profiler
crates/            Rust+PyO3 core — paged latent cache, scheduler, request router
kernels/           Triton (FP8 flash), HIP (FP8 GEMM), CPU AMX kernels
scripts/           runners (run_cosmos / run_wan / run_vjepa2 / run_vjepa2_ac) + diagnostics
docs/adr/          architecture decision records
docs/walkthroughs/ line-by-line inference tour
```

## License

Proprietary — Copyright (c) 2026 Mitesh Shah. All rights reserved. See
[LICENSE](LICENSE). No rights are granted to use, copy, modify, or distribute
this software without prior written permission.
