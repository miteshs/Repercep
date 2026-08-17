# Repercep Runtime

*(formerly "Mirage" — renamed 2026-07-12 to the company name, Repercep.AI; the
old name also collided with Decart's MirageLSD product. Older session docs and
provenance records keep the historical name where it is part of the record.)*

**A vendor-neutral inference engine and metered gateway for every model
class** — LLMs, VLAs, video diffusion, and world models — with **AMD Instinct
MI300X** (`gfx942`, CDNA3) as the lead target rather than an afterthought.

The company strategy is `docs/PIVOT_2026_07_PLATFORM_STRATEGY.md`: an
inference cloud that competes on cost per unit of customer work, built from
two measured levers — software efficiency on the workload shapes the
incumbents serve badly, and alt-silicon arbitrage on hardware none of them
price at all. This repo is the engine underneath it.

## Why MI300X first

On NVIDIA silicon you compete with bundled, subsidized NIM/TensorRT-LLM; on
MI300X there was no production-grade serving stack for these workloads at all
([ADR-0001](docs/adr/0001-mi300x-first-hardware-target.md)). The codebase
stays vendor-neutral (Protocol-based backends,
[ADR-0003](docs/adr/0003-vendor-neutral-backend-protocol.md)), so every result
below exists on both vendors and the NVIDIA path is not a port.

That bet is now **measured rather than assumed**: on dense Qwen2.5 models
under vLLM, a single MI300X matched or beat a single H100 at every workload
shape tested — `R = 1.02–1.38×` output throughput at ~60% of the hourly price,
a **1.7–2.3× advantage in cost per million output tokens**
([`docs/LLM_SILICON_GATE_RESULT.md`](docs/LLM_SILICON_GATE_RESULT.md)). Read
§5 of that document before quoting any of it: the coverage is dense Qwen2.5
only, one box per vendor, and the cost half is provider-confounded.

## Status

**Pre-alpha.** Six models across three serving regimes run end-to-end through
one engine on **AMD MI300X**, **NVIDIA H100**, and **Intel CPU (AMX)**, behind
one vendor-neutral backend Protocol.

Headline numbers, all measured by us — read
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the honest apples-to-apples
accounting before trusting any of them, and note that **every speedup here
travels with the workload it was measured on**:

| Result | Measured | Where |
|---|---|---|
| MI300X vs H100, LLM output throughput | **1.02–1.38×**, 1.7–2.3× cheaper/Mtok | `LLM_SILICON_GATE_RESULT.md` |
| Shared-prefix candidate batching (token-VLA decode) | **5.3–9.9×** both vendors, exact parity | `VLA_ON_{H100,MI300X}.md` |
| Adaptive step-skip cache (video diffusion) | **3.2–3.8×** | `COSMOS_ON_*.md` |
| Cosmos H100 / MI300X | 99.6 s (3.81× ref) / 142 s | `COSMOS_ON_*.md` |
| Resident sessions per GPU (16.5B policy model) | **1 on H100 vs 6 on MI300X** | `DREAMZERO_LEVERS_2026_07.md` |
| Cosmos 3 Nano policy, MI300X | 3610 ms/chunk, ROCm gate passed | `COSMOS3_ON_MI300X.md` |

Cosmos MI300X is, to our knowledge, the first publicly reported Cosmos
benchmark on any AMD GPU. Wan-2.2-A14B runs both 14B MoE experts resident in
the MI300X's 192 GiB — a config an 80 GiB H100 cannot hold without offload.

**Negative results are published, not buried.** The best-of-N LLM lever was
measured at 1.5× rather than the hoped 3×, then withdrawn from the business
plan entirely when the gateway could not capture it
([`docs/LLM_BESTOFN_RESULT.md`](docs/LLM_BESTOFN_RESULT.md)).
[ADR-0009](docs/adr/0009-kv-latent-reuse.md) contains two GPU-verified
negative findings. This is the practice the company trades on; see
[`docs/POSITIONING.md`](docs/POSITIONING.md) for what is defensible to claim.

### What is built

- **Vendor-neutral backend layer** (ROCm / CUDA / CPU) with an attention
  abstraction, autotuned FP8 Triton flash kernels (gfx942 / Hopper / Ada), and
  CPU AMX kernels.
- **World-model path** — the Repercep-native denoise loop with adaptive
  caching, the Cosmos/Wan engines, and an `InteractiveWorldModel` seam
  ([ADR-0008](docs/adr/0008-interactive-world-model-seam.md)) carrying V-JEPA
  2-AC energy-MPC planning, LingBot-VA 2.0, DreamZero, and Cosmos 3 Nano.
- **LLM path** — a co-located OpenAI-compatible reverse proxy to vLLM/SGLang
  ([`docs/LLM_PROXY.md`](docs/LLM_PROXY.md)). Deliberately not a reimplemented
  LLM engine: the upstream's continuous batching and paged attention are
  commodity and inheriting them beats writing a worse copy.
- **Metered gateway** — per-customer API keys, exact token metering, an
  append-only usage ledger, and monthly quotas
  ([`docs/METERED_GATEWAY.md`](docs/METERED_GATEWAY.md)). This is what makes
  the runtime billable.
- **Rust+PyO3 core** (paged latent cache, scheduler, request router) and a
  FastAPI serving surface (v1 sync, v2 router path, `/v2/world/session`).

**👉 Picking up the project?** Read
[`docs/SESSION_25_HANDOFF.md`](docs/SESSION_25_HANDOFF.md) first (latest
state), then [`docs/HANDOFF.md`](docs/HANDOFF.md) for deeper orientation. New
to how inference works here? Start the line-by-line tour at
[`docs/walkthroughs/`](docs/walkthroughs/).

## Documentation map

**Strategy and business** (confidential — see each file's header)
- [`docs/PIVOT_2026_07_PLATFORM_STRATEGY.md`](docs/PIVOT_2026_07_PLATFORM_STRATEGY.md) — the current company strategy
- [`docs/BUSINESS_PLAN_2026.md`](docs/BUSINESS_PLAN_2026.md) — market, pricing, unit economics, the raise
- [`docs/INFERENCE_MOAT_TECHNIQUES.md`](docs/INFERENCE_MOAT_TECHNIQUES.md) — the technique ledger behind the cost claim
- [`docs/POSITIONING.md`](docs/POSITIONING.md) — what's defensible to claim, and what isn't

**Measurement**
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — how numbers are measured (read before trusting a benchmark)
- [`docs/BENCHMARK_PROGRAM.md`](docs/BENCHMARK_PROGRAM.md) — what the gate program settled, and what is deliberately not being measured next
- [`docs/CONTROL_LOOP_BENCH.md`](docs/CONTROL_LOOP_BENCH.md) — the control-loop serving leaderboard
- [`docs/LLM_SILICON_GATE_RESULT.md`](docs/LLM_SILICON_GATE_RESULT.md) / [`docs/LLM_BESTOFN_RESULT.md`](docs/LLM_BESTOFN_RESULT.md) — the two pre-registered LLM gates
- [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) — strategy + measured ledger

**Engineering**
- [`docs/architecture.md`](docs/architecture.md) — the component map
- [`docs/METERED_GATEWAY.md`](docs/METERED_GATEWAY.md) — keys, metering, quotas
- [`docs/LLM_PROXY.md`](docs/LLM_PROXY.md) — the co-located LLM path
- [`docs/adr/`](docs/adr/) — architecture decision records (0001–0009)
- [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) — chronological log of work, decisions, findings
- [`docs/walkthroughs/`](docs/walkthroughs/) — line-by-line tour of Cosmos inference, outermost → kernel
- [`docs/walkthroughs-vjepa2-ac/`](docs/walkthroughs-vjepa2-ac/) — the same, for the interactive / energy-based path

## The marketing site is a different repo

repercep.ai is **not** built from this repository. It lives in the separate
**public** repo `github.com/miteshs/repercep-site` and deploys via GitHub Pages.
A local `site/` directory here is a gitignored mirror that goes stale silently —
edit the site repo, not this one, and treat anything under `site/` as
untrustworthy. This repo is private and holds confidential material; never copy
from here into a site path, including an unreferenced file, because Pages still
serves it.

## Requirements

- An AMD GPU host with **ROCm 7.x** installed (`/dev/kfd` readable by your user).
- **Python 3.11+**.
- ~30 GB free disk for weights.

## Setup

`torch` must come from the ROCm wheel index that matches the system ROCm — the
default PyPI `torch` is CUDA-only and will not work.

```bash
# 1. environment (uv manages the venv; python3-venv is not required)
pip install --user uv
uv venv --python 3.12 .venv

# 2. ROCm PyTorch — match the index to your system ROCm (here: 7.2)
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/rocm7.2

# 3. Repercep + extras
make install        # == uv pip install --python .venv -e ".[models,serving,dev]"

# 4. verify the GPU is visible
make check-gpu
make info
```

If `check-gpu` reports no GPU, the device nodes are likely not readable:

```bash
sudo usermod -aG render,video "$USER"   # then log out / back in
```

## Serving

```bash
# Mint a key for a customer, then serve with metering on.
repercep keys create --db gateway.db --customer acme --monthly-token-quota 5000000

REPERCEP_GATEWAY_DB=gateway.db \
REPERCEP_LLM_ENABLED=true \
REPERCEP_LLM_UPSTREAM_URL=http://127.0.0.1:8001 \
  uvicorn --factory repercep.serving.app:create_app_from_config

repercep usage --db gateway.db     # the invoice query
```

Full recipe, including the co-located vLLM-on-ROCm sidecar, in
[`docs/METERED_GATEWAY.md`](docs/METERED_GATEWAY.md) and
[`docs/LLM_PROXY.md`](docs/LLM_PROXY.md).

## Developing

```bash
make lint        # ruff
make typecheck   # mypy --strict
make test        # pytest (GPU tests skip cleanly without a GPU)
```

CI runs all three plus a CPU-only end-to-end control-loop smoke on every PR
and every push to `main` (`.github/workflows/ci.yml`).

The non-kernel codebase is held to `mypy --strict`; strong typing is a
deliberate force-multiplier for AI-assisted development. The kernel layer is a
separate codebase with different review standards.

## Layout

```
src/repercep/
  hardware.py      framework-agnostic device/dtype domain types
  config.py        runtime configuration (REPERCEP_* env vars)
  cli.py           `repercep info | keys | usage`
  backend/         vendor-neutral compute backends (rocm / cuda / cpu)
  attention/       attention ops behind the AttentionOp Protocol (+ FP8/AMX bridges)
  runtime/         denoise loop + adaptive cache, scheduler, router, types,
                   and the interactive (action-conditioned) seam
  models/          Cosmos, Wan-2.2, V-JEPA 2-AC, LingBot-VA, DreamZero,
                   Cosmos 3 Nano, and the token-VLA engine
  serving/         HTTP (v1/v2) + /v2/world/session WebSocket, the LLM proxy,
                   and the tenancy / metering / usage layer
  bench/           benchmark harness + per-stage profiler
crates/            Rust+PyO3 core — paged latent cache, scheduler, request router
kernels/           Triton (FP8 flash), HIP (FP8 GEMM), CPU AMX kernels
scripts/           model runners, GPU benchmark harnesses, diagnostics
docs/adr/          architecture decision records
docs/walkthroughs/ line-by-line inference tour
```

## License

Proprietary — Copyright (c) 2026 Mitesh Shah. All rights reserved. See
[LICENSE](LICENSE). No rights are granted to use, copy, modify, or distribute
this software without prior written permission.
