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

**Pre-alpha, and running.** Cosmos-Predict-7B generates video end-to-end on the
MI300X through the Mirage runtime, with the Cosmos safety guardrail integrated.
Landed: the vendor-neutral backend layer, attention abstraction, Runtime types +
frame-streaming serving API, the Cosmos-Predict-7B engine, a per-stage profiler
and benchmark harness, and the first optimization (`torch.compile` on the DiT).

**👉 Picking up the project? Start with [`docs/HANDOFF.md`](docs/HANDOFF.md).**

Other docs:
- [`docs/architecture.md`](docs/architecture.md) — the component map
- [`docs/COSMOS_ON_MI300X.md`](docs/COSMOS_ON_MI300X.md) — publish-ready writeup with measured numbers
- [`docs/OPTIMIZATION.md`](docs/OPTIMIZATION.md) — strategy + measured ledger
- [`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) — chronological log of work, decisions, findings
- [`docs/adr/`](docs/adr/) — architecture decision records

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
  backend/         vendor-neutral compute backends — ROCmBackend is the lead
  attention/       attention ops behind the AttentionOp Protocol
  runtime/         (next) inference engine, latent cache, scheduler
  models/          (next) Cosmos-Predict-7B loader + model code
  serving/         (next) frame-streaming HTTP/gRPC API
  bench/           (next) benchmark harness vs naive PyTorch + Diffusers
docs/adr/          architecture decision records
scripts/           standalone diagnostics
```

## License

Apache-2.0. The OSS core carries no field-of-use restrictions — see
[ADR-0001](docs/adr/0001-mi300x-first-hardware-target.md) and the
implementation plan's OSS-first GTM section.
