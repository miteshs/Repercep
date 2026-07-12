# Session 14 close — next-session pickup

**Date:** 2026-05-24 · **HEAD on `origin/session-14-cuda-port`:**
(pending push) · **Branch:** `session-14-cuda-port` (PR-ready) ·
**Working tree:** clean

Read this *before* `docs/HANDOFF.md`. This doc is the focused
"what happened today, what to do next" cut; `HANDOFF.md` is the
full orientation layer.

---

## What landed today

**NVIDIA H100 SXM5 port — architecture + benchmarks in a single
session.** ADR-0006 lands and closes the "Revisit if" clause of
ADR-0001. AMD MI300X **remains the lead workload** — the 142 s /
2.68× headline, `COSMOS_ON_MI300X.md`, the verification campaign
from Session 13 are all unchanged.

### Settled headline numbers

**Cosmos-Predict-7B Text2World, 121 f @ 1280×704, 36 steps, BF16,
seed=0** — measured on a single H100 SXM5 80GB HBM3 (`sm_90`, 132
SMs, CUDA 13.0 driver / torch 2.8.0+cu128):

| Config | Wall | vs NVIDIA pub. (~380 s) | vs Repercep MI300X | Peak HBM |
|---|--:|--:|--:|--:|
| Baseline (no cache) | 446.3 s | 0.85× | 1.05× faster than 470 s | 52.5 GiB |
| **Adaptive cache (thr=0.30)** | **138.4 s** | **2.75×** | 1.11× faster than 154 s | 52.5 GiB |
| Adaptive + FP8 Hopper Triton (cold) | 307.5 s | 1.24× | 0.46× (loss) | 52.5 GiB |
| Adaptive + FP8 Hopper Triton (warm) | 184.9 s | 2.05× | 0.77× (loss) | 52.5 GiB |
| Smoke (17 f / 8 steps, warm) | 10.7 s | — | comparable to MI300X 45 s | 28.4 GiB |

**Three meaningful findings from these numbers:**

1. **Silicon delta is 5–11 %, not 24 %.** The old
   `docs/METHODOLOGY.md` §3 claim "MI300X is 1.24× slower than H100
   at the same compute" was *stack* difference, not silicon: 470
   (Repercep diffusers path on MI300X) vs ~380 (NVIDIA's optimized
   TE + Apex + flash-attn-3 path on H100). Stack-vs-stack on the
   same silicon now measures 1.054× on baseline (470 vs 446.3) and
   1.113× on adaptive cache (154 vs 138.4). The 2.68× MI300X claim
   is **strengthened, not weakened** — the win is overwhelmingly
   the optimization stack.

2. **Repercep on H100 with adaptive cache alone beats NVIDIA's
   published H100 reference by 2.75×.** No FP8 needed. The
   TeaCache-style adaptive cache (loop-level, vendor-neutral) is
   the dominant optimization, and cuDNN-FA3 (via SDPA) is the
   attention floor on Hopper that the Repercep stack inherits for
   free.

3. **The FP8 Hopper Triton kernel is correct but slower than
   cuDNN-FA3 on Hopper** (F29 in BUILD_LOG.md). Per-call attention
   is 1.4–2.0× SDPA on the production shape, which makes the
   adaptive + FP8 path a NET LOSS (184.9 s warm vs 138.4 s
   adaptive-only). On MI300X the same kernel wins because aotriton
   is the comparator; on H100 cuDNN-FA3 is the much harder bar.
   The right Hopper FP8 path is TransformerEngine; the Triton path
   ships for parity (per-vendor sibling kernels per F25) and needs
   WGMMA-shaped autotune work to actually win on H100.

### Wan-2.2-T2V-A14B on H100 — deferred to Session 16

**Blocked by F30: HF Hub parallel downloader trips RunPod /workspace
FUSE quota.** Wan-2.2-T2V-A14B is ~118 GB across 39 files; the
default concurrent-write downloader hits "Disk quota exceeded
(os error 122)" mid-stream. The error is transient (1 GB sequential
writes succeed; the FUSE quota is well above 118 GB), but the
parallel-write pattern overwhelms the backend at scale. Session 16
recovery: `hf download --max-workers 1` to serialize writes, then
run `scripts/run_wan.py --frames 81 --steps 40` against the warm
cache. **This is plumbing, not code.** The WanEngine path through
`CUDABackend` is verified by existing Repercep tests.

### Code shipped (committed on `session-14-cuda-port`)

3 commits on the branch:

```
4dbe33b  fix(diffusers_backend): vendor-aware FP8 kernel selection
c90cbab  Session 14 followups: WAN_ON_H100 stub + Makefile uv fix + live CUDA parity tests
a202a1c  Session 14: NVIDIA H100 SXM5 backend — architecture, kernels, docs
```

Files (paths relative to repo root):

- `src/repercep/backend/cuda.py` — `CUDABackend` satisfying the
  Backend Protocol; sm_XX → DeviceArch; H100/H200 disambiguation
  by HBM; capabilities advertise FP8 (e4m3fn + e5m2, NOT fnuz —
  F25).
- `src/repercep/attention/hopper_flash.py` — `HopperFlashAttention`
  wrapping `flash_attn_interface.flash_attn_func` (FA-3, Hopper-
  only) with `flash_attn.flash_attn_func` (FA-2) fallback. FA-2
  installed Session 14; FA-3 source build is Session 16+.
- `src/repercep/attention/fp8_hopper_triton.py` +
  `kernels/triton_kernels/fp8_flash_attn_hopper.py` — Hopper FP8
  Triton FA-2 kernel (sibling of gfx942). Separate autotune cache
  at `~/.cache/repercep/fp8_autotune_hopper.json`.
- `src/repercep/attention/transformer_engine.py` — optional
  `TransformerEngineAttention` wrapping
  `transformer_engine.pytorch.DotProductAttention`. Loads cleanly
  when TE absent (F28: TE install on Hopper has a cu13/cu12 + torch
  ABI hazard; not yet exercised end-to-end).
- `src/repercep/attention/registry.py` — NVIDIA vendor branch in
  `select_attention_op`. `REPERCEP_FP8_ATTENTION` grows `{te,
  transformer_engine}` subvalues alongside `{triton, scaled_mm}`.
- `src/repercep/attention/diffusers_backend.py` — fixed in the
  third commit; the diffusers FP8 bridge now dispatches to the
  vendor-correct kernel (caught mid-sweep when Phase 3 first ran).
- `src/repercep/backend/registry.py` — `CUDABackend()` appended to
  `_ALL_BACKENDS`. ROCm precedes CUDA so the "MI300X is lead"
  framing holds on dual-vendor hosts (rare).
- `pyproject.toml` — `[nvidia]` optional dep group; cu128 index
  marker; `flash_attn_interface` + `transformer_engine` added to
  mypy `ignore_missing_imports`.
- `scripts/check_gpu.py` — vendor-neutral (ROCm OR CUDA).
- `Makefile` — UV path now resolves via `command -v` (was
  hardcoded to `~/.local/bin/uv`, broke on `/usr/bin/uv`).
- `tests/test_backend_cuda.py` + `tests/test_attention_cuda.py` —
  sibling test files for CUDA path; 13 new cases, GPU-skip
  cleanly. Two live correctness checks on this H100:
  `HopperFlashAttention` FA-2 vs SDPA = 0.0 rel diff;
  `FP8HopperTritonAttention` vs SDPA = 0.034 rel diff.
- `tests/test_attention_fp8.py` +
  `tests/test_attention_diffusers_backend.py` — gated on
  `torch.version.hip` so the AMD-only `fp8e4b8` kernel doesn't try
  to compile on NVIDIA (F25).
- `tests/test_backend.py` — sentinel for "unknown backend" switched
  from `"cuda"` (now real) to `"tpu"`.
- `docs/adr/0006-cuda-backend.md` (NEW) — full ADR mirroring 0001.
- `docs/COSMOS_ON_H100.md` (NEW) — measured numbers replace all
  the TBD markers from the original framework version.
- `docs/WAN_ON_H100.md` (NEW) — Wan port doc, deferred to Session
  16 per F30.
- `docs/HANDOFF.md` — Session 14 added to §10 timeline.
- `docs/BUILD_LOG.md` — full Session 14 entry + findings F25–F30.

### Quality gate (all green)

- `mypy --strict`: 56 source files, no issues.
- `ruff check`: clean across `src/`, `tests/`, `scripts/`.
- `pytest`: **143 passed, 12 skipped** (vendor-conditional skips
  on the AMD-only FP8 paths, run cleanly on this NVIDIA host).
- Rust crates: built + installed via `make rust-install` (after
  fixing the UV path in the Makefile).

---

## What's open after today (priority-ranked)

1. **(highest signal) Multi-prompt variance on H100.** Run
   `scripts/verify_timing.py --N 3 --prompts 5` for a confidence
   band on the 138.4 s and 446.3 s headlines. ~25 min GPU.
   Expected: ±0.5 % on no-cache (Phase 1 was cleanly noise-free),
   ±3–5 % on adaptive (matching MI300X Session 13 variance).
2. **TE FP8 path on H100** — unblock F28 first (pin TE +
   matching torch in a single `uv pip install` so the cu12 vs
   cu13 ABI mismatch doesn't fire). Then add Phase 4 = adaptive +
   TE. Expected ~115–130 s (cuDNN-FA3 with FP8 recipe should beat
   the BF16 cuDNN-FA3 floor). If TE matches adaptive-alone, the
   Triton path becomes purely research surface.
3. **Wan-2.2-T2V-A14B on H100** — unblock F30 via `hf download
   --max-workers 1`, then run the 17 f / 8 step smoke + 81 f /
   40 step quality reference. Compare against the Wan team's
   1041.5 s H100 reference (with offload + FP8 weight conversion)
   and against the MI300X 1700 s projection (BF16, both experts
   resident — apples-to-apples with what we'd measure on H100).
4. **FP8 Hopper kernel autotune (F29).** Expand the autotune grid
   with WGMMA-aware tiles (M ∈ {64, 128, 192, 256} × N ∈ {64,
   128, 256} × num_stages ∈ {2, 3, 4, 5}), add TMA-based K/V
   loads via `tl.make_tensor_descriptor`. Goal: beat cuDNN-FA3 at
   the Cosmos production shape, restoring
   `REPERCEP_FP8_ATTENTION=1` as a perf-on setting on Hopper.
5. **FA-3 from source on Hopper** — `cd
   flash-attention/hopper && python setup.py install` (~30 min
   build). Currently HopperFlashAttention falls back to FA-2 (1.09×
   over SDPA at our test shape); FA-3 would give more. Not on the
   critical path — SDPA already routes to cuDNN-FA3 inside
   PyTorch.
6. **MI300X-side open list from Session 13 unchanged** — FVD at
   N ≥ 1000, HIP FP8 correctness, continuous batching, action
   conditioning, bare-metal MI300X validation, OSS announce push.
   None of these moved in Session 14.

---

## Quick reproducers (for next session)

```bash
# H100 environment setup (one-time per fresh box)
git clone https://github.com/miteshs/Mirage.git && cd Repercep
git checkout session-14-cuda-port    # (or main, once merged)
uv venv --python 3.12 .venv
uv pip install --python .venv torch==2.8.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv -e ".[models,serving,dev]"

# Build Rust crates (after fixing rustup if not present)
make rust-install        # builds + installs repercep-cache/router/scheduler

# Sanity (should report H100, ~790 TFLOP/s BF16)
make check-gpu

# Big-volume cache (RunPod tip — without this, HF cache goes to overlay)
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub

# Headline reproducer (~138 s warm)
.venv/bin/python scripts/run_cosmos.py \
    --frames 121 --steps 36 --native-loop \
    --cache-mode adaptive --cache-adaptive-threshold 0.30 \
    --cache-force-full-every 16

# Multi-prompt variance band (~25 min GPU)
.venv/bin/python scripts/verify_timing.py --N 3 --prompts 5

# Wan-A14B (after unblocking F30)
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers --max-workers 1
.venv/bin/python scripts/run_wan.py --frames 81 --steps 40
```

---

## Commits this session (Session 14, on `session-14-cuda-port`)

```
4dbe33b  fix(diffusers_backend): vendor-aware FP8 kernel selection
c90cbab  Session 14 followups: WAN_ON_H100 stub + Makefile uv fix + live CUDA parity tests
a202a1c  Session 14: NVIDIA H100 SXM5 backend — architecture, kernels, docs
```

Plus a pending commit with this Session 14 close doc + the
measured Cosmos numbers in `COSMOS_ON_H100.md` + the F27–F30
findings in `BUILD_LOG.md`.

---

## When you resume — start here

1. Read this doc (you're doing it).
2. Read `docs/HANDOFF.md` §"TL;DR for a new session" (it points
   at the same open items; this doc supersedes it on the ranking).
3. Pick one of the open items above. (1) and (2) are tractable in
   a single session; (3) needs the F30 download workaround first;
   (4) is a multi-day kernel project.
4. PR `session-14-cuda-port → main` is ready to review at
   <https://github.com/miteshs/Mirage/pull/new/session-14-cuda-port>.

Have a good day.
