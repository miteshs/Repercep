# Session 20 close — F40 fix-path 1 verified on H100, headlines reproduce, AMX CI smoke

**Date:** 2026-05-25 · **Working tree:** `main`, 5 commits ahead of
`origin/main` at session start (Session 19 close).  Session 20 adds 5 more
commits, all green: `ruff` clean, `mypy --strict` on edited files clean,
**`pytest -q` = 233 passed / 24 skipped** (up from 229 / 24 at session
start — the 4 new tests are the AMX capability-gate routing suite).

This session resumed from the Session 19 H100 handoff (codex session
delta).  Goal: verify the Wan F40 fix on a real H100 stack and re-run
the load-bearing headlines before the session window closed.  Both
landed.  CPU AMX got a parallel quality-of-life win: kernels can now
CI-compile-smoke on AMX-less VMs.

---

## TL;DR

**F40 fix-path 1 (Wan diffusers bridge) is confirmed working on H100.**
`scripts/trace_wan_attention.py` on Wan TI2V-5B 17f/8 under
`MIRAGE_FP8_ATTENTION=fa` reports `mirage_fp8_attention (dispatcher
entry) = 960` — up from 0 in Session 17's F40 baseline.  The Session
19 `wan_processor.maybe_install_mirage_wan_attention(pipe)` install
is doing exactly what F40 needed it to.  Verdict "PARTIAL" because
`flash-attn` isn't in this venv, so the 960 bridge dispatches all
fall through `_native_fallback` → SDPA (still cuDNN-FA-3 underneath
on Hopper).  Wiring is correct; only the FA-3 Python wrapper is
absent.  Recorded as **F45** in `docs/BUILD_LOG.md`.

**Cosmos H100 headline reproduces on a fresh env.**
121f / 36 step / adaptive thr=0.30 / `MIRAGE_FP8_ATTENTION=fa` lands
at **101.8 s / 52.5 GiB peak HBM** — within +2.2 s of Session 16's
99.6 ± 3.9 s mean (well inside the 3.86 % multi-prompt spread).
**3.73× NVIDIA's published ~380 s reference** (vs 3.81× Session 16),
same regime.  No FA-3 wheel installed; the bridge fall-through is
cuDNN-FA-3 SDPA, which is itself the path NVIDIA's own reference takes
on Hopper.  The Cosmos claim is robust across session boundaries.
Recorded as **F46** in `docs/BUILD_LOG.md`.

**Wan-2.2 TI2V-5B 17f/8 smoke: 6.0 s / 42.6 GiB peak.**
End-to-end through the `mirage.attention.wan_processor` install path
on H100.  No regressions on the smoke; the PROFILE row confirms DiT
dominates at 60 % of total wall time.

**AMX CI smoke unlocked.**  All three AMX kernels (BF16 / INT8 / FP16)
now build cleanly under `MIRAGE_AMX_FORCE_BUILD=1` on the hypervisor-
AMX-masked dev VM — verified end-to-end.  Gcc 13 + `-march=sapphirerapids`
(BF16/INT8) + `-march=graniterapids` (FP16) compile every intrinsic in
the kernel sources.  4 new capability-gate routing tests in
`tests/test_attention_cpu.py` verify the registry's INTEL branch picks
the right candidate under each `MIRAGE_AMX_ATTENTION` env value without
needing real AMX silicon.

---

## What landed this session

### 1. Wan attention dispatch type-cleanup (commit `42778cd`)

Codex-session leftover: `src/mirage/models/wan.py` had two
`# type: ignore[no-untyped-call]` annotations queued on the
`AutoencoderKLWan.from_pretrained` / `WanPipeline.from_pretrained`
calls — diffusers ships only partial type info and mypy --strict was
flagging both.  Committed as the intended Session 19 close-out.

### 2. F40 fix-path 1 confirmed on H100 — F45 (commit `8c0f47e`)

Ran `scripts/trace_wan_attention.py --small --frames 17 --steps 8`
on H100 with `MIRAGE_FP8_ATTENTION=fa`.  Result:

```
=== ATTENTION DISPATCHER COUNTERS ===
  torch.F.scaled_dot_product_attention (direct)   1100
  mirage_fp8_attention (dispatcher entry)         960
  native_fallback -> SDPA (fallback)              960
```

- **Bridge engagement count: 0 (F40 baseline, Session 17) → 960 (now).**
  The Wan dispatcher is no longer a dead lever.
- The 1100 "direct" SDPA calls are non-WanTransformer attention — the
  `AutoencoderKLWan` decoder and the UMT5-XXL text encoder both call
  `torch.nn.functional.scaled_dot_product_attention` directly without
  going through diffusers' attention dispatcher.  Not load-bearing for
  the headline; the DiT dominates wall time.
- "PARTIAL" verdict is because `flash-attn` isn't installed.  When
  it is, all 960 bridge calls take the FA-3 path automatically with
  no further wiring changes.

This was the highest-leverage Wan move in `docs/POSITIONING.md`
"What we recommend doing in the next 2 weeks" — now closed.

### 3. AMX CI compile-smoke + capability-gate tests (commits `7e14f18`, `735011f`)

Background context: per Session 18 close, every dev VM has `amx_*`
CPUID flags masked by the hypervisor, so the AMX BF16/INT8/FP16
kernels could neither be compile-tested nor have their registry
routing logic verified.  Each kernel's `setup.py` hard-exited on the
missing flag.

`7e14f18` adds a `MIRAGE_AMX_FORCE_BUILD=1` early-return bypass to:

- `kernels/cpu/amx_attn/setup.py` (BF16, `-march=sapphirerapids
  -mamx-bf16 -mamx-tile`)
- `kernels/cpu/amx_int8_attn/setup.py` (INT8, same march
  + `-mamx-int8`)
- `kernels/cpu/amx_fp16_attn/setup.py` (FP16, `-march=graniterapids
  -mamx-fp16`)
- `Makefile` umbrella `kernels-cpu-{bf16,int8,fp16}` targets mirror
  the bypass.

**End-to-end smoke validated:** `MIRAGE_AMX_FORCE_BUILD=1 make
kernels-cpu` produces three `_native.cpython-312-x86_64-linux-gnu.so`
artifacts on this VM.  All three import cleanly in Python.  Runtime
invocation will SIGILL (no AMX exposure) — expected and out of
scope.

`735011f` adds 4 capability-gate routing tests to
`tests/test_attention_cpu.py`:

- `test_int8_routing_when_amx_int8_detected`
- `test_fp16_routing_when_amx_fp16_detected`
- `test_bf16_routing_when_amx_bf16_detected`
- `test_int8_fallthrough_when_amx_int8_absent`

The helper `_force_wrapper_available()` patches both gates per
wrapper (the module-level `_detect_*` cpuinfo function and the
class-level `available` property), so the test isolates the
*routing* logic from the silicon-detection logic.  All 4 pass on
this AMX-less VM.

This closes the CPU-side audit's only "movable today" item that
wasn't blocked on silicon.

### 4. Cosmos H100 + Wan TI2V-5B headline reproduction — F46 (commit `246995f`)

User-requested before session close.

**Cosmos-Predict1-7B on H100, 121f / 36 steps, adaptive cache
thr=0.30 + `MIRAGE_FP8_ATTENTION=fa`:**

```
RESULT {
  "model": "cosmos-predict1-7b-text2world",
  "device": "NVIDIA H100 80GB HBM3",
  "load_seconds": 22.1, "generate_seconds": 101.8,
  "peak_hbm_gib": 52.5, "frames": 121, "steps": 36,
  "output": "benchmark-results/cosmos_h100_headline_rerun.mp4"
}
```

- 101.8 s vs Session 16's 99.6 ± 3.9 s mean = Δ +2.2 s = +2.2 %,
  well inside the published 3.9 s std.  **3.73× NVIDIA pub
  reference** (vs Session 16's 3.81×) — same regime.
- Peak HBM 52.5 GiB matches Session 16 exactly.
- Load time 22.1 s — significantly faster than Session 16's ~3 min
  cold load, because the model now sits on the MooseFS-backed
  `/workspace/.cache/huggingface` (F30 path resolved with a
  symlink from `~/.cache/huggingface/hub`).

**Wan-2.2 TI2V-5B 17f / 8 step smoke:**

```
RESULT {"generate_seconds": 6.0, "peak_hbm_gib": 42.6, ...}
PROFILE {"text_encode_s": 0.257, "dit_loop_s": 3.6,
         "vae_decode_s": 1.496, "dit_calls": 16}
```

- 6.0 s wall, DiT loop 60 % of total.  This is the *right* TI2V-5B
  number — the small variant is ~6× faster than A14B's smoke at the
  same shape because there's no MoE expert swap and the transformer
  is ~14B smaller.

**What we did NOT re-run:** Wan-2.2-A14B 81f/40 (Session 17's
1552.8 s headline).  The A14B repo (~118 GB) isn't on disk; the
user scoped the re-verification to Cosmos + TI2V-5B by explicit
choice.  The A14B number remains as-is from Session 17.

### 5. Operational cleanup

- Killed an orphan `flash-attn 2.8.3` source build the prior codex
  session had detached (266 cicc/nvcc compile processes consuming
  the box for 15+ minutes).  HopperFlashAttention falls back to
  cuDNN-FA-3 SDPA cleanly without it; the Cosmos + Wan headlines
  reproduce without flash-attn in the venv.  Box went from 274
  active processes back to ~6.
- Diagnosed + fixed a 100 % full root overlay caused by HF cache
  duplication.  `~/.cache/huggingface/hub/` was a real directory
  holding 32 GB of Wan + 14 GB partial Cosmos, while
  `/workspace/.cache/huggingface/hub/` held its own copies.
  Wiped the duplicate, symlinked `~/.cache/huggingface/hub` →
  `/workspace/.cache/huggingface/hub`, and same for `xet`.  Overlay
  went from 50 G used / 232 K free → 4.8 G used / 46 G free.
  Future Cosmos / Wan downloads land in the MooseFS-backed
  `/workspace` automatically.

---

## Quality gates

- `ruff check src tests scripts` — **clean**.
- `mypy --strict src/mirage` — **clean across all 49 source files**.
  This includes the previously-pre-existing `runtime/quantize.py`
  lines 171 + 250 errors around the lazy `nn.Module` class — fixed
  this session with class-level annotations for the
  `register_buffer`-backed attributes and an `Any`-typed local for the
  lazily-built class (PyTorch-canonical pattern).  First time the
  whole `src/mirage` tree is `mypy --strict` clean since the
  `QuantizedLinearModule` lazy pattern landed.
- `pytest -q` — **233 passed / 24 skipped** (was 229 / 24 at session
  start; the +4 are the new AMX capability-gate tests).
- `MIRAGE_AMX_FORCE_BUILD=1 make kernels-cpu` — **all three kernels
  build to `_native*.so`** on this AMX-less VM.

---

## What's open after today (priority-ranked)

1. **Install `flash-attn` (or `flash-attn-3`) in the H100 venv +
   re-run F45's trace.**  Expected counter: `native_fallback → FA-3
   = 960` instead of `→ SDPA = 960`.  Then benchmark
   `MIRAGE_FP8_ATTENTION=fa` Wan TI2V-5B 17f/8 wall against the
   un-bridged baseline to quantify the FA-3 lift on Wan.  Cosmos's
   3.81 × already includes the FA-3 lift via the cuDNN dispatch on
   Hopper, but Wan's bridge now reaching FA-3 directly is a separate
   compounding lever.  ~half-day on this pod once flash-attn builds.

2. **Wan-shaped adaptive cache.**  Per `docs/POSITIONING.md`, the
   highest-leverage open lever now that F40 is unblocked.  The
   Cosmos-shaped cache hardcodes `CosmosTransformer3DModel` block
   topology; the Wan-equivalent in `mirage.runtime.denoise` needs a
   gate that respects the MoE high-noise / low-noise expert boundary
   (`pipe.config.boundary_ratio`).  ~1-2 weeks; turns Wan H100 1552 s
   into ~700-800 s projected and re-positions the story from
   "Cosmos-specific cache lever" to "world-model-family-general cache
   lever proven on two diffusion-video families."

3. **FVD with N ≥ 50 prompts on a held-out Cosmos eval set.**  Per
   POSITIONING.md, required before external publication of the
   3.81 × claim.  The pixel-LPIPS work that landed in Session 17
   (LPIPS 0.61 H100) is necessary but not sufficient for the
   distribution-level "cache is quality-preserved" reading.  ~1 week
   of GPU time.

4. **Wan-2.2-A14B 81f/40 re-verification.**  Session 17's
   1552.8 s / 72.6 GiB headline was *not* re-run this session by
   explicit user scope choice (118 GB repo download + 26-min
   generation).  When session time allows, this is the third leg of
   the headline triple.

5. **F40 fix-path coverage for the VAE + text encoder.**  The 1100
   direct SDPA calls observed in F45's trace are the
   `AutoencoderKLWan` decoder + UMT5-XXL text encoder bypassing the
   dispatcher entirely.  Not load-bearing for the headline (DiT
   dominates), but if FP8 ever wants to cover those paths, a
   `WanEngine.load()`-scoped monkeypatch (F40 fix-path 2) is the
   route.  Defer until the bridge is shown to actually win on the DiT
   path first.

6. **Bare-metal AMX measurement (hardware-blocked, ongoing).**  The
   three AMX kernels now have CI compile-smoke coverage + routing-
   test coverage on AMX-less VMs; the only thing they still lack is
   actual silicon for headline numbers.  When a Sapphire / Emerald /
   Granite Rapids host opens up, the COSMOS_ON_CPU + WAN_ON_CPU
   "Pending headline numbers" tables are ready to be filled.

7. **Ada FP8 wiring decision (F42 vs F44).**  Two correct kernels
   are in tree.  Not addressed this session — user scope was H100
   work.  Half-day decision + wiring whenever an Ada/L40S host is
   next on the agenda.

8. **Production serving driver hardening.**  `mirage.serving.driver`
   exists but end-to-end FastAPI / gRPC under load is unmeasured.
   Per POSITIONING.md the productionization angle is a separate axis
   from the headline wall-time story; not urgent.

---

## Strategic posture going into next session

Per `docs/POSITIONING.md`'s 2-week recommendation, three of the four
priority items have now closed:

- ✅ **Cosmos H100 quality measurement** (Session 17, F23-equivalent
  on Hopper).
- ✅ **F40 fix-path 1** (Sessions 19 + 20, now empirically verified).
- 🟡 **Wan-shaped adaptive cache** — open, ~1-2 weeks, highest open
  leverage.
- 🟡 **FVD with N ≥ 50 prompts** — open, ~1 week of GPU time.

Decision for next session: pick *one* of the two open levers and
commit.  Both are 1-2 weeks; doing both in parallel is overcommit.
The Wan cache widens the wedge; the FVD work bullet-proofs the
existing wedge.  Strategic call belongs to whoever picks up.

---

## Quick reproducers (for next session)

```bash
# Set HF auth (Cosmos is gated)
export HF_TOKEN=$(cat ~/extra.sh | grep HF_TOKEN | cut -d= -f2)

# Wan F40 dispatch trace — expect bridge=960, fallback→SDPA=960
MIRAGE_FP8_ATTENTION=fa PYTHONPATH=$(pwd)/src .venv/bin/python \
    scripts/trace_wan_attention.py --small --frames 17 --steps 8

# Cosmos H100 headline — expect ~100 s ± 4 s, 52.5 GiB peak
MIRAGE_FP8_ATTENTION=fa PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$(pwd)/src .venv/bin/python -u scripts/run_cosmos.py \
        --frames 121 --steps 36 --native-loop \
        --cache-mode adaptive --cache-adaptive-threshold 0.30 \
        --cache-force-full-every 16

# Wan TI2V-5B smoke — expect ~6 s, 42.6 GiB peak
MIRAGE_FP8_ATTENTION=fa PYTHONPATH=$(pwd)/src .venv/bin/python \
    scripts/run_wan.py --small --frames 17 --steps 8 --profile

# AMX CI compile-smoke + routing tests (works on AMX-less VMs)
MIRAGE_AMX_FORCE_BUILD=1 make kernels-cpu
PYTHONPATH=$(pwd)/src .venv/bin/python -m pytest \
    tests/test_attention_cpu.py tests/test_amx_int8.py \
    tests/test_amx_fp16.py -q
```

---

## When you resume — start here

1. Read this doc.
2. Skim `docs/POSITIONING.md` "Strategic options matrix" — pick the
   *story* you want to tell, then the lane.
3. If continuing the Wan track: `flash-attn` install → re-run F45
   trace → quantify the FA-3 lift on Wan.  Then Wan-shaped cache.
4. If continuing the Cosmos / publication track: FVD with N ≥ 50
   prompts on a held-out eval set.
5. CPU AMX work is mostly correctness-only without real silicon at
   this point.  The CI smoke + routing tests landed cover the
   movable-today scope; everything else is hardware-blocked.

Have a good day.
