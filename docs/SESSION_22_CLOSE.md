# Session 22 close — fresh MI300X pod bring-up reproduces Cosmos cold-cache budget, AMX CI smoke validated on AMX-masked Emerald Rapids, Wan TI2V-5B pre-staged for Session 23

**Date:** 2026-05-25 · **HEAD entering session:** `ea4da33` on `main`, in
sync with `origin/main` (Session 21 handoff).  **HEAD at close:** this
session adds 2 commits, both docs only — no source changes.

This session resumed on a *new* MI300X VF pod (not the H100 pod the
Session 21 handoff was written from).  Goal was twofold and shifted
mid-session:

1. Stand up a fresh MI300X pod end-to-end against the public
   `docs/COSMOS_ON_MI300X.md` playbook and reproduce one Cosmos run
   (done).
2. Originally: pick a Session 21 lane (Wan-shaped cache vs FVD N≥50)
   and commit.  Mid-session the scope shifted to **both** lanes
   simultaneously, then was deliberately deferred to Session 23 once
   the budget math was clear (see § "Lane A + FVD scope decision"
   below).  No Lane A or FVD code landed in Session 22.

---

## TL;DR

**Fresh MI300X pod reproduces `COSMOS_ON_MI300X.md` cold-cache
budget within noise.**  121 f / 36 step / adaptive thr=0.30 /
force_full=16 lands at **420.2 s wall / 52.5 GiB peak HBM** on first
cold use.  That matches the doc's decomposition (~270 s one-time
ROCm autotune + ~150 s adaptive cache work) almost exactly — the
public playbook reproduces on a fresh pod with no fixups.

**AMX CI compile-smoke + 32 routing-test design validates on this
silicon.**  Session 20's `MIRAGE_AMX_FORCE_BUILD=1 make kernels-cpu`
builds all three AMX kernels (BF16 `-march=sapphirerapids`, INT8
`-march=sapphirerapids`, FP16 `-march=graniterapids`) on the
hypervisor-AMX-masked Emerald Rapids VM hosting this pod.  `pytest -q
tests/test_attention_cpu.py tests/test_amx_int8.py tests/test_amx_fp16.py`
→ **32 passed / 1 skipped in 2.04 s**.  This is the second
independent confirmation (after Session 20 on the H100 pod's dev VM)
that the AMX CI path works on AMX-masked silicon.

**Wan TI2V-5B (~32 GiB) pre-staged on disk for Session 23.**
`Wan-AI/Wan2.2-TI2V-5B-Diffusers` snapshot is in
`~/.cache/huggingface/hub`, ready for Lane A's smoke validation.
Wan-A14B (~118 GiB) is **not** downloaded; deferred per Session 23
handoff's "stretch only" framing.

**Lane A + FVD scope decision recorded.**  Both lanes are go for
Session 23; the budget math (esp. FVD N≥1000 = ~7 days continuous
GPU on a single MI300X VF) is documented in
`docs/SESSION_23_HANDOFF.md` so the next session can pace itself.

---

## What landed this session

### 1. Fresh MI300X pod environment

- ROCm 7.2.0 (HIP 7.2.53211) already on host.
- `uv 0.11.16`, `.venv` at `/home/mshah/Mirage/.venv`, Python 3.12.3.
- `torch 2.12.0+rocm7.2`, `torchvision 0.27.0+rocm7.2`,
  `triton-rocm 3.7.0` (one minor bump from Session 11's `torch 2.12.0+
  rocm7.2 / triton 3.6.0` on the original MI300X pod that produced the
  `COSMOS_ON_MI300X.md` headline numbers — same major dep stack).
- `make install` brings `diffusers 0.37.1`, `transformers 5.9.0`,
  `accelerate 1.13.0`, `safetensors 0.7.0`, plus the dev / serving
  extras, exactly matching Session 21's "Env shape" notes (modulo the
  H100 pod's CUDA path → ROCm).

**One env-fix worth flagging in the README:** on a fresh pod, the
user is not in the `render` or `video` groups, so `make check-gpu`
fails with "no GPU visible (ROCm wheel)" until
`sudo usermod -aG render,video "$USER"` + re-login.  This is already
covered in `README.md` § Setup, but tripped this session — the
agent's shells inherited the *old* gid set even after the user's
shell got the new groups (had to wrap GPU commands in
`sg render -c "sg video -c '...'"` for the rest of the session).
A follow-up `claude` restart would pick up the new groups cleanly;
worth a one-liner note in the README's troubleshooting section.

### 2. Cosmos cold e2e on fresh pod

Single `scripts/run_cosmos.py` invocation, cold first run on this
pod (full ROCm autotune storm + first-shape kernel compile):

```text
[mirage] backend=rocm  device=AMD Instinct MI300X VF  192 GiB  arch=amd:gfx942
[mirage] loading Cosmos-Predict-7B (~38 GB), guardrail=False ...
[mirage] model loaded in 10.8s
[mirage] generating 121 frames @ 1280x704, 36 steps, seed 0 ...
[mirage] RESULT {
  "model": "cosmos-predict1-7b-text2world",
  "device": "AMD Instinct MI300X VF",
  "frames": 121,
  "resolution": "1280x704",
  "steps": 36,
  "load_seconds": 10.8,
  "generate_seconds": 420.2,
  "frames_per_second": 0.288,
  "seconds_per_step": 11.67,
  "peak_hbm_gib": 52.5,
  "output": "benchmark-results/cosmos_cold_run1.mp4"
}
```

Cross-check against `docs/COSMOS_ON_MI300X.md`:

| Source | Wall | Peak HBM | Notes |
|---|--:|--:|---|
| `COSMOS_ON_MI300X.md` cold (no cache) | 738 s | 52.5 GiB | autotune dominates ~273 s of that |
| `COSMOS_ON_MI300X.md` adaptive thr=0.30 (warmup-separated) | 151.4 s | 52.5 GiB | steady-state |
| This pod, cold-with-cache (one shot, fresh) | **420.2 s** | **52.5 GiB** | ≈ 270 s autotune + ~150 s adaptive cache work |

The 420 s ≈ 270 + 150 decomposition holds within run-to-run noise,
which is the strongest "the public playbook is reproducible"
signal we have.  Peak HBM is bit-identical to the doc.  No
fixups, no environment surgery, no diff to the codebase between
clone and first run.

**Output is in `benchmark-results/cosmos_cold_run1.mp4`** (2.0 MiB,
121 frames at 1280×704) — git-ignored per `.gitignore`'s
`benchmark-results/` entry; not committed.

### 3. AMX CI smoke on AMX-masked Emerald Rapids VM

Host CPU is `Intel(R) Xeon(R) Platinum 8568Y+` — silicon family is
Emerald Rapids (`intel:emr` per `mirage.cli info`), which would
normally expose AMX_BF16 + AMX_INT8.  This pod is a **20-core KVM
slice** and the hypervisor masks the AMX flags entirely:

```text
$ grep -m1 ^flags /proc/cpuinfo | tr ' ' '\n' | grep -E '^amx'
(no output)
$ lscpu | grep -i hypervisor
Hypervisor vendor:                       KVM
```

What IS exposed: AVX-512 BF16 + AVX-512 FP16 + AVX-512 VNNI +
AVX-VNNI.  AMX-masked.

**Session 20 designed the CI smoke for exactly this case.**
`MIRAGE_AMX_FORCE_BUILD=1 make kernels-cpu` bypasses the CPUID gate
and runs the compile-only smoke; all three kernels build:

- `kernels/cpu/amx_attn/_native.cpython-312-x86_64-linux-gnu.so`
  (BF16, `-march=sapphirerapids -mamx-bf16 -mamx-tile -mamx-int8`)
- `kernels/cpu/amx_int8_attn/_native.cpython-312-x86_64-linux-gnu.so`
  (INT8, same march)
- `kernels/cpu/amx_fp16_attn/_native.cpython-312-x86_64-linux-gnu.so`
  (FP16, `-march=graniterapids -mamx-fp16 ...`)

`pytest -q tests/test_attention_cpu.py tests/test_amx_int8.py
tests/test_amx_fp16.py` → **32 passed / 1 skipped in 2.04 s**.  This
is the second independent verification (after Session 20's H100-pod
dev VM) that the AMX CI path holds on AMX-masked silicon.

**Headline AMX numbers remain hardware-blocked** (Session 21's
lower-priority item #8 stands).  We can compile and route-test on
this VM; we cannot run the kernels for headline measurement.  A
bare-metal Sapphire / Emerald / Granite Rapids host is still the
gating prereq.

### 4. Wan TI2V-5B pre-staged for Session 23

`Wan-AI/Wan2.2-TI2V-5B-Diffusers` snapshot downloaded into
`~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/`
— **32 GiB on disk**, snapshot
`b8fff7315c768468a5333511427288870b2e9635`.  TI2V-5B is the
non-MoE Wan variant; Session 23's Lane A scaffold smoke-validates
on this before touching the MoE-aware A14B path
(~118 GiB, deferred).

### 5. Lane A + FVD scope decision

The strategic discussion mid-session settled on **both** Session 21
"open" lanes simultaneously as the Session 23 commitment:

- **Lane A** (Wan-shaped adaptive cache) — widens the wedge from
  Cosmos-specific to world-model-family-general.  This is the
  POSITIONING.md "structural moat" play.
- **FVD N≥1000** — bullet-proofs every cache claim against the
  literature-standard arbiter.  Without this, every published
  cache speedup carries an unmeasured-at-distribution-level
  quality caveat (the issue COSMOS_ON_MI300X.md § "Quantitative
  cache quality" already flags).

Budget math that forced the deferral to Session 23 (not Session 22):

| FVD N | GPU-hours | Wall (single MI300X VF) |
|---:|--:|--:|
| 100 | 17 hr | overnight |
| **1000** | **171 hr** | **~7 days continuous** |
| 10000 | ~1700 hr | weeks, needs a fleet |

N=1000 on a single VF is not a session-bounded task — it's a
multi-day batch.  The right scaffolding is a resumable
checkpoint-after-each-clip harness so the batch survives kills
and restarts.  Session 23 will land that scaffolding *first*,
then start the batch.

**No Session 22 code was written for either lane** — only docs.
Two reasons: (a) the budget math made it clear FVD N=1000 isn't a
session-window task, and (b) the user asked to stop Session 22
mid-stride and commit the state cleanly rather than half-land
Lane A's `denoise_wan_video`.  Session 23 picks both up clean.

---

## What's open after Session 22

Everything Session 21 flagged remains open; the Session 23 handoff
re-states Lane A + FVD as the explicit commitment for the next pod.
The lower-priority Session 21 items (flash-attn install + FA-3
trace, Wan A14B 81f/40 re-verification, F40 VAE/text-encoder
coverage, bare-metal AMX measurement, Ada FP8, production serving
driver) all stand unchanged.

The Session 22-specific open items:

1. **`denoise_wan_video` in `src/mirage/runtime/denoise.py`** —
   mirror `denoise_cosmos_video` with MoE-aware boundary swap.
   Session 23 starts here.  Template + Wan pipeline signature both
   studied this session; implementation deferred.
2. **`WanConfig.cache_mode` / `cache_adaptive_threshold` /
   `cache_force_full_every`** — wire the same knob set the Cosmos
   engine exposes.  Currently `WanConfig` has `cache_skip_every`
   and `use_native_loop` as forward-compat-only fields.
3. **`scripts/fvd_batch.py`** — resumable batch generator.  Doesn't
   exist; Session 23 lands it.
4. **1000-prompt corpus** — `docs/eval/fvd_prompts_v1.jsonl` (or
   similar) with provenance / seed.  Session 23 generates this
   deterministically from a seed so the corpus is itself
   reproducible.

---

## Verified on this MI300X pod (re-runnable)

```bash
cd /home/mshah/Mirage
export HF_TOKEN=$(grep '^export HF_TOKEN' ~/extra.sh | sed 's/.*HF_TOKEN=//')

# Env sanity
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
sg render -c "sg video -c '.venv/bin/python scripts/check_gpu.py'"
sg render -c "sg video -c '.venv/bin/python -m mirage.cli info'"

# Cosmos cold e2e (this is the run that produced the 420.2 s number)
sg render -c "sg video -c 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$(pwd)/src HF_TOKEN=$HF_TOKEN \
    .venv/bin/python -u scripts/run_cosmos.py \
        --frames 121 --steps 36 --native-loop \
        --cache-mode adaptive --cache-adaptive-threshold 0.30 \
        --cache-force-full-every 16 \
        --out benchmark-results/cosmos_cold_run1.mp4'"

# AMX CI smoke + routing tests (works on AMX-masked VMs)
MIRAGE_AMX_FORCE_BUILD=1 make kernels-cpu
.venv/bin/python -m pytest tests/test_attention_cpu.py \
    tests/test_amx_int8.py tests/test_amx_fp16.py -q

# What's cached on disk for Session 23
du -sh ~/.cache/huggingface/hub/models--nvidia--Cosmos-1.0-Diffusion-7B-Text2World
du -sh ~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B-Diffusers
```

The `sg render -c "sg video -c '...'"` wrapping is *this pod
specifically* — the agent's shells inherited the pre-`usermod` gid
set after the user added themselves to render/video.  A
fresh terminal session (or `claude` restart) wouldn't need the
wrapping.  Session 23 should either (a) restart the agent so the new
groups apply, or (b) keep using `sg`.

---

## Headline numbers table (post-Session 22)

No new headlines — Session 22 was reproducibility + scaffolding.
The published headlines stand:

| Workload | Config | Wall | Peak HBM | vs ref |
|---|---|--:|--:|--:|
| Cosmos MI300X (no cache, warmup-sep) | 121f/36 | 469.84 s ±0.32 s | 52.5 GiB | 0.81× H100 ref |
| Cosmos MI300X (adaptive thr=0.30, warmup-sep) | 121f/36 | 154.48 s ±5.96 s | 52.5 GiB | 2.47× H100 ref |
| Cosmos MI300X (adaptive + FP8 autotuned, warmup-sep) | 121f/36 | 142.0 s | 52.5 GiB | **2.68× H100 ref** |
| Cosmos H100 (no cache, Session 20) | 121f/36 | 320.9 s | 52.5 GiB | 1.18× NVIDIA pub |
| Cosmos H100 (adaptive thr=0.30, Session 20) | 121f/36 | 101.8 s | 52.5 GiB | **3.73× NVIDIA pub** |
| Wan A14B MI300X (no cache, cold, Session 11) | 81f/40 | 2576 s (contested) | 85.1 GiB | n/a |
| Wan TI2V-5B H100 (smoke, Session 20) | 17f/8 | 6.0 s | 42.6 GiB | n/a |
| **Cosmos MI300X (this pod, cold-with-cache, Session 22)** | **121f/36** | **420.2 s** | **52.5 GiB** | reproducibility check |

---

## When you resume — start at `docs/SESSION_23_HANDOFF.md`

Session 23 has a concrete two-track plan.  Both tracks are go;
neither is started.
