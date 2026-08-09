# Cosmos 3 Nano Policy on MI300X — reference-stack numbers (2026-08-09)

**The port's headline question, answered: Cosmos 3 Nano's policy path runs on
AMD.** No CUDA-only dependency, no stubbing, no source patches — diffusers'
`Cosmos3OmniPipeline` on stock ROCm PyTorch, first try.

These are **reference-stack** numbers — the diffusers pipeline called directly,
*not* through the Repercep seam (`models/cosmos3.py`'s pipeline half is still
Phase 1). That is why there is no row in `CONTROL_LOOP_BENCH.md`: putting one
there would claim a provenance this run does not have. Raw lines:
`docs/results/cosmos3_nano_policy_mi300x_2026-08-09.json`. Plan and the
pre-registered gates: `docs/COSMOS3_PORT_PLAN.md`.

## 1. Environment

| | |
|---|---|
| GPU | AMD Instinct MI300X VF (gfx942), 191.7 GiB HBM |
| Host | AMD Developer Cloud (DigitalOcean-backed), `gpu-mi300x1-192gb-devcloud`, `atl1`, $1.99/hr |
| ROCm | **7.0.2**, torch **2.10.0+rocm7.0** (HIP 7.0.51831), Python 3.12.3 |
| diffusers | **0.40.0.dev0 @ `d6726f38`** (main — no released wheel carries `Cosmos3OmniPipeline`) |
| transformers | 5.14.1 |
| Model | `nvidia/Cosmos3-Nano-Policy-DROID`, bf16, **public and ungated** (32.94 GiB) |
| Guardrail | disabled (`enable_safety_checker=False`) |

Setup deviates from `amd-developer-cloud-access`'s recipe in one way worth
keeping: **a plain venv with torch installed into it**, rather than a
system-level torch plus `venv --system-site-packages`. That sidesteps the
RECORD-less Debian `typing_extensions` failure entirely. `apt install
python3.12-venv` is still required first, and torch still must come from
`--index-url https://download.pytorch.org/whl/rocm7.0`.

## 2. The ROCm gate — passed, and cheaply

`scripts/cosmos3_rocm_probe.py` stages the checks by cost so the 33 GB download
is last. Stage 2 is the one the port rested on:

```
[PASS] stage 0: torch + device -- AMD Instinct MI300X VF, torch 2.10.0+rocm7.0, ROCm 7.0.51831
[PASS] stage 1: diffusers + Cosmos 3 symbols -- diffusers 0.40.0.dev0
[PASS] stage 2: CUDA-only dependency check -- 0 modules pulled, no CUDA-only imports
       pipeline module: diffusers.pipelines.cosmos.pipeline_cosmos3_omni
[PASS] stage 3: construct from config -- Cosmos3OmniTransformer, 15.17B params (meta)
```

Zero `transformer_engine` / `apex` / `flashinfer` imports, and no `flash_attn`
either — so unlike DreamZero there was not even a guarded fallback to rely on.
The port plan's §6 risk ("if the pipeline pulls TransformerEngine the AMD claim
is gone") did not fire.

MIOpen logs `Error [Init] Not found :<N>-DeviceGroupedConvFwd...` repeatedly
during the first calls. These are **kernel-selection misses that fall back
successfully**, not failures — output is correct and steady-state timing is
stable. They are the visible edge of the cold-start tax in §4.

## 3. Architecture facts, read off the shipped config

Worth recording because most are absent from the model card and two correct
guesses made in the port plan:

| | |
|---|---|
| transformer | `Cosmos3OmniTransformer`, **15.17B params** (the "16B" headline includes VAE + vision encoder) |
| reasoner backbone | **`model_type: qwen3_vl_text`** — the reasoning tower is Qwen3-VL-derived |
| MoE | **`use_moe: true`** |
| VAE | **`AutoencoderKLWan`** — the *Wan* VAE, the same family `models/wan.py` already serves |
| vision encoder | `Qwen3VLVisionModel` |
| attention | `joint_attn_implementation: two_way`, `qk_norm_for_diffusion: true` |
| positions | `unified_3d_mrope`, `max_position_embeddings: 262144` |
| dims | hidden 4096, intermediate 12288, head_dim 128, latent_channel 48, latent_patch_size 2 |
| **action register** | **`action_dim: 64` / `max_action_dim: 64` (padded)** — DROID's *used* width is 10 |

**The action-width correction matters.** `Cosmos3Config.action_dim = 10` in the
scaffold is the *wire* width; the model's register is **64-wide, zero-padded**
across embodiments — exactly the pattern DreamZero has (32 padded / 8 used) and
which its config models as `action_dim` + `used_action_dim`. Phase 1 should
mirror that split rather than carry one number. The pipeline returned `(16, 10)`,
confirming it de-pads on the way out, so the scaffold is not *wrong* today — but
it is under-described, and a post-trained checkpoint on another embodiment would
expose that.

## 4. Latency

Steady state, 16-action policy chunk at 480p / 30 denoise steps / `flow_shift=5.0`:

| | ms | note |
|---|---:|---|
| **chunk latency (steady state)** | **3610** | median of 7 samples, spread **±0.3%** |
| ├ VAE decode (`AutoencoderKLWan`) | 452 | 12.5% of the chunk |
| └ encode + 30 denoise steps | 3160 | |
| peak HBM | 33.59 GiB | resident weights 29.7 GiB |
| weight load | 82.1 s | from local disk cache |

### 4.1 The cold-start tax is the largest this repo has measured

| call | ms | vs steady state |
|---|---:|---:|
| first call ever on a fresh box (cold MIOpen disk cache) | **253,159** | **70×** |
| first call in a fresh process (warm MIOpen cache) | 15,454 | 4.3× |
| steady state | 3,610 | 1× |

DreamZero's MI300X row recorded a 49.6 s first call (~8.4× warm) and called it
"larger than the ~3× rule of thumb elsewhere in this repo's ROCm numbers." This
is four minutes, and it is **70×**. Two distinct effects, now separated: MIOpen
autotune results persist on disk, so the catastrophic case is once per *machine
image*, while the 4.3× is once per *process*.

**This is an operational finding, not a benchmark artifact, and it is the most
directly Repercep-relevant thing in this run.** An inference cloud that
cold-starts a Cosmos 3 worker onto fresh ROCm capacity pays four minutes before
the first token of useful work. Any autoscaling story on AMD has to ship a
pre-warmed MIOpen cache in the image, or it does not work. That belongs in the
deployment playbook.

### 4.2 Real-time framing — it does not make it, and by how much

A 16-action chunk at 15 FPS covers **1.067 s** of robot time and takes 3.61 s to
produce: **3.4× slower than real time**, 4.43 actions/s produced against 15
required. At these settings the model is a simulation/eval engine, not a
closed-loop controller. Levers not yet tried (denoise steps below 30, the 256p
tier, `torch.compile`) are the Phase-2 ladder and are where that gap gets
attacked.

### 4.3 Against the other DROID model on this GPU

| | Cosmos3-Nano-Policy-DROID | DreamZero-DROID |
|---|---|---|
| chunk latency (MI300X) | **3610 ms** | 5646.6 ms |
| actions/chunk | 16 | 24 |
| **ms/action** | **226** | 235 |
| weights resident | 29.7 GiB | 42.78 GiB |
| per-session marginal HBM | ~0 (stateless) | 23.05 GiB |
| resident sessions/GPU | **weights-bound, not state-bound** | 6 |

**Read this as serving cost on one robot, never as a policy comparison.** Same
DROID hardware, different action representations — Cosmos 3 emits 10D
end-effector pose deltas, DreamZero 8D joint positions (port plan §1.1).

The structurally interesting column is the last two. DreamZero pays 23 GiB of KV
*per session*, capping an H100 at one session and an MI300X at six. Cosmos 3 is
stateless per chunk (port plan §2.1), so a session costs one RGB frame and **one
29.7 GiB weight copy serves all of them** — concurrency is bounded by batching
and compute, not by memory per session. On a 191.7 GiB MI300X that is a much
better shape.

## 5. The pre-registered batching gate — and the fake number it nearly produced

Port plan §5 pre-registered this before any measurement, predicting
prefix-sharing near 1× and warning that any win would be plain batch efficiency,
which is stock and therefore playbook rather than moat.

**Actual verdict: not measurable — the stock interface exposes no
candidate-batching axis at all.** `Cosmos3OmniPipeline.__call__` has no
`num_videos_per_prompt` / `num_images_per_prompt` parameter, and
`CosmosActionCondition` carries a single `image` that drives the latent batch.

Passing `prompt=[p]*N` produced, at N = 1, 2, 4, 8:

- **1** candidate returned every time,
- wall time **3595–3610 ms** — statistically identical to N=1,
- peak HBM **33.59 GiB** — identical to N=1.

The batch was silently ignored. **A naive `batched_ms / N` reading of that run
gives 2.01× / 4.00× / 8.01×, and it is entirely fake** — an unchanged runtime
divided by N. It is recorded here, and in the results JSON, specifically so it
is never re-derived by someone reading the raw lines and mistaken for a result.
Constant wall time *and* constant peak memory across a 8× batch sweep is the
tell; either alone might have been explained away.

**What this changes strategically, stated carefully.** In July the best-of-N
lever died because `n=N` was stock in every LLM engine — we had the lever but
could not capture it. Here the opposite holds: candidate fan-out is **not**
available through the stock interface, so a customer cannot get it from anyone
by passing a flag, and implementing it would genuinely be ours. That is a
*reason to build*, not a result. It requires modifying the pipeline's latent
preparation, and **whether it then yields a win is still completely unmeasured**
— Phase 2, with the sequential baseline (8 × 3.61 s = 28.9 s for eight
candidates) as the number to beat. Nothing about batching goes in any external
artifact until that is done.

## 6. Caveats on these numbers

- **Synthetic conditioning.** A flat 640×540 grey canvas, not the real
  three-camera DROID composite. Latency is dominated by fixed-size denoise work
  so the timing should hold, but the *outputs* are meaningless and nothing about
  trajectory quality can be read off this run.
- **One machine, one session.** No multi-seed averaging across boxes, no
  concurrency curve. The ±0.3% spread is within-process repeatability, not
  run-to-run variance.
- **No H100 row yet**, so every cross-vendor statement here is deferred rather
  than made. MI300X went first deliberately (port plan §6) because ROCm was the
  gate the port depended on.
- **Guardrail disabled.** Fine for latency; a deployment claim would need it on
  and re-measured, since it adds a text check before and a video check after.
- **`step_mode` still unverified.** Whether the post-trained Policy-DROID
  checkpoint also serves `forward_dynamics` was not tested this run — it is what
  makes the seam's `step()` action-conditioned at all (module docstring).

## 7. What this run settles, and what it does not

**Settles:** Cosmos 3 Nano's policy path runs clean on MI300X/ROCm with no
patches. 3.61 s/chunk steady state, 226 ms/action, 29.7 GiB resident, stateless
sessions. A 70× cold-start tax that any AMD autoscaling design has to plan
around. The AMD claim in the port plan is **earned** — as of this run nobody
else has published a Cosmos 3 ROCm number, and NVIDIA's own
`inference_benchmarks.md` still has no policy row on any hardware.

**Does not settle:** anything comparative against H100; anything about
trajectory quality; anything about candidate batching beyond "the stock
interface cannot do it"; anything about the levers ladder.
