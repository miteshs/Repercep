# MoE silicon gate — **H100 leg only, NO VERDICT** (2026-08-16)

**This carries no verdict on anything.** `R = MI300X ÷ H100` and only the
denominator exists. Nothing here may be quoted as a result, a scope widening,
or a scope restriction until the MI300X leg runs. The dense gate's H100-only
commit (`98fe07a`) said the same thing in the same place, for the same reason.

- **Plan (pre-registered, binding):** `docs/LLM_MOE_GATE_PLAN.md`, including
  the §2.1a model decision committed before any box was created.
- Raw: `docs/results/moe_gate_h100_2026-08-16.json`. Harness:
  `scripts/bench_llm_silicon_gate.py`.

---

## 1. The H100 numbers

`deepseek-ai/DeepSeek-V2-Lite` (15.7B total / 2.4B active, MLA + 64 routed
experts), bf16, TP=1, single H100 80GB HBM3.

| shape | prompt | decode | conc | H100 output tok/s | equal-work |
|---|---:|---:|---:|---:|:--:|
| S1-interactive | 512 | 512 | 1 | **248.74** | ✅ 512/512 |
| S2-interactive-load | 512 | 512 | 32 | **3632.64** | ✅ 16384/16384 |
| S3-batch | 4096 | 128 | 32 | **2282.68** | ✅ 4096/4096 |

Run-to-run spread across 5 timed repeats: **1.7% / 16.4% / 3.8%** (S1/S2/S3).
S2's spread is materially wider than anything in the dense gate (which was
≤0.5% on H100), so **an MI300X difference on S2 under ~16% is not signal.**
Recorded now rather than discovered when it becomes convenient.

Equal work is enforced structurally (`min_tokens == max_tokens` + `ignore_eos`),
so every request emitted exactly the requested count rather than being checked
afterwards.

## 2. Environment — record it, per plan §4

| | H100 leg |
|---|---|
| vLLM | **0.26.0** (same as the dense gate's H100 leg) |
| torch | **2.11.0+cu130** (dense gate: same 2.11.0) |
| MoE backend | **TRITON** unquantized, chosen from `['TRITON', 'BATCHED_TRITON', 'FlashInfer TRTLLM', 'FlashInfer CUTLASS']` |
| attention | vLLM default |
| TP / GPUs | 1 / 1 |

**On the MoE backend.** vLLM initially fell back to TRITON because
`libnvrtc.so.13` was off the loader path and `deep_gemm` failed to import. That
was fixed before the timed run — the library ships in the `nvidia/cu13` pip
package and only needed `LD_LIBRARY_PATH` — and vLLM **still** selects TRITON,
which is the expected choice for *unquantized bf16* MoE (DeepGEMM's fused path
is an FP8 route). So TRITON here is a legitimate default rather than a
handicap. It was worth fixing anyway: leaving H100 on a degraded kernel path
would have biased the ratio toward AMD, the direction we have least licence to
be wrong in.

**The MI300X leg must record its own MoE backend**, and if the two differ
materially that is a caveat on the number and possibly the finding itself.

## 3. The §2.1a memory arithmetic, confirmed empirically

The plan rejected Qwen3-30B-A3B on a calculation. vLLM's own startup accounting
on the H100 confirms the calculation was right:

| | predicted (§2.1a) | measured |
|---|---:|---:|
| weights | ~31 GB | **29.32 GiB** |
| KV cache available | ~41 GB | **40.06 GiB** |
| S3-batch KV needed | 4.0 GiB | fits, ~10× headroom |

**Nothing in this gate can be attributed to memory capacity on either side.**
That was the entire point of the model choice, and it is now a measured fact
rather than an argument.

## 4. Accuracy smoke

20 greedy prompts captured and stored in the artifact for the cross-vendor diff
(plan §6 of the dense gate, reused). **Not yet compared** — the comparison
needs the AMD outputs.

## 5. What the MI300X leg needs

Blocking, and neither half is a keyboard problem:

1. **A `gpu-mi300x1-192gb-devcloud` instance**, created through the
   `devcloud.amd.com` web console. That site is blocked for browser
   automation here, so it is a manual step.
2. **`rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0`**
   — Docker works on that box, and `pip install vllm` on ROCm is a trap
   (PyPI wheels are CUDA-only and clobber ROCm torch).

Then the identical command, changing only the output path:

```bash
python bench_llm_silicon_gate.py \
  --model deepseek-ai/DeepSeek-V2-Lite --trust-remote-code \
  --max-model-len 8192 --out /workspace/out/moe_gate_mi300x.json
```

**Pre-declared, so it cannot be rationalised later:** the AMD leg will run
vLLM **0.23.x** against H100's **0.26.0**, the same asymmetry as the dense
gate. Per that gate's §6b the handicap runs *against* AMD, so `R ≥ 0.75`
despite it is robust and a lower bound, while `R < 0.75` is **not** decisive
on its own and requires a version-matched H100 re-run before outcome 3 is
declared. **MLA on ROCm under 0.23.x is the least certain part of this gate**
(plan §2.1a); if it will not serve, that is a first-class finding, not a
failed experiment.

## 6. Infrastructure findings (cost: ~50 min, ~$2.75)

Recorded because each cost real time and will otherwise be re-derived:

- **`runpodctl pod create` is still broken**; the REST API at
  `rest.runpod.io/v1/pods` placed the pod with a `machineId` immediately.
  Unchanged since 2026-08-09.
- **SSH uses `~/.runpod/ssh/runpodctl-ssh-key`, not the account
  `id_ed25519`** — a correction to the prior note, which described the AMD
  Dev Cloud box's key and did not transfer to RunPod.
- **`pip install vllm` collides with Debian-packaged `PyJWT`** (no RECORD
  file → `uninstall-no-record-file`). Fix: `pip install --ignore-installed
  PyJWT` first, then vLLM. Same family as the AMD box's `typing_extensions`
  trap.
- **`torchcodec` breaks `import vllm`** on this image — it is built against a
  different torch/CUDA and cannot resolve `libtorch.so` / `libnvrtc.so.13`.
  Installing ffmpeg does *not* fix it. `pip uninstall torchcodec` does, and
  costs nothing for a text-only benchmark.
- **Killing a vLLM run does not free the GPU.** The engine renames its
  process title to `VLLM::EngineCore`, so it survives a `pkill` matching the
  launching script and holds ~76 GiB indefinitely; the next run then dies with
  a misleading *"Free memory on device (3.96/79.18 GiB)... reduce GPU memory
  used by other processes."* `nvidia-smi` is no help — it reports **host**
  PIDs that cannot be killed from inside the container. Kill by name:
  `pkill -f 'VLLM::EngineCor[e]'`.
