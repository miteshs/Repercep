# Co-located LLM proxy

Repercep can reverse-proxy an OpenAI-compatible LLM server (vLLM / SGLang)
running **alongside** the world-model runtime — same box, same auth, same
MI300X. This lets an embodied-AI design partner run their whole stack —
perception-generation (Cosmos/Wan), control-planning (V-JEPA-AC / LingBot-VA),
**and** the language/VLM head — behind one gateway on any silicon, without
Repercep pretending to be an LLM engine.

It is **off by default**. This is deployment coverage, not a headline.

## The one load-bearing design decision

The proxy **bypasses** the world-model path (`router → Scheduler → EngineDriver`,
`serving/driver.py`). That driver is single-engine, one-request-at-a-time by
design. A vLLM/SGLang upstream already does continuous batching + paged
attention *internally*; routing its traffic through our single-driver scheduler
would **serialize exactly what the upstream parallelizes** — strictly worse than
talking to it directly.

So `serving/llm_proxy.py` is a thin, stateless async passthrough that reuses
only the gateway's **auth** and **deployment surface**. Do not "integrate" it
into the scheduler later; that would be a regression, not an upgrade.

## Endpoints (mounted under the gateway's bearer auth)

| Route | Method | Behaviour |
|---|---|---|
| `/v1/chat/completions` | POST | streamed passthrough (SSE when `stream: true`) |
| `/v1/completions` | POST | streamed passthrough |
| `/v1/models` | GET | upstream model listing |
| `/v1/llm/health` | GET | upstream readiness probe (200/503) |

`/health` remains the gateway's own always-200 liveness; `/v1/llm/health` is the
upstream *readiness* signal (it needs the weights loaded to answer).

## Configuration (`REPERCEP_*`)

| Env var | Default | Meaning |
|---|---|---|
| `REPERCEP_LLM_ENABLED` | `false` | mount the proxy |
| `REPERCEP_LLM_UPSTREAM_URL` | `http://127.0.0.1:8001` | vLLM/SGLang OpenAI base URL |
| `REPERCEP_LLM_UPSTREAM_TIMEOUT_S` | `600` | per-request timeout |
| `REPERCEP_LLM_UPSTREAM_API_KEY` | _unset_ | bearer for a secured upstream |

Security: the client's own `Authorization` (the *gateway* token) is **never**
forwarded upstream. If the upstream is secured (`vllm serve --api-key`), set
`REPERCEP_LLM_UPSTREAM_API_KEY` and the proxy injects it.

## Co-location on MI300X (the vendor-neutral part, for free)

vLLM and SGLang both ship first-class ROCm support, so the LLM head runs on the
same Instinct box as the control engine:

```bash
# vLLM on ROCm as a sidecar OpenAI server on :8001
docker run -d --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host rocm/vllm:latest \
  vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8001 --tensor-parallel-size 1 --quantization fp8

# Repercep gateway, proxy enabled, launched from env:
REPERCEP_LLM_ENABLED=true \
REPERCEP_LLM_UPSTREAM_URL=http://127.0.0.1:8001 \
  uvicorn --factory repercep.serving.app:create_app_from_config
```

SGLang is a drop-in alternative (`python -m sglang.launch_server --port 8001 …`);
the proxy only speaks the OpenAI surface, so either works unchanged.

## Candidate fusing — the one thing we *do* add (opt-in, off by default)

`serving/llm_fusing.py` coalesces **concurrent, equivalent** `/v1/completions`
requests into one upstream call with `n=N`, then hands each caller its own
choice reshaped as a normal `n=1` response.

**Why this is worth adding when everything else here is inherited.** The
measurement in `docs/LLM_BESTOFN_RESULT.md` found the win does *not* come from
prefill arithmetic — it comes from a cache-timing race. N prefix-sharing
requests issued simultaneously **all miss the prefix cache together**, because
none has finished prefilling to populate it. Automatic prefix caching is a
*temporal* optimization and a simultaneous fan-out defeats it. One `n=N` request
shares the prefill structurally and cannot lose that race.

So this is not "we beat vLLM" — it is "we deliver vLLM's own best path to
callers who emit N separate requests," which is the shape agent frameworks
produce. **Measured 1.5× at a 2k shared prefix with 32-token decodes at moderate
concurrency; 1.04–2.5× across the shape grid.** Read the result doc before
quoting a number — it is strongly workload-dependent.

| Env var | Default | Meaning |
|---|---|---|
| `REPERCEP_LLM_FUSING_ENABLED` | `false` | Turn fusing on |
| `REPERCEP_LLM_FUSING_WINDOW_MS` | `8.0` | Coalescing window. Longer catches more peers and adds that latency to the first arrival |
| `REPERCEP_LLM_FUSING_MAX_BATCH` | `32` | Cap per fused call; `1` disables fusing |

**Safety rules, each of which exists for a reason:**

- **Greedy is never fused.** vLLM rejects `n>1` under `temperature=0` outright,
  so a fused greedy batch would 400 for callers whose individual requests were
  perfectly valid. Learned from the measurement, not the docs.
- Streaming, `n>1`, `best_of>1`, list prompts and **any unrecognised field** are
  passed straight through. An enabled fuser can never make a request *fail* that
  the passthrough would have served; the worst case is that it isn't coalesced.
- A lone request in a window is sent as the plain `n=1` call it already is — its
  response is never reshaped. **It does, however, wait out the window first**,
  which is real added latency on an idle gateway: the cost of the lever with
  none of the benefit. `window_ms` is the dial, and this trade is not yet
  measured end-to-end (`docs/LLM_BESTOFN_PLAN.md` §10).
- If the upstream 4xx's *because* we fused, every caller is retried individually
  — that failure is ours, not theirs.
- Per-caller `usage` is reported as if unfused (what they'd have been billed
  alone); the true shared cost is disclosed additively in `repercep_fusion`,
  which never replaces a standard field.

Chat completions are **not** fused: they carry message structure this has not
been measured on.

## What you inherit for free (and deliberately did not rebuild)

Continuous batching, paged attention, prefix/APC caching, AWQ/GPTQ/FP8-KV quant,
speculative decoding, structured output, and a moving OpenAI-compat surface —
all from the upstream. The 80% of LLM serving that is hard is also the 80% that
is commodity; co-locating inherits it instead of re-implementing a worse copy.
