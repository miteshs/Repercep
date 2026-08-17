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

## Metering

When the gateway runs with `REPERCEP_GATEWAY_DB` set, every call proxied here
is attributed to the calling customer's API key and token-metered into the
usage ledger. The proxy stays ignorant of *who* the caller is — it reports
what was spent and lets the gateway resolve attribution.

The passthrough remains byte-identical to the upstream's response, with one
deliberate exception on the streaming path: where the client did not specify
`stream_options`, the proxy asks the upstream for a usage chunk and then
suppresses that chunk, so billing is exact and the client's stream is
unchanged. See **`docs/METERED_GATEWAY.md`** for the full rationale, the
estimated-token disclosure rule, and the operator CLI.

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

## What you inherit for free (and deliberately did not rebuild)

Continuous batching, paged attention, prefix/APC caching, AWQ/GPTQ/FP8-KV quant,
speculative decoding, structured output, and a moving OpenAI-compat surface —
all from the upstream. The 80% of LLM serving that is hard is also the 80% that
is commodity; co-locating inherits it instead of re-implementing a worse copy.
