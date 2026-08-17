# The metered gateway — per-customer keys, token metering, quotas

**Status: shipped 2026-08-16.** This is the P0 slice from
`BUSINESS_PLAN_2026.md` §8 — the minimum that turns a runtime into something a
design partner can be given a key to and invoiced for. It is deliberately
*not* the full P0: there is no self-serve signup, no autoscaling, no payment
integration. Those are P1 and they are not what blocks a first paying
customer.

## Why this exists

The gateway's auth was one shared bearer token. On one box with one design
partner that is fine. The moment there are two customers it fails at
everything that matters commercially:

| Question | Shared token | Per-customer keys |
|---|---|---|
| Who made this call? | unanswerable | `key_id` → `customer` |
| Revoke one caller | rotate everybody | revoke one key |
| What do we invoice? | nothing is recorded | the usage ledger |
| Cap a customer's spend | impossible | `--monthly-token-quota` |

Everything below follows from wanting those four answers, and nothing more.

## Turning it on

One setting. `REPERCEP_GATEWAY_DB` points at a SQLite file that holds both the
keys and the usage ledger:

```bash
# Mint a key for a design partner (CLI-only — see "no HTTP admin surface").
repercep keys create --db /var/lib/repercep/gateway.db \
  --customer acme-robotics --label "design partner" \
  --monthly-token-quota 5000000

# Serve, with the co-located LLM proxy metered behind it.
REPERCEP_GATEWAY_DB=/var/lib/repercep/gateway.db \
REPERCEP_LLM_ENABLED=true \
REPERCEP_LLM_UPSTREAM_URL=http://127.0.0.1:8001 \
  uvicorn --factory repercep.serving.app:create_app_from_config
```

The customer then uses the key as an ordinary OpenAI bearer:

```bash
curl https://gw.example/v1/chat/completions \
  -H "Authorization: Bearer rpc_1cb4b4aa_..." \
  -d '{"model":"qwen2.5-7b","messages":[{"role":"user","content":"hi"}]}'
```

### Operator commands

```bash
repercep keys create --db DB --customer NAME [--label L] [--monthly-token-quota N]
repercep keys list   --db DB [--customer NAME] [--active]
repercep keys revoke --db DB KEY_ID
repercep usage       --db DB [--customer NAME] [--all-time]
```

`repercep usage` is the invoice query:

```
KEY ID     CUSTOMER                CALLS       PROMPT   COMPLETION        TOTAL        EST
1cb4b4aa   acme-robotics               5        4,800        1,457        6,257         97
cac3883d   globex                      1          800          210        1,010          0
TOTAL                                  6        5,600        1,667        7,267         97
```

## Four decisions worth knowing about

### 1. Keys are stored hashed and shown once

`repercep keys create` prints the only copy of the plaintext key that will
ever exist; the database keeps SHA-256 of it. A stolen database file yields no
working credentials. Plain SHA-256 rather than a slow KDF is deliberate: an
API key is 32 bytes of CSPRNG output with no dictionary to attack, so
stretching would only add latency to every authenticated request.

The wire format is `rpc_<key_id>_<secret>`. The `key_id` travels inside the
key so authentication is one indexed lookup rather than a hash against every
stored row, and so support tickets and logs can name a key without ever
handling its secret half.

### 2. Streaming calls are metered exactly, and the client cannot tell

This is the only genuinely subtle part of the implementation, and it is worth
the paragraph because billing accuracy rests on it.

An OpenAI-compatible server reports token usage on a streaming request **only
if the request set `stream_options.include_usage`**. Most clients don't set
it, which would leave the most common shape of traffic unbillable. Three
options existed:

1. **Estimate from chunk counts.** Wrong by a few percent — acceptable for a
   dashboard, not for an invoice.
2. **Inject `include_usage` and forward the resulting chunk.** Exact, but the
   client now receives a trailing chunk it did not ask for.
3. **Inject `include_usage`, then suppress the one chunk the injection
   caused.** Exact billing, byte-identical stream from the client's side.

We do (3). `serving/metering.py` parses the SSE stream line by line, forwards
every line the instant it completes (so time-to-first-token is unaffected),
and drops exactly the terminal usage chunk it caused. A client that asked for
usage itself still gets its chunk; a usage field attached to a chunk that also
carries content is never dropped. `tests/test_metering.py` pins the stream
byte-identical across chunk boundaries from 1 byte to 1 KiB.

### 3. Estimated tokens are marked, not laundered

Where exact counts genuinely cannot be obtained — an upstream that sends no
usage, or a client that disconnects mid-stream — the row is written with
`exact = 0` and a chunk-count estimate, and prompt tokens stay `0` rather than
being invented. `repercep usage` reports those tokens in a separate `EST`
column and prints a warning when any are present.

**If you invoice from a period containing estimated tokens, say so on the
invoice.** A customer who later discovers that a number they paid against was
a guess is a customer who stops believing the other numbers too — which is the
whole asset this company trades on.

### 4. No HTTP surface can mint or list keys

Key management is CLI-only, so a leaked customer key cannot be escalated into
minting more keys or enumerating other customers. The single HTTP reporting
route, `/v1/usage`, returns **only the calling key's own** month-to-date
spend. Cross-customer reporting requires shell access to the box.

## Quotas

`--monthly-token-quota N` rejects calls with **429** once `N` tokens have been
billed to that key in the current UTC month. Enforcement is at call
boundaries against already-billed usage, because a call's cost is unknowable
until it completes — so **a customer can overshoot its quota by at most one
call.** That is deliberate; the alternative (reserving a worst-case
`max_tokens` up front) would reject calls that would have fitted.
`tests/test_gateway_metered.py` pins the overshoot behaviour so it stays a
decision rather than a surprise.

## What is recorded

One append-only row per billable call: timestamp, key, customer, route, model,
prompt/completion tokens, the `exact` flag, HTTP status, and latency. Rows are
never updated or deleted — revoking a key leaves a tombstone rather than
removing it, so an invoice stays reconstructible and a billing dispute stays
winnable. Failed calls are recorded with their status and zero tokens, which
makes "we were down" answerable from the same table.

## Migration and compatibility

- A gateway with **no** `REPERCEP_GATEWAY_DB` and no `api_token` behaves
  exactly as before: open, unauthenticated, unmetered. Local dev is unchanged.
- The legacy shared `api_token` **still works** alongside per-customer keys, so
  callers migrate without a flag day. Its usage is metered under the
  `legacy-shared` key id, which is precisely the attribution problem
  per-customer keys exist to fix — treat rows under it as a migration backlog.
- The world-model WebSocket (`/v2/world/session`) accepts per-customer keys
  too. Session traffic is **not** token-metered: tokens are the wrong unit for
  a closed-loop control session, and per-second/per-session metering for the
  non-LLM classes is P2 work (`BUSINESS_PLAN_2026.md` §8).

## The scale ceiling, stated up front

**One SQLite file on one box.** It does not survive the gateway becoming more
than one process on more than one machine, and it is not a billing system of
record. The replacement is Postgres and it is P1 work. This exists so P0 can
bill a design partner correctly — not so it can run a public cloud.

Related: `docs/LLM_PROXY.md` (what sits behind the meter),
`docs/BUSINESS_PLAN_2026.md` §8 (where this sits in the plan).
