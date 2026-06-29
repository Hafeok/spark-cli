# Running on the Spark

How to run real autonomous work on a DGX Spark box (or any Linux machine with a
served model). `spark` is the **executor** — it orchestrates a model server and a
verification gate through two seams. This guide wires both to real infrastructure.

```
   developer ── spark mode set queue ──▶  box residency: QUEUE
        │                                 └─▶ launch vLLM container on the box over SSH (SPARK_SSH_TARGET)
   spark admit unit.json  ──▶  queue (binding-homogeneity guard)
        │
   spark serve  ──┬─▶  Worker   = the materialized host  (else SPARK_OPENAI_BASE_URL | SPARK_WORKER_CMD)
                  └─▶  Oracle    = a protected test/gate  (SPARK_ORACLE_CMD)
                          │
                          ▼
                  durable verdict log  .spark/verdicts.jsonl
```

---

## 1. Prerequisites

```bash
# build the executor
cargo build --release          # → target/release/spark, target/release/spark-conform
export PATH="$PWD/target/release:$PATH"
```

On the box you also need **one served model** reachable over HTTP, and a
**protected oracle** command (the worker must not be able to write it).

---

## 2. Serve a model on the box

`spark` only needs an OpenAI-compatible `/v1/chat/completions` endpoint. You can
let the switch start it for you, or manage the server yourself.

### Option A — let the switch start it (vLLM over SSH)

Set `SPARK_SSH_TARGET` and `spark mode set` **materializes the residency**: it
retires any live host, launches the mode's model as a **vLLM container** on the box
over SSH, polls `/v1` until it answers, and only then serves. The switch becomes a
real start/stop of VRAM, and `spark serve` auto-targets the host.

```bash
export SPARK_SSH_TARGET=dev@spark-abcd.local     # enables the built-in SshVllmHost
export SPARK_QUEUE_MODEL=qwen2.5-coder-7b        # model vLLM loads in QUEUE mode
export SPARK_EXPLORER_MODEL=llama-3.1-70b        # model vLLM loads in EXPLORER mode
export SPARK_VLLM_IMAGE=vllm/vllm-openai:latest  # optional; the container image
export SPARK_VLLM_PORT=8000                      # optional; the served port
export SPARK_VLLM_ARGS="--quantization awq"      # optional; extra vLLM flags

spark mode set queue
#   materializing residency: launching `vllm/vllm-openai:latest` (qwen2.5-coder-7b) on dev@spark-abcd.local ...
#   residency ready at http://spark-abcd.local:8000
```

Under the hood it runs `docker run -d --gpus all … vllm/vllm-openai … --model … --port …`
(detached, so the server survives the SSH session) and force-removes the container on
the next flip. Prerequisite: **key-based SSH** to the box (NVIDIA Sync or your own
key) and Docker with the NVIDIA runtime on it.

### Option B — manage the server yourself

Leave `SPARK_SSH_TARGET` unset and run any OpenAI-compatible server on the box, then
point `spark` at it with `SPARK_OPENAI_BASE_URL` (§3).

**llama.cpp (`llama-server`)**

```bash
llama-server \
  -m /models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 --parallel 8        # --parallel N enables batched serving
```

**vLLM**

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --quantization awq --host 127.0.0.1 --port 8080 --api-key spark-local
```

Confirm it answers:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":false}' | head -c 200
```

> **Residency & the switch.** `spark mode set queue` flips the box's *residency
> state* into QUEUE and enforces the one-binding-at-a-time invariants. It does not
> itself load the GGUF — your model server owns VRAM. Keep **one** server resident
> per mode: that is the QUEUE-vs-EXPLORER mutual exclusion in practice.

---

## 3. Point the Worker seam at it

The built-in `OpenAiWorker` talks to the server directly (one resident model serves
every cell — no per-cell process spawn). Selected automatically when
`SPARK_OPENAI_BASE_URL` is set:

```bash
export SPARK_OPENAI_BASE_URL=http://127.0.0.1:8080   # server root, NOT including /v1
export SPARK_OPENAI_MODEL=qwen2.5-coder-7b           # optional; else the unit's binding.model
export SPARK_OPENAI_API_KEY=spark-local              # optional bearer token (vLLM --api-key)
```

`temperature` and `max_tokens` set in a unit's `model_binding.params` are forwarded
to the request.

**Worker precedence** (first match wins):

| Condition | Worker |
|---|---|
| a residency materialized by `spark mode set` (host ready) | `OpenAiWorker` → the on-box vLLM host |
| `SPARK_OPENAI_BASE_URL` set | `OpenAiWorker` (HTTP, persistent server) |
| else `SPARK_WORKER_CMD` set | `CommandWorker` (shells out per cell, prompt on stdin) |
| else | `StubWorker` (deterministic, offline) |

> `OpenAiWorker` speaks plain `http://` (on-box localhost needs no TLS). For a
> remote TLS endpoint, implement `Worker` with a TLS client — it drops in without
> any spec change.

---

## 4. Point the Oracle seam at a protected gate

The oracle is the verifier the **worker cannot write** (ADR-076). It runs as a shell
command; non-zero exit = cell rejected. The unit's sandbox workspace is the CWD-ish
target — reference it via `$SANDBOX` in your command if you templatize it, or run a
fixed project gate:

```bash
export SPARK_ORACLE_CMD='cargo test --quiet'        # or: pytest -q, make check, ./gate.sh
```

If unset, `serve` uses a trivially-passing **protected** oracle (`worker_writable:
false`) so the loop runs offline. A `CommandOracle` with `worker_writable: true`
**fails closed** — an oracle the worker can write is never trusted, even if it exits 0.

---

## 5. Run it

```bash
spark mode set queue
spark admit unit.json            # repeat to enqueue more frozen WorkUnits
spark serve                      # isolated drain over the whole queue
```

`serve` prints which worker it selected, then per unit:

```
worker: OpenAI HTTP @ http://127.0.0.1:8080
  accepted   wu-hello  (sandbox provisioned → worker → oracle → logged → torn down)
isolated-drained 1 unit-attempt(s); durable log holds 1 verdict(s) at .spark/verdicts.jsonl
```

Inspect results:

```bash
spark status                     # box mode + read-model views
spark stream                     # emitted VerdictEvents
cat .spark/verdicts.jsonl        # the durable, append-only log (idempotent by bundle_hash)
```

### EXPLORER mode

One large model, serial — for discovery rather than batched units. With Option A
configured, the flip swaps the container for you (retire the QUEUE host, launch
`SPARK_EXPLORER_MODEL`); otherwise swap the resident server yourself first. Then:

```bash
spark mode set explorer          # retires the QUEUE vLLM host, launches the EXPLORER model
spark explore                    # produces a discovery record (candidate structure, NOT accepted code)
```

---

## 6. What each step enforces

| Step | Invariant / guard |
|---|---|
| `mode set` | `inv-distinct-mode` — a no-op flip is refused; flips are a deliberate human act |
| host launch | `inv-containerized-host` — vLLM runs only as a container; `inv-single-serving-host` — the prior host is retired first |
| host ready | `inv-ready-needs-launch` — only a launching host is confirmed ready, and only a ready host serves work |
| host retire | `inv-nothing-to-retire` — the live host is stopped (freeing VRAM) before the next launches |
| `admit` | binding-homogeneity — a mixed-binding unit is a decomposition defect, never dispatched |
| sandbox provision | `inv-undeclared-network` — only declared destinations reachable |
| credential lease | `inv-lease-needs-sandbox` — authority bound to the live sandbox |
| batch | `inv-batch-homogeneous` / `inv-batch-empty` — confound-free inference call |
| gate | `inv-oracle-writable` (ADR-076) — gate only against a worker-unwritable oracle |
| emit | `inv-idempotent-append` — at most one verdict per `bundle_hash` |
| teardown | sandbox destroyed, lease revoked — nothing standing survives the unit |

---

## 7. Files & state

| Path | Contents |
|---|---|
| `.spark/state.json` | persisted Engine (queue, views, sequence) |
| `.spark/verdicts.jsonl` | durable, append-only verdict log |
| `.spark/sandboxes/<unit>/` | per-unit workspace (removed at verdict) |

---

## 8. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `host launched but /v1 did not answer in time` | vLLM still loading weights, wrong `SPARK_VLLM_PORT`, or no NVIDIA runtime on the box |
| `residency materialization failed: ssh …` | no key-based SSH to `SPARK_SSH_TARGET`, or Docker missing on the box |
| `worker: offline stub …` printed | no materialized host and neither `SPARK_OPENAI_BASE_URL` nor `SPARK_WORKER_CMD` set |
| `connect 127.0.0.1:8080: …` | model server not up / wrong port |
| `non-200 from …` | bad model name, missing/incorrect API key |
| `only http:// urls supported` | `OpenAiWorker` is `http`-only; use a TLS `Worker` for `https` |
| units `escalated` then `halted` | the oracle rejected; check `SPARK_ORACLE_CMD` runs green by hand |
| `rejected '…': inv-…` on admit | the WorkUnit isn't binding-homogeneous — fix the decomposition |

See [`production-seams.md`](production-seams.md) for the architecture behind each seam.
