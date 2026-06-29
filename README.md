# spark-cli

A Rust implementation of the **Spark Execution Framework** — the execution pillar
for running autonomous AI development work on a single NVIDIA DGX Spark box.

`spark-cli` is the developer switch and executor loop. It admits frozen
**WorkUnits**, runs their sealed cell-DAGs through a verification gate, and emits
**VerdictEvents** — *computed*-done, never *claimed*-done. The whole thing is
spec-driven: the domain model (the **What**), the architecture (the **How**), and
the behavioural conformance suite all live under `.product/` and are kept in lockstep
with the code.

---

## The box has one switch

The Spark is bandwidth-bound and its VRAM holds **one residency at a time**. The
developer throws a switch between two mutually exclusive modes:

| Mode | What runs | Shape |
|---|---|---|
| **QUEUE** | many small models, batched inference | high-throughput, many units in flight |
| **EXPLORER** | one large model, serial | deep, single discovery session |
| *OFF* | nothing resident | the default |

The flip is a deliberate **human act** — no machine-rate process triggers it,
because swapping a residency is expensive. (Opus-class *OFF-BOX* work is not a box
mode; it lives elsewhere.)

---

## Core concepts

- **WorkUnit** — travels in by-value, identified by `bundle_hash`. Its cells form a
  sealed DAG. **Binding-homogeneity invariant**: every cell of a unit requires the
  *same* SPMC model binding (model identity + served quantization + params).
  Quantization is load-bearing on the Spark; a heterogeneous unit is a decomposition
  defect and is never dispatched.
- **Verification** — each cell is gated by a **protected oracle** the worker *cannot
  write* (ADR-076). A worker that can write its own oracle has no verifier. Verdicts
  are `accepted` / `rejected` / `escalate`; consequences are `advance` / `halt` /
  `retry` / `escalate`. Escalation is **unit-atomic** — the whole unit moves one
  binding up the ladder, never a single cell.
- **VerdictEvent** — travels out fire-and-forget to a durable, append-only log.

---

## Architecture (11 crates)

```
interface     WorkUnit / Cell / ModelBinding / VerdictEvent  (by-value contract, bundle_hash)
switch        Box Control     — the developer switch (QUEUE ⇄ EXPLORER), distinct-mode guard
queue         Work-Unit Queue — admission (homogeneity guard), priority, escalation ladder
execution     Execution       — sealed cell-DAG walk, verdict reduction, + oracle-run gate
exploration   Exploration     — single serial discovery session (EXPLORER)
serving       Model Serving   — VRAM residency + batched inference + Worker seam      ← production
sandbox       Isolation       — per-unit ephemeral sandbox + brokered credentials     ← production
stream        Verdict Stream  — durable, append-only, idempotent verdict log          ← production
host          Serving Host    — vLLM residency materialized on the box over SSH       ← production
executor      Engine          — composes everything; persists to .spark/state.json
cli           spark / spark-conform binaries
```

Each bounded context is one crate. Every aggregate is a **decider**
(`decide() -> Result<Vec<Event>, &'static str>`, where `Err` is the violated
invariant id); every read-model is a **projector** (an event fold).

---

## Install & build

```bash
cargo build --release        # produces target/release/spark and spark-conform
cargo test                   # 79 tests across the workspace
```

## CLI usage

```bash
spark mode set queue                 # throw the developer switch into QUEUE
spark admit work-unit.json           # admit a frozen WorkUnit (homogeneity guard)
spark run                            # drain the queue (in-memory demo path)
spark serve                          # drain isolated: sandbox + creds + worker + oracle + durable log
spark status                         # box mode + read-model views
spark stream                         # print the emitted VerdictEvents
spark mode set explorer && spark explore   # run a discovery session (EXPLORER only)
```

A WorkUnit is JSON; `acceptance_class` is kebab-case (`auto-commit-if-green` |
`needs-verdict`). See [`docs/production-seams.md`](docs/production-seams.md) for a
full example and the `spark serve` pipeline.

State persists to `.spark/state.json`; the durable verdict log to
`.spark/verdicts.jsonl`; per-unit sandboxes under `.spark/sandboxes/`.

### Pointing `serve` at a real model

`spark serve` runs each unit's frontier through a **Worker** and gates it with a
**protected Oracle** — both wired by environment variable:

```bash
# 1. an OpenAI-compatible model server on the box (llama-server / vLLM / TGI)
export SPARK_OPENAI_BASE_URL=http://127.0.0.1:8080     # → built-in OpenAiWorker
export SPARK_OPENAI_MODEL=qwen2.5-coder-7b             # optional
# 2. the protected gate the worker cannot write (ADR-076)
export SPARK_ORACLE_CMD='cargo test --quiet'
spark mode set queue && spark admit unit.json && spark serve
```

Worker precedence: a residency materialized by `spark mode set` → `SPARK_OPENAI_BASE_URL`
(HTTP) → `SPARK_WORKER_CMD` (shell) → offline `StubWorker`. **Full box setup:
[`docs/running-on-spark.md`](docs/running-on-spark.md).**

### Let the switch start the model (vLLM over SSH)

When `SPARK_SSH_TARGET` is set, `spark mode set` *physically materializes* the
residency: it retires any live host, then launches the mode's model as a **vLLM
container** on the box over SSH, polls its `/v1` endpoint, and only serves once it
answers. The switch becomes a real start/stop of VRAM, not a flag — and `spark
serve` then auto-targets that host.

```bash
export SPARK_SSH_TARGET=dev@spark-abcd.local      # → built-in SshVllmHost backend
export SPARK_QUEUE_MODEL=qwen2.5-coder-7b         # model vLLM loads in QUEUE
spark mode set queue                              # launches the container, waits until ready
```

---

## Spec-driven: the `.product/` graph

The implementation is derived from, and continuously checked against, a Product
Framework model:

- **What** — 8 bounded contexts, 13 entities (11 aggregate roots, each a decider),
  events / commands / invariants / read-models.
- **How** — 9 decisions, 10 principles, 12 patterns; the application contract and the
  Rust crate layout.
- **Deciders & projectors** — every aggregate's guarded state machine, proven
  **sound & complete** by simulation against its scenarios.
- **Behavioural conformance (§6.3)** — `spark-conform` replays the *realised* Rust
  deciders against the spec's scenario oracle. All 11 deciders are conformant.
- **Deliverables** — acceptance criteria wired to named, passing `cargo test`s and
  **computed**-done.

Re-run the gates:

```bash
product domain validate                       # What graph conformant
product how validate                          # How contract conformant
product archetype check spark-cli             # crate layout matches the tree
CONF="$PWD/target/release/spark-conform"
product decider conform box-decider --runner "$CONF box-decider"   # §6.3, per decider
product deliverable done deliverable-serving  # computed-done %
```

---

## Status

- 79/79 tests pass · release build green
- 11/11 deciders behaviourally conformant
- 11/11 deliverables computed-done
- domain / how / archetype all conformant

Physical infrastructure (GPU model serving, microVM isolation, the on-box vLLM
container) is implemented behind the `Worker`, `SandboxRuntime`, `CredentialBroker`,
and `ResidencyHost` **trait seams** with working local backends — a real server, a
container runtime, or the SSH/vLLM host drops in without touching the spec.
