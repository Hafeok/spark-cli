# Production seams — from demo to working product

The demo executor proved the control flow (switch → admit → walk → verdict). Going
to a working product meant closing five seams where the demo used a stub. Each seam
is now a **modeled bounded context** in `.product/` *and* a **working Rust crate**
with a trait boundary so real infrastructure drops in without changing the spec.

| Seam | Demo stub | Production crate | Trait seam | Local backend |
|---|---|---|---|---|
| Model serving | enum-flip, no model | `spark-serving` | `Worker` | `StubWorker` / `CommandWorker` |
| Batched inference | one call per cell | `spark-serving` | — (`schedule_batches`) | homogeneous batch packing |
| Isolation | no sandbox | `spark-sandbox` | `SandboxRuntime` | `LocalSandbox` (per-unit dir) |
| Credentials | none | `spark-sandbox` | `CredentialBroker` | `LocalBroker` (leased token) |
| Verdict stream | in-memory `Vec` | `spark-stream` | — (`DurableLog`) | append-only JSONL on disk |
| Protected oracle | pass-all closure | `spark-execution` | `Oracle` | `CommandOracle` (ADR-076) |
| Residency (vLLM host) | enum-flip, no process | `spark-host` | `ResidencyHost` | `LocalProcessHost` / `SshVllmHost` |

---

## 1. Model Serving (`ctx-serving` → `spark-serving`)

Two aggregates govern what the box can run *right now*:

- **`resident-set`** — bindings currently in VRAM. `LoadBinding` is guarded by
  `inv-vram-budget` (a binding loads only if it fits the remaining budget);
  `EvictBinding` by `inv-nothing-resident`.
- **`work-batch`** — a batch of ready cells, **all sharing the resident binding**.
  `FormBatch` is guarded by `inv-batch-homogeneous` (no mixed-binding confound) and
  `inv-batch-empty`; `DispatchBatch` by `inv-batch-not-formed`. This is the
  binding-homogeneity invariant at the *batch* grain — the regime the bandwidth-bound
  box excels at.

```rust
pub trait Worker {
    fn invoke(&self, binding: &ModelBinding, prompt: &str) -> Result<String, String>;
}
```

- `StubWorker` — deterministic, offline; lets the whole pipeline run with no model.
- `CommandWorker` — shells out to a served model named by `$SPARK_WORKER_CMD`
  (e.g. a llama.cpp / vLLM CLI), prompt on stdin, artifact on stdout. Dependency-free.
- A GPU batching server is a drop-in third implementation.

## 2. Isolation (`ctx-sandbox` → `spark-sandbox`)

- **`sandbox`** — the per-unit ephemeral boundary: private writable workspace, frozen
  bundle read-only, network restricted to declared destinations. `ProvisionSandbox`
  is guarded by `inv-undeclared-network`; `TeardownSandbox` by `inv-no-sandbox`. The
  sealed cell-DAG runs entirely inside one sandbox — that is *why* the cell interior
  is sealed: it is one isolation domain.
- **`credential-lease`** — short-lived, least-privilege credentials exchanged from a
  unit's grant-reference, bound to its sandbox lifetime. `ExchangeCredential` is
  guarded by `inv-lease-needs-sandbox`; `RevokeCredential` by `inv-nothing-to-revoke`.
  The executor never injects a standing secret — authority derives from the unit.

```rust
pub trait SandboxRuntime {
    fn provision(&self, unit_ref: &str) -> std::io::Result<PathBuf>;
    fn teardown(&self, workspace: &Path) -> std::io::Result<()>;
}
pub trait CredentialBroker {
    fn exchange(&self, grant_ref: &str) -> String;
    fn revoke(&self, lease: &str);
}
```

`LocalSandbox` gives a real per-unit directory under `.spark/sandboxes/`, removed at
verdict. A container/microVM runtime is a drop-in for hard isolation.

## 3. Verdict Stream (`ctx-stream` → `spark-stream`)

- **`verdict-log`** — durable, append-only log of `VerdictEvent`s. `AppendVerdict` is
  guarded by `inv-idempotent-append` (at most once per `bundle_hash`, so at-least-once
  emission never duplicates); `ReconcileDeliverable` by `inv-nothing-to-reconcile`.
  Temporal decoupling: a night verdict waits in the log until a morning reader.

`DurableLog` is a real JSON-lines file that recovers its seen-hash set on reopen, so
idempotency holds across restarts.

## 4. Protected oracle (`oracle-run` in `ctx-execution` → `spark-execution`)

- **`oracle-run`** — one execution of a protected oracle gating a cell. `RunGate` is
  guarded by `inv-oracle-writable` (ADR-076): a gate may run only against an oracle
  the cell-worker has **no write capability over**. A worker that can write its own
  oracle is not verified.

```rust
pub struct CommandOracle { pub command: String, pub worker_writable: bool }
```

`CommandOracle` runs an external check (e.g. `cargo test <filter>`). If
`worker_writable` is true it **fails closed** — an unverified gate is never trusted,
even if the command passes.

## 5. Serving Host (`ctx-host` → `spark-host`)

The demo's switch flipped an enum; no model ever loaded. The **`serving-host`**
aggregate makes the switch *materialize the residency physically*. `LaunchHost` is
guarded by `inv-containerized-host` (a host runs **only as a container** — a
bare-metal launch is refused) and `inv-single-serving-host` (a host launches only
when none is live, so the prior one is retired first — the box's one-residency rule
in hardware). `ConfirmHostReady` is guarded by `inv-ready-needs-launch`, and only a
ready host serves work; `RetireHost` by `inv-nothing-to-retire`.

```rust
pub trait ResidencyHost {
    fn launch(&self, spec: &HostSpec) -> std::io::Result<HostHandle>;
    fn retire(&self, handle: &HostHandle) -> std::io::Result<()>;
}
```

- `SshVllmHost` — launches **vLLM as a detached container** on the box over SSH
  (`docker run -d --gpus all … vllm/vllm-openai …`), then `probe_ready` polls the
  `/v1` endpoint until it answers. The reused `OpenAiWorker` dispatches to it.
- `LocalProcessHost` — a dev stand-in for an already-running local server.

`Engine::launch_residency` retires any live host, launches the new one, waits for
readiness, and confirms it — each transition `serving-host-decider`-gated. `spark
mode set` calls it when `SPARK_SSH_TARGET` is configured.

---

## The `spark serve` pipeline

`Engine::drain_one_isolated` composes all four seams for each unit:

1. **Provision** a per-unit `LocalSandbox` (sandbox decider, declared-network guard).
2. **Exchange** the unit's grant-reference for a brokered lease bound to the sandbox.
3. **Batch & run**: pack the frontier by binding (`schedule_batches`), form/dispatch
   each batch (work-batch decider), invoke the `Worker`; artifacts land *inside* the
   sandbox.
4. **Gate**: walk the sealed cell-DAG against the protected `Oracle`; reduce
   cell-verdicts to a unit-verdict.
5. **Emit**: append the `VerdictEvent` to the durable, idempotent `DurableLog`.
6. **Teardown**: destroy the sandbox and revoke the lease — nothing standing survives.

```bash
spark mode set queue
spark admit work-unit.json
spark serve
#   accepted   wu-demo-1  (sandbox provisioned → worker → oracle → logged → torn down)
# isolated-drained 1 unit-attempt(s); durable log holds 1 verdict(s) at .spark/verdicts.jsonl
```

### Environment variables

| Var | Effect | Default |
|---|---|---|
| `SPARK_WORKER_CMD` | shell command serving the model (prompt on stdin) | `StubWorker` (offline) |
| `SPARK_ORACLE_CMD` | shell command for the protected gate | a passing protected oracle |

### Example WorkUnit

```json
{
  "unit_ref": "wu-demo-1",
  "parent_deliverable": "deliverable-serving",
  "bundle_hash": "sha256:demo1",
  "spmc_bundle": {},
  "model_binding": { "model": "coder", "quantization": "q4_k_m", "params": {} },
  "tier": "light",
  "acceptance_class": "auto-commit-if-green",
  "ladder_position": 0,
  "cell_graph": [
    { "cell_id": "test", "binding": { "model": "coder", "quantization": "q4_k_m", "params": {} }, "depends_on": [] },
    { "cell_id": "impl", "binding": { "model": "coder", "quantization": "q4_k_m", "params": {} }, "depends_on": ["test"] }
  ],
  "environment": { "network": [], "workspace": "ws" },
  "credential_grant": "grant-demo",
  "tool_grants": []
}
```

---

## Conformance for the new deciders

All seven production deciders are proven sound & complete and behaviourally conformant:

```bash
CONF="$PWD/target/release/spark-conform"
for d in resident-set-decider work-batch-decider sandbox-decider \
         credential-lease-decider verdict-log-decider oracle-run-decider \
         serving-host-decider; do
  product decider conform "$d" --runner "$CONF $d"
done
```

Their deliverables (`deliverable-serving`, `-sandbox`, `-stream`, `-oracle`, `-host`)
are computed-done with acceptance criteria wired to named passing tests.
