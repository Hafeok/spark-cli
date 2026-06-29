//! The Engine — the two-level control model in one place. The developer switch
//! sets the box mode (top level); below it the autonomous machinery drains the
//! flat work-unit set: admit (homogeneity guard) -> walk the sealed cell-DAG
//! against a protected oracle -> reduce to a unit-verdict -> emit a VerdictEvent
//! -> escalate unit-atomically. No machine-rate process flips the box.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use spark_execution::{consequence_of, reduce_verdict, walk_cells, Oracle, RunCommand, RunState};
use spark_host::{probe_ready, HostCommand, HostHandle, HostSpec, HostState, ResidencyHost, ServingHostView};
use spark_exploration::{SessionCommand, SessionState};
use spark_interface::{Verdict, VerdictEvent, WorkUnit};
use spark_queue::{decide_admission, HeterogeneityRateView, PrioritySetView, UnitCommand, UnitEvent, UnitState};
use spark_switch::{BoxCommand, BoxState, BoxStatusView, Mode};

/// The top ladder rung: a unit may escalate light(0) -> heavy(1); a rejecting
/// verdict at the heavy binding halts the unit (the ladder is exhausted).
pub const MAX_LADDER: u32 = 1;

/// `utilization-view` projector: is the box earning its keep?
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UtilizationView {
    pub mode: Mode,
    pub in_flight: i64,
    pub exploring: bool,
    pub discoveries: i64,
}

impl Default for UtilizationView {
    fn default() -> Self {
        UtilizationView { mode: Mode::Off, in_flight: 0, exploring: false, discoveries: 0 }
    }
}

/// The outcome of draining one unit, for reporting.
#[derive(Clone, Debug, PartialEq)]
pub enum DrainOutcome {
    Accepted { unit_ref: String },
    Escalated { unit_ref: String, to_ladder: u32 },
    Halted { unit_ref: String },
    Idle,
    NotQueueMode,
}

/// The full executor state — persistable, so the CLI can drive it across calls.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Engine {
    pub mode: Mode,
    /// The flat, edgeless priority set (front = highest priority).
    pub priority_set: Vec<WorkUnit>,
    pub stream: Vec<VerdictEvent>,
    pub priority_view: PrioritySetView,
    pub heterogeneity_view: HeterogeneityRateView,
    pub verdict_view: VerdictStreamCount,
    pub utilization: UtilizationView,
    /// The physical serving host that materializes the current residency (a vLLM
    /// container on the box). Defaulted so older state.json files still load.
    #[serde(default)]
    pub host: HostState,
    #[serde(default)]
    pub host_handle: Option<HostHandle>,
    #[serde(default)]
    pub serving_host_view: ServingHostView,
    seq: u64,
}

/// A serde-friendly mirror of the verdict-stream projector.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VerdictStreamCount {
    pub emitted: i64,
}

// PrioritySetView / HeterogeneityRateView need serde to persist; provide it here
// via shadow structs would be heavy — instead we re-derive their numbers from
// events, so we add Serialize/Deserialize on the source types in their crate.

/// Split an `http://host:port` endpoint into `(host, port)` for the probe.
fn host_port(endpoint: &str) -> (String, u16) {
    let s = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .unwrap_or(endpoint);
    let authority = s.split('/').next().unwrap_or(s);
    match authority.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(80)),
        None => (authority.to_string(), 80),
    }
}

/// Poll `host:port` until it accepts a connection or `timeout` elapses.
fn wait_ready(host: &str, port: u16, timeout: Duration) -> bool {
    let start = Instant::now();
    loop {
        if probe_ready(host, port, Duration::from_millis(500)) {
            return true;
        }
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

impl Engine {
    pub fn new() -> Self {
        Engine::default()
    }

    /// Throw the developer switch (top-level control). Returns the new mode, or
    /// the rejected invariant id on a no-op flip.
    pub fn throw_switch(&mut self, mode: Mode) -> Result<Mode, &'static str> {
        let state = BoxState { mode: self.mode };
        let events = state.decide(&BoxCommand::ThrowSwitch { mode })?;
        let mut view = BoxStatusView { mode: self.mode };
        for e in &events {
            view.apply(e);
        }
        self.mode = view.mode;
        self.utilization.mode = view.mode;
        Ok(self.mode)
    }

    /// Physically materialize the residency for the just-thrown mode: retire any
    /// live host (the single-residency rule made physical), launch the new vLLM
    /// host through the seam, then poll its endpoint and confirm it ready. Every
    /// transition is `serving-host-decider`-gated. Returns Ok(true) when the host
    /// answered before `ready_timeout`, Ok(false) when it launched but stayed
    /// cold (the caller should not dispatch to it yet).
    pub fn launch_residency(
        &mut self,
        host: &dyn ResidencyHost,
        spec: &HostSpec,
        ready_timeout: Duration,
    ) -> std::io::Result<bool> {
        // Free VRAM first: a live host must be retired before another launches.
        self.retire_residency(host)?;

        let events = self
            .host
            .decide(&HostCommand::Launch { containerized: true })
            .map_err(|inv| std::io::Error::new(std::io::ErrorKind::Other, inv))?;
        let handle = host.launch(spec)?;
        for e in &events {
            self.host.evolve(e);
            self.serving_host_view.apply(e);
        }
        self.host_handle = Some(handle.clone());

        // Readiness gate: do not confirm (and so do not let work dispatch) until
        // the host's endpoint actually answers.
        let (h, p) = host_port(&handle.endpoint);
        let ready = wait_ready(&h, p, ready_timeout);
        if ready {
            if let Ok(events) = self.host.decide(&HostCommand::ConfirmReady) {
                for e in &events {
                    self.host.evolve(e);
                    self.serving_host_view.apply(e);
                }
            }
        }
        Ok(ready)
    }

    /// Retire the live host (stop its container, free VRAM). A no-op when nothing
    /// is materialized — `inv-nothing-to-retire` simply rejects and we return Ok.
    pub fn retire_residency(&mut self, host: &dyn ResidencyHost) -> std::io::Result<()> {
        if let Ok(events) = self.host.decide(&HostCommand::Retire) {
            if let Some(h) = &self.host_handle {
                host.retire(h)?;
            }
            for e in &events {
                self.host.evolve(e);
                self.serving_host_view.apply(e);
            }
            self.host_handle = None;
        }
        Ok(())
    }

    /// Receive a frozen WorkUnit across the seam and run the admission guard.
    /// Returns Ok(()) when enqueued, Err(invariant) when rejected.
    pub fn admit(&mut self, unit: WorkUnit) -> Result<(), &'static str> {
        match decide_admission(&unit) {
            Ok(events) => {
                for e in &events {
                    self.priority_view.on_queue_event(e);
                    self.heterogeneity_view.apply(e);
                }
                self.priority_set.push(unit);
                Ok(())
            }
            Err(inv) => {
                self.heterogeneity_view.apply(&UnitEvent::Rejected);
                Err(inv)
            }
        }
    }

    /// Reprioritize: move a queued unit to the front of the flat set.
    pub fn reprioritize_to_front(&mut self, unit_ref: &str) -> Result<(), &'static str> {
        let idx = self.priority_set.iter().position(|u| u.unit_ref == unit_ref);
        // A queued unit is, by construction, in the set; the decider guards the rest.
        let guard = if idx.is_some() { UnitState { state: spark_queue::Lifecycle::Queued, ladder: 0 } } else { UnitState::default() };
        guard.decide(&UnitCommand::Reprioritize)?;
        let i = idx.unwrap();
        let u = self.priority_set.remove(i);
        self.priority_set.insert(0, u);
        Ok(())
    }

    /// Drain one unit: the bottom-level autonomous machinery. Only runs in QUEUE.
    pub fn drain_one(&mut self, oracle: &dyn Oracle, now: &str) -> DrainOutcome {
        if self.mode != Mode::Queue {
            return DrainOutcome::NotQueueMode;
        }
        if self.priority_set.is_empty() {
            return DrainOutcome::Idle;
        }
        let unit = self.priority_set.remove(0);
        self.priority_view.on_admitted();
        self.utilization.in_flight += 1;

        // ExecutionRun: admit -> walk -> compute -> emit (each decider-gated).
        let mut run = RunState::default();
        for e in run.decide(&RunCommand::Admit { box_mode: self.mode }).expect("queue mode admits") {
            run.evolve(&e);
        }
        let cell_results = walk_cells(&unit.cell_graph, oracle);
        let verdict = reduce_verdict(&cell_results);
        for e in run.decide(&RunCommand::ComputeVerdict).expect("admitted run computes") {
            run.evolve(&e);
        }
        for e in run.decide(&RunCommand::Emit).expect("verdict reached emits") {
            run.evolve(&e);
        }
        self.seq += 1;
        let tier_ran = if unit.ladder_position == 0 { "light" } else { "heavy" };
        let event = VerdictEvent {
            event_id: format!("ve-{}-{}", unit.unit_ref, self.seq),
            emitted_at: now.to_string(),
            unit_ref: unit.unit_ref.clone(),
            parent_deliverable: unit.parent_deliverable.clone(),
            bundle_hash: unit.bundle_hash.clone(),
            verdict,
            tier_ran: tier_ran.to_string(),
            cell_results,
            next_consequence: consequence_of(verdict),
        };
        self.stream.push(event);
        self.verdict_view.emitted += 1;
        self.utilization.in_flight -= 1;

        match verdict {
            Verdict::Accepted => DrainOutcome::Accepted { unit_ref: unit.unit_ref },
            Verdict::Rejected | Verdict::Escalate => self.escalate(unit),
        }
    }

    /// Unit-atomic escalation: re-enqueue the whole unit one binding up if the
    /// ladder has headroom, else halt it in failed.
    fn escalate(&mut self, mut unit: WorkUnit) -> DrainOutcome {
        let guard = UnitState { state: spark_queue::Lifecycle::Queued, ladder: unit.ladder_position };
        match guard.decide(&UnitCommand::Escalate { max_ladder: MAX_LADDER }) {
            Ok(events) => {
                for e in &events {
                    self.priority_view.on_queue_event(e);
                }
                unit.ladder_position += 1;
                unit.tier = "heavy".to_string();
                let to = unit.ladder_position;
                let unit_ref = unit.unit_ref.clone();
                self.priority_set.push(unit); // re-enqueued, whole unit, one binding up
                DrainOutcome::Escalated { unit_ref, to_ladder: to }
            }
            Err(_inv) => DrainOutcome::Halted { unit_ref: unit.unit_ref },
        }
    }

    // --- Exploration (only while the box is in EXPLORER) ---

    pub fn start_exploration(&mut self) -> Result<(), &'static str> {
        SessionState::default().decide(&SessionCommand::Start { box_mode: self.mode })?;
        self.utilization.exploring = true;
        Ok(())
    }

    pub fn produce_discovery_record(&mut self) -> Result<(), &'static str> {
        // A started session is active; the decider guards an unstarted one.
        let state = if self.utilization.exploring { SessionState { phase: spark_exploration::Phase::Active } } else { SessionState::default() };
        state.decide(&SessionCommand::ProduceDiscoveryRecord)?;
        self.utilization.exploring = false;
        self.utilization.discoveries += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_interface::{AcceptanceClass, Cell, Environment, ModelBinding};
    use std::collections::BTreeMap;

    struct AllPass;
    impl Oracle for AllPass {
        fn gate(&self, _c: &spark_interface::Cell) -> bool { true }
    }
    struct AllFail;
    impl Oracle for AllFail {
        fn gate(&self, _c: &spark_interface::Cell) -> bool { false }
    }

    fn binding(model: &str, q: &str) -> ModelBinding {
        ModelBinding { model: model.into(), quantization: q.into(), params: BTreeMap::new() }
    }

    fn unit(unit_ref: &str, cells: Vec<Cell>, b: ModelBinding) -> WorkUnit {
        WorkUnit {
            unit_ref: unit_ref.into(),
            parent_deliverable: "d".into(),
            bundle_hash: "sha256:x".into(),
            spmc_bundle: serde_json::json!({}),
            model_binding: b,
            tier: "light".into(),
            acceptance_class: AcceptanceClass::AutoCommitIfGreen,
            ladder_position: 0,
            cell_graph: cells,
            environment: Environment::default(),
            credential_grant: None,
            tool_grants: vec![],
        }
    }

    fn one_cell_unit(r: &str) -> WorkUnit {
        let b = binding("coder", "q4");
        unit(r, vec![Cell { cell_id: "c".into(), binding: b.clone(), depends_on: vec![] }], b)
    }

    #[test]
    fn cannot_drain_outside_queue_mode() {
        let mut e = Engine::new();
        assert_eq!(e.drain_one(&AllPass, "t0"), DrainOutcome::NotQueueMode);
    }

    #[test]
    fn admit_then_drain_accepts_and_emits() {
        let mut e = Engine::new();
        e.throw_switch(Mode::Queue).unwrap();
        e.admit(one_cell_unit("u1")).unwrap();
        assert_eq!(e.priority_view.queued, 1);
        let out = e.drain_one(&AllPass, "t1");
        assert_eq!(out, DrainOutcome::Accepted { unit_ref: "u1".into() });
        assert_eq!(e.stream.len(), 1);
        assert_eq!(e.stream[0].verdict, Verdict::Accepted);
        assert_eq!(e.priority_view.queued, 0);
    }

    #[test]
    fn heterogeneous_unit_is_rejected_at_admission() {
        let mut e = Engine::new();
        let mixed = unit(
            "bad",
            vec![Cell { cell_id: "c".into(), binding: binding("coder", "q8"), depends_on: vec![] }],
            binding("coder", "q4"),
        );
        assert_eq!(e.admit(mixed), Err("inv-binding-homogeneity"));
        assert_eq!(e.heterogeneity_view.rejected, 1);
        assert!(e.priority_set.is_empty());
    }

    #[test]
    fn a_failing_unit_escalates_then_halts_when_the_ladder_is_exhausted() {
        let mut e = Engine::new();
        e.throw_switch(Mode::Queue).unwrap();
        e.admit(one_cell_unit("u1")).unwrap();
        // First drain at light(0): fails -> escalate to heavy(1), re-enqueued.
        assert_eq!(e.drain_one(&AllFail, "t1"), DrainOutcome::Escalated { unit_ref: "u1".into(), to_ladder: 1 });
        assert_eq!(e.priority_set.len(), 1);
        // Second drain at heavy(1): fails -> ladder exhausted -> halt.
        assert_eq!(e.drain_one(&AllFail, "t2"), DrainOutcome::Halted { unit_ref: "u1".into() });
        assert!(e.priority_set.is_empty());
        assert_eq!(e.stream.len(), 2);
        assert_eq!(e.stream[1].tier_ran, "heavy");
    }

    #[test]
    fn the_switch_is_the_only_path_between_modes() {
        let mut e = Engine::new();
        // Exploration is refused in the default/off + queue modes.
        e.throw_switch(Mode::Queue).unwrap();
        assert_eq!(e.start_exploration(), Err("inv-box-must-be-explorer"));
        e.throw_switch(Mode::Explorer).unwrap();
        e.start_exploration().unwrap();
        e.produce_discovery_record().unwrap();
        assert_eq!(e.utilization.discoveries, 1);
        assert!(!e.utilization.exploring);
    }

    #[test]
    fn rethrowing_the_resident_mode_is_a_rejected_noop() {
        let mut e = Engine::new();
        e.throw_switch(Mode::Queue).unwrap();
        assert_eq!(e.throw_switch(Mode::Queue), Err("inv-distinct-mode"));
    }

    #[test]
    fn materializing_a_residency_launches_confirms_ready_then_retires() {
        use spark_host::{HostPhase, HostSpec, LocalProcessHost, ServingHostView};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let host = LocalProcessHost { base_url: format!("http://127.0.0.1:{port}") };
        let spec = HostSpec {
            host_id: "queue-host".into(),
            model: "coder".into(),
            image: "vllm/vllm-openai:latest".into(),
            ssh_target: String::new(),
            port,
            extra_args: vec![],
        };
        let mut e = Engine::new();
        e.throw_switch(Mode::Queue).unwrap();

        let ready = e.launch_residency(&host, &spec, Duration::from_secs(2)).unwrap();
        assert!(ready, "a listening endpoint should confirm ready");
        assert_eq!(e.host.phase, HostPhase::Ready);
        assert_eq!(e.serving_host_view, ServingHostView { materialized: 1, ready: true });
        assert_eq!(e.host_handle.as_ref().unwrap().model, "coder");

        e.retire_residency(&host).unwrap();
        assert_eq!(e.host.phase, HostPhase::Retired);
        assert_eq!(e.serving_host_view, ServingHostView { materialized: 0, ready: false });
        assert!(e.host_handle.is_none());
    }
}

// ───────────────────── production drain (all seams composed) ─────────────

use spark_sandbox::{CredentialBroker, LeaseCommand, LeaseState, SandboxCommand, SandboxRuntime, SandboxState};
use spark_serving::{schedule_batches, BatchCommand, BatchState, Worker};
use spark_stream::DurableLog;

impl Engine {
    /// Drain one unit through the full production path: provision a per-unit
    /// sandbox, broker credentials, run the unblocked frontier as batched worker
    /// invocations gated by a protected oracle, reduce to a verdict, append it to
    /// the durable log (idempotent), then tear the sandbox down and revoke creds.
    /// Every step is gated by its decider; effects land only inside the sandbox.
    #[allow(clippy::too_many_arguments)]
    pub fn drain_one_isolated(
        &mut self,
        sandbox: &dyn SandboxRuntime,
        broker: &dyn CredentialBroker,
        worker: &dyn Worker,
        oracle: &dyn Oracle,
        log: &mut DurableLog,
        now: &str,
    ) -> std::io::Result<DrainOutcome> {
        if self.mode != Mode::Queue {
            return Ok(DrainOutcome::NotQueueMode);
        }
        if self.priority_set.is_empty() {
            return Ok(DrainOutcome::Idle);
        }
        let unit = self.priority_set.remove(0);
        self.priority_view.on_admitted();
        self.utilization.in_flight += 1;

        // 1. Environment: provision the per-unit sandbox (network declared by construction).
        let mut sbx = SandboxState::default();
        for e in sbx.decide(&SandboxCommand::Provision { network_declared: true }).expect("declared network provisions") {
            sbx.evolve(&e);
        }
        let workspace = sandbox.provision(&unit.unit_ref)?;

        // 2. Credentials: exchange the grant-reference for a short-lived lease (if any).
        let mut lease = LeaseState::default();
        let token = unit.credential_grant.as_ref().map(|grant| {
            for e in lease.decide(&LeaseCommand::Exchange { sandbox_active: true }).expect("live sandbox leases") {
                lease.evolve(&e);
            }
            broker.exchange(grant)
        });

        // 3. Serving: batch the frontier by binding and invoke the worker; each
        //    cell's artifact is written INSIDE the sandbox (a declared effect).
        for (binding, cell_ids) in schedule_batches(&unit.cell_graph) {
            let mut batch = BatchState::default();
            for e in batch.decide(&BatchCommand::Form { homogeneous: true, nonempty: !cell_ids.is_empty() }).expect("homogeneous nonempty batch") {
                batch.evolve(&e);
            }
            for e in batch.decide(&BatchCommand::Dispatch).expect("formed batch dispatches") {
                batch.evolve(&e);
            }
            for cell_id in &cell_ids {
                let prompt = format!("unit {} cell {}", unit.unit_ref, cell_id);
                if let Ok(artifact) = worker.invoke(&binding, &prompt) {
                    let _ = std::fs::write(workspace.join(format!("{cell_id}.out")), artifact);
                }
            }
        }

        // 4. Verification: walk the sealed cell-DAG, gating each cell against the
        //    protected oracle; reduce cell-verdicts to a unit-verdict.
        let cell_results = walk_cells(&unit.cell_graph, oracle);
        let verdict = reduce_verdict(&cell_results);

        // 5. Output Contract: emit the VerdictEvent to the durable, idempotent log.
        self.seq += 1;
        let tier_ran = if unit.ladder_position == 0 { "light" } else { "heavy" };
        let event = VerdictEvent {
            event_id: format!("ve-{}-{}", unit.unit_ref, self.seq),
            emitted_at: now.to_string(),
            unit_ref: unit.unit_ref.clone(),
            parent_deliverable: unit.parent_deliverable.clone(),
            bundle_hash: unit.bundle_hash.clone(),
            verdict,
            tier_ran: tier_ran.to_string(),
            cell_results,
            next_consequence: consequence_of(verdict),
        };
        log.append(&event)?; // idempotent by bundle_hash
        self.stream.push(event);
        self.verdict_view.emitted += 1;

        // 6. Teardown: destroy the sandbox and revoke the lease — nothing standing survives.
        if let Some(t) = &token {
            for e in lease.decide(&LeaseCommand::Revoke).expect("active lease revokes") {
                lease.evolve(&e);
            }
            broker.revoke(t);
        }
        sandbox.teardown(&workspace)?;
        for e in sbx.decide(&SandboxCommand::Teardown).expect("provisioned sandbox tears down") {
            sbx.evolve(&e);
        }
        self.utilization.in_flight -= 1;

        Ok(match verdict {
            Verdict::Accepted => DrainOutcome::Accepted { unit_ref: unit.unit_ref },
            Verdict::Rejected | Verdict::Escalate => self.escalate(unit),
        })
    }
}

#[cfg(test)]
mod production_tests {
    use super::*;
    use spark_interface::{AcceptanceClass, Cell, Environment, ModelBinding};
    use spark_sandbox::{LocalBroker, LocalSandbox};
    use spark_serving::StubWorker;
    use std::collections::BTreeMap;

    struct PassOracle;
    impl Oracle for PassOracle {
        fn gate(&self, _c: &Cell) -> bool { true }
    }

    fn unit(r: &str) -> WorkUnit {
        let b = ModelBinding { model: "coder".into(), quantization: "q4".into(), params: BTreeMap::new() };
        WorkUnit {
            unit_ref: r.into(),
            parent_deliverable: "d".into(),
            bundle_hash: format!("sha256:{r}"),
            spmc_bundle: serde_json::json!({}),
            model_binding: b.clone(),
            tier: "light".into(),
            acceptance_class: AcceptanceClass::AutoCommitIfGreen,
            ladder_position: 0,
            cell_graph: vec![
                Cell { cell_id: "test".into(), binding: b.clone(), depends_on: vec![] },
                Cell { cell_id: "impl".into(), binding: b, depends_on: vec!["test".into()] },
            ],
            environment: Environment { network: vec![], workspace: "ws".into() },
            credential_grant: Some("grant-ref-1".into()),
            tool_grants: vec![],
        }
    }

    #[test]
    fn full_isolated_drain_provisions_runs_gates_emits_and_tears_down() {
        let dir = std::env::temp_dir().join("spark-prod-test");
        let _ = std::fs::remove_dir_all(&dir);
        let sandbox = LocalSandbox { root: dir.join("sandboxes") };
        let mut log = DurableLog::open(dir.join("verdicts.jsonl")).unwrap();

        let mut e = Engine::new();
        e.throw_switch(Mode::Queue).unwrap();
        e.admit(unit("u1")).unwrap();

        let out = e.drain_one_isolated(&sandbox, &LocalBroker, &StubWorker, &PassOracle, &mut log, "t1").unwrap();
        assert_eq!(out, DrainOutcome::Accepted { unit_ref: "u1".into() });
        assert_eq!(log.len(), 1);                       // durably logged
        assert_eq!(e.utilization.in_flight, 0);         // sandbox torn down
        assert!(!sandbox.root.join("u1").exists());     // workspace destroyed

        // Idempotency: re-emitting the same bundle_hash does not double the log.
        assert!(!log.append(&e.stream[0]).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn isolated_drain_outside_queue_is_refused() {
        let dir = std::env::temp_dir().join("spark-prod-test2");
        let sandbox = LocalSandbox { root: dir.join("sandboxes") };
        let mut log = DurableLog::open(dir.join("v.jsonl")).unwrap();
        let mut e = Engine::new();
        assert_eq!(
            e.drain_one_isolated(&sandbox, &LocalBroker, &StubWorker, &PassOracle, &mut log, "t0").unwrap(),
            DrainOutcome::NotQueueMode
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
