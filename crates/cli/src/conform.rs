//! §6.3 behavioural-conformance runner. Reads a JSON array of {given, when}
//! requests on stdin and writes a JSON array of {emit|reject} outcomes on
//! stdout, by replaying the *realised* Rust deciders. `product decider conform
//! <name> --runner 'spark-conform <name>'` compares these to the oracle.

use std::io::{Read, Write};

use serde_json::{json, Value};

use spark_exploration::{SessionCommand, SessionState};
use spark_execution::{OracleCommand, OracleRunState, RunCommand, RunState};
use spark_interface::WorkUnit;
use spark_queue::{UnitCommand, UnitEvent, UnitState};
use spark_host::{HostCommand, HostEvent, HostState};
use spark_sandbox::{LeaseCommand, LeaseEvent, LeaseState, SandboxCommand, SandboxEvent, SandboxState};
use spark_serving::{BatchCommand, BatchEvent, BatchState, BindingCommand, BindingEvent, ResidentState};
use spark_stream::{LogCommand, LogEvent, LogState};
use spark_switch::{BoxCommand, BoxEvent, BoxState, Mode};

fn mode(v: &Value, key: &str) -> Mode {
    match v.get(key).and_then(Value::as_str) {
        Some("queue") => Mode::Queue,
        Some("explorer") => Mode::Explorer,
        _ => Mode::Off,
    }
}
fn mode_str(m: Mode) -> &'static str {
    match m {
        Mode::Off => "off",
        Mode::Queue => "queue",
        Mode::Explorer => "explorer",
    }
}

/// (event_id, payload) of one `given` entry.
fn ev(e: &Value) -> (String, Value) {
    match e {
        Value::String(s) => (s.clone(), json!({})),
        o => (
            o.get("event").and_then(Value::as_str).unwrap_or("").to_string(),
            o.get("with").cloned().unwrap_or_else(|| json!({})),
        ),
    }
}
/// (command_id, payload) of the `when` entry.
fn cmd(c: &Value) -> (String, Value) {
    match c {
        Value::String(s) => (s.clone(), json!({})),
        o => (
            o.get("command").and_then(Value::as_str).unwrap_or("").to_string(),
            o.get("with").cloned().unwrap_or_else(|| json!({})),
        ),
    }
}

fn emit(ids: Vec<Value>) -> Value {
    json!({ "emit": ids })
}
fn reject(inv: &str) -> Value {
    json!({ "reject": inv })
}

fn box_decider(given: &[Value], when: &Value) -> Value {
    let mut st = BoxState::default();
    for g in given {
        let (id, p) = ev(g);
        if id == "box-mode-changed" {
            st.evolve(&BoxEvent::ModeChanged { mode: mode(&p, "mode") });
        }
    }
    let (id, p) = cmd(when);
    if id != "throw-switch" {
        return reject("unknown-command");
    }
    let m = mode(&p, "mode");
    match st.decide(&BoxCommand::ThrowSwitch { mode: m }) {
        Ok(_) => emit(vec![json!({ "event": "box-mode-changed", "with": { "mode": mode_str(m) } })]),
        Err(inv) => reject(inv),
    }
}

fn replay_unit(given: &[Value]) -> UnitState {
    let mut st = UnitState::default();
    for g in given {
        let (id, _) = ev(g);
        let e = match id.as_str() {
            "work-unit-enqueued" => Some(UnitEvent::Enqueued),
            "work-unit-reprioritized" => Some(UnitEvent::Reprioritized),
            "unit-escalated" => Some(UnitEvent::Escalated),
            "work-unit-rejected" => Some(UnitEvent::Rejected),
            "unit-halted" => Some(UnitEvent::Halted),
            _ => None,
        };
        if let Some(e) = e {
            st.evolve(&e);
        }
    }
    st
}

fn unit_event_id(e: &UnitEvent) -> &'static str {
    match e {
        UnitEvent::Enqueued => "work-unit-enqueued",
        UnitEvent::Reprioritized => "work-unit-reprioritized",
        UnitEvent::Escalated => "unit-escalated",
        UnitEvent::Rejected => "work-unit-rejected",
        UnitEvent::Halted => "unit-halted",
    }
}

fn work_unit_decider(given: &[Value], when: &Value) -> Value {
    let st = replay_unit(given);
    let (id, p) = cmd(when);
    let command = match id.as_str() {
        "admit-work-unit" => UnitCommand::Admit { homogeneous: p.get("homogeneous").and_then(Value::as_bool).unwrap_or(false) },
        "reprioritize-work-unit" => UnitCommand::Reprioritize,
        "escalate-unit" => UnitCommand::Escalate { max_ladder: p.get("max_ladder").and_then(Value::as_u64).unwrap_or(0) as u32 },
        _ => return reject("unknown-command"),
    };
    match st.decide(&command) {
        Ok(events) => emit(events.iter().map(|e| json!(unit_event_id(e))).collect()),
        Err(inv) => reject(inv),
    }
}

fn execution_run_decider(given: &[Value], when: &Value) -> Value {
    use spark_execution::RunEvent;
    let mut st = RunState::default();
    for g in given {
        let (id, _) = ev(g);
        let e = match id.as_str() {
            "unit-admitted" => Some(RunEvent::UnitAdmitted),
            "unit-verdict-computed" => Some(RunEvent::UnitVerdictComputed),
            "verdict-emitted" => Some(RunEvent::VerdictEmitted),
            _ => None,
        };
        if let Some(e) = e {
            st.evolve(&e);
        }
    }
    let (id, p) = cmd(when);
    let command = match id.as_str() {
        "admit-to-executor" => RunCommand::Admit { box_mode: mode(&p, "box_mode") },
        "compute-unit-verdict" => RunCommand::ComputeVerdict,
        "emit-verdict" => RunCommand::Emit,
        _ => return reject("unknown-command"),
    };
    let name = |e: &RunEvent| match e {
        RunEvent::UnitAdmitted => "unit-admitted",
        RunEvent::UnitVerdictComputed => "unit-verdict-computed",
        RunEvent::VerdictEmitted => "verdict-emitted",
    };
    match st.decide(&command) {
        Ok(events) => emit(events.iter().map(|e| json!(name(e))).collect()),
        Err(inv) => reject(inv),
    }
}

fn exploration_decider(given: &[Value], when: &Value) -> Value {
    use spark_exploration::SessionEvent;
    let mut st = SessionState::default();
    for g in given {
        let (id, _) = ev(g);
        if id == "exploration-started" {
            st.evolve(&SessionEvent::ExplorationStarted);
        } else if id == "discovery-record-produced" {
            st.evolve(&SessionEvent::DiscoveryRecordProduced);
        }
    }
    let (id, p) = cmd(when);
    let command = match id.as_str() {
        "start-exploration" => SessionCommand::Start { box_mode: mode(&p, "box_mode") },
        "produce-discovery-record" => SessionCommand::ProduceDiscoveryRecord,
        _ => return reject("unknown-command"),
    };
    let name = |e: &SessionEvent| match e {
        SessionEvent::ExplorationStarted => "exploration-started",
        SessionEvent::DiscoveryRecordProduced => "discovery-record-produced",
    };
    match st.decide(&command) {
        Ok(events) => emit(events.iter().map(|e| json!(name(e))).collect()),
        Err(inv) => reject(inv),
    }
}

fn bool_of(p: &Value, key: &str) -> bool {
    p.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn resident_set_decider(given: &[Value], when: &Value) -> Value {
    let mut st = ResidentState::default();
    for g in given {
        match ev(g).0.as_str() {
            "binding-loaded" => st.evolve(&BindingEvent::BindingLoaded),
            "binding-evicted" => st.evolve(&BindingEvent::BindingEvicted),
            _ => {}
        }
    }
    let (id, p) = cmd(when);
    let command = match id.as_str() {
        "load-binding" => BindingCommand::Load { within_budget: bool_of(&p, "within_budget") },
        "evict-binding" => BindingCommand::Evict,
        _ => return reject("unknown-command"),
    };
    let name = |e: &BindingEvent| match e {
        BindingEvent::BindingLoaded => "binding-loaded",
        BindingEvent::BindingEvicted => "binding-evicted",
    };
    match st.decide(&command) {
        Ok(es) => emit(es.iter().map(|e| json!(name(e))).collect()),
        Err(inv) => reject(inv),
    }
}

fn work_batch_decider(given: &[Value], when: &Value) -> Value {
    let mut st = BatchState::default();
    for g in given {
        match ev(g).0.as_str() {
            "batch-formed" => st.evolve(&BatchEvent::BatchFormed),
            "batch-dispatched" => st.evolve(&BatchEvent::BatchDispatched),
            _ => {}
        }
    }
    let (id, p) = cmd(when);
    let command = match id.as_str() {
        "form-batch" => BatchCommand::Form { homogeneous: bool_of(&p, "homogeneous"), nonempty: bool_of(&p, "nonempty") },
        "dispatch-batch" => BatchCommand::Dispatch,
        _ => return reject("unknown-command"),
    };
    let name = |e: &BatchEvent| match e {
        BatchEvent::BatchFormed => "batch-formed",
        BatchEvent::BatchDispatched => "batch-dispatched",
    };
    match st.decide(&command) {
        Ok(es) => emit(es.iter().map(|e| json!(name(e))).collect()),
        Err(inv) => reject(inv),
    }
}

fn sandbox_decider(given: &[Value], when: &Value) -> Value {
    let mut st = SandboxState::default();
    for g in given {
        match ev(g).0.as_str() {
            "sandbox-provisioned" => st.evolve(&SandboxEvent::SandboxProvisioned),
            "sandbox-destroyed" => st.evolve(&SandboxEvent::SandboxDestroyed),
            _ => {}
        }
    }
    let (id, p) = cmd(when);
    let command = match id.as_str() {
        "provision-sandbox" => SandboxCommand::Provision { network_declared: bool_of(&p, "network_declared") },
        "teardown-sandbox" => SandboxCommand::Teardown,
        _ => return reject("unknown-command"),
    };
    let name = |e: &SandboxEvent| match e {
        SandboxEvent::SandboxProvisioned => "sandbox-provisioned",
        SandboxEvent::SandboxDestroyed => "sandbox-destroyed",
    };
    match st.decide(&command) {
        Ok(es) => emit(es.iter().map(|e| json!(name(e))).collect()),
        Err(inv) => reject(inv),
    }
}

fn credential_lease_decider(given: &[Value], when: &Value) -> Value {
    let mut st = LeaseState::default();
    for g in given {
        match ev(g).0.as_str() {
            "credential-leased" => st.evolve(&LeaseEvent::CredentialLeased),
            "credential-revoked" => st.evolve(&LeaseEvent::CredentialRevoked),
            _ => {}
        }
    }
    let (id, p) = cmd(when);
    let command = match id.as_str() {
        "exchange-credential" => LeaseCommand::Exchange { sandbox_active: bool_of(&p, "sandbox_active") },
        "revoke-credential" => LeaseCommand::Revoke,
        _ => return reject("unknown-command"),
    };
    let name = |e: &LeaseEvent| match e {
        LeaseEvent::CredentialLeased => "credential-leased",
        LeaseEvent::CredentialRevoked => "credential-revoked",
    };
    match st.decide(&command) {
        Ok(es) => emit(es.iter().map(|e| json!(name(e))).collect()),
        Err(inv) => reject(inv),
    }
}

fn verdict_log_decider(given: &[Value], when: &Value) -> Value {
    let mut st = LogState::default();
    for g in given {
        match ev(g).0.as_str() {
            "verdict-appended" => st.evolve(&LogEvent::VerdictAppended),
            "deliverable-reconciled" => st.evolve(&LogEvent::DeliverableReconciled),
            _ => {}
        }
    }
    let (id, p) = cmd(when);
    let command = match id.as_str() {
        "append-verdict" => LogCommand::Append { already_logged: bool_of(&p, "already_logged") },
        "reconcile-deliverable" => LogCommand::Reconcile,
        _ => return reject("unknown-command"),
    };
    let name = |e: &LogEvent| match e {
        LogEvent::VerdictAppended => "verdict-appended",
        LogEvent::DeliverableReconciled => "deliverable-reconciled",
    };
    match st.decide(&command) {
        Ok(es) => emit(es.iter().map(|e| json!(name(e))).collect()),
        Err(inv) => reject(inv),
    }
}

fn oracle_run_decider(_given: &[Value], when: &Value) -> Value {
    let st = OracleRunState::default();
    let (id, p) = cmd(when);
    let command = match id.as_str() {
        "run-gate" => OracleCommand::RunGate { oracle_protected: bool_of(&p, "oracle_protected") },
        _ => return reject("unknown-command"),
    };
    match st.decide(&command) {
        Ok(es) => emit(es.iter().map(|_| json!("gate-confirmed")).collect()),
        Err(inv) => reject(inv),
    }
}

fn serving_host_decider(given: &[Value], when: &Value) -> Value {
    let mut st = HostState::default();
    for g in given {
        match ev(g).0.as_str() {
            "host-launched" => st.evolve(&HostEvent::HostLaunched),
            "host-ready" => st.evolve(&HostEvent::HostReady),
            "host-retired" => st.evolve(&HostEvent::HostRetired),
            _ => {}
        }
    }
    let (id, p) = cmd(when);
    let command = match id.as_str() {
        "launch-host" => HostCommand::Launch { containerized: bool_of(&p, "containerized") },
        "confirm-host-ready" => HostCommand::ConfirmReady,
        "retire-host" => HostCommand::Retire,
        _ => return reject("unknown-command"),
    };
    let name = |e: &HostEvent| match e {
        HostEvent::HostLaunched => "host-launched",
        HostEvent::HostReady => "host-ready",
        HostEvent::HostRetired => "host-retired",
    };
    match st.decide(&command) {
        Ok(es) => emit(es.iter().map(|e| json!(name(e))).collect()),
        Err(inv) => reject(inv),
    }
}

fn main() {
    let _ = WorkUnit::is_binding_homogeneous; // keep interface linked for clarity
    let which = std::env::args().nth(1).unwrap_or_default();
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).expect("read stdin");
    let requests: Vec<Value> = serde_json::from_str(&input).expect("requests JSON array");
    let outcomes: Vec<Value> = requests
        .iter()
        .map(|r| {
            let given = r.get("given").and_then(Value::as_array).cloned().unwrap_or_default();
            let when = r.get("when").cloned().unwrap_or(Value::Null);
            match which.as_str() {
                "box-decider" => box_decider(&given, &when),
                "work-unit-decider" => work_unit_decider(&given, &when),
                "execution-run-decider" => execution_run_decider(&given, &when),
                "exploration-session-decider" => exploration_decider(&given, &when),
                "resident-set-decider" => resident_set_decider(&given, &when),
                "work-batch-decider" => work_batch_decider(&given, &when),
                "sandbox-decider" => sandbox_decider(&given, &when),
                "credential-lease-decider" => credential_lease_decider(&given, &when),
                "verdict-log-decider" => verdict_log_decider(&given, &when),
                "oracle-run-decider" => oracle_run_decider(&given, &when),
                "serving-host-decider" => serving_host_decider(&given, &when),
                _ => reject("unknown-decider"),
            }
        })
        .collect();
    let mut out = std::io::stdout();
    out.write_all(serde_json::to_string(&outcomes).unwrap().as_bytes()).unwrap();
}
