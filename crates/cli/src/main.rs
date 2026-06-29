//! `spark` — the developer-facing CLI over the executor Engine. The developer
//! switch is a deliberate human act here (top-level control); the executor loop
//! runs below it. State persists to `.spark/state.json` between invocations.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use spark_execution::{CommandOracle, Oracle};
use spark_executor::{DrainOutcome, Engine};
use spark_host::{HostPhase, HostSpec, SshVllmHost};
use spark_interface::{Cell, WorkUnit};
use spark_sandbox::{LocalBroker, LocalSandbox};
use spark_serving::{CommandWorker, OpenAiWorker, StubWorker, Worker};
use spark_stream::DurableLog;
use spark_switch::Mode;

/// The default protected oracle for the demo loop: it passes every cell. A real
/// deployment plugs in test execution the worker cannot write. Crucially the
/// worker never implements this trait — the oracle is external by construction.
struct PassingOracle;
impl Oracle for PassingOracle {
    fn gate(&self, _cell: &Cell) -> bool {
        true
    }
}

fn state_path() -> PathBuf {
    PathBuf::from(".spark/state.json")
}

fn load() -> Engine {
    match std::fs::read_to_string(state_path()) {
        Ok(t) => serde_json::from_str(&t).unwrap_or_default(),
        Err(_) => Engine::default(),
    }
}

fn save(engine: &Engine) -> std::io::Result<()> {
    std::fs::create_dir_all(".spark")?;
    std::fs::write(state_path(), serde_json::to_string_pretty(engine).expect("serialize engine"))
}

fn now() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    format!("t{secs}")
}

fn parse_mode(s: &str) -> Result<Mode, String> {
    match s {
        "queue" => Ok(Mode::Queue),
        "explorer" => Ok(Mode::Explorer),
        other => Err(format!("unknown mode '{other}' (use queue | explorer)")),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    match refs.as_slice() {
        ["mode", "set", m] => cmd_mode_set(m),
        ["status"] => cmd_status(),
        ["admit", file] => cmd_admit(file),
        ["run"] => cmd_run(),
        ["serve"] => cmd_serve(),
        ["explore"] => cmd_explore(),
        ["stream"] => cmd_stream(),
        _ => {
            eprintln!(
                "spark — Spark execution engine\n\n\
                 usage:\n  \
                 spark mode set <queue|explorer>   throw the developer switch\n  \
                 spark status                      show box mode + views\n  \
                 spark admit <work-unit.json>      admit a frozen WorkUnit (homogeneity guard)\n  \
                 spark run                         drain the work-unit set (QUEUE only)\n  \
                 spark serve                       drain isolated: sandbox+creds+worker+oracle+durable log\n  \
                 spark explore                     run a discovery session (EXPLORER only)\n  \
                 spark stream                      print the emitted VerdictEvents"
            );
            ExitCode::from(2)
        }
    }
}

fn cmd_mode_set(m: &str) -> ExitCode {
    let mode = match parse_mode(m) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let mut e = load();
    match e.throw_switch(mode) {
        Ok(now) => {
            println!("box -> {now:?} (residency loaded; the other mode is unloaded from VRAM)");
            // Physically materialize the residency if a box is configured: the
            // switch starts/stops the real vLLM container, not just a flag.
            materialize_if_configured(&mut e, mode);
            save(&e).ok();
            ExitCode::SUCCESS
        }
        Err(inv) => {
            eprintln!("rejected: {inv} (already in {mode:?} — flips are expensive, a no-op flip is refused)");
            ExitCode::from(1)
        }
    }
}

fn mode_lower(m: Mode) -> &'static str {
    match m {
        Mode::Off => "off",
        Mode::Queue => "queue",
        Mode::Explorer => "explorer",
    }
}

/// Build a `HostSpec` for the given mode from the environment, or `None` when no
/// box is configured (the offline/dev path — logical switch only). The model
/// served differs by mode (small coder for QUEUE, a large model for EXPLORER).
fn host_spec_for(mode: Mode) -> Option<HostSpec> {
    let ssh_target = std::env::var("SPARK_SSH_TARGET").ok()?;
    let model = match mode {
        Mode::Queue => std::env::var("SPARK_QUEUE_MODEL").ok(),
        Mode::Explorer => std::env::var("SPARK_EXPLORER_MODEL").ok(),
        Mode::Off => None,
    }
    .or_else(|| std::env::var("SPARK_VLLM_MODEL").ok())
    .unwrap_or_else(|| "default".into());
    let port = std::env::var("SPARK_VLLM_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(8000);
    let image = std::env::var("SPARK_VLLM_IMAGE").unwrap_or_else(|_| "vllm/vllm-openai:latest".into());
    let extra_args = std::env::var("SPARK_VLLM_ARGS")
        .ok()
        .map(|s| s.split_whitespace().map(String::from).collect())
        .unwrap_or_default();
    Some(HostSpec { host_id: format!("{}-host", mode_lower(mode)), model, image, ssh_target, port, extra_args })
}

/// Retire the prior host and launch + ready the new mode's vLLM container over
/// SSH. A no-op (logical switch only) when `SPARK_SSH_TARGET` is unset.
fn materialize_if_configured(e: &mut Engine, mode: Mode) {
    let Some(spec) = host_spec_for(mode) else {
        return;
    };
    let host = SshVllmHost::default();
    eprintln!("materializing residency: launching `{}` ({}) on {} ...", spec.image, spec.model, spec.ssh_target);
    match e.launch_residency(&host, &spec, Duration::from_secs(180)) {
        Ok(true) => {
            let url = e.host_handle.as_ref().map(|h| h.endpoint.as_str()).unwrap_or("?");
            println!("residency ready at {url}");
        }
        Ok(false) => eprintln!("host launched but /v1 did not answer in time — check the box; not dispatching yet"),
        Err(err) => eprintln!("residency materialization failed: {err}"),
    }
}

fn cmd_status() -> ExitCode {
    let e = load();
    println!("box mode:        {:?}", e.mode);
    println!("queued units:    {}", e.priority_view.queued);
    println!("verdicts:        {}", e.verdict_view.emitted);
    println!(
        "heterogeneity:   {}/{} rejected ({:.0}%)",
        e.heterogeneity_view.rejected,
        e.heterogeneity_view.admitted + e.heterogeneity_view.rejected,
        e.heterogeneity_view.rate() * 100.0
    );
    println!(
        "utilization:     in_flight={} exploring={} discoveries={}",
        e.utilization.in_flight, e.utilization.exploring, e.utilization.discoveries
    );
    println!(
        "serving host:    phase={:?} ready={} endpoint={}",
        e.host.phase,
        e.serving_host_view.ready,
        e.host_handle.as_ref().map(|h| h.endpoint.as_str()).unwrap_or("-")
    );
    ExitCode::SUCCESS
}

fn cmd_admit(file: &str) -> ExitCode {
    let text = match std::fs::read_to_string(file) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("cannot read {file}: {err}");
            return ExitCode::from(2);
        }
    };
    let unit: WorkUnit = match serde_json::from_str(&text) {
        Ok(u) => u,
        Err(err) => {
            eprintln!("invalid WorkUnit JSON: {err}");
            return ExitCode::from(2);
        }
    };
    let unit_ref = unit.unit_ref.clone();
    let mut e = load();
    match e.admit(unit) {
        Ok(()) => {
            save(&e).ok();
            println!("admitted '{unit_ref}' -> queued (binding-homogeneous)");
            ExitCode::SUCCESS
        }
        Err(inv) => {
            save(&e).ok();
            eprintln!("rejected '{unit_ref}': {inv} (a decomposition defect, never dispatched)");
            ExitCode::from(1)
        }
    }
}

fn cmd_run() -> ExitCode {
    let mut e = load();
    if e.mode != Mode::Queue {
        eprintln!("box is {:?}; the executor drains only in QUEUE. `spark mode set queue` first.", e.mode);
        return ExitCode::from(1);
    }
    let oracle = PassingOracle;
    let mut drained = 0;
    loop {
        match e.drain_one(&oracle, &now()) {
            DrainOutcome::Accepted { unit_ref } => println!("  accepted   {unit_ref}"),
            DrainOutcome::Escalated { unit_ref, to_ladder } => {
                println!("  escalated  {unit_ref} -> ladder {to_ladder} (whole unit, one binding up)")
            }
            DrainOutcome::Halted { unit_ref } => println!("  halted     {unit_ref} (ladder exhausted)"),
            DrainOutcome::Idle => break,
            DrainOutcome::NotQueueMode => break,
        }
        drained += 1;
        if drained > 10_000 {
            break; // safety backstop
        }
    }
    save(&e).ok();
    println!("drained {drained} unit-attempt(s); {} verdict(s) on the stream", e.stream.len());
    ExitCode::SUCCESS
}

/// The production drain: each unit runs inside a per-unit `LocalSandbox` with a
/// brokered credential lease, its frontier executed by a `Worker` (a real served
/// model via `$SPARK_WORKER_CMD`, else the deterministic stub) and gated by a
/// protected oracle (`$SPARK_ORACLE_CMD` the worker cannot write, else pass-all),
/// with every verdict appended to a durable, idempotent log on disk.
fn cmd_serve() -> ExitCode {
    let mut e = load();
    if e.mode != Mode::Queue {
        eprintln!("box is {:?}; isolated drain runs only in QUEUE. `spark mode set queue` first.", e.mode);
        return ExitCode::from(1);
    }
    let sandbox = LocalSandbox::default();
    let broker = LocalBroker;
    // Worker precedence: a residency materialized by `spark mode set` (the vLLM
    // host on the box) → an OpenAI-compatible server named by env → a per-cell
    // shell command → the deterministic offline stub.
    let worker: Box<dyn Worker> = if e.host.phase == HostPhase::Ready && e.host_handle.is_some() {
        let h = e.host_handle.clone().unwrap();
        eprintln!("worker: materialized vLLM host @ {}", h.endpoint);
        Box::new(OpenAiWorker { base_url: h.endpoint, model: h.model, api_key: std::env::var("SPARK_OPENAI_API_KEY").ok() })
    } else if let Some(http) = OpenAiWorker::from_env() {
        eprintln!("worker: OpenAI HTTP @ {}", http.base_url);
        Box::new(http)
    } else if let Ok(cmd) = std::env::var("SPARK_WORKER_CMD") {
        eprintln!("worker: command `{cmd}`");
        Box::new(CommandWorker { command: cmd })
    } else {
        eprintln!("worker: offline stub (set SPARK_OPENAI_BASE_URL or SPARK_WORKER_CMD for a real model)");
        Box::new(StubWorker)
    };
    // A protected oracle (worker_writable: false). With no command, fall back to
    // a trivially-passing protected oracle so the loop runs offline.
    let oracle: Box<dyn Oracle> = match std::env::var("SPARK_ORACLE_CMD") {
        Ok(cmd) => Box::new(CommandOracle { command: cmd, worker_writable: false }),
        Err(_) => Box::new(CommandOracle { command: "true".into(), worker_writable: false }),
    };
    let mut log = match DurableLog::open(".spark/verdicts.jsonl") {
        Ok(l) => l,
        Err(err) => {
            eprintln!("cannot open durable log: {err}");
            return ExitCode::from(2);
        }
    };

    let mut drained = 0;
    loop {
        let outcome = match e.drain_one_isolated(&sandbox, &broker, worker.as_ref(), oracle.as_ref(), &mut log, &now()) {
            Ok(o) => o,
            Err(err) => {
                eprintln!("isolated drain failed: {err}");
                save(&e).ok();
                return ExitCode::from(1);
            }
        };
        match outcome {
            DrainOutcome::Accepted { unit_ref } => println!("  accepted   {unit_ref}  (sandbox provisioned → worker → oracle → logged → torn down)"),
            DrainOutcome::Escalated { unit_ref, to_ladder } => println!("  escalated  {unit_ref} -> ladder {to_ladder}"),
            DrainOutcome::Halted { unit_ref } => println!("  halted     {unit_ref} (ladder exhausted)"),
            DrainOutcome::Idle | DrainOutcome::NotQueueMode => break,
        }
        drained += 1;
        if drained > 10_000 {
            break;
        }
    }
    save(&e).ok();
    println!("isolated-drained {drained} unit-attempt(s); durable log holds {} verdict(s) at .spark/verdicts.jsonl", log.len());
    ExitCode::SUCCESS
}

fn cmd_explore() -> ExitCode {
    let mut e = load();
    if let Err(inv) = e.start_exploration() {
        eprintln!("cannot explore: {inv} (the box must be in EXPLORER)");
        return ExitCode::from(1);
    }
    e.produce_discovery_record().ok();
    save(&e).ok();
    println!("discovery record produced (candidate structure — NOT accepted code)");
    ExitCode::SUCCESS
}

fn cmd_stream() -> ExitCode {
    let e = load();
    if e.stream.is_empty() {
        println!("(no verdicts emitted yet)");
        return ExitCode::SUCCESS;
    }
    for v in &e.stream {
        println!(
            "{}  unit={}  verdict={:?}  tier={}  next={:?}  hash={}",
            v.event_id, v.unit_ref, v.verdict, v.tier_ran, v.next_consequence, v.bundle_hash
        );
    }
    ExitCode::SUCCESS
}
