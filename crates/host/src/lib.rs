//! Serving Host — the physical materialization of a box residency: a vLLM
//! server launched as a container on the box over SSH that actually backs the
//! logical resident set. Where `spark-switch` sets the logical mode and
//! `spark-serving` tracks logical VRAM residency, this crate owns the real
//! process: launch the container, wait until its `/v1` endpoint answers, serve,
//! then retire it to free VRAM.
//!
//! It realises `serving-host-decider`, the `serving-host-view` projector, and
//! the `ResidencyHost` seam — a `LocalProcessHost` standing in for an
//! already-running local server, and an `SshVllmHost` that launches vLLM in a
//! container on the box over SSH. The reused `OpenAiWorker` (in `spark-serving`)
//! dispatches inference at the endpoint a launched host returns.

use std::net::{TcpStream, ToSocketAddrs};
use std::process::Command;
use std::time::Duration;

use serde::{Deserialize, Serialize};

// ───────────────────────── serving-host-decider ─────────────────────────

/// The host lifecycle phase: nothing launched, container started but not yet
/// answering, health-ready and serving, or retired (container stopped).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostPhase {
    Offline,
    Launching,
    Ready,
    Retired,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostEvent {
    HostLaunched,
    HostReady,
    HostRetired,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostCommand {
    /// `containerized` = the launch request runs the server as a container.
    /// A bare-metal launch (`false`) is refused — vLLM always runs containerized.
    Launch { containerized: bool },
    ConfirmReady,
    Retire,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostState {
    pub phase: HostPhase,
}
impl Default for HostState {
    fn default() -> Self {
        HostState { phase: HostPhase::Offline }
    }
}
impl HostState {
    pub fn evolve(&mut self, e: &HostEvent) {
        match e {
            HostEvent::HostLaunched => self.phase = HostPhase::Launching,
            HostEvent::HostReady => self.phase = HostPhase::Ready,
            HostEvent::HostRetired => self.phase = HostPhase::Retired,
        }
    }
    pub fn decide(&self, c: &HostCommand) -> Result<Vec<HostEvent>, &'static str> {
        match c {
            // Guard order mirrors the decider: containerized first, then the
            // single-residency rule — so a non-containerized request is refused
            // even when a host is already live.
            HostCommand::Launch { containerized } => {
                if !*containerized {
                    return Err("inv-containerized-host");
                }
                match self.phase {
                    HostPhase::Offline | HostPhase::Retired => Ok(vec![HostEvent::HostLaunched]),
                    _ => Err("inv-single-serving-host"),
                }
            }
            HostCommand::ConfirmReady => {
                if self.phase == HostPhase::Launching {
                    Ok(vec![HostEvent::HostReady])
                } else {
                    Err("inv-ready-needs-launch")
                }
            }
            HostCommand::Retire => {
                if matches!(self.phase, HostPhase::Launching | HostPhase::Ready) {
                    Ok(vec![HostEvent::HostRetired])
                } else {
                    Err("inv-nothing-to-retire")
                }
            }
        }
    }
}

// ───────────────────────── serving-host-view projector ──────────────────

/// The current materialized host folded from its events: how many hosts are
/// live (0 or 1 by `inv-single-serving-host`) and whether the live one is ready.
/// This is what the executor reads to learn whether to dispatch.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServingHostView {
    pub materialized: i64,
    pub ready: bool,
}
impl ServingHostView {
    pub fn apply(&mut self, e: &HostEvent) {
        match e {
            HostEvent::HostLaunched => {
                self.materialized += 1;
                self.ready = false;
            }
            HostEvent::HostReady => self.ready = true,
            HostEvent::HostRetired => {
                self.materialized -= 1;
                self.ready = false;
            }
        }
    }
}

// ───────────────────────── ResidencyHost seam ───────────────────────────

/// What residency to materialize: one binding, on a box reached over SSH, served
/// by a container image on a port. The homogeneous binding is reduced to the
/// served `model` name vLLM is asked to load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostSpec {
    pub host_id: String,
    pub model: String,
    pub image: String,
    pub ssh_target: String,
    pub port: u16,
    /// Extra vLLM server flags, e.g. `--quantization awq`, `--max-model-len 8192`.
    pub extra_args: Vec<String>,
}

/// A live host: the endpoint base URL clients dispatch against, the served model
/// name (so a worker can target it), plus what the backend needs to retire it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostHandle {
    pub endpoint: String,
    pub model: String,
    pub container: String,
    pub ssh_target: String,
}

/// The seam that materializes a residency physically. `launch` brings a server
/// up and hands back its handle; the caller then polls readiness ([`probe_ready`])
/// before `confirm-host-ready`. `retire` stops it and frees VRAM. A real GPU box
/// uses [`SshVllmHost`]; [`LocalProcessHost`] stands in for an already-running
/// local server in development.
pub trait ResidencyHost {
    fn launch(&self, spec: &HostSpec) -> std::io::Result<HostHandle>;
    fn retire(&self, handle: &HostHandle) -> std::io::Result<()>;
}

/// Probe whether a host endpoint is accepting connections — a dependency-free
/// readiness check (TCP connect to host:port) the caller loops on before
/// confirming the host ready, so no batch is dispatched against a cold server.
pub fn probe_ready(host: &str, port: u16, timeout: Duration) -> bool {
    let Ok(mut addrs) = (host, port).to_socket_addrs() else {
        return false;
    };
    addrs.any(|addr| TcpStream::connect_timeout(&addr, timeout).is_ok())
}

/// Dev backend: assumes a model server is already running locally (the human ran
/// `vllm serve` / `llama-server`, or `SPARK_OPENAI_BASE_URL` points at it).
/// `launch` returns its endpoint without managing a process; `retire` is a no-op.
pub struct LocalProcessHost {
    pub base_url: String,
}
impl Default for LocalProcessHost {
    fn default() -> Self {
        LocalProcessHost { base_url: "http://127.0.0.1:8000".into() }
    }
}
impl ResidencyHost for LocalProcessHost {
    fn launch(&self, spec: &HostSpec) -> std::io::Result<HostHandle> {
        Ok(HostHandle {
            endpoint: self.base_url.clone(),
            model: spec.model.clone(),
            container: format!("local-{}", spec.host_id),
            ssh_target: String::new(),
        })
    }
    fn retire(&self, _handle: &HostHandle) -> std::io::Result<()> {
        Ok(())
    }
}

/// Production backend: launches vLLM as a detached container on the box over SSH.
/// The remote command is built by a pure function ([`SshVllmHost::launch_command`])
/// so it is unit-testable without a box; `launch`/`retire` shell out to `ssh`.
pub struct SshVllmHost {
    /// The ssh binary (overridable in tests); default `ssh`.
    pub ssh: String,
}
impl Default for SshVllmHost {
    fn default() -> Self {
        SshVllmHost { ssh: "ssh".into() }
    }
}
impl SshVllmHost {
    /// A stable, shell-safe container name derived from the host id.
    pub fn container_name(spec: &HostSpec) -> String {
        let safe: String =
            spec.host_id.chars().map(|c| if c.is_alphanumeric() || c == '-' { c } else { '-' }).collect();
        format!("spark-vllm-{safe}")
    }

    /// The remote `docker run` invocation that starts a **detached** vLLM
    /// container (`-d`) serving an OpenAI-compatible endpoint on `spec.port`.
    /// Detachment is what keeps the server alive after the SSH session closes.
    pub fn launch_command(spec: &HostSpec) -> String {
        let name = Self::container_name(spec);
        let mut parts = vec![
            "docker run -d --rm".to_string(),
            "--gpus all".to_string(),
            format!("--name {name}"),
            format!("-p {0}:{0}", spec.port),
            spec.image.clone(),
            "--host 0.0.0.0".to_string(),
            format!("--port {}", spec.port),
            format!("--model {}", spec.model),
        ];
        parts.extend(spec.extra_args.iter().cloned());
        parts.join(" ")
    }

    /// The endpoint clients reach: the ssh host (sans `user@`) on `spec.port`.
    pub fn endpoint(spec: &HostSpec) -> String {
        let host = spec.ssh_target.rsplit('@').next().unwrap_or(&spec.ssh_target);
        format!("http://{host}:{}", spec.port)
    }

    /// The remote command that stops and removes the container, freeing VRAM.
    pub fn retire_command(handle: &HostHandle) -> String {
        format!("docker rm -f {}", handle.container)
    }
}
impl ResidencyHost for SshVllmHost {
    fn launch(&self, spec: &HostSpec) -> std::io::Result<HostHandle> {
        let status = Command::new(&self.ssh).arg(&spec.ssh_target).arg(Self::launch_command(spec)).status()?;
        if !status.success() {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("ssh launch failed: {status}")));
        }
        Ok(HostHandle {
            endpoint: Self::endpoint(spec),
            model: spec.model.clone(),
            container: Self::container_name(spec),
            ssh_target: spec.ssh_target.clone(),
        })
    }
    fn retire(&self, handle: &HostHandle) -> std::io::Result<()> {
        let status =
            Command::new(&self.ssh).arg(&handle.ssh_target).arg(Self::retire_command(handle)).status()?;
        if !status.success() {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("ssh retire failed: {status}")));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay(es: &[HostEvent]) -> HostState {
        let mut s = HostState::default();
        for e in es {
            s.evolve(e);
        }
        s
    }

    fn spec() -> HostSpec {
        HostSpec {
            host_id: "queue-coder".into(),
            model: "qwen2.5-coder-7b".into(),
            image: "vllm/vllm-openai:latest".into(),
            ssh_target: "dev@spark-abcd.local".into(),
            port: 8000,
            extra_args: vec!["--quantization".into(), "awq".into()],
        }
    }

    // ── decider: the four deliverable criteria ───────────────────────────

    #[test]
    fn containerized_host_launches_when_nothing_is_resident() {
        assert_eq!(
            replay(&[]).decide(&HostCommand::Launch { containerized: true }),
            Ok(vec![HostEvent::HostLaunched])
        );
    }

    #[test]
    fn bare_metal_launch_is_refused() {
        assert_eq!(
            replay(&[]).decide(&HostCommand::Launch { containerized: false }),
            Err("inv-containerized-host")
        );
    }

    #[test]
    fn second_launch_is_refused_while_one_is_live() {
        assert_eq!(
            replay(&[HostEvent::HostLaunched]).decide(&HostCommand::Launch { containerized: true }),
            Err("inv-single-serving-host")
        );
    }

    #[test]
    fn confirm_ready_requires_launching() {
        // launching → ready is allowed; offline is refused.
        assert_eq!(
            replay(&[HostEvent::HostLaunched]).decide(&HostCommand::ConfirmReady),
            Ok(vec![HostEvent::HostReady])
        );
        assert_eq!(replay(&[]).decide(&HostCommand::ConfirmReady), Err("inv-ready-needs-launch"));
    }

    #[test]
    fn retire_requires_a_live_host() {
        // a ready host retires; nothing-live is refused.
        assert_eq!(
            replay(&[HostEvent::HostLaunched, HostEvent::HostReady]).decide(&HostCommand::Retire),
            Ok(vec![HostEvent::HostRetired])
        );
        assert_eq!(replay(&[]).decide(&HostCommand::Retire), Err("inv-nothing-to-retire"));
    }

    #[test]
    fn relaunch_is_allowed_after_retire() {
        assert_eq!(
            replay(&[HostEvent::HostLaunched, HostEvent::HostRetired])
                .decide(&HostCommand::Launch { containerized: true }),
            Ok(vec![HostEvent::HostLaunched])
        );
    }

    // ── projector ────────────────────────────────────────────────────────

    #[test]
    fn view_folds_materialization_and_readiness() {
        let mut v = ServingHostView::default();
        for e in [HostEvent::HostLaunched, HostEvent::HostReady] {
            v.apply(&e);
        }
        assert_eq!(v, ServingHostView { materialized: 1, ready: true });
        v.apply(&HostEvent::HostRetired);
        assert_eq!(v, ServingHostView { materialized: 0, ready: false });
    }

    // ── SSH/vLLM backend: pure command construction ──────────────────────

    #[test]
    fn ssh_launch_command_is_a_detached_vllm_container() {
        let cmd = SshVllmHost::launch_command(&spec());
        assert!(cmd.starts_with("docker run -d --rm"), "must be detached: {cmd}");
        assert!(cmd.contains("--gpus all"));
        assert!(cmd.contains("--name spark-vllm-queue-coder"));
        assert!(cmd.contains("-p 8000:8000"));
        assert!(cmd.contains("vllm/vllm-openai:latest"));
        assert!(cmd.contains("--model qwen2.5-coder-7b"));
        assert!(cmd.contains("--quantization awq"));
    }

    #[test]
    fn endpoint_drops_the_ssh_user() {
        assert_eq!(SshVllmHost::endpoint(&spec()), "http://spark-abcd.local:8000");
    }

    #[test]
    fn retire_command_force_removes_the_container() {
        let h = HostHandle {
            endpoint: "http://spark-abcd.local:8000".into(),
            model: "qwen2.5-coder-7b".into(),
            container: "spark-vllm-queue-coder".into(),
            ssh_target: "dev@spark-abcd.local".into(),
        };
        assert_eq!(SshVllmHost::retire_command(&h), "docker rm -f spark-vllm-queue-coder");
    }

    // ── readiness probe ──────────────────────────────────────────────────

    #[test]
    fn probe_detects_a_listening_then_closed_port() {
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(probe_ready("127.0.0.1", port, Duration::from_millis(200)));
        drop(listener);
        assert!(!probe_ready("127.0.0.1", port, Duration::from_millis(200)));
    }
}
