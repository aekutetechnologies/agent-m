//! Slack transport: the remote-human seam for agent-m.
//!
//! Outbound = Slack Web API (`chat.postMessage`, plain HTTPS POST via
//! reqwest). Inbound = Socket Mode (a long-lived websocket opened through
//! `apps.connections.open`), which needs no public endpoint.
//!
//! `SlackTransport` is the trait every remote-human feature uses (ask tool,
//! approval gate, progress notifier). `SlackClient` talks to real Slack;
//! `FakeTransport` records posts and injects events so tests never touch the
//! network.
//!
//! Env: `SLACK_APP_TOKEN` (xapp-…, Socket Mode) and `SLACK_BOT_TOKEN`
//! (xoxb-…, Web API).

use agent_m_agent::{AgentEvent, ClosureAskGate, Permission, RiskPolicy, ToolCallInfo};
use anyhow::Result;
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

/// An inbound Slack event the harness cares about.
#[allow(dead_code)] // `user`/`ts` are used by Phase 2 (approvals, audit)
#[derive(Debug, Clone)]
pub enum Inbound {
    /// A message in a channel or DM.
    Message { channel: String, user: String, text: String, ts: String },
}

#[async_trait]
pub trait SlackTransport: Send + Sync {
    /// Post a message to a channel (for a DM, `channel` is the user id).
    async fn post_message(&self, channel: &str, text: &str) -> Result<(), String>;
    /// Blocking event loop: connect and deliver inbound events to `on_event`.
    async fn run(&self, on_event: Arc<dyn Fn(Inbound) + Send + Sync>)
        -> Result<(), String>;
}

/// Real Slack transport: Socket Mode inbound (websocket), Web API outbound.
pub struct SlackClient {
    app_token: String, // xapp-1-… (Socket Mode)
    bot_token: String, // xoxb-… (Web API)
    http: reqwest::Client,
}

impl SlackClient {
    pub fn from_env() -> Result<Self, String> {
        let app_token = std::env::var("SLACK_APP_TOKEN").map_err(|_| {
            "SLACK_APP_TOKEN is not set (Socket Mode app-level token, xapp-…)".to_string()
        })?;
        let bot_token = std::env::var("SLACK_BOT_TOKEN").map_err(|_| {
            "SLACK_BOT_TOKEN is not set (bot user OAuth token, xoxb-…)".to_string()
        })?;
        Ok(Self { app_token, bot_token, http: reqwest::Client::new() })
    }

    async fn open_socket_url(&self) -> Result<String, String> {
        let resp = self
            .http
            .post("https://slack.com/api/apps.connections.open")
            .bearer_auth(&self.app_token)
            .send()
            .await
            .map_err(|e| format!("apps.connections.open: {e}"))?
            .json::<serde_json::Value>()
            .await
            .map_err(|e| format!("parse apps.connections.open: {e}"))?;
        if resp["ok"].as_bool() == Some(true) {
            resp["url"]
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| "apps.connections.open returned no url".to_string())
        } else {
            Err(format!(
                "slack error: {}",
                resp["error"].as_str().unwrap_or("unknown")
            ))
        }
    }
}

#[async_trait]
impl SlackTransport for SlackClient {
    async fn post_message(&self, channel: &str, text: &str) -> Result<(), String> {
        let resp = self
            .http
            .post("https://slack.com/api/chat.postMessage")
            .bearer_auth(&self.bot_token)
            .json(&serde_json::json!({ "channel": channel, "text": text }))
            .send()
            .await
            .map_err(|e| format!("chat.postMessage: {e}"))?
            .json::<serde_json::Value>()
            .await
            .map_err(|e| format!("parse chat.postMessage: {e}"))?;
        if resp["ok"].as_bool() == Some(true) {
            Ok(())
        } else {
            Err(format!(
                "slack error: {}",
                resp["error"].as_str().unwrap_or("unknown")
            ))
        }
    }

    async fn run(&self, on_event: Arc<dyn Fn(Inbound) + Send + Sync>)
        -> Result<(), String> {
        let url = self.open_socket_url().await?;
        let (mut ws, _) = tokio_tungstenite::connect_async(&url)
            .await
            .map_err(|e| format!("socket connect: {e}"))?;
        loop {
            match ws.next().await {
                Some(Ok(msg)) => {
                    let text = msg
                        .into_text()
                        .map_err(|e| format!("websocket text: {e}"))?
                        .to_string();
                    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
                        continue;
                    };
                    match value["type"].as_str() {
                        // Socket Mode greeting.
                        Some("hello") => {}
                        Some("events_api") => {
                            let envelope = value["envelope_id"].as_str().unwrap_or("");
                            let ack =
                                serde_json::json!({ "type": "ack", "envelope_id": envelope })
                                    .to_string();
                            ws.send(tokio_tungstenite::tungstenite::Message::Text(
                                ack.into(),
                            ))
                            .await
                            .map_err(|e| format!("websocket ack: {e}"))?;
                            let event = &value["payload"]["event"];
                            if event["type"].as_str() == Some("message") {
                                let channel = event["channel"].as_str().unwrap_or("").to_string();
                                let user = event["user"].as_str().unwrap_or("").to_string();
                                let text = event["text"].as_str().unwrap_or("").to_string();
                                let ts = event["ts"].as_str().unwrap_or("").to_string();
                                on_event(Inbound::Message { channel, user, text, ts });
                            }
                        }
                        _ => {}
                    }
                }
                Some(Err(e)) => return Err(format!("websocket error: {e}")),
                None => return Ok(()),
            }
        }
    }
}

/// In-memory transport for tests (and later, dry-run mode): records posted
/// messages and lets tests inject inbound events.
#[allow(dead_code)] // used by tests; dry-run mode in Phase 2
pub struct FakeTransport {
    posted: Mutex<Vec<(String, String)>>,
    inbox: Mutex<VecDeque<Inbound>>,
    wake: Notify,
}

impl Default for FakeTransport {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)] // used by tests; dry-run mode in Phase 2
impl FakeTransport {
    pub fn new() -> Self {
        Self {
            posted: Mutex::new(Vec::new()),
            inbox: Mutex::new(VecDeque::new()),
            wake: Notify::new(),
        }
    }

    /// Inject an inbound event, as if it arrived over the websocket.
    pub fn push(&self, event: Inbound) {
        self.inbox.lock().unwrap().push_back(event);
        self.wake.notify_one();
    }

    /// The (channel, text) pairs posted so far, in order.
    pub fn posted(&self) -> Vec<(String, String)> {
        self.posted.lock().unwrap().clone()
    }
}

#[async_trait]
impl SlackTransport for FakeTransport {
    async fn post_message(&self, channel: &str, text: &str) -> Result<(), String> {
        self.posted
            .lock()
            .unwrap()
            .push((channel.to_string(), text.to_string()));
        Ok(())
    }

    async fn run(&self, on_event: Arc<dyn Fn(Inbound) + Send + Sync>)
        -> Result<(), String> {
        loop {
            let item = self.inbox.lock().unwrap().pop_front();
            if let Some(item) = item {
                on_event(item);
                continue;
            }
            tokio::select! {
                _ = self.wake.notified() => {}
                _ = tokio::time::sleep(Duration::from_millis(50)) => {}
            }
        }
    }
}

/// Parse a reply of the form `ask-3 approve the plan` into (id, answer).
pub fn split_answer(text: &str) -> Option<(String, String)> {
    let mut words = text.split_whitespace();
    let id = words.next()?;
    if !id.starts_with("ask-") {
        return None;
    }
    let answer = words.collect::<Vec<_>>().join(" ");
    if answer.is_empty() {
        None
    } else {
        Some((id.to_string(), answer))
    }
}

/// Handle one inbound message: resolve a pending ask, otherwise echo back.
/// This is the DM handler for `agent-m slack` (Phase 1); Phase 2 adds
/// reactions → approval decisions on top.
pub async fn handle_inbound(
    inbound: Inbound,
    human: &crate::human::HumanChannel,
    transport: &dyn SlackTransport,
) {
    let Inbound::Message { channel, text, .. } = inbound;
    if let Some((id, answer)) = split_answer(&text) {
        if human.resolve(&id, answer.clone()) {
            let _ = transport
                .post_message(&channel, &format!("✅ answered `{id}`: {answer}"))
                .await;
        }
        return;
    }
    let reply = format!("agent-m online. You said: {text}");
    let _ = transport.post_message(&channel, &reply).await;
}

/// Map an agent event to a short Slack line (`None` = not worth posting).
/// This is the progress notifier's text rule; wiring into a running agent
/// lands in Phase 2 (daemon/flow progress posts).
pub fn event_to_slack_text(event: &AgentEvent) -> Option<String> {
    use AgentEvent::*;
    match event {
        ToolExecutionStart { name, .. } => Some(format!("🔧 `{name}`…")),
        ToolExecutionEnd { outcome, .. } if outcome.is_error => {
            Some("✗ a tool failed — see the session log".to_string())
        }
        TurnEnd { tool_results, .. } => {
            Some(format!("✅ turn complete ({tool_results} tool call(s))"))
        }
        Notice { message } => Some(format!("⚠️ {message}")),
        _ => None,
    }
}

/// Subscribe to agent events and post a compact summary to Slack.
/// `#[allow(dead_code)]`: wired into a running agent in Phase 2 (the daemon
/// and flow runner), kept here with tests so the text rules are locked in.
#[allow(dead_code)]
pub fn start_progress_notifier(
    agent: &mut agent_m_agent::Agent,
    transport: Arc<dyn SlackTransport>,
    channel: String,
) {
    agent.subscribe(move |event| {
        if let Some(text) = event_to_slack_text(event) {
            let transport = transport.clone();
            let channel = channel.clone();
            tokio::spawn(async move {
                let _ = transport.post_message(&channel, &text).await;
            });
        }
    });
}

/// Remote human channel: Slack transport + question registry + the event
/// loop that resolves answers. One instance per process; attach it to the
/// ask gate, the permission gate, and the flow progress notifier.
pub struct RemoteHuman {
    /// Pending-question registry shared with `handle_inbound`.
    pub human: Arc<crate::human::HumanChannel>,
    /// The underlying transport (posting progress lines).
    pub transport: Arc<dyn SlackTransport>,
}

/// Ask one High/Critical approval over Slack, mirroring `gate::ask_human`
/// for remote use. Used by `RemoteHuman::permission_closure` and directly
/// by the daemon/repl gate wiring.
pub fn ask_slack_permission(
    remote: Arc<RemoteHuman>,
    channel: String,
    policy: RiskPolicy,
    call: ToolCallInfo,
) -> Pin<Box<dyn Future<Output = Permission> + Send>> {
    Box::pin(async move {
        let risk = policy.risk(&call);
        let consequence = policy.consequence(&call);
        let args_str = serde_json::to_string(&call.arguments).unwrap_or_default();
        let prompt = format!(
            "⚠️  [Security Gate] Tool Execution Requested:\n    Tool: {}\n    Args: {}\n    Risk Level: {}\n    Consequence: {}",
            call.name,
            args_str,
            risk.as_deref().unwrap_or("High"),
            consequence.unwrap_or_default()
        );
        let options = Some(vec!["Approve".to_string(), "Deny".to_string()]);
        match remote
            .human
            .ask(remote.transport.as_ref(), &channel, &prompt, options, None)
            .await
        {
            Ok(answer)
                if answer.eq_ignore_ascii_case("approve")
                    || answer.eq_ignore_ascii_case("allow")
                    || answer.eq_ignore_ascii_case("1") =>
            {
                Permission::Allowed
            }
            _ => Permission::Denied("Denied over remote channel.".to_string()),
        }
    })
}

impl RemoteHuman {
    /// Connect the event loop and return the channel handle. The loop runs
    /// on a spawned task until the process ends.
    pub fn start(transport: Arc<dyn SlackTransport>) -> Arc<Self> {
        let human = Arc::new(crate::human::HumanChannel::new());
        let t = transport.clone();
        let t_loop = transport.clone();
        let h = human.clone();
        tokio::spawn(async move {
            let on_event = Arc::new(move |inbound: Inbound| {
                let h = h.clone();
                let t = t.clone();
                tokio::spawn(async move {
                    handle_inbound(inbound, &h, t.as_ref()).await;
                });
            });
            if let Err(e) = t_loop.run(on_event).await {
                eprintln!("remote channel ended: {e}");
            }
        });
        Arc::new(Self { human, transport })
    }

    /// An `AskGate` for the ask tool: posts the question and returns the
    /// first `ask-N <answer>` reply.
    pub fn ask_gate(
        &self,
        channel: &str,
    ) -> ClosureAskGate<
        impl Fn(
                String,
                Option<Vec<String>>,
                bool,
            ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>>
            + Send
            + Sync
            + 'static,
    > {
        let human = self.human.clone();
        let transport = self.transport.clone();
        let channel = channel.to_string();
        ClosureAskGate::new(move |question, options, multi_select| {
            let human = human.clone();
            let transport = transport.clone();
            let channel = channel.clone();
            Box::pin(async move {
                // Slack answers are free text; multi-select is not enforced.
                let _ = multi_select;
                human
                    .ask(transport.as_ref(), &channel, &question, options, None)
                    .await
            })
        })
    }

    /// The ask closure for `LevelGate`: High/Critical calls go to Slack as
    /// an approve/deny question, mirroring `gate::ask_human` for remote use.
    pub fn permission_closure(
        &self,
        policy: RiskPolicy,
        channel: &str,
    ) -> impl Fn(ToolCallInfo) -> Pin<Box<dyn Future<Output = Permission> + Send>>
        + Send
        + Sync
        + 'static {
        let remote = Arc::new(Self {
            human: self.human.clone(),
            transport: self.transport.clone(),
        });
        let channel = channel.to_string();
        move |call: ToolCallInfo| {
            ask_slack_permission(remote.clone(), channel.clone(), policy.clone(), call)
        }
    }
}

/// A flow progress callback that posts step transitions to Slack.
pub fn slack_progress(
    transport: Arc<dyn SlackTransport>,
    channel: String,
) -> Arc<dyn Fn(agent_m_flow::FlowProgress) + Send + Sync> {
    Arc::new(move |p: agent_m_flow::FlowProgress| {
        let transport = transport.clone();
        let channel = channel.clone();
        tokio::spawn(async move {
            let icon = match p.status.as_str() {
                "running" => "🔄",
                "succeeded" => "✅",
                "failed" => "❌",
                _ => "⏸️",
            };
            let _ = transport
                .post_message(
                    &channel,
                    &format!("{icon} step {} `{}`", p.step_index + 1, p.step_name),
                )
                .await;
        });
    })
}

/// Parse a pickup trigger from a DM: `pick up PROJ-123`, `pickup PROJ-123`,
/// or a bare `pick up` / `pickup` (auto-pick the next ticket).
///
/// Returns:
/// - `None` — not a trigger (echo the message as usual);
/// - `Some(None)` — auto-pick (no ticket key in the DM);
/// - `Some(Some(key))` — work on this specific ticket.
pub fn parse_pickup_trigger(text: &str) -> Option<Option<String>> {
    let trimmed = text.trim().trim_matches(['.', '!', '?']);
    let lowered = trimmed.to_lowercase();
    if lowered == "pick up" || lowered == "pickup" {
        return Some(None);
    }
    for prefix in ["pick up ", "pickup "] {
        if lowered.starts_with(prefix) {
            let rest = trimmed[prefix.len()..].trim();
            // A ticket key is a single word containing a dash (PROJ-42).
            let valid = rest.split_whitespace().count() == 1 && rest.contains('-');
            return if valid {
                Some(Some(rest.to_string()))
            } else {
                None
            };
        }
    }
    None
}

/// Compact flow result for a summary DM: ticket + verdict, per-step status,
/// the PR link when the `pr` step ran, and the first failure's error.
pub fn flow_summary(ticket: &str, run: &agent_m_flow::FlowRun) -> String {
    use agent_m_flow::StepStatus;
    let failed = run
        .steps
        .iter()
        .any(|s| s.status == StepStatus::Failed);
    let verdict = if failed { "FAILED" } else { "OK" };
    let icon = if failed { "❌" } else { "✅" };
    let mut lines = vec![format!("{icon} {ticket} — {} {verdict}", run.flow_name)];
    if run.fix_rounds > 0 {
        lines.push(format!("fix rounds: {}", run.fix_rounds));
    }
    for step in &run.steps {
        let mark = match step.status {
            StepStatus::Succeeded => "✓",
            StepStatus::Failed => "✗",
            StepStatus::Skipped => "⏸",
            _ => "·",
        };
        let mut line = format!("{mark} {}", step.name);
        if let Some(error) = &step.error {
            line.push_str(&format!(" — {error}"));
        }
        lines.push(line);
    }
    if let Some(pr) = run.steps.iter().find(|s| s.name == "pr") {
        if let Some(content) = pr
            .output
            .as_ref()
            .and_then(|o| o.get("content"))
            .and_then(serde_json::Value::as_str)
        {
            // The github-create-pr output is `PR created: <url> (#id)`;
            // our prefix would make it "🔗 PR: PR created: …", so strip it.
            let content = content.strip_prefix("PR created: ").unwrap_or(content);
            if content.contains("http") {
                lines.push(format!("🔗 PR: {content}"));
            }
        }
    }
    lines.join("\n")
}

/// Handle one inbound message with pickup-trigger routing (Phase 6): ask
/// replies resolve first, then `pick up [TICKET]` DM triggers `on_pickup`,
/// everything else is echoed.
pub async fn handle_inbound_orchestrated(
    inbound: Inbound,
    human: &crate::human::HumanChannel,
    transport: &dyn SlackTransport,
    on_pickup: Arc<dyn Fn(Option<String>, String) + Send + Sync>,
) {
    let Inbound::Message { channel, text, .. } = inbound;
    if let Some((id, answer)) = split_answer(&text) {
        if human.resolve(&id, answer.clone()) {
            let _ = transport
                .post_message(&channel, &format!("✅ answered `{id}`: {answer}"))
                .await;
        }
        return;
    }
    if let Some(ticket) = parse_pickup_trigger(&text) {
        on_pickup(ticket, channel.clone());
        return;
    }
    let reply = format!("agent-m online. You said: {text}");
    let _ = transport.post_message(&channel, &reply).await;
}

impl RemoteHuman {
    /// Build the channel handle directly around a transport (used by the
    /// orchestrator, which must hand the handle to pickup runs before the
    /// event loop starts).
    pub fn new(transport: Arc<dyn SlackTransport>) -> Arc<Self> {
        Arc::new(Self {
            human: Arc::new(crate::human::HumanChannel::new()),
            transport,
        })
    }

    /// The Phase 6 orchestrator loop: the event loop resolves ask replies
    /// *and* routes `pick up [TICKET]` DMs to `on_pickup`. The loop runs on
    /// a spawned task until the process ends.
    pub fn start_orchestrator(
        remote: Arc<RemoteHuman>,
        on_pickup: Arc<dyn Fn(Option<String>, String) + Send + Sync>,
    ) -> Arc<RemoteHuman> {
        let h = remote.human.clone();
        let t = remote.transport.clone();
        let t_loop = remote.transport.clone();
        let op = on_pickup.clone();
        tokio::spawn(async move {
            let on_event = Arc::new(move |inbound: Inbound| {
                let h = h.clone();
                let t = t.clone();
                let op = op.clone();
                tokio::spawn(async move {
                    handle_inbound_orchestrated(inbound, &h, t.as_ref(), op).await;
                });
            });
            if let Err(e) = t_loop.run(on_event).await {
                eprintln!("remote channel ended: {e}");
            }
        });
        remote
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::human::HumanChannel;

    /// Wait (up to 2s) for the `n`-th posted *question* (text starts with ❓)
    /// and return its `ask-N` id. Acks (`✅ answered ask-N`) are skipped.
    /// Panics if the question never appears.
    async fn wait_for_ask_id(fake: &FakeTransport, n: usize) -> String {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            let posted = fake.posted();
            let questions: Vec<_> = posted
                .iter()
                .filter(|(_, t)| t.starts_with('❓'))
                .collect();
            if let Some((_, text)) = questions.get(n) {
                break text
                    .split_whitespace()
                    .find(|w| w.starts_with("ask-"))
                    .expect("posted question contains an ask id")
                    .to_string();
            }
            if tokio::time::Instant::now() >= deadline {
                panic!("question {n} was never posted");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    #[test]
    fn split_answer_parses_ask_replies() {
        assert_eq!(
            split_answer("ask-3 approve"),
            Some(("ask-3".to_string(), "approve".to_string()))
        );
        assert_eq!(
            split_answer("ask-12 yes please"),
            Some(("ask-12".to_string(), "yes please".to_string()))
        );
        assert_eq!(split_answer("hello there"), None);
        assert_eq!(split_answer("ask-5"), None);
    }

    #[test]
    fn event_to_slack_text_maps_relevant_events() {
        let start = AgentEvent::ToolExecutionStart {
            tool_call_id: "1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({}),
        };
        assert_eq!(
            event_to_slack_text(&start).unwrap(),
            "🔧 `bash`…"
        );
        let notice = AgentEvent::Notice {
            message: "model error: boom".into(),
        };
        assert!(event_to_slack_text(&notice).unwrap().contains("model error"));
        // Events that should stay silent.
        assert!(event_to_slack_text(&AgentEvent::AgentStart).is_none());
    }

    #[tokio::test]
    async fn ask_posts_then_resolves_from_inbound_reply() {
        let transport = Arc::new(FakeTransport::new());
        let human = Arc::new(HumanChannel::new());

        let ask_handle = tokio::spawn({
            let transport = transport.clone();
            let human = human.clone();
            async move {
                human
                    .ask(
                        transport.as_ref(),
                        "U123",
                        "Approve the plan?",
                        None,
                        Some(Duration::from_secs(5)),
                    )
                    .await
            }
        });

        // Wait (up to 2s) for the question to be posted.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        let id = loop {
            if let Some(text) = transport.posted().first().map(|(_, t)| t.clone()) {
                break text
                    .split_whitespace()
                    .find(|w| w.starts_with("ask-"))
                    .expect("posted question contains an ask id")
                    .to_string();
            }
            if tokio::time::Instant::now() >= deadline {
                panic!("question was never posted");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        };

        // Simulate the event loop delivering the human's reply.
        handle_inbound(
            Inbound::Message {
                channel: "U123".into(),
                user: "U9".into(),
                text: format!("{id} approved"),
                ts: "1".into(),
            },
            &human,
            transport.as_ref(),
        )
        .await;

        let answer = ask_handle.await.expect("ask task panicked").expect("ask failed");
        assert_eq!(answer, "approved");
    }

    #[tokio::test]
    async fn handle_inbound_echoes_non_answers() {
        let transport = Arc::new(FakeTransport::new());
        let human = HumanChannel::new();
        handle_inbound(
            Inbound::Message {
                channel: "C1".into(),
                user: "U1".into(),
                text: "hi".into(),
                ts: "2".into(),
            },
            &human,
            transport.as_ref(),
        )
        .await;
        assert_eq!(
            transport.posted()[0].1,
            "agent-m online. You said: hi"
        );
    }

    #[tokio::test]
    async fn ask_times_out_without_a_reply() {
        let transport = Arc::new(FakeTransport::new());
        let human = HumanChannel::new();
        let err = human
            .ask(transport.as_ref(), "U1", "question", None, Some(Duration::from_millis(100)))
            .await;
        assert!(err.is_err());
        assert_eq!(human.pending_count(), 0);
    }

    #[tokio::test]
    async fn remote_ask_gate_round_trip() {
        use agent_m_agent::AskGate;
        let fake: Arc<FakeTransport> = Arc::new(FakeTransport::new());
        let transport: Arc<dyn SlackTransport> = fake.clone();
        let remote = RemoteHuman::start(transport.clone());
        let gate = remote.ask_gate("U123");

        let ask_handle = tokio::spawn(async move { gate.ask("Approve the plan?".into(), None, false).await });

        // The question is posted; extract its ask id.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        let id = loop {
            if let Some(text) = fake.posted().first().map(|(_, t)| t.clone()) {
                break text
                    .split_whitespace()
                    .find(|w| w.starts_with("ask-"))
                    .expect("posted question contains an ask id")
                    .to_string();
            }
            if tokio::time::Instant::now() >= deadline {
                panic!("question was never posted");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        };

        // The RemoteHuman event loop turns this inbound message into a resolve.
        fake.push(Inbound::Message {
            channel: "U123".into(),
            user: "U9".into(),
            text: format!("{id} yes do it"),
            ts: "1".into(),
        });

        let answer = ask_handle.await.expect("ask task panicked").expect("ask failed");
        assert_eq!(answer, "yes do it");
    }

    #[tokio::test]
    async fn remote_permission_closure_approve_and_deny() {
        let fake: Arc<FakeTransport> = Arc::new(FakeTransport::new());
        let transport: Arc<dyn SlackTransport> = fake.clone();
        let remote = RemoteHuman::start(transport.clone());
        let policy = RiskPolicy::default();
        let closure = remote.permission_closure(policy.clone(), "C1");

        let call = ToolCallInfo {
            tool_call_id: "t1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({ "command": "rm -rf /tmp/x" }),
        };

        // Approve path.
        let approve = tokio::spawn({
            let closure = closure;
            let call = call.clone();
            async move { (closure)(call).await }
        });
        let id = wait_for_ask_id(&fake, 0).await;
        fake.push(Inbound::Message {
            channel: "C1".into(),
            user: "U9".into(),
            text: format!("{id} Approve"),
            ts: "2".into(),
        });
        assert_eq!(approve.await.expect("approval task panicked"), Permission::Allowed);

        // Deny path: a fresh closure for a fresh ask id.
        let closure = remote.permission_closure(policy, "C1");
        let deny = tokio::spawn({
            let closure = closure;
            let call = call;
            async move { (closure)(call).await }
        });
        let id = wait_for_ask_id(&fake, 1).await;
        fake.push(Inbound::Message {
            channel: "C1".into(),
            user: "U9".into(),
            text: format!("{id} Deny"),
            ts: "3".into(),
        });
        assert!(matches!(deny.await.expect("deny task panicked"), Permission::Denied(_)));
    }

    #[tokio::test]
    async fn slack_progress_posts_step_lines() {
        let fake: Arc<FakeTransport> = Arc::new(FakeTransport::new());
        let transport: Arc<dyn SlackTransport> = fake.clone();
        let progress = slack_progress(transport.clone(), "C1".into());
        progress(agent_m_flow::FlowProgress {
            step_index: 0,
            step_name: "plan".into(),
            status: agent_m_flow::StepStatus::Running,
        });
        progress(agent_m_flow::FlowProgress {
            step_index: 0,
            step_name: "plan".into(),
            status: agent_m_flow::StepStatus::Succeeded,
        });
        // The spawned posts are fire-and-forget; give them a beat.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let posted = fake.posted();
        assert_eq!(posted.len(), 2);
        assert!(posted[0].1.contains("🔄 step 1 `plan`"));
        assert!(posted[1].1.contains("✅ step 1 `plan`"));
    }

    #[test]
    fn parse_pickup_trigger_matches_forms() {
        // Bare auto-pick.
        assert_eq!(parse_pickup_trigger("pick up"), Some(None));
        assert_eq!(parse_pickup_trigger("Pickup"), Some(None));
        assert_eq!(parse_pickup_trigger("pick up!"), Some(None));
        // With a ticket key.
        assert_eq!(
            parse_pickup_trigger("pick up PROJ-42"),
            Some(Some("PROJ-42".to_string()))
        );
        assert_eq!(
            parse_pickup_trigger("Pickup PROJ-7."),
            Some(Some("PROJ-7".to_string()))
        );
        // Not triggers.
        assert_eq!(parse_pickup_trigger("hello"), None);
        assert_eq!(parse_pickup_trigger("pick up the trash"), None);
        assert_eq!(parse_pickup_trigger("pickup please"), None);
        assert_eq!(parse_pickup_trigger("pick"), None);
    }

    #[test]
    fn flow_summary_reports_pr_link_and_failures() {
        use agent_m_flow::{FlowRun, FlowContext, StepRecord, StepStatus};
        use agent_m_flow::StepStatus::*;
        let run = FlowRun {
            flow_name: "agentic-dev".to_string(),
            fix_rounds: 2,
            context: FlowContext::new(),
            steps: vec![
                StepRecord {
                    name: "verify".into(),
                    step_type: "verify".into(),
                    status: Failed,
                    output: None,
                    error: Some(
                        "fix budget exhausted (used 2 of 5 fix rounds); last failure:\nboom"
                            .to_string(),
                    ),
                },
                StepRecord {
                    name: "pr".into(),
                    step_type: "tool".into(),
                    status: Succeeded,
                    output: Some(serde_json::json!({
                        "content": "PR created: https://github.com/acme/app/pull/7 (#7)"
                    })),
                    error: None,
                },
            ],
        };
        let summary = flow_summary("PROJ-42", &run);
        assert!(summary.contains("❌ PROJ-42 — agentic-dev FAILED"), "{summary}");
        assert!(summary.contains("fix rounds: 2"), "{summary}");
        assert!(summary.contains("✗ verify"), "{summary}");
        assert!(summary.contains("fix budget exhausted"), "{summary}");
        assert!(summary.contains("🔗 PR: https://github.com/acme/app/pull/7 (#7)"), "{summary}");

        // Green run: no error lines, ✅ verdict, no PR line when absent.
        let green = FlowRun {
            flow_name: "agentic-dev".to_string(),
            fix_rounds: 0,
            context: FlowContext::new(),
            steps: vec![StepRecord {
                name: "verify".into(),
                step_type: "verify".into(),
                status: StepStatus::Succeeded,
                output: None,
                error: None,
            }],
        };
        let summary = flow_summary("PROJ-9", &green);
        assert!(summary.contains("✅ PROJ-9 — agentic-dev OK"), "{summary}");
        assert!(!summary.contains('✗'), "{summary}");
        assert!(!summary.contains("PR:"), "{summary}");
    }

    #[tokio::test]
    async fn orchestrated_inbound_routes_pickups_and_echoes() {
        let transport = Arc::new(FakeTransport::new());
        let human = Arc::new(HumanChannel::new());
        let triggers: Arc<Mutex<Vec<(Option<String>, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let on_pickup: Arc<dyn Fn(Option<String>, String) + Send + Sync> = Arc::new({
            let triggers = triggers.clone();
            move |ticket, channel| triggers.lock().unwrap().push((ticket, channel))
        });

        // A ticket DM routes to on_pickup and posts nothing else.
        handle_inbound_orchestrated(
            Inbound::Message {
                channel: "U123".into(),
                user: "U9".into(),
                text: "pick up PROJ-7".into(),
                ts: "1".into(),
            },
            &human,
            transport.as_ref(),
            on_pickup.clone(),
        )
        .await;
        assert_eq!(
            triggers.lock().unwrap().clone(),
            vec![(Some("PROJ-7".to_string()), "U123".to_string())]
        );
        assert!(transport.posted().is_empty(), "no echo for a trigger");

        // Bare pickup → auto-pick.
        handle_inbound_orchestrated(
            Inbound::Message {
                channel: "U9".into(),
                user: "U9".into(),
                text: "pickup".into(),
                ts: "2".into(),
            },
            &human,
            transport.as_ref(),
            on_pickup.clone(),
        )
        .await;
        assert_eq!(
            triggers.lock().unwrap().last().cloned(),
            Some((None, "U9".to_string()))
        );

        // Ordinary messages still echo.
        handle_inbound_orchestrated(
            Inbound::Message {
                channel: "U9".into(),
                user: "U9".into(),
                text: "hello".into(),
                ts: "3".into(),
            },
            &human,
            transport.as_ref(),
            on_pickup.clone(),
        )
        .await;
        assert_eq!(
            transport.posted().last().map(|(_, t)| t.as_str()),
            Some("agent-m online. You said: hello")
        );
    }

    #[tokio::test]
    async fn orchestrated_inbound_still_resolves_asks() {
        let transport = Arc::new(FakeTransport::new());
        let human = Arc::new(HumanChannel::new());
        let triggers: Arc<Mutex<Vec<(Option<String>, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let on_pickup: Arc<dyn Fn(Option<String>, String) + Send + Sync> = Arc::new({
            let triggers = triggers.clone();
            move |ticket, channel| triggers.lock().unwrap().push((ticket, channel))
        });

        let ask_handle = tokio::spawn({
            let transport = transport.clone();
            let human = human.clone();
            async move {
                human
                    .ask(
                        transport.as_ref(),
                        "U123",
                        "Approve the plan?",
                        None,
                        Some(Duration::from_secs(5)),
                    )
                    .await
            }
        });

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        let id = loop {
            if let Some(text) = transport.posted().first().map(|(_, t)| t.clone()) {
                break text
                    .split_whitespace()
                    .find(|w| w.starts_with("ask-"))
                    .expect("posted question contains an ask id")
                    .to_string();
            }
            if tokio::time::Instant::now() >= deadline {
                panic!("question was never posted");
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        };

        handle_inbound_orchestrated(
            Inbound::Message {
                channel: "U123".into(),
                user: "U9".into(),
                text: format!("{id} approved"),
                ts: "4".into(),
            },
            &human,
            transport.as_ref(),
            on_pickup,
        )
        .await;

        let answer = ask_handle.await.expect("ask task panicked").expect("ask failed");
        assert_eq!(answer, "approved");
        assert!(triggers.lock().unwrap().is_empty(), "no pickup triggered");
    }
}
