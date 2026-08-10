//! Flow executor tests: tool/ask/prompt/condition steps, failure abort,
//! verify fix loop, and the agent loop driving.

use agent_m_agent::{
    AgentOptions, AlwaysAllowGate, AskGate, ClosureAskGate, Mode, Tool, ToolContext, ToolError,
    ToolOutcome,
};
use agent_m_ai::{
    AiError, ChatRequest, ContentPart, ModelSpec, Provider, StopReason, StreamEvent, Usage,
};
use agent_m_flow::{FlowContext, FlowDeps, FlowStep, load_flow, run_flow};
use async_trait::async_trait;
use futures_util::stream;
use futures_util::stream::BoxStream;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

// --- Fake provider ---------------------------------------------------------

struct FakeLlm {
    responses: std::sync::Mutex<std::collections::VecDeque<Vec<StreamEvent>>>,
}

impl FakeLlm {
    fn new(scripts: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            responses: std::sync::Mutex::new(scripts.into()),
        }
    }
    fn text(text: &str) -> Vec<StreamEvent> {
        vec![
            StreamEvent::TextDelta {
                delta: text.to_string(),
            },
            StreamEvent::Done {
                message: agent_m_ai::AssistantMessage {
                    content: vec![ContentPart::Text {
                        text: text.to_string(),
                    }],
                    usage: Some(Usage {
                        input_tokens: 10,
                        output_tokens: 8,
                        cache_read_tokens: 0,
                        cache_creation_tokens: 0,
                        total_tokens: 18,
                        cost: 0.0,
                    }),
                    stop_reason: StopReason::Stop,
                    error_message: None,
                    model: "fake".to_string(),
                },
            },
        ]
    }
}

#[async_trait]
impl Provider for FakeLlm {
    fn id(&self) -> &str {
        "fake"
    }
    fn display_name(&self) -> &str {
        "Fake"
    }
    fn api_key(&self) -> Option<&str> {
        Some("fake")
    }
    fn set_api_key(&mut self, _key: String) {}
    fn models(&self) -> &[ModelSpec] {
        &[]
    }
    async fn stream_chat(
        &self,
        _request: ChatRequest,
    ) -> Result<BoxStream<'static, StreamEvent>, AiError> {
        let events = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_default();
        Ok(Box::pin(stream::iter(events)))
    }
}

// --- Scripted bash stub ----------------------------------------------------

struct BashStub {
    /// (command, outcome) pairs; each call pops one.
    script: std::sync::Mutex<std::collections::VecDeque<(String, ToolOutcome)>>,
    calls: std::sync::Mutex<Vec<String>>,
}

impl BashStub {
    fn new(script: Vec<(String, ToolOutcome)>) -> Self {
        Self {
            script: std::sync::Mutex::new(script.into()),
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Tool for BashStub {
    fn name(&self) -> &str {
        "bash"
    }
    fn description(&self) -> String {
        "run a command".to_string()
    }
    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "command": { "type": "string" } } })
    }
    async fn execute(
        &self,
        arguments: Value,
        _context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let command = arguments
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        self.calls.lock().unwrap().push(command.clone());
        let (expected, outcome) = self
            .script
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| ("".to_string(), ToolOutcome::success("ok")));
        assert_eq!(expected, command, "unexpected command {command}");
        Ok(outcome)
    }
}

// --- Jira stub -------------------------------------------------------------

struct JiraStub;

#[async_trait]
impl Tool for JiraStub {
    fn name(&self) -> &str {
        "jira-stub"
    }
    fn description(&self) -> String {
        "look up a jira ticket".to_string()
    }
    fn parameters(&self) -> Value {
        json!({ "type": "object", "properties": { "query": { "type": "string" } } })
    }
    async fn execute(
        &self,
        arguments: Value,
        _context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let query = arguments.get("query").and_then(Value::as_str).unwrap_or("");
        Ok(ToolOutcome::success(format!("jira: {query}")))
    }
}

// --- Fixtures --------------------------------------------------------------

fn options() -> AgentOptions {
    AgentOptions {
        model: "fake".to_string(),
        system_prompt: "You are a test agent.".to_string(),
        tools: vec![],
        permission_gate: Arc::new(AlwaysAllowGate),
        max_turns: 5,
        cwd: PathBuf::from("."),
        mode: Mode::Build,
        ask_gate: None,
        context_window: None,
    }
}

#[tokio::test]
async fn tool_and_condition_and_ask_steps() {
    let bash = Arc::new(BashStub::new(vec![(
        "git status".to_string(),
        ToolOutcome::success("clean"),
    )]));
    let gate: Arc<dyn AskGate> = Arc::new(ClosureAskGate::new(|_q, _o| {
        Box::pin(async { Ok("approved".to_string()) })
    }));
    let deps = FlowDeps {
        provider: Arc::new(FakeLlm::new(vec![])),
        agent_options: options(),
        tools: vec![Arc::new(JiraStub), bash.clone()],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: Some(gate),
        state_dir: None,
        on_progress: None,
    };
    let flow = load_flow(&PathBuf::from("tests/fixtures/basic.yml")).expect("load");
    let mut context = FlowContext::new();
    context.set("ticket", json!("TICKET-42"));
    let run = run_flow(&flow, &mut context, &deps).await.expect("run");

    let names: Vec<&str> = run.steps.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["jira", "plan", "gate", "gate", "ship"]);
    assert!(
        run.steps
            .iter()
            .all(|s| s.status == agent_m_flow::StepStatus::Succeeded)
    );
    // Tool output is available via ${steps.jira.output}.
    assert_eq!(
        context
            .get("steps.jira.output.content")
            .and_then(Value::as_str),
        Some("jira: TICKET-42")
    );
    // Ask answer stored.
    assert_eq!(
        context
            .get("steps.gate.output.answer")
            .and_then(Value::as_str),
        Some("approved")
    );
}

#[tokio::test]
async fn failed_step_aborts_the_flow() {
    let bash = Arc::new(BashStub::new(vec![(
        "false".to_string(),
        ToolOutcome::error("boom"),
    )]));
    let deps = FlowDeps {
        provider: Arc::new(FakeLlm::new(vec![])),
        agent_options: options(),
        tools: vec![bash.clone()],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: None,
        state_dir: None,
        on_progress: None,
    };
    let flow = load_flow(&PathBuf::from("tests/fixtures/failing.yml")).expect("load");
    let mut context = FlowContext::new();
    let error = run_flow(&flow, &mut context, &deps)
        .await
        .expect_err("must abort");
    assert!(error.to_string().contains("failing"), "got: {error}");
}

#[tokio::test]
async fn prompt_step_uses_fresh_agent_and_captures_output() {
    let llm = Arc::new(FakeLlm::new(vec![FakeLlm::text("the plan is: 1. fix it")]));
    let deps = FlowDeps {
        provider: llm,
        agent_options: options(),
        tools: vec![],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: None,
        state_dir: None,
        on_progress: None,
    };
    let flow = agent_m_flow::Flow {
        name: "single".to_string(),
        description: None,
        steps: vec![FlowStep::Prompt {
            name: "plan".to_string(),
            mode: Some("plan".to_string()),
            message: Some("Plan ${ticket}".to_string()),
            model: None,
        }],
    };
    let mut context = FlowContext::new();
    context.set("ticket", json!("ABC"));
    let run = run_flow(&flow, &mut context, &deps).await.expect("run");
    assert_eq!(run.steps[0].status, agent_m_flow::StepStatus::Succeeded);
    let output = run.steps[0]
        .output
        .as_ref()
        .and_then(|o| o.as_str())
        .unwrap_or("");
    assert!(output.contains("1. fix it"), "got: {output}");
}

#[tokio::test]
async fn destructive_command_denied_in_flow() {
    let bash = Arc::new(BashStub::new(vec![]));
    let deps = FlowDeps {
        provider: Arc::new(FakeLlm::new(vec![])),
        agent_options: options(),
        tools: vec![bash.clone()],
        // --yes in print mode: destructive commands are still denied.
        permission_gate: Arc::new(agent_m_agent::DangerousCommandGate(AlwaysAllowGate)),
        ask_gate: None,
        state_dir: None,
        on_progress: None,
    };
    let flow = agent_m_flow::Flow {
        name: "danger".to_string(),
        description: None,
        steps: vec![FlowStep::Tool {
            name: "wipe".to_string(),
            tool: "bash".to_string(),
            arguments: Some(json!({ "command": "rm -rf /tmp/x" })),
        }],
    };
    let mut context = FlowContext::new();
    let error = run_flow(&flow, &mut context, &deps)
        .await
        .expect_err("must abort");
    assert!(
        error.to_string().contains("permission denied"),
        "got: {error}"
    );
    assert!(bash.calls.lock().unwrap().is_empty(), "tool must not run");
}

#[tokio::test]
async fn on_progress_emits_running_then_final_per_step() {
    let bash = Arc::new(BashStub::new(vec![(
        "git status".to_string(),
        ToolOutcome::success("clean"),
    )]));
    let gate: Arc<dyn AskGate> = Arc::new(ClosureAskGate::new(|_q, _o| {
        Box::pin(async { Ok("approved".to_string()) })
    }));
    let progress = Arc::new(std::sync::Mutex::new(Vec::new()));
    let collector = progress.clone();
    let deps = FlowDeps {
        provider: Arc::new(FakeLlm::new(vec![])),
        agent_options: options(),
        tools: vec![Arc::new(JiraStub), bash.clone()],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: Some(gate),
        state_dir: None,
        on_progress: Some(Arc::new(move |p| {
            collector.lock().unwrap().push((
                p.step_index,
                p.step_name,
                p.status.as_str().to_string(),
            ));
        })),
    };
    let flow = load_flow(&PathBuf::from("tests/fixtures/basic.yml")).expect("load");
    let mut context = FlowContext::new();
    context.set("ticket", json!("TICKET-42"));
    run_flow(&flow, &mut context, &deps).await.expect("run");
    let events = progress.lock().unwrap().clone();
    // Top-level steps: jira, plan, gate (condition), ship.
    let top: Vec<(&str, &str)> = events
        .iter()
        .map(|(_, name, status)| (name.as_str(), status.as_str()))
        .collect();
    assert_eq!(
        top,
        vec![
            ("jira", "running"),
            ("jira", "succeeded"),
            ("plan", "running"),
            ("plan", "succeeded"),
            ("gate", "running"),
            ("gate", "succeeded"),
            ("ship", "running"),
            ("ship", "succeeded"),
        ]
    );
}

#[tokio::test]
async fn writes_state_artifacts() {
    let bash = Arc::new(BashStub::new(vec![(
        "git status".to_string(),
        ToolOutcome::success("clean"),
    )]));
    let state = tempfile::tempdir().unwrap();
    let deps = FlowDeps {
        provider: Arc::new(FakeLlm::new(vec![])),
        agent_options: options(),
        tools: vec![bash.clone()],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: None,
        state_dir: Some(state.path().to_path_buf()),
        on_progress: None,
    };
    let flow = agent_m_flow::Flow {
        name: "state-demo".to_string(),
        description: None,
        steps: vec![FlowStep::Tool {
            name: "step1".to_string(),
            tool: "bash".to_string(),
            arguments: Some(json!({ "command": "git status" })),
        }],
    };
    let mut context = FlowContext::new();
    run_flow(&flow, &mut context, &deps).await.expect("run");
    let state_file = state.path().join("state-demo/STATE.md");
    let context_file = state.path().join("state-demo/CONTEXT.json");
    assert!(state_file.is_file(), "STATE.md written");
    let text = std::fs::read_to_string(&state_file).unwrap();
    assert!(text.contains("step1"), "step listed: {text}");
    assert!(context_file.is_file(), "CONTEXT.json written");
}

#[test]
fn strategic_compaction_threshold() {
    // ECC default 0.5: compact at turn boundaries above 50% of the window.
    assert!(agent_m_flow::should_compact(51_000, 64_000, 0.5));
    assert!(!agent_m_flow::should_compact(31_000, 64_000, 0.5));
    assert!(agent_m_flow::should_compact(58_000, 64_000, 0.9));
    assert!(
        !agent_m_flow::should_compact(64_000, 0, 0.5),
        "no window → no compact"
    );
}

#[tokio::test]
async fn verify_loops_fixes_until_command_passes() {
    let bash = Arc::new(BashStub::new(vec![
        (
            "cargo test".to_string(),
            ToolOutcome::error("test failure 1"),
        ),
        ("cargo test".to_string(), ToolOutcome::success("all green")),
    ]));
    let llm = Arc::new(FakeLlm::new(vec![FakeLlm::text("fixed the test")]));
    let deps = FlowDeps {
        provider: llm,
        agent_options: options(),
        tools: vec![bash.clone()],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: None,
        state_dir: None,
        on_progress: None,
    };
    let flow = agent_m_flow::Flow {
        name: "verify-demo".to_string(),
        description: None,
        steps: vec![FlowStep::Verify {
            name: "check".to_string(),
            command: Some("cargo test".to_string()),
            max_fix_rounds: 3,
        }],
    };
    let mut context = FlowContext::new();
    let run = run_flow(&flow, &mut context, &deps).await.expect("run");
    assert_eq!(run.steps[0].status, agent_m_flow::StepStatus::Succeeded);
    assert_eq!(
        run.steps[0]
            .output
            .as_ref()
            .and_then(|o| o.get("fix_rounds"))
            .and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(bash.calls.lock().unwrap().len(), 2, "run + one fix re-run");
}
