//! Flow executor tests: tool/ask/prompt/condition steps, failure abort,
//! verify fix loop, and the agent loop driving.

use agent_m_agent::{
    AgentOptions, AlwaysAllowGate, AskGate, ClosureAskGate, Mode, RiskPolicy, Tool, ToolContext,
    ToolError, ToolOutcome,
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
    /// The last user-message text of every request the agent sent (so tests
    /// can assert the fix prompt carries the failure output).
    prompts: std::sync::Mutex<Vec<String>>,
}

impl FakeLlm {
    fn new(scripts: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            responses: std::sync::Mutex::new(scripts.into()),
            prompts: std::sync::Mutex::new(Vec::new()),
        }
    }
    fn prompts(&self) -> Vec<String> {
        self.prompts.lock().unwrap().clone()
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
                    trust: Default::default(),
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
        request: ChatRequest,
    ) -> Result<BoxStream<'static, StreamEvent>, AiError> {
        let last_user = request.messages.iter().rev().find_map(|message| match message {
            agent_m_ai::LlmMessage::User { content, .. } => Some(content.clone()),
            _ => None,
        });
        if let Some(text) = last_user {
            self.prompts.lock().unwrap().push(text);
        }
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

// --- Recording stub for external tools (github/jira ops) --------------------

struct ToolStub {
    tool_name: String,
    reply: String,
    calls: std::sync::Mutex<Vec<Value>>,
}

impl ToolStub {
    fn new(tool_name: &str, reply: &str) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            reply: reply.to_string(),
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }
    fn calls(&self) -> Vec<Value> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl Tool for ToolStub {
    fn name(&self) -> &str {
        &self.tool_name
    }
    fn description(&self) -> String {
        "recording stub".to_string()
    }
    fn parameters(&self) -> Value {
        json!({ "type": "object" })
    }
    async fn execute(
        &self,
        arguments: Value,
        _context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        self.calls.lock().unwrap().push(arguments.clone());
        Ok(ToolOutcome::success(self.reply.clone()))
    }
}

// --- Fixtures --------------------------------------------------------------

fn options() -> AgentOptions {
    AgentOptions {
        model: "fake".to_string(),
        system_prompt: "You are a test agent.".to_string(),
        harness_block: None,
        tools: vec![],
        max_turns: 5,
        cwd: PathBuf::from("."),
        mode: Mode::Build,
        ask_gate: None,
        context_window: None,
        variant: None,
        output_dir: None,
        trust: agent_m_agent::TrustPolicy::default(),
        risk_policy: None,
        delegate_depth: 0,
    }
}

#[tokio::test]
async fn tool_and_condition_and_ask_steps() {
    let bash = Arc::new(BashStub::new(vec![(
        "git status".to_string(),
        ToolOutcome::success("clean"),
    )]));
    let gate: Arc<dyn AskGate> = Arc::new(ClosureAskGate::new(|_q, _o, _m| {
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
        error_budget: None,
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
        permission_gate: Arc::new(agent_m_agent::DangerousCommandGate::new(
            Arc::new(RiskPolicy {
                cwd: PathBuf::from("."),
                opaque_tools: vec![],
            }),
            AlwaysAllowGate,
        )),
        ask_gate: None,
        state_dir: None,
        on_progress: None,
    };
    let flow = agent_m_flow::Flow {
        name: "danger".to_string(),
        description: None,
        error_budget: None,
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
    let gate: Arc<dyn AskGate> = Arc::new(ClosureAskGate::new(|_q, _o, _m| {
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
        error_budget: None,
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
        error_budget: None,
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

#[tokio::test]
async fn verify_stops_and_reports_on_fix_round_cap() {
    // Never passes: run + 2 fix rounds, then stop-and-report with the last
    // failure in the error. No ship step follows (flow aborts).
    let bash = Arc::new(BashStub::new(vec![
        ("cargo test".to_string(), ToolOutcome::error("boom 1")),
        ("cargo test".to_string(), ToolOutcome::error("boom 2")),
        ("cargo test".to_string(), ToolOutcome::error("boom 3")),
    ]));
    let llm = Arc::new(FakeLlm::new(vec![
        FakeLlm::text("fix 1"),
        FakeLlm::text("fix 2"),
    ]));
    let deps = FlowDeps {
        provider: llm.clone(),
        agent_options: options(),
        tools: vec![bash.clone()],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: None,
        state_dir: None,
        on_progress: None,
    };
    let flow = agent_m_flow::Flow {
        name: "cap-demo".to_string(),
        description: None,
        error_budget: None,
        steps: vec![FlowStep::Verify {
            name: "check".to_string(),
            command: Some("cargo test".to_string()),
            max_fix_rounds: 2,
        }],
    };
    let mut context = FlowContext::new();
    let error = run_flow(&flow, &mut context, &deps)
        .await
        .expect_err("must abort");
    assert!(
        error.to_string().contains("fix rounds exhausted (2/2)"),
        "got: {error}"
    );
    // Stop-and-report: the last failure is in the error message.
    assert!(
        error.to_string().contains("boom 3"),
        "last failure must be reported: {error}"
    );
    assert_eq!(bash.calls.lock().unwrap().len(), 3, "run + 2 fix re-runs");
    // The failure output fed each fix prompt.
    let prompts = llm.prompts();
    assert!(prompts.iter().any(|p| p.contains("boom 1")), "got: {prompts:?}");
    assert!(prompts.iter().any(|p| p.contains("boom 2")), "got: {prompts:?}");
    // Fix rounds remaining is communicated to the model.
    assert!(
        prompts.iter().any(|p| p.contains("Fix rounds remaining: 1")),
        "budget-aware prompt expected: {prompts:?}"
    );
}

#[tokio::test]
async fn verify_failure_feeds_the_fix_prompt() {
    let bash = Arc::new(BashStub::new(vec![
        (
            "cargo test".to_string(),
            ToolOutcome::error("test failure 1"),
        ),
        ("cargo test".to_string(), ToolOutcome::success("all green")),
    ]));
    let llm = Arc::new(FakeLlm::new(vec![FakeLlm::text("fixed the test")]));
    let deps = FlowDeps {
        provider: llm.clone(),
        agent_options: options(),
        tools: vec![bash.clone()],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: None,
        state_dir: None,
        on_progress: None,
    };
    let flow = agent_m_flow::Flow {
        name: "verify-feed".to_string(),
        description: None,
        error_budget: None,
        steps: vec![FlowStep::Verify {
            name: "check".to_string(),
            command: Some("cargo test".to_string()),
            max_fix_rounds: 3,
        }],
    };
    let mut context = FlowContext::new();
    let run = run_flow(&flow, &mut context, &deps).await.expect("run");
    assert_eq!(run.steps[0].status, agent_m_flow::StepStatus::Succeeded);
    assert_eq!(run.fix_rounds, 1);
    let prompts = llm.prompts();
    assert_eq!(prompts.len(), 1, "one fix prompt: {prompts:?}");
    assert!(
        prompts[0].contains("test failure 1"),
        "failure output must seed the fix prompt: {}",
        prompts[0]
    );
    assert!(
        prompts[0].contains("Fix rounds remaining: 2"),
        "remaining budget must be visible: {}",
        prompts[0]
    );
}

#[tokio::test]
async fn error_budget_caps_fix_rounds_across_verify_steps() {
    // error_budget: 1. Verify A spends the only fix round and still fails;
    // verify B has zero budget left → runs once, cannot fix, and reports the
    // budget exhaustion instead of burning model calls.
    let bash = Arc::new(BashStub::new(vec![
        ("t A".to_string(), ToolOutcome::error("A fails")),
        ("t A".to_string(), ToolOutcome::error("A still fails")),
        ("t B".to_string(), ToolOutcome::error("B fails")),
    ]));
    let deps = FlowDeps {
        provider: Arc::new(FakeLlm::new(vec![FakeLlm::text("fix A")])),
        agent_options: options(),
        tools: vec![bash.clone()],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: None,
        state_dir: None,
        on_progress: None,
    };
    let flow = agent_m_flow::Flow {
        name: "budget-demo".to_string(),
        description: None,
        error_budget: Some(1),
        steps: vec![
            FlowStep::Verify {
                name: "a".to_string(),
                command: Some("t A".to_string()),
                max_fix_rounds: 3,
            },
            FlowStep::Verify {
                name: "b".to_string(),
                command: Some("t B".to_string()),
                max_fix_rounds: 3,
            },
        ],
    };
    let mut context = FlowContext::new();
    let error = run_flow(&flow, &mut context, &deps)
        .await
        .expect_err("must abort");
    // Stop-and-report at the cap: A spends the only fix round, still fails,
    // and the flow aborts there — verify B never even runs its check.
    assert!(
        error.to_string().contains("fix budget exhausted (used 1 of 1 fix rounds)"),
        "got: {error}"
    );
    assert!(
        error.to_string().contains("A still fails"),
        "last failure reported: {error}"
    );
    // A: run + 1 fix. B never ran.
    assert_eq!(bash.calls.lock().unwrap().len(), 2, "got: {:?}", bash.calls);
}

/// The agentic-dev shape end-to-end with stubbed externals: green verify →
/// PR → Jira In Review transition + comment with the PR link.
#[tokio::test]
async fn agentic_dev_close_out_transitions_and_comments_after_pr() {
    let bash = Arc::new(BashStub::new(vec![
        (
            "git clone acme/app work".to_string(),
            ToolOutcome::success("cloned"),
        ),
        (
            "cd work && cargo test".to_string(),
            ToolOutcome::success("all green"),
        ),
    ]));
    let transition = Arc::new(ToolStub::new("jira-transition", "transitioned PROJ-42"));
    let comment = Arc::new(ToolStub::new("jira-comment", "commented on PROJ-42"));
    let pr = Arc::new(ToolStub::new(
        "github-create-pr",
        "PR created: https://github.com/acme/app/pull/7 (#7)",
    ));
    let gate: Arc<dyn AskGate> = Arc::new(ClosureAskGate::new(|_q, _o, _m| {
        Box::pin(async { Ok("approved".to_string()) })
    }));
    let deps = FlowDeps {
        provider: Arc::new(FakeLlm::new(vec![
            FakeLlm::text("the plan"),
            FakeLlm::text("implemented"),
        ])),
        agent_options: options(),
        tools: vec![
            Arc::new(JiraStub),
            bash.clone(),
            pr.clone(),
            transition.clone(),
            comment.clone(),
        ],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: Some(gate),
        state_dir: None,
        on_progress: None,
    };
    let flow = agent_m_flow::Flow {
        name: "agentic-dev".to_string(),
        description: None,
        error_budget: Some(5),
        steps: vec![
            FlowStep::Tool {
                name: "jira-ticket".to_string(),
                tool: "jira-stub".to_string(),
                arguments: Some(json!({ "query": "${ticket}" })),
            },
            FlowStep::Tool {
                name: "clone".to_string(),
                tool: "bash".to_string(),
                arguments: Some(json!({ "command": "git clone ${repo} work" })),
            },
            FlowStep::Prompt {
                name: "plan".to_string(),
                mode: Some("plan".to_string()),
                message: Some("Plan: ${steps.jira-ticket.output.content}".to_string()),
                model: None,
            },
            FlowStep::Ask {
                name: "review-plan".to_string(),
                question: "Approve?".to_string(),
                options: None,
                on_reject: "stop".to_string(),
            },
            FlowStep::Phase {
                name: "execute".to_string(),
                prompt: Some("Implement: ${steps.plan.output}".to_string()),
                steps: vec![],
            },
            FlowStep::Verify {
                name: "verify".to_string(),
                command: Some("cd work && cargo test".to_string()),
                max_fix_rounds: 3,
            },
            FlowStep::Ask {
                name: "review-implementation".to_string(),
                question: "Ship?".to_string(),
                options: None,
                on_reject: "stop".to_string(),
            },
            FlowStep::Tool {
                name: "pr".to_string(),
                tool: "github-create-pr".to_string(),
                arguments: Some(json!({ "title": "${ticket}" })),
            },
            FlowStep::Tool {
                name: "transition".to_string(),
                tool: "jira-transition".to_string(),
                arguments: Some(json!({ "key": "${ticket}", "transitionId": "${transitionId}" })),
            },
            FlowStep::Tool {
                name: "comment".to_string(),
                tool: "jira-comment".to_string(),
                arguments: Some(json!({ "key": "${ticket}", "body": "PR ready: ${steps.pr.output.content}" })),
            },
        ],
    };
    let mut context = FlowContext::new();
    context.set("ticket", json!("PROJ-42"));
    context.set("repo", json!("acme/app"));
    context.set("transitionId", json!("31"));
    let run = run_flow(&flow, &mut context, &deps).await.expect("run");
    let names: Vec<&str> = run
        .steps
        .iter()
        .map(|s| s.name.as_str())
        .filter(|n| !n.contains(':')) // skip nested phase records
        .collect();
    assert_eq!(
        names,
        vec![
            "jira-ticket", "clone", "plan", "review-plan", "execute", "verify",
            "review-implementation", "pr", "transition", "comment"
        ]
    );
    assert!(
        run.steps
            .iter()
            .all(|s| s.status == agent_m_flow::StepStatus::Succeeded),
        "all steps succeeded"
    );
    assert_eq!(run.fix_rounds, 0, "green verify spends no budget");
    // Close-out: transition called with ticket + configured transition id.
    let transition_calls = transition.calls();
    assert_eq!(transition_calls.len(), 1);
    assert_eq!(
        transition_calls[0].get("key").and_then(Value::as_str),
        Some("PROJ-42")
    );
    assert_eq!(
        transition_calls[0]
            .get("transitionId")
            .and_then(Value::as_str),
        Some("31")
    );
    // Comment carries the PR link from the pr step output.
    let comment_calls = comment.calls();
    assert_eq!(comment_calls.len(), 1);
    let body = comment_calls[0].get("body").and_then(Value::as_str).unwrap();
    assert!(
        body.contains("https://github.com/acme/app/pull/7"),
        "PR link in comment: {body}"
    );
}

/// `agent-m pickup` runs the flow inside a worktree checkout (worktree=true):
/// the clone step is skipped in favor of an in-place echo, and verify runs
/// `cargo test` at cwd instead of `cd work && cargo test`. Standalone
/// `--flow` runs keep the plain clone + cd. Both branches keep the `clone`
/// and `verify` step names so `${steps.clone.output}` and
/// `${steps.verify.output}` resolve downstream.
#[tokio::test]
async fn worktree_mode_skips_clone_and_runs_inplace_verify() {
    let bash = Arc::new(BashStub::new(vec![
        (
            "echo 'worktree checkout: /repo'".to_string(),
            ToolOutcome::success("worktree checkout: /repo"),
        ),
        (
            "cargo test".to_string(),
            ToolOutcome::success("all green"),
        ),
    ]));
    let gate: Arc<dyn AskGate> = Arc::new(ClosureAskGate::new(|_q, _o, _m| {
        Box::pin(async { Ok("approved".to_string()) })
    }));
    let deps = FlowDeps {
        provider: Arc::new(FakeLlm::new(vec![
            FakeLlm::text("the plan"),
            FakeLlm::text("implemented"),
        ])),
        agent_options: options(),
        tools: vec![bash.clone()],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: Some(gate),
        state_dir: None,
        on_progress: None,
    };
    let flow = agent_m_flow::Flow {
        name: "agentic-dev".to_string(),
        description: None,
        error_budget: Some(5),
        steps: vec![
            FlowStep::Condition {
                name: "workspace".to_string(),
                if_condition: "${worktree} == true".to_string(),
                then: vec![FlowStep::Tool {
                    name: "clone".to_string(),
                    tool: "bash".to_string(),
                    arguments: Some(json!({ "command": "echo 'worktree checkout: ${cwd}'" })),
                }],
                else_: Some(vec![FlowStep::Tool {
                    name: "clone".to_string(),
                    tool: "bash".to_string(),
                    arguments: Some(json!({ "command": "git clone ${repo} work" })),
                }]),
            },
            FlowStep::Ask {
                name: "review-plan".to_string(),
                question: "Approve?".to_string(),
                options: None,
                on_reject: "stop".to_string(),
            },
            FlowStep::Condition {
                name: "verify-env".to_string(),
                if_condition: "${worktree} == true".to_string(),
                then: vec![FlowStep::Verify {
                    name: "verify".to_string(),
                    command: Some("cargo test".to_string()),
                    max_fix_rounds: 3,
                }],
                else_: Some(vec![FlowStep::Verify {
                    name: "verify".to_string(),
                    command: Some("cd work && cargo test".to_string()),
                    max_fix_rounds: 3,
                }]),
            },
        ],
    };

    // Worktree mode: in-place checkout, no clone, `cargo test` at cwd.
    let mut context = FlowContext::new();
    context.set("cwd", json!("/repo"));
    context.set("worktree", json!("true"));
    let run = run_flow(&flow, &mut context, &deps).await.expect("run");
    assert!(
        run.steps
            .iter()
            .all(|s| s.status == agent_m_flow::StepStatus::Succeeded),
        "all steps succeeded"
    );
    let calls = bash.calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 2, "clone echo + inplace verify: {calls:?}");
    assert!(calls[0].contains("worktree checkout"), "{calls:?}");
    assert_eq!(calls[1], "cargo test", "{calls:?}");

    // Standalone mode: plain clone + cd work verify.
    bash.calls.lock().unwrap().clear();
    *bash.script.lock().unwrap() = [
        (
            "git clone acme/app work".to_string(),
            ToolOutcome::success("cloned"),
        ),
        (
            "cd work && cargo test".to_string(),
            ToolOutcome::success("all green"),
        ),
    ]
    .into();
    let mut context = FlowContext::new();
    context.set("cwd", json!("/elsewhere"));
    context.set("repo", json!("acme/app"));
    run_flow(&flow, &mut context, &deps).await.expect("run");
    let calls = bash.calls.lock().unwrap().clone();
    assert_eq!(calls.len(), 2, "{calls:?}");
    assert_eq!(calls[0], "git clone acme/app work", "{calls:?}");
    assert_eq!(calls[1], "cd work && cargo test", "{calls:?}");
}

#[tokio::test]
async fn red_verify_never_ships() {
    let bash = Arc::new(BashStub::new(vec![
        (
            "cd work && cargo test".to_string(),
            ToolOutcome::error("tests fail"),
        ),
        (
            "cd work && cargo test".to_string(),
            ToolOutcome::error("tests fail"),
        ),
    ]));
    let pr = Arc::new(ToolStub::new("github-create-pr", "PR created: x (#1)"));
    let transition = Arc::new(ToolStub::new("jira-transition", "transitioned"));
    let deps = FlowDeps {
        provider: Arc::new(FakeLlm::new(vec![FakeLlm::text("fix 1")])),
        agent_options: options(),
        tools: vec![bash.clone(), pr.clone(), transition.clone()],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: None,
        state_dir: None,
        on_progress: None,
    };
    let flow = agent_m_flow::Flow {
        name: "red-flow".to_string(),
        description: None,
        error_budget: Some(1),
        steps: vec![
            FlowStep::Verify {
                name: "verify".to_string(),
                command: Some("cd work && cargo test".to_string()),
                max_fix_rounds: 1,
            },
            FlowStep::Tool {
                name: "pr".to_string(),
                tool: "github-create-pr".to_string(),
                arguments: Some(json!({})),
            },
            FlowStep::Tool {
                name: "transition".to_string(),
                tool: "jira-transition".to_string(),
                arguments: Some(json!({})),
            },
        ],
    };
    let mut context = FlowContext::new();
    let error = run_flow(&flow, &mut context, &deps)
        .await
        .expect_err("must abort");
    assert!(error.to_string().contains("fix budget exhausted"), "{error}");
    assert!(pr.calls().is_empty(), "no PR on red verify");
    assert!(transition.calls().is_empty(), "no transition on red verify");
}

#[tokio::test]
async fn delegate_step_returns_parsed_json_and_feeds_downstream() {
    // The delegate step spawns a fresh sub-agent; the FakeLlm's single
    // scripted reply IS that sub-agent's JSON answer.
    let llm = Arc::new(FakeLlm::new(vec![FakeLlm::text(
        r#"{"risk":"low","entrypoints":["src/main.rs"]}"#,
    )]));
    let bash = Arc::new(BashStub::new(vec![(
        "echo low".to_string(),
        ToolOutcome::success("low"),
    )]));
    let deps = FlowDeps {
        provider: llm.clone(),
        agent_options: options(),
        tools: vec![bash.clone()],
        permission_gate: Arc::new(AlwaysAllowGate),
        ask_gate: None,
        state_dir: None,
        on_progress: None,
    };
    let flow = agent_m_flow::Flow {
        name: "structured".to_string(),
        description: None,
        error_budget: None,
        steps: vec![
            FlowStep::Delegate {
                name: "analysis".to_string(),
                prompt: "Map entrypoints and rate the risk".to_string(),
                schema: Some(json!({ "type": "object" })),
                tools: None,
                max_turns: 4,
                on_invalid: "stop".to_string(),
            },
            FlowStep::Tool {
                name: "echo-risk".to_string(),
                tool: "bash".to_string(),
                arguments: Some(json!({ "command": "echo ${steps.analysis.output.json.risk}" })),
            },
        ],
    };
    let mut context = FlowContext::new();
    let run = run_flow(&flow, &mut context, &deps).await.expect("run");
    assert_eq!(run.steps.len(), 2);
    assert_eq!(run.steps[0].status, agent_m_flow::StepStatus::Succeeded);
    // Parsed JSON + pretty-printed text are both exposed.
    let output = run.steps[0].output.as_ref().expect("delegate output");
    assert_eq!(output["json"]["risk"], json!("low"));
    assert_eq!(output["json"]["entrypoints"][0], json!("src/main.rs"));
    assert_eq!(output["isError"], json!(false));
    assert!(output["content"].as_str().unwrap_or("").contains("\"risk\""));
    // The sub-agent's prompt ran in a fresh context window.
    assert_eq!(llm.prompts(), vec!["Map entrypoints and rate the risk"]);
    // Downstream refs resolve to the parsed JSON fields.
    assert_eq!(
        context
            .get("steps.analysis.output.json.risk")
            .and_then(Value::as_str),
        Some("low")
    );
    assert_eq!(bash.calls.lock().unwrap()[0], "echo low");
}

#[tokio::test]
async fn delegate_step_invalid_json_aborts_the_flow() {
    let llm = Arc::new(FakeLlm::new(vec![FakeLlm::text(
        "I explored the codebase. Entrypoints are src/main.rs.",
    )]));
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
        name: "strict".to_string(),
        description: None,
        error_budget: None,
        steps: vec![FlowStep::Delegate {
            name: "analysis".to_string(),
            prompt: "Map entrypoints".to_string(),
            schema: None,
            tools: None,
            max_turns: 4,
            on_invalid: "stop".to_string(),
        }],
    };
    let mut context = FlowContext::new();
    let error = run_flow(&flow, &mut context, &deps)
        .await
        .expect_err("must abort on invalid JSON");
    assert!(
        error.to_string().contains("did not produce valid JSON"),
        "got: {error}"
    );
}

#[tokio::test]
async fn delegate_step_on_invalid_continue_keeps_flow_running() {
    let llm = Arc::new(FakeLlm::new(vec![FakeLlm::text(
        "no json here, just prose",
    )]));
    let bash = Arc::new(BashStub::new(vec![(
        "echo after".to_string(),
        ToolOutcome::success("after"),
    )]));
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
        name: "tolerant".to_string(),
        description: None,
        error_budget: None,
        steps: vec![
            FlowStep::Delegate {
                name: "analysis".to_string(),
                prompt: "Map entrypoints".to_string(),
                schema: None,
                tools: None,
                max_turns: 4,
                on_invalid: "continue".to_string(),
            },
            FlowStep::Tool {
                name: "after".to_string(),
                tool: "bash".to_string(),
                arguments: Some(json!({ "command": "echo after" })),
            },
        ],
    };
    let mut context = FlowContext::new();
    let run = run_flow(&flow, &mut context, &deps).await.expect("flow continues");
    assert_eq!(run.steps[0].status, agent_m_flow::StepStatus::Failed);
    assert!(
        run.steps[0]
            .error
            .as_deref()
            .unwrap_or("")
            .contains("did not produce valid JSON")
    );
    assert_eq!(run.steps[1].status, agent_m_flow::StepStatus::Succeeded);
    assert_eq!(bash.calls.lock().unwrap()[0], "echo after");
}
