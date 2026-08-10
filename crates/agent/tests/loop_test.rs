//! Deterministic agent-loop tests against a scripted fake provider.

use agent_m_agent::{
    Agent, AgentEvent, AgentOptions, AlwaysAllowGate, BoolGate, InterruptHandle, Permission,
    PermissionGate, SessionMessage, Tool, ToolCallInfo, ToolContext, ToolError, ToolOutcome,
};
use agent_m_ai::{
    AiError, AssistantMessage, CacheStats, ChatRequest, ContentPart, ModelSpec, Provider,
    StopReason, StreamEvent, Usage,
};
use async_trait::async_trait;
use futures_util::stream::{self, BoxStream};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

/// A provider that plays back scripted responses, one per stream_chat call.
struct FakeLlm {
    responses: Mutex<VecDeque<Vec<StreamEvent>>>,
}

impl FakeLlm {
    fn new(responses: Vec<Vec<StreamEvent>>) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
        }
    }

    fn text_response(text: &str, cache_read_tokens: u64) -> Vec<StreamEvent> {
        vec![
            StreamEvent::Start,
            StreamEvent::TextDelta {
                delta: text.to_string(),
            },
            StreamEvent::Done {
                message: AssistantMessage {
                    content: vec![ContentPart::Text {
                        text: text.to_string(),
                    }],
                    usage: Some(Usage {
                        input_tokens: 10,
                        output_tokens: 5,
                        cache_read_tokens,
                        cache_creation_tokens: 10 - cache_read_tokens,
                        total_tokens: 15,
                        cost: 0.0,
                    }),
                    stop_reason: StopReason::Stop,
                    error_message: None,
                    model: "fake".to_string(),
                },
            },
        ]
    }

    fn tool_response(name: &str, arguments: Value) -> Vec<StreamEvent> {
        vec![
            StreamEvent::Start,
            StreamEvent::ToolCallDelta {
                index: 0,
                id: "call_1".to_string(),
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
            StreamEvent::Done {
                message: AssistantMessage {
                    content: vec![ContentPart::ToolCall {
                        id: "call_1".to_string(),
                        name: name.to_string(),
                        arguments,
                    }],
                    usage: None,
                    stop_reason: StopReason::ToolUse,
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

/// A trivial tool the fake model can call.
struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }
    fn description(&self) -> String {
        "echoes its text argument".to_string()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        })
    }
    async fn execute(
        &self,
        arguments: Value,
        _context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let text = arguments
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default();
        Ok(ToolOutcome::success(format!("echoed: {text}")))
    }
}

fn options() -> AgentOptions {
    AgentOptions {
        model: "fake".to_string(),
        system_prompt: "You are a test agent.".to_string(),
        tools: vec![Arc::new(EchoTool)],
        max_turns: 5,
        cwd: PathBuf::from("."),
        mode: agent_m_agent::Mode::Build,
        ask_gate: None,
        context_window: None,
    }
}

fn event_label(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::AgentStart => "agent_start",
        AgentEvent::TurnStart { .. } => "turn_start",
        AgentEvent::MessageStart { .. } => "message_start",
        AgentEvent::MessageUpdate { .. } => "message_update",
        AgentEvent::MessageEnd { .. } => "message_end",
        AgentEvent::ToolExecutionStart { .. } => "tool_execution_start",
        AgentEvent::ToolExecutionEnd { .. } => "tool_execution_end",
        AgentEvent::TurnEnd { .. } => "turn_end",
        AgentEvent::Compacted { .. } => "compacted",
        AgentEvent::AgentEnd { .. } => "agent_end",
        AgentEvent::Notice { .. } => "notice",
        AgentEvent::FlowStep { .. } => "flow_step",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn emits_pi_event_ordering_through_tool_calls() {
    let fake = Arc::new(FakeLlm::new(vec![
        FakeLlm::tool_response("echo", json!({ "text": "hi" })),
        FakeLlm::text_response("done", 8),
    ]));
    let mut agent = Agent::new(fake, options(), Arc::new(AlwaysAllowGate));

    let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let capture = events.clone();
    agent.subscribe(move |event| capture.lock().unwrap().push(event.clone()));

    agent.prompt("say hi".to_string()).await.expect("prompt");

    let events = events.lock().unwrap();
    let labels: Vec<&str> = events.iter().map(event_label).collect();
    assert_eq!(
        labels,
        vec![
            "agent_start",
            "message_start", // user
            "message_end",
            "turn_start",
            "message_start", // assistant (tool call turn)
            "message_update",
            "message_end",
            "tool_execution_start",
            "tool_execution_end",
            "turn_end",
            "turn_start", // final text turn
            "message_start",
            "message_update",
            "message_end",
            "turn_end",
            "agent_end",
        ]
    );

    // The tool result was recorded in the context.
    let messages: Vec<&SessionMessage> = agent.messages().iter().collect();
    assert!(matches!(
        messages[2],
        SessionMessage::ToolResult { name, is_error: false, .. } if name == "echo"
    ));

    // Cache stats flowed through to AgentEnd.
    let last = events.last().unwrap();
    match last {
        AgentEvent::AgentEnd { cache_stats, .. } => {
            assert_eq!(cache_stats.hit_tokens, 8);
            assert_eq!(cache_stats.miss_tokens, 2);
        }
        _ => panic!("expected agent_end, got {last:?}"),
    }
}

#[tokio::test]
async fn denied_tool_call_becomes_error_result() {
    let fake = Arc::new(FakeLlm::new(vec![
        FakeLlm::tool_response("echo", json!({ "text": "hi" })),
        FakeLlm::text_response("understood", 0),
    ]));
    let deny = Arc::new(BoolGate::new(|_| false));
    let mut agent = Agent::new(fake, options(), deny);

    let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let capture = events.clone();
    agent.subscribe(move |event| capture.lock().unwrap().push(event.clone()));

    agent.prompt("do it".to_string()).await.expect("prompt");

    let events = events.lock().unwrap();
    let execution_end = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::ToolExecutionEnd { outcome, .. } => Some(outcome),
            _ => None,
        })
        .expect("tool execution end");
    assert!(execution_end.is_error);
    assert!(execution_end.content.contains("Permission denied"));

    let tool_result = agent
        .messages()
        .iter()
        .find_map(|message| match message {
            SessionMessage::ToolResult { content, .. } => Some(content.clone()),
            _ => None,
        })
        .expect("tool result message");
    assert!(tool_result.contains("Permission denied"));
}

#[tokio::test]
async fn model_error_is_reported_as_notice_and_run_ends() {
    let fake = Arc::new(FakeLlm::new(vec![vec![
        StreamEvent::Start,
        StreamEvent::Error {
            message: "connection reset".to_string(),
        },
    ]]));
    let mut agent = Agent::new(fake, options(), Arc::new(AlwaysAllowGate));

    let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let capture = events.clone();
    agent.subscribe(move |event| capture.lock().unwrap().push(event.clone()));

    agent.prompt("hi".to_string()).await.expect("prompt");

    let events = events.lock().unwrap();
    let labels: Vec<&str> = events.iter().map(event_label).collect();
    assert!(labels.contains(&"notice"), "got {labels:?}");
    assert_eq!(labels.last(), Some(&"agent_end"));

    let final_message = agent.messages().last().unwrap();
    match final_message {
        SessionMessage::Assistant { stop_reason, .. } => {
            assert_eq!(*stop_reason, StopReason::Error);
        }
        other => panic!("expected assistant message, got {other:?}"),
    }
}

#[tokio::test]
async fn cache_stats_accumulate_across_turns() {
    let fake = Arc::new(FakeLlm::new(vec![
        FakeLlm::text_response("first", 5),
        FakeLlm::text_response("second", 3),
    ]));
    let mut agent = Agent::new(fake, options(), Arc::new(AlwaysAllowGate));

    let stats: Arc<Mutex<Option<CacheStats>>> = Arc::new(Mutex::new(None));
    let capture = stats.clone();
    agent.subscribe(move |event| {
        if let AgentEvent::AgentEnd { cache_stats, .. } = event {
            *capture.lock().unwrap() = Some(cache_stats.clone());
        }
    });

    agent.prompt("one".to_string()).await.expect("prompt");
    agent.prompt("two".to_string()).await.expect("prompt");

    let stats = stats.lock().unwrap().clone().expect("agent_end seen");
    assert_eq!(stats.requests, 2);
    assert_eq!(stats.hit_tokens, 8);
    // First turn: read 5 / miss 5; second turn: read 3 / miss 7.
    assert_eq!(stats.miss_tokens, 12);
}

#[tokio::test]
async fn interrupt_aborts_the_stream() {
    /// A provider that sleeps before its first delta, so an interrupt can land
    /// mid-stream.
    struct SlowLlm;
    #[async_trait]
    impl Provider for SlowLlm {
        fn id(&self) -> &str {
            "slow"
        }
        fn display_name(&self) -> &str {
            "Slow"
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
            Ok(Box::pin(stream::unfold(0u8, |state| async move {
                match state {
                    0 => {
                        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                        Some((
                            StreamEvent::TextDelta {
                                delta: "slow text".to_string(),
                            },
                            1,
                        ))
                    }
                    1 => Some((
                        StreamEvent::Done {
                            message: AssistantMessage {
                                content: vec![ContentPart::Text {
                                    text: "slow text".to_string(),
                                }],
                                usage: None,
                                stop_reason: StopReason::Stop,
                                error_message: None,
                                model: "slow".to_string(),
                            },
                        },
                        2,
                    )),
                    _ => None,
                }
            })))
        }
    }

    let mut agent = Agent::new(Arc::new(SlowLlm), options(), Arc::new(AlwaysAllowGate));
    let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let capture = events.clone();
    agent.subscribe(move |event| capture.lock().unwrap().push(event.clone()));
    let handle: InterruptHandle = agent.interrupt_handle();
    let interrupt_task = tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        handle.interrupt();
    });

    agent.prompt("hi".to_string()).await.expect("prompt");
    interrupt_task.await.unwrap();

    // The aborted reply is emitted as MessageEnd with StopReason::Aborted but
    // is NOT persisted to history (half-streamed content must not pollute the
    // byte-stable prefix), so assert via the event stream.
    let message_end = events
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find_map(|event| match event {
            AgentEvent::MessageEnd {
                message: message @ SessionMessage::Assistant { .. },
            } => Some(message.clone()),
            _ => None,
        })
        .expect("assistant message_end seen");
    match message_end {
        SessionMessage::Assistant { stop_reason, .. } => {
            assert_eq!(stop_reason, StopReason::Aborted);
        }
        other => panic!("expected assistant message, got {other:?}"),
    }
    // History holds only the user message.
    assert_eq!(agent.messages().len(), 1);
}

/// A mutating tool stub — must be hidden in plan mode.
struct BashStub;

#[async_trait]
impl Tool for BashStub {
    fn name(&self) -> &str {
        "bash"
    }
    fn description(&self) -> String {
        "run a shell command".to_string()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "command": { "type": "string" } },
            "required": ["command"]
        })
    }
    async fn execute(
        &self,
        _arguments: Value,
        _context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        Ok(ToolOutcome::success("ran"))
    }
}

/// A read-only tool stub — allowed in plan mode.
struct ReadStub;

#[async_trait]
impl Tool for ReadStub {
    fn name(&self) -> &str {
        "read"
    }
    fn description(&self) -> String {
        "read a file".to_string()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        })
    }
    async fn execute(
        &self,
        _arguments: Value,
        _context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        Ok(ToolOutcome::success("contents"))
    }
}

#[tokio::test]
async fn plan_mode_hides_mutating_tools() {
    let mut agent = Agent::new(
        Arc::new(FakeLlm::new(vec![
            FakeLlm::tool_response("bash", json!({ "command": "ls" })),
            FakeLlm::tool_response("ls", json!({ "path": "." })),
            FakeLlm::tool_response("search", json!({ "query": "cache_hit" })),
            FakeLlm::text_response("finished", 8),
        ])),
        AgentOptions {
            model: "fake".to_string(),
            system_prompt: "You are a test agent.".to_string(),
            // The full registered set, as the CLI now provides: read-only
            // tools (which plan mode keeps) plus mutating ones (which it
            // hides).
            tools: vec![
                Arc::new(ReadStub),
                Arc::new(LsStub),
                Arc::new(SearchStub),
                Arc::new(BashStub),
            ],
            max_turns: 6,
            cwd: PathBuf::from("."),
            mode: agent_m_agent::Mode::Plan,
            ask_gate: None,
            context_window: None,
        },
        Arc::new(AlwaysAllowGate),
    );
    let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let capture = events.clone();
    agent.subscribe(move |event| capture.lock().unwrap().push(event.clone()));

    agent
        .prompt("plan the work".to_string())
        .await
        .expect("prompt");

    let events = events.lock().unwrap();
    let outcomes: Vec<&ToolOutcome> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { outcome, .. } => Some(outcome),
            _ => None,
        })
        .collect();
    assert_eq!(
        outcomes.len(),
        3,
        "bash rejected, ls+search allowed: {outcomes:?}"
    );
    assert!(
        outcomes[0].is_error && outcomes[0].content.contains("not available in plan mode"),
        "plan mode must reject bash, got: {}",
        outcomes[0].content
    );
    assert!(
        !outcomes[1].is_error && outcomes[1].content == "listing",
        "plan mode must allow ls, got: {}",
        outcomes[1].content
    );
    assert!(
        !outcomes[2].is_error && outcomes[2].content == "hits",
        "plan mode must allow search, got: {}",
        outcomes[2].content
    );
    // The plan-mode system prompt must be used: switch back and forth.
    let _ = agent.mode();
    agent.set_mode(agent_m_agent::Mode::Build);
    assert_eq!(agent.mode(), agent_m_agent::Mode::Build);
}

/// A read-only search stub — must be allowed in plan mode.
struct SearchStub;

#[async_trait]
impl Tool for SearchStub {
    fn name(&self) -> &str {
        "search"
    }
    fn description(&self) -> String {
        "search the codebase index".to_string()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"]
        })
    }
    async fn execute(
        &self,
        _arguments: Value,
        _context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        Ok(ToolOutcome::success("hits"))
    }
}

/// A read-only directory-listing stub — must be allowed in plan mode.
struct LsStub;

#[async_trait]
impl Tool for LsStub {
    fn name(&self) -> &str {
        "ls"
    }
    fn description(&self) -> String {
        "list a directory".to_string()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        })
    }
    async fn execute(
        &self,
        _arguments: Value,
        _context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        Ok(ToolOutcome::success("listing"))
    }
}

/// A stub for the `ask` tool (the real one lives in agent-m-tools).
struct AskStub;

#[async_trait]
impl Tool for AskStub {
    fn name(&self) -> &str {
        "ask"
    }
    fn description(&self) -> String {
        "ask the user a question".to_string()
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "question": { "type": "string" } },
            "required": ["question"]
        })
    }
    async fn execute(
        &self,
        arguments: Value,
        context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let question = arguments
            .get("question")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let Some(gate) = &context.ask_gate else {
            return Ok(ToolOutcome::error("ask requires the interactive UI"));
        };
        match gate.ask(question.to_string(), None).await {
            Ok(answer) => Ok(ToolOutcome::success(format!("User answer: {answer}"))),
            Err(message) => Ok(ToolOutcome::error(format!("ask cancelled: {message}"))),
        }
    }
}

#[tokio::test]
async fn ask_tool_returns_user_answer_and_continues() {
    let gate = agent_m_agent::ClosureAskGate::new(|_question, _options| {
        Box::pin(async { Ok("blue".to_string()) })
    });
    let mut agent = Agent::new(
        Arc::new(FakeLlm::new(vec![
            FakeLlm::tool_response("ask", json!({ "question": "Which color?" })),
            FakeLlm::text_response("done", 8),
        ])),
        AgentOptions {
            model: "fake".to_string(),
            system_prompt: "You are a test agent.".to_string(),
            tools: vec![Arc::new(AskStub)],
            max_turns: 4,
            cwd: PathBuf::from("."),
            mode: agent_m_agent::Mode::Build,
            ask_gate: Some(Arc::new(gate)),
            context_window: None,
        },
        Arc::new(AlwaysAllowGate),
    );
    let events: Arc<Mutex<Vec<AgentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let capture = events.clone();
    agent.subscribe(move |event| capture.lock().unwrap().push(event.clone()));

    agent
        .prompt("pick a color".to_string())
        .await
        .expect("prompt");

    let events = events.lock().unwrap();
    let outcomes: Vec<&ToolOutcome> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::ToolExecutionEnd { outcome, .. } => Some(outcome),
            _ => None,
        })
        .collect();
    assert_eq!(outcomes.len(), 1);
    assert!(
        outcomes[0].content.contains("User answer: blue"),
        "got: {}",
        outcomes[0].content
    );
}

#[tokio::test]
async fn compaction_replaces_older_messages_with_summary() {
    let mut agent = Agent::new(
        Arc::new(FakeLlm::new(vec![
            FakeLlm::text_response("first reply", 8),
            FakeLlm::text_response("second reply", 8),
            FakeLlm::text_response("summary of the conversation", 0),
        ])),
        AgentOptions {
            model: "fake".to_string(),
            system_prompt: "You are a test agent.".to_string(),
            tools: vec![Arc::new(EchoTool)],
            max_turns: 5,
            cwd: PathBuf::from("."),
            mode: agent_m_agent::Mode::Build,
            ask_gate: None,
            context_window: Some(64_000),
        },
        Arc::new(AlwaysAllowGate),
    );
    agent.prompt("hello".to_string()).await.expect("prompt 1");
    agent.prompt("again".to_string()).await.expect("prompt 2");
    assert_eq!(agent.messages().len(), 4);

    let summary = agent.summarize_and_compact(2).await.expect("compact");
    assert!(
        summary.contains("summary of the conversation"),
        "got: {summary}"
    );

    let messages = agent.messages();
    // [summary, user, assistant] — the two oldest (first user+assistant) were
    // replaced by the summary.
    assert_eq!(messages.len(), 3, "messages: {messages:?}");
    assert!(matches!(&messages[0], SessionMessage::Summary { .. }));
    // context usage reflects the provider-reported input tokens.
    let (tokens, window) = agent.context_usage();
    assert_eq!(tokens, 10);
    assert_eq!(window, Some(64_000));
}

fn bash_call(command: &str) -> ToolCallInfo {
    ToolCallInfo {
        tool_call_id: "t1".to_string(),
        name: "bash".to_string(),
        arguments: json!({ "command": command }),
    }
}

#[test]
fn risk_policy_detects_destructive_commands() {
    let policy = agent_m_agent::RiskPolicy {
        cwd: PathBuf::from("/work"),
        opaque_tools: vec![],
    };
    assert!(policy.risk(&bash_call("rm -rf /tmp/x")).is_some());
    assert!(policy.risk(&bash_call("sudo rm -rf /")).is_some());
    assert!(
        policy
            .risk(&bash_call("git checkout --force main"))
            .is_some()
    );
    assert!(policy.risk(&bash_call("git reset --hard HEAD")).is_some());
    assert!(policy.risk(&bash_call("git clean -fd")).is_some());
    assert!(
        policy
            .risk(&bash_call("find . -name '*.tmp' -delete"))
            .is_some()
    );
    assert!(policy.risk(&bash_call("dd if=x of=/dev/sda")).is_some());
    // Benign commands pass.
    assert!(policy.risk(&bash_call("ls -la")).is_none());
    assert!(policy.risk(&bash_call("rm file.txt")).is_none());
    assert!(policy.risk(&bash_call("git status")).is_none());
    // Non-bash tools are never flagged by command risk.
    let read = ToolCallInfo {
        tool_call_id: "t2".to_string(),
        name: "read".to_string(),
        arguments: json!({ "path": "rm -rf /" }),
    };
    assert!(policy.risk(&read).is_none());
}

#[tokio::test]
async fn selective_gate_asks_only_for_destructive_commands() {
    let policy = Arc::new(agent_m_agent::RiskPolicy {
        cwd: PathBuf::from("/work"),
        opaque_tools: vec![],
    });
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ToolCallInfo>();
    let gate = agent_m_agent::SelectiveAskGate::new(policy, move |call: ToolCallInfo| {
        let tx = tx.clone();
        Box::pin(async move {
            let _ = tx.send(call);
            Permission::Allowed
        })
    });
    // Benign: auto-approved without asking.
    let permission = gate.authorize(&bash_call("ls -la")).await;
    assert_eq!(permission, Permission::Allowed);
    assert!(rx.try_recv().is_err(), "benign commands must not ask");
    // Destructive: routed through the ask closure.
    let permission = gate.authorize(&bash_call("rm -rf /tmp/x")).await;
    assert_eq!(permission, Permission::Allowed);
    let asked = rx.try_recv().expect("destructive commands must ask");
    assert_eq!(asked.name, "bash");
    assert!(
        asked
            .arguments
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("")
            .contains("rm -rf")
    );
}

#[tokio::test]
async fn dangerous_gate_denies_destructive_under_auto_approve() {
    let policy = Arc::new(agent_m_agent::RiskPolicy {
        cwd: PathBuf::from("/work"),
        opaque_tools: vec![],
    });
    let gate = agent_m_agent::DangerousCommandGate::new(policy, AlwaysAllowGate);
    match gate.authorize(&bash_call("rm -rf /tmp/x")).await {
        Permission::Denied(message) => {
            assert!(
                message.contains("recursive delete"),
                "reason should name the risk: {message}"
            )
        }
        other => panic!("destructive command must be denied under auto-approve, got {other:?}"),
    }
    let permission = gate.authorize(&bash_call("git status")).await;
    assert_eq!(permission, Permission::Allowed);
}

#[tokio::test]
async fn read_only_tools_bypass_the_inner_gate() {
    // Inner denies everything; only read-only tool names should skip it.
    let gate = agent_m_agent::ReadOnlyAutoApproveGate::new(BoolGate::new(|_| false));
    let read_call = ToolCallInfo {
        tool_call_id: "t1".to_string(),
        name: "read".to_string(),
        arguments: json!({ "path": "src/main.rs" }),
    };
    assert_eq!(gate.authorize(&read_call).await, Permission::Allowed);
    let ls_call = ToolCallInfo {
        tool_call_id: "t2".to_string(),
        name: "ls".to_string(),
        arguments: json!({}),
    };
    assert_eq!(gate.authorize(&ls_call).await, Permission::Allowed);

    let write_call = ToolCallInfo {
        tool_call_id: "t3".to_string(),
        name: "write".to_string(),
        arguments: json!({ "path": "src/main.rs", "content": "" }),
    };
    assert!(matches!(
        gate.authorize(&write_call).await,
        Permission::Denied(_)
    ));
}

#[tokio::test]
async fn benign_shell_commands_auto_approve_without_yes() {
    // The TUI's actual interactive gate shape: SelectiveAskGate wrapped in
    // ReadOnlyAutoApproveGate, regardless of --yes. A model that runs `ls`
    // or `cat` via `bash` instead of the dedicated tools must not prompt.
    let policy = Arc::new(agent_m_agent::RiskPolicy {
        cwd: PathBuf::from("/work"),
        opaque_tools: vec![],
    });
    let gate = agent_m_agent::ReadOnlyAutoApproveGate::new(agent_m_agent::SelectiveAskGate::new(
        policy,
        |_call: ToolCallInfo| Box::pin(async { Permission::Denied("should not ask".to_string()) }),
    ));
    assert_eq!(
        gate.authorize(&bash_call("ls -la")).await,
        Permission::Allowed
    );
    assert_eq!(
        gate.authorize(&bash_call("cat README.md")).await,
        Permission::Allowed
    );
    assert_eq!(
        gate.authorize(&bash_call("git status")).await,
        Permission::Allowed
    );
    // A risky command must still ask (here: denied, proving it reached the ask closure).
    assert_eq!(
        gate.authorize(&bash_call("rm -rf /tmp/x")).await,
        Permission::Denied("should not ask".to_string())
    );
}
