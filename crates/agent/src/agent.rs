//! The agent loop: user message in, streamed assistant replies and tool calls
//! out, with pi-compatible event ordering.

use agent_m_ai::{
    AiError, CacheStats, ChatRequest, ContentPart, Provider, StopReason, StreamEvent, ToolSpec,
};
use futures_util::StreamExt;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::Instrument;

use crate::message::SessionMessage;
use crate::tool::{Permission, PermissionGate, Tool, ToolCallInfo, ToolContext, ToolOutcome};

/// The agent's working mode. Plan mode is read-only: mutating tools are
/// hidden and the model is asked to produce a numbered plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    #[default]
    Build,
    Plan,
}

impl Mode {
    pub fn as_str(&self) -> &'static str {
        match self {
            Mode::Build => "build",
            Mode::Plan => "plan",
        }
    }
}

/// Appended to the system prompt in plan mode (same idea as pi's plan-mode
/// extension): the model must produce a numbered `Plan:` list and may not
/// mutate anything. The available tools are named explicitly so the model
/// does not invent `bash` calls or try to read directories.
const PLAN_MODE_BLOCK: &str = "\n\nYou are in PLAN MODE. You may only read, search, and ask questions — you cannot modify files or run state-changing commands. Available tools: `ls` (list a directory — use this instead of reading a directory), `read` (read one file), `view_outline` (fast AST outline of a file), `grep` (search file contents), `find` (locate files), `ask` (ask the user a question). There is no `bash` tool in plan mode. Explore the codebase, then create a detailed numbered plan under a heading `Plan:`, one item per line (`1. step`). Each step must be a concrete, verifiable action. Do not execute the plan yet.";

/// The tools a plan-mode agent may call (read-only + ask).
pub(crate) const PLAN_TOOLS: &[&str] = &[
    "read",
    "view_outline",
    "grep",
    "find",
    "ls",
    "ask",
    "search",
    "web_fetch",
    "web_search",
];

/// Static trust-metadata instruction appended to every system prompt
/// (check.md principles 2/3/4/9/10). Static text keeps the prefix byte-stable.
const TRUST_BLOCK: &str = "\n\nEnd each reply with a <trust> block the harness machine-reads (it is never shown to the user): <confidence>0-100</confidence>, <reason>, <expected_outcome>, <evidence> with <item file=\"…\" line=\"…\">note</item> entries, <uncertainty>, <plan> with <item> steps, and <estimated_time>. Omit any field you cannot answer honestly; do not invent evidence or confidence.";

/// The `delegate` tool: spawn a fresh-context sub-agent (check.md-inspired
/// subagents — parallel/isolated work with its own context window).
const DELEGATE_SPEC: &str = r#"{"type":"object","properties":{"prompt":{"type":"string","description":"Self-contained task for the sub-agent"},"tools":{"type":"array","items":{"type":"string"},"description":"Restrict the sub-agent to these tools (default: the parent's set minus delegate)"},"max_turns":{"type":"integer","description":"Sub-agent turn budget (default 4)"}},"required":["prompt"]}"#;

/// Instructions for the compaction summarizer (memory across sessions).
const SUMMARY_PROMPT: &str = "Summarize the conversation above for continuation by a coding agent. Keep it concise but complete: the goal, key decisions, files touched, important tool results, user preferences, and open questions. 300 words or fewer.";

/// Pull the `<trust>` block out of the last text part (the model is
/// instructed to end the reply with it) and return (trust, parts-without).
/// Truncate `ToolResult` content in `messages` to `max_chars` in-place.
/// Used as a pre-step before retrying a compaction summarization that failed
/// because the old messages were too large.
fn clamp_tool_results(messages: &mut [SessionMessage], max_chars: usize) {
    for msg in messages.iter_mut() {
        if let SessionMessage::ToolResult { content, .. } = msg
            && content.len() > max_chars
        {
            // String::truncate takes a *byte* index and panics if it lands
            // inside a multi-byte UTF-8 char — walk back to a char boundary
            // (review: compaction-failure recovery must never crash the
            // agent on non-ASCII tool output).
            let mut boundary = max_chars;
            while !content.is_char_boundary(boundary) {
                boundary -= 1;
            }
            content.truncate(boundary);
            content.push_str("\n…[clamped for compaction]");
        }
    }
}

fn extract_trust(parts: &[ContentPart]) -> (agent_m_ai::TrustData, Vec<ContentPart>) {
    let mut result = parts.to_vec();
    let Some(last_text) = result.iter_mut().rev().find_map(|part| match part {
        ContentPart::Text { text } => Some(text),
        _ => None,
    }) else {
        return (agent_m_ai::TrustData::default(), result);
    };
    let (trust, cleaned) = agent_m_ai::extract_trust_block(last_text);
    *last_text = cleaned;
    (trust, result)
}

/// Configuration for one agent run.
#[derive(Clone)]
pub struct AgentOptions {
    /// Model id sent to the provider, e.g. `deepseek-chat`.
    pub model: String,
    /// Fixed system prompt. Assembled once and byte-stable for the session.
    pub system_prompt: String,
    /// Continual-Harness block (memories/notes/skills), injected between the
    /// base prompt and the trust suffix. Rebuilt only on refine/rollback.
    pub harness_block: Option<String>,
    /// The tools the model may call (already allow/deny filtered).
    pub tools: Vec<Arc<dyn Tool>>,
    /// Safety cap on the number of model turns (including tool-call turns).
    pub max_turns: usize,
    /// Working directory for tools.
    pub cwd: PathBuf,
    /// Plan or build mode.
    pub mode: Mode,
    /// The `ask` tool's user gate (None → ask fails with a clear message).
    pub ask_gate: Option<Arc<dyn crate::tool::AskGate>>,
    /// The model's context window in tokens (used for compaction + display).
    pub context_window: Option<u64>,
    /// Selected reasoning-effort variant (`default`/`low`/`high`/`max`),
    /// forwarded to providers that support `reasoning_effort`.
    pub variant: Option<String>,
    /// Directory for offloading large tool outputs outside the context window.
    /// When set, bash/grep results over 10KB are written here and replaced with
    /// a 2KB preview + path hint. None → plain truncation (existing behavior).
    pub output_dir: Option<std::path::PathBuf>,
}

/// Events emitted by the agent loop, in pi's ordering:
/// `AgentStart`, per turn `TurnStart`, `MessageStart/Update/End`,
/// `ToolExecutionStart/End`, `TurnEnd`, … then `AgentEnd`.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    AgentStart,
    TurnStart {
        turn: usize,
    },
    MessageStart {
        kind: crate::message::SessionMessageKind,
    },
    MessageUpdate {
        delta: StreamEvent,
    },
    MessageEnd {
        message: SessionMessage,
    },
    ToolExecutionStart {
        tool_call_id: String,
        name: String,
        arguments: serde_json::Value,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        outcome: ToolOutcome,
    },
    /// The background refinement planner finished. The proposal is JSON
    /// (the agent crate does not depend on the tui crate's types).
    RefineResult {
        proposal_json: String,
    },
    TurnEnd {
        message: SessionMessage,
        tool_results: usize,
    },
    AgentEnd {
        messages: usize,
        cache_stats: CacheStats,
    },
    /// Non-fatal notice (stream error, max turns reached, ...).
    Notice {
        message: String,
    },
    /// The conversation was compacted: older messages were replaced by a
    /// summary (the cross-session memory mechanism).
    Compacted {
        summary: String,
        messages_removed: usize,
    },
    /// A flow step's live status change (index into the top-level steps).
    FlowStep {
        index: usize,
        name: String,
        status: String,
    },
}

/// A cloneable handle that interrupts a running agent (escape key, tests).
#[derive(Clone)]
pub struct InterruptHandle {
    interrupted: Arc<AtomicBool>,
    interrupt_notify: Arc<tokio::sync::Notify>,
}

impl InterruptHandle {
    pub fn interrupt(&self) {
        self.interrupted.store(true, Ordering::SeqCst);
        self.interrupt_notify.notify_one();
    }
}

/// Errors that stop a prompt before any stream could be produced.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("{0}")]
    Provider(#[from] AiError),
}

/// An event listener attached via [`Agent::subscribe`].
type AgentListener = Arc<dyn Fn(&AgentEvent) + Send + Sync>;

/// The agent: owns the conversation, streams replies, executes tool calls.
pub struct Agent {
    provider: Arc<dyn Provider>,
    options: AgentOptions,
    gate: Arc<dyn PermissionGate>,
    /// Byte-stable tool specs, computed once at construction.
    tool_specs: Vec<ToolSpec>,
    messages: Vec<SessionMessage>,
    cache_stats: CacheStats,
    listeners: Vec<AgentListener>,
    interrupted: Arc<AtomicBool>,
    interrupt_notify: Arc<tokio::sync::Notify>,
    /// Provider-reported prompt tokens of the most recent turn (≈ context size).
    last_input_tokens: u64,
    /// Per-session read dedup cache shared across all ToolContext instances.
    read_cache: crate::tool::ReadCache,
}

impl Agent {
    pub fn new(
        provider: Arc<dyn Provider>,
        options: AgentOptions,
        gate: Arc<dyn PermissionGate>,
    ) -> Self {
        let tool_specs = Self::tool_specs_for(&options);
        Self {
            provider,
            options,
            gate,
            tool_specs,
            messages: Vec::new(),
            cache_stats: CacheStats::default(),
            listeners: Vec::new(),
            interrupted: Arc::new(AtomicBool::new(false)),
            interrupt_notify: Arc::new(tokio::sync::Notify::new()),
            last_input_tokens: 0,
            read_cache: std::sync::Arc::new(
                std::sync::Mutex::new(std::collections::HashMap::new()),
            ),
        }
    }

    /// The tools active in the current mode (plan mode filters to read-only).
    fn active_tools(options: &AgentOptions) -> Vec<Arc<dyn Tool>> {
        options
            .tools
            .iter()
            .filter(|tool| {
                if options.mode == Mode::Plan {
                    PLAN_TOOLS.contains(&tool.name())
                } else {
                    true
                }
            })
            .cloned()
            .collect()
    }

    fn tool_specs_for(options: &AgentOptions) -> Vec<ToolSpec> {
        let mut specs: Vec<ToolSpec> = Self::active_tools(options)
            .iter()
            .map(|tool| crate::tool::tool_spec(tool.as_ref()))
            .collect();
        if options.mode == Mode::Build {
            specs.push(ToolSpec {
                name: "delegate".to_string(),
                description: "Delegate a self-contained task to a fresh sub-agent with its own context window and tool budget. Returns the sub-agent's final answer. Use for isolated research, review, or implementation subtasks you should not block on.".to_string(),
                parameters: serde_json::from_str(DELEGATE_SPEC).unwrap_or_default(),
            });
        }
        specs
    }

    /// The system prompt for the current mode (byte-stable within a mode).
    fn current_system_prompt(&self) -> String {
        let mut prompt = self.raw_system_prompt();
        if let Some(harness) = &self.options.harness_block
            && !harness.is_empty()
        {
            prompt.push_str(harness);
        }
        prompt.push_str(TRUST_BLOCK);
        prompt
    }

    /// The base system prompt (without the mode block), so the trust
    /// instruction stays a stable suffix.
    fn raw_system_prompt(&self) -> String {
        match self.options.mode {
            Mode::Build => self.options.system_prompt.clone(),
            Mode::Plan => format!("{}{}", self.options.system_prompt, PLAN_MODE_BLOCK),
        }
    }

    /// Switch modes: rebuilds the tool set and system prompt. The provider's
    /// prefix cache takes one miss at the switch, then stays byte-stable.
    pub fn set_mode(&mut self, mode: Mode) {
        self.options.mode = mode;
        self.tool_specs = Self::tool_specs_for(&self.options);
    }

    pub fn mode(&self) -> Mode {
        self.options.mode
    }

    /// Request cancellation of the current stream. Takes effect on the next
    /// stream event; the run ends with `StopReason::Aborted`.
    pub fn interrupt(&self) {
        self.interrupt_handle().interrupt();
    }

    /// A cloneable handle for interrupting runs from other tasks (the TUI's
    /// escape key, tests, timers).
    pub fn interrupt_handle(&self) -> InterruptHandle {
        InterruptHandle {
            interrupted: self.interrupted.clone(),
            interrupt_notify: self.interrupt_notify.clone(),
        }
    }

    /// Switch the model used for subsequent turns (used by `/model`).
    pub fn set_model(&mut self, model: impl Into<String>) {
        self.options.model = model.into();
    }

    pub fn set_variant(&mut self, variant: Option<String>) {
        self.options.variant = variant;
    }

    /// Replace the Continual-Harness prompt block (memories/notes/skills).
    /// The base prompt + trust suffix stay immutable; the harness layer sits
    /// between them, so one prefix-cache miss happens at apply, then the
    /// prompt is byte-stable again.
    pub fn set_harness_block(&mut self, block: String) {
        self.options.harness_block = if block.trim().is_empty() {
            None
        } else {
            Some(block)
        };
    }

    pub fn variant(&self) -> Option<&str> {
        self.options.variant.as_deref()
    }

    pub fn model(&self) -> &str {
        &self.options.model
    }

    pub fn options(&self) -> &AgentOptions {
        &self.options
    }

    pub fn options_mut(&mut self) -> &mut AgentOptions {
        &mut self.options
    }



    pub fn subscribe(&mut self, listener: impl Fn(&AgentEvent) + Send + Sync + 'static) {
        self.listeners.push(Arc::new(listener));
    }

    pub fn messages(&self) -> &[SessionMessage] {
        &self.messages
    }

    /// Restore a previous conversation (session resume). Appended to any
    /// messages already present.
    pub fn restore_messages(&mut self, messages: Vec<SessionMessage>) {
        self.messages.extend(messages);
    }

    pub fn cache_stats(&self) -> &CacheStats {
        &self.cache_stats
    }

    /// Current context usage: (provider-reported prompt tokens, window).
    pub fn context_usage(&self) -> (u64, Option<u64>) {
        (self.last_input_tokens, self.options.context_window)
    }

    /// Compact the conversation: summarize the oldest messages with the
    /// provider and replace them with a [`SessionMessage::Summary`], keeping
    /// the most recent `keep_messages`. Returns the summary text (empty if
    /// there was nothing worth compacting). The summary persists to the
    /// session log, which is the cross-session memory mechanism.
    pub async fn summarize_and_compact(&mut self, keep_messages: usize) -> Result<String, AiError> {
        let keep = keep_messages.max(1);
        let mut split = self.messages.len().saturating_sub(keep);
        // Gap 4: advance split forward past any ToolResult at the boundary to
        // avoid orphaned results (a ToolResult with no preceding Assistant call).
        while split < self.messages.len()
            && matches!(self.messages[split], SessionMessage::ToolResult { .. })
        {
            split += 1;
        }
        if split < 2 {
            return Ok(String::new());
        }
        let older = self.messages.drain(..split).collect::<Vec<_>>();

        // Gap 5: three-stage compaction fallback.
        // Stage 0: normal summarization of the older messages.
        // Stage 1: clamp ToolResult content to 5 K and retry.
        // Stage 2: keep only head+tail (30%/30%) of older and clamp.
        let summary = match self.try_summarize(older.clone()).await {
            Ok(s) => Ok(s),
            Err(_) => {
                let mut clamped = older.clone();
                clamp_tool_results(&mut clamped, 5_000);
                match self.try_summarize(clamped).await {
                    Ok(s) => Ok(s),
                    Err(_) => {
                        let head = older.len() * 3 / 10;
                        let tail = older.len() * 3 / 10;
                        let tail_start = older.len().saturating_sub(tail);
                        let mut reduced: Vec<SessionMessage> = older[..head]
                            .iter()
                            .chain(&older[tail_start..])
                            .cloned()
                            .collect();
                        clamp_tool_results(&mut reduced, 5_000);
                        self.try_summarize(reduced).await
                    }
                }
            }
        };

        match summary {
            Ok(text) if !text.trim().is_empty() => {
                self.messages
                    .insert(0, SessionMessage::Summary { text: text.clone() });
                self.emit(&AgentEvent::Compacted {
                    summary: text.clone(),
                    messages_removed: split,
                });
                Ok(text)
            }
            Ok(_) => {
                self.messages.splice(0..0, older);
                Ok(String::new())
            }
            Err(error) => {
                self.messages.splice(0..0, older);
                Err(error)
            }
        }
    }

    /// Run one summarization attempt against `messages`. Returns the summary
    /// text or the first API error; does NOT touch `self.messages`.
    async fn try_summarize(&self, messages: Vec<SessionMessage>) -> Result<String, AiError> {
        let mut summary_messages: Vec<agent_m_ai::LlmMessage> = messages
            .iter()
            .map(SessionMessage::to_llm_message)
            .collect();
        summary_messages.push(agent_m_ai::LlmMessage::User {
            content: SUMMARY_PROMPT.to_string(),
            images: Vec::new(),
        });
        let request = ChatRequest {
            model: self.options.model.clone(),
            system: "You produce conversation summaries for a coding agent.".to_string(),
            messages: summary_messages,
            tools: vec![],
            temperature: None,
            variant: self.options.variant.clone(),
        };
        let stream = self.provider.stream_chat(request).await?;
        futures_util::pin_mut!(stream);
        let mut summary = String::new();
        let mut failed = false;
        while let Some(event) = stream.next().await {
            match event {
                StreamEvent::TextDelta { delta } => summary.push_str(&delta),
                StreamEvent::Error { .. } => {
                    failed = true;
                    break;
                }
                _ => {}
            }
        }
        if failed {
            return Err(AiError::Api("stream error during compaction".into()));
        }
        Ok(summary)
    }

    fn emit(&self, event: &AgentEvent) {
        for listener in &self.listeners {
            listener(event);
        }
    }

    /// Run one user prompt through the loop: stream the assistant reply,
    /// execute any tool calls, and repeat until the model stops calling tools.
    ///
    /// Boxed future: `run_tool` → `run_delegate` → `prompt` recurses (the
    /// delegate spawns a sub-agent), so the future must be opaque.
    pub fn prompt(
        &mut self,
        text: String,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), AgentError>> + Send + '_>>
    {
        let span = tracing::info_span!(
            "agent.prompt",
            model = %self.options.model,
            mode = %self.options.mode.as_str(),
        );
        Box::pin(async move { self.prompt_inner(text).await }.instrument(span))
    }

    /// Like `prompt`, with image attachments (data URIs) for vision models.
    pub fn prompt_with_images(
        &mut self,
        text: String,
        images: Vec<String>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), AgentError>> + Send + '_>>
    {
        Box::pin(async move {
            self.interrupted.store(false, Ordering::SeqCst);
            self.emit(&AgentEvent::AgentStart);
            let user_message = SessionMessage::User {
                content: text,
                images,
            };
            self.emit(&AgentEvent::MessageStart {
                kind: crate::message::SessionMessageKind::User,
            });
            self.messages.push(user_message.clone());
            self.emit(&AgentEvent::MessageEnd {
                message: user_message.clone(),
            });
            self.run_turns().await
        })
    }

    async fn prompt_inner(&mut self, text: String) -> Result<(), AgentError> {
        self.interrupted.store(false, Ordering::SeqCst);
        self.emit(&AgentEvent::AgentStart);

        let user_message = SessionMessage::User {
            content: text,
            images: Vec::new(),
        };
        self.emit(&AgentEvent::MessageStart {
            kind: crate::message::SessionMessageKind::User,
        });
        self.messages.push(user_message.clone());
        self.emit(&AgentEvent::MessageEnd {
            message: user_message.clone(),
        });

        self.run_turns().await
    }

    /// Drive the turn loop: stream replies, execute tool calls, repeat until
    /// the model stops or the turn budget is exhausted. The user message was
    /// already pushed to `self.messages`.
    async fn run_turns(&mut self) -> Result<(), AgentError> {
        let mut turn = 0;
        loop {
            if turn >= self.options.max_turns {
                self.emit(&AgentEvent::Notice {
                    message: format!(
                        "agent-m has been working on this problem for a while ({} turns). \
                         It can continue to iterate, or you can send a new message to refine your prompt.",
                        self.options.max_turns
                    ),
                });
                let should_continue = if let Some(gate) = &self.options.ask_gate {
                    matches!(
                        gate.ask(
                            "Continue for more turns?".into(),
                            Some(vec!["yes".into(), "no".into()]),
                            false,
                        )
                        .await,
                        Ok(ref ans) if ans.trim().eq_ignore_ascii_case("yes")
                    )
                } else {
                    false
                };
                if should_continue {
                    self.options.max_turns += 20;
                } else {
                    break;
                }
            }
            turn += 1;
            self.emit(&AgentEvent::TurnStart { turn });
            tracing::info!(turn, model = %self.options.model, "turn_start");

            let request = ChatRequest {
                model: self.options.model.clone(),
                system: self.current_system_prompt(),
                messages: self
                    .messages
                    .iter()
                    .map(SessionMessage::to_llm_message)
                    .collect(),
                tools: self.tool_specs.clone(),
                temperature: None,
                variant: self.options.variant.clone(),
            };

            let mut stream = match self.provider.stream_chat(request).await {
                Ok(stream) => stream,
                Err(error) => {
                    self.emit(&AgentEvent::Notice {
                        message: format!("provider error: {error}"),
                    });
                    break;
                }
            };

            let mut text = String::new();
            let mut thinking = String::new();
            let mut tool_calls: Vec<(usize, ToolCallInfo)> = Vec::new();
            let mut usage = None;
            let mut stop_reason = StopReason::Stop;
            let mut stream_error: Option<String> = None;

            self.emit(&AgentEvent::MessageStart {
                kind: crate::message::SessionMessageKind::Assistant,
            });
            loop {
                if self.interrupted.load(Ordering::SeqCst) {
                    stop_reason = StopReason::Aborted;
                    stream_error = Some("interrupted by user".to_string());
                    break;
                }
                tokio::select! {
                    event = stream.next() => {
                        let Some(event) = event else { break };
                        match event {
                    StreamEvent::Start => {}
                    StreamEvent::TextDelta { delta } => {
                        text.push_str(&delta);
                        self.emit(&AgentEvent::MessageUpdate {
                            delta: StreamEvent::TextDelta { delta },
                        });
                    }
                    StreamEvent::ThinkingDelta { delta } => {
                        thinking.push_str(&delta);
                        self.emit(&AgentEvent::MessageUpdate {
                            delta: StreamEvent::ThinkingDelta { delta },
                        });
                    }
                    StreamEvent::ToolCallDelta {
                        index,
                        id,
                        name,
                        arguments,
                    } => {
                        let call = tool_calls
                            .iter_mut()
                            .find(|(existing_index, _)| *existing_index == index);
                        match call {
                            Some((_, existing)) => {
                                if !id.is_empty() {
                                    existing.tool_call_id = id.clone();
                                }
                                if !name.is_empty() {
                                    existing.name = name.clone();
                                }
                                existing.arguments = serde_json::from_str(&arguments)
                                    .unwrap_or(serde_json::Value::String(arguments.clone()));
                            }
                            None => tool_calls.push((
                                index,
                                ToolCallInfo {
                                    tool_call_id: id,
                                    name,
                                    arguments: serde_json::from_str(&arguments)
                                        .unwrap_or(serde_json::Value::String(arguments.clone())),
                                },
                            )),
                        }
                        let (_, current) = tool_calls
                            .iter()
                            .find(|(existing_index, _)| *existing_index == index)
                            .expect("tool call just upserted");
                        self.emit(&AgentEvent::MessageUpdate {
                            delta: StreamEvent::ToolCallDelta {
                                index,
                                id: current.tool_call_id.clone(),
                                name: current.name.clone(),
                                arguments: serde_json::to_string(&current.arguments)
                                    .unwrap_or_default(),
                            },
                        });
                    }
                    StreamEvent::Error { message } => {
                        stream_error = Some(message);
                        stop_reason = StopReason::Error;
                    }
                    StreamEvent::Done { message } => {
                        usage = message.usage.clone();
                        stop_reason = message.stop_reason;
                        if let Some(error) = message.error_message {
                            stream_error = Some(error);
                            stop_reason = StopReason::Error;
                        }
                    }
                        }
                    }
                    _ = self.interrupt_notify.notified() => {
                        if self.interrupted.load(Ordering::SeqCst) {
                            stop_reason = StopReason::Aborted;
                            stream_error = Some("interrupted by user".to_string());
                            break;
                        }
                    }
                }
            }

            let mut final_parts: Vec<ContentPart> = Vec::new();
            if !text.is_empty() {
                final_parts.push(ContentPart::Text { text });
            }
            if !thinking.is_empty() {
                final_parts.push(ContentPart::Thinking { thinking });
            }
            tool_calls.sort_by_key(|(index, _)| *index);
            for (_, call) in &tool_calls {
                final_parts.push(ContentPart::ToolCall {
                    id: call.tool_call_id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });
            }

            // Best-effort trust metadata: the model ends the reply with a
            // <trust> block (last text part); strip it and parse it.
            let (trust, final_parts) = extract_trust(&final_parts);
            let assistant_message = SessionMessage::Assistant {
                content: final_parts,
                usage: usage.clone(),
                stop_reason,
                model: self.options.model.clone(),
                trust,
            };
            if let Some(usage) = &usage {
                self.cache_stats.record(usage);
                self.last_input_tokens = usage.input_tokens;
                tracing::info!(
                    turn,
                    input_tokens = usage.input_tokens,
                    output_tokens = usage.output_tokens,
                    cache_hit_tokens = usage.cache_read_tokens,
                    "turn_end"
                );
            }

            if stop_reason == StopReason::Aborted {
                // A half-streamed reply must not pollute the conversation
                // history (its partial tool-call JSON would break the
                // byte-stable prefix) and its tool calls must not run.
                self.emit(&AgentEvent::MessageEnd {
                    message: assistant_message.clone(),
                });
                self.emit(&AgentEvent::TurnEnd {
                    message: assistant_message,
                    tool_results: 0,
                });
                break;
            }

            self.messages.push(assistant_message.clone());
            self.emit(&AgentEvent::MessageEnd {
                message: assistant_message.clone(),
            });

            if let Some(error) = stream_error {
                self.emit(&AgentEvent::Notice {
                    message: format!("model error: {error}"),
                });
            }

            let has_tool_calls = !tool_calls.is_empty();
            if has_tool_calls {
                let mut tool_results = 0;
                for (_, call) in tool_calls {
                    self.emit(&AgentEvent::ToolExecutionStart {
                        tool_call_id: call.tool_call_id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    });
                    let outcome = self
                        .run_tool(&call)
                        .instrument(tracing::info_span!(
                            "agent.tool",
                            tool.name = %call.name,
                            tool.call_id = %call.tool_call_id,
                        ))
                        .await;
                    tool_results += 1;
                    self.emit(&AgentEvent::ToolExecutionEnd {
                        tool_call_id: call.tool_call_id.clone(),
                        outcome: outcome.clone(),
                    });
                    self.messages.push(SessionMessage::ToolResult {
                        tool_call_id: call.tool_call_id.clone(),
                        name: call.name.clone(),
                        content: outcome.content,
                        is_error: outcome.is_error,
                    });
                }
                self.emit(&AgentEvent::TurnEnd {
                    message: assistant_message,
                    tool_results,
                });
                continue;
            }

            self.emit(&AgentEvent::TurnEnd {
                message: assistant_message,
                tool_results: 0,
            });
            break;
        }

        let messages = self.messages.len();
        let cache_stats = self.cache_stats.clone();
        self.emit(&AgentEvent::AgentEnd {
            messages,
            cache_stats,
        });
        Ok(())
    }

    /// Spawn a fresh sub-agent for a delegated task: same provider and gate,
    /// a fresh context window, an optional tool subset, and a turn budget.
    /// Returns the sub-agent's final answer as the tool outcome.
    async fn run_delegate(&self, arguments: &Value) -> ToolOutcome {
        let Some(prompt) = arguments.get("prompt").and_then(Value::as_str) else {
            return ToolOutcome::error("delegate: missing `prompt` argument");
        };
        let max_turns = arguments
            .get("max_turns")
            .and_then(Value::as_u64)
            .unwrap_or(4)
            .clamp(1, 16) as usize;
        // Tool subset: `tools` restricts; otherwise the parent's set minus
        // delegate (no recursion).
        let tools: Vec<Arc<dyn Tool>> = match arguments.get("tools").and_then(Value::as_array) {
            Some(names) => self
                .options
                .tools
                .iter()
                .filter(|tool| {
                    tool.name() != "delegate"
                        && names.iter().any(|name| name.as_str() == Some(tool.name()))
                })
                .cloned()
                .collect(),
            None => self
                .options
                .tools
                .iter()
                .filter(|tool| tool.name() != "delegate")
                .cloned()
                .collect(),
        };
        let sub_options = AgentOptions {
            tools,
            max_turns,
            ..self.options.clone()
        };
        // Run the sub-agent on its own task: a fresh future root, so the
        // async recursion (prompt → run_tool → delegate → prompt) never
        // nests in the parent's stack.
        let provider = self.provider.clone();
        let gate = self.gate.clone();
        let sub_prompt = prompt.to_string();
        let sub = tokio::task::spawn(async move {
            let mut sub = Agent::new(provider, sub_options, gate);
            let _ = sub.prompt(sub_prompt).await;
            sub
        })
        .await;
        let sub = match sub {
            Ok(sub) => sub,
            Err(_) => return ToolOutcome::error("delegate: sub-agent task failed"),
        };

        // The sub-agent's final answer = its last assistant text.
        let mut answer = String::new();
        for message in sub.messages() {
            if let SessionMessage::Assistant { content, .. } = message {
                for part in content {
                    if let ContentPart::Text { text } = part {
                        answer.push_str(text);
                        answer.push('\n');
                    }
                }
            }
        }
        let answer = answer.trim().to_string();
        if answer.is_empty() {
            return ToolOutcome::error("delegate: sub-agent produced no answer");
        }
        ToolOutcome {
            content: answer,
            is_error: false,
        }
    }

    async fn run_tool(&self, call: &ToolCallInfo) -> ToolOutcome {
        match self.gate.authorize(call).await {
            Permission::Denied(reason) => ToolOutcome::error(format!(
                "Permission denied for tool `{}`: {reason}",
                call.name
            )),
            Permission::Allowed if call.name == "delegate" => {
                self.run_delegate(&call.arguments).await
            }
            Permission::Allowed => {
                let gated = self.options.mode == Mode::Plan
                    && !PLAN_TOOLS.contains(&call.name.as_str())
                    && self
                        .options
                        .tools
                        .iter()
                        .any(|tool| tool.name() == call.name);
                let Some(tool) = self.options.tools.iter().find(|tool| {
                    tool.name() == call.name
                        && (self.options.mode == Mode::Build || PLAN_TOOLS.contains(&tool.name()))
                }) else {
                    return ToolOutcome::error(if gated {
                        format!("`{}` is not available in plan mode", call.name)
                    } else {
                        format!("Unknown tool `{}`", call.name)
                    });
                };
                let context = ToolContext {
                    cwd: self.options.cwd.clone(),
                    ask_gate: self.options.ask_gate.clone(),
                    output_dir: self.options.output_dir.clone(),
                    read_cache: self.read_cache.clone(),
                };
                match tool.execute(call.arguments.clone(), &context).await {
                    Ok(outcome) => outcome,
                    Err(error) => ToolOutcome::error(error.to_string()),
                }
            }
        }
    }
}

#[cfg(test)]
mod clamp_tests {
    use super::*;

    fn tool_result(content: &str) -> SessionMessage {
        SessionMessage::ToolResult {
            tool_call_id: "call_1".to_string(),
            name: "bash".to_string(),
            content: content.to_string(),
            is_error: false,
        }
    }

    #[test]
    fn clamp_tool_results_truncates_on_char_boundary() {
        // 5000-byte boundary lands inside a 3-byte UTF-8 char (é = 0xC3 0xA9):
        // a byte-truncate would panic; the char-boundary walk must not.
        let mut content = "é".repeat(2500); // 5000 bytes exactly
        content.push_str("trailing");
        let mut messages = vec![tool_result(&content)];
        clamp_tool_results(&mut messages, 5000);
        let clamped = match &messages[0] {
            SessionMessage::ToolResult { content, .. } => content,
            _ => unreachable!(),
        };
        assert!(clamped.is_char_boundary(5000) || clamped.len() <= 5000);
        assert!(clamped.ends_with("[clamped for compaction]"));
        assert!(clamped.starts_with("éé"));
    }

    #[test]
    fn clamp_tool_results_leaves_short_output_untouched() {
        let mut messages = vec![tool_result("short")];
        clamp_tool_results(&mut messages, 5000);
        match &messages[0] {
            SessionMessage::ToolResult { content, .. } => assert_eq!(content, "short"),
            _ => unreachable!(),
        }
    }
}
