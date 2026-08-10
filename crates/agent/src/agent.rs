//! The agent loop: user message in, streamed assistant replies and tool calls
//! out, with pi-compatible event ordering.

use agent_m_ai::{
    AiError, CacheStats, ChatRequest, ContentPart, Provider, StopReason, StreamEvent, ToolSpec,
};
use futures_util::StreamExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
const PLAN_MODE_BLOCK: &str = "\n\nYou are in PLAN MODE. You may only read, search, and ask questions — you cannot modify files or run state-changing commands. Available tools: `ls` (list a directory — use this instead of reading a directory), `read` (read one file), `grep` (search file contents), `find` (locate files), `ask` (ask the user a question). There is no `bash` tool in plan mode. Explore the codebase, then create a detailed numbered plan under a heading `Plan:`, one item per line (`1. step`). Each step must be a concrete, verifiable action. Do not execute the plan yet.";

/// The tools a plan-mode agent may call (read-only + ask).
const PLAN_TOOLS: &[&str] = &["read", "grep", "find", "ls", "ask", "search"];

/// Instructions for the compaction summarizer (memory across sessions).
const SUMMARY_PROMPT: &str = "Summarize the conversation above for continuation by a coding agent. Keep it concise but complete: the goal, key decisions, files touched, important tool results, user preferences, and open questions. 300 words or fewer.";

/// Configuration for one agent run.
#[derive(Clone)]
pub struct AgentOptions {
    /// Model id sent to the provider, e.g. `deepseek-chat`.
    pub model: String,
    /// Fixed system prompt. Assembled once and byte-stable for the session.
    pub system_prompt: String,
    /// The tools the model may call (already allow/deny filtered).
    pub tools: Vec<Arc<dyn Tool>>,
    /// Gate consulted before each tool call.
    pub permission_gate: Arc<dyn PermissionGate>,
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
    /// Byte-stable tool specs, computed once at construction.
    tool_specs: Vec<ToolSpec>,
    messages: Vec<SessionMessage>,
    cache_stats: CacheStats,
    listeners: Vec<AgentListener>,
    interrupted: Arc<AtomicBool>,
    interrupt_notify: Arc<tokio::sync::Notify>,
    /// Provider-reported prompt tokens of the most recent turn (≈ context size).
    last_input_tokens: u64,
}

impl Agent {
    pub fn new(provider: Arc<dyn Provider>, options: AgentOptions) -> Self {
        let tool_specs = Self::tool_specs_for(&options);
        Self {
            provider,
            options,
            tool_specs,
            messages: Vec::new(),
            cache_stats: CacheStats::default(),
            listeners: Vec::new(),
            interrupted: Arc::new(AtomicBool::new(false)),
            interrupt_notify: Arc::new(tokio::sync::Notify::new()),
            last_input_tokens: 0,
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
        Self::active_tools(options)
            .iter()
            .map(|tool| crate::tool::tool_spec(tool.as_ref()))
            .collect()
    }

    /// The system prompt for the current mode (byte-stable within a mode).
    fn current_system_prompt(&self) -> String {
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

    pub fn model(&self) -> &str {
        &self.options.model
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
        let split = self.messages.len().saturating_sub(keep);
        if split < 2 {
            return Ok(String::new());
        }
        let older = self.messages.drain(..split).collect::<Vec<_>>();
        let mut summary_messages: Vec<agent_m_ai::LlmMessage> =
            older.iter().map(SessionMessage::to_llm_message).collect();
        summary_messages.push(agent_m_ai::LlmMessage::User {
            content: SUMMARY_PROMPT.to_string(),
        });
        let request = ChatRequest {
            model: self.options.model.clone(),
            system: "You produce conversation summaries for a coding agent.".to_string(),
            messages: summary_messages,
            tools: vec![],
            temperature: None,
        };
        let stream = match self.provider.stream_chat(request).await {
            Ok(stream) => stream,
            Err(error) => {
                // Restore the removed messages; compaction failed safely.
                self.messages.splice(0..0, older);
                return Err(error);
            }
        };
        futures_util::pin_mut!(stream);
        let mut summary = String::new();
        let mut failed = false;
        while let Some(event) = stream.next().await {
            match event {
                StreamEvent::TextDelta { delta } => summary.push_str(&delta),
                // A mid-stream error leaves a truncated summary; restore the
                // removed messages and report failure.
                StreamEvent::Error { .. } => {
                    failed = true;
                    break;
                }
                _ => {}
            }
        }
        if failed || summary.trim().is_empty() {
            self.messages.splice(0..0, older);
            return Ok(String::new());
        }
        self.messages.insert(
            0,
            SessionMessage::Summary {
                text: summary.clone(),
            },
        );
        self.emit(&AgentEvent::Compacted {
            summary: summary.clone(),
            messages_removed: split,
        });
        Ok(summary)
    }

    fn emit(&self, event: &AgentEvent) {
        for listener in &self.listeners {
            listener(event);
        }
    }

    /// Run one user prompt through the loop: stream the assistant reply,
    /// execute any tool calls, and repeat until the model stops calling tools.
    pub async fn prompt(&mut self, text: String) -> Result<(), AgentError> {
        self.interrupted.store(false, Ordering::SeqCst);
        self.emit(&AgentEvent::AgentStart);

        let user_message = SessionMessage::User { content: text };
        self.emit(&AgentEvent::MessageStart {
            kind: crate::message::SessionMessageKind::User,
        });
        self.messages.push(user_message.clone());
        self.emit(&AgentEvent::MessageEnd {
            message: user_message,
        });

        let mut turn = 0;
        loop {
            if turn >= self.options.max_turns {
                self.emit(&AgentEvent::Notice {
                    message: format!(
                        "max turns ({}) reached; ending the run",
                        self.options.max_turns
                    ),
                });
                break;
            }
            turn += 1;
            self.emit(&AgentEvent::TurnStart { turn });

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

            let assistant_message = SessionMessage::Assistant {
                content: final_parts,
                usage: usage.clone(),
                stop_reason,
                model: self.options.model.clone(),
            };
            if let Some(usage) = &usage {
                self.cache_stats.record(usage);
                self.last_input_tokens = usage.input_tokens;
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
                    let outcome = self.run_tool(&call).await;
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

    async fn run_tool(&self, call: &ToolCallInfo) -> ToolOutcome {
        match self.options.permission_gate.authorize(call).await {
            Permission::Denied(reason) => ToolOutcome::error(format!(
                "Permission denied for tool `{}`: {reason}",
                call.name
            )),
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
                };
                match tool.execute(call.arguments.clone(), &context).await {
                    Ok(outcome) => outcome,
                    Err(error) => ToolOutcome::error(error.to_string()),
                }
            }
        }
    }
}
