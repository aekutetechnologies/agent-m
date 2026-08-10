//! agent-m-agent: agent loop, session messages, tool interface, and events.
//!
//! The loop mirrors pi's `AgentSession` event ordering (`agent_start` →
//! `turn_start` → `message_start/update/end` → `tool_execution_*` →
//! `turn_end` → … → `agent_end`) and its `StreamFn` contract: provider
//! failures are encoded as `StopReason::Error`/`Aborted` messages, never
//! thrown. The context builder keeps the request prefix byte-stable across
//! turns (fixed system prompt + tool schemas, append-only messages), which is
//! what makes provider prefix caching effective.

mod agent;
mod context;
mod message;
mod risk;
mod tool;

pub use agent::{Agent, AgentEvent, AgentOptions, InterruptHandle, Mode};
pub use context::{InstructionFile, discover_instructions, render_instructions};
pub use message::{SessionMessage, SessionMessageKind};
pub use risk::RiskPolicy;
pub use tool::{
    AlwaysAllowGate, AskGate, BoolGate, ClosureAskGate, ClosureGate, DangerousCommandGate,
    DenyAllGate, Permission, PermissionGate, ReadOnlyAutoApproveGate, SelectiveAskGate, Tool,
    ToolCallInfo, ToolContext, ToolError, ToolOutcome, ToolRegistry, UnavailableAskGate, tool_spec,
};
