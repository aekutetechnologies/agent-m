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
mod checkpoint;
mod context;
mod message;
mod risk;
mod tool;
mod trust_policy;
mod worktree;

pub use agent::{
    Agent, AgentEvent, AgentOptions, DelegateOutcome, DelegateRequest, InterruptHandle, Mode,
    run_delegate_sub,
};
pub use checkpoint::{create_checkpoint, is_git_repo, restore_checkpoint};
pub use context::{InstructionFile, discover_instructions, render_instructions};
pub use message::{SessionMessage, SessionMessageKind};
pub use risk::{RiskAssessment, RiskLevel, RiskPolicy};
pub use tool::{
    AlwaysAllowGate, AskGate, AutonomyLevel, BoolGate, ClosureAskGate, ClosureGate,
    DangerousCommandGate, DenyAllGate, LevelGate, Permission, PermissionGate,
    ReadOnlyAutoApproveGate, SelectiveAskGate, TierGate, Tool, ToolCallInfo, ToolContext,
    ToolError, ToolOutcome, ToolRegistry, UnavailableAskGate, tool_spec,
};
pub use trust_policy::{TrustDecision, TrustIssues, TrustMode, TrustPolicy, assess, check_evidence, enforce};
pub use worktree::{create_worktree, list_worktrees, remove_worktree};
