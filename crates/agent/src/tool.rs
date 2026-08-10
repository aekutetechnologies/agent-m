//! The tool interface, tool outcomes, and the permission gate.

use async_trait::async_trait;
use serde_json::Value;
use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

/// Asks the user a question mid-run (the `ask` tool). The UI implements the
/// gate; a `None` gate makes `ask` fail with a clear message.
pub trait AskGate: Send + Sync {
    fn ask(
        &self,
        question: String,
        options: Option<Vec<String>>,
    ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>>;
}

/// An ask gate backed by a closure (used by the TUI to route to a dialog).
pub struct ClosureAskGate<F>(F)
where
    F: Fn(
            String,
            Option<Vec<String>>,
        ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync;

impl<F> ClosureAskGate<F>
where
    F: Fn(
            String,
            Option<Vec<String>>,
        ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync,
{
    pub fn new(f: F) -> Self {
        Self(f)
    }
}

impl<F> AskGate for ClosureAskGate<F>
where
    F: Fn(
            String,
            Option<Vec<String>>,
        ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>>
        + Send
        + Sync,
{
    fn ask(
        &self,
        question: String,
        options: Option<Vec<String>>,
    ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> {
        (self.0)(question, options)
    }
}

/// The default gate: the ask tool cannot work without a UI.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnavailableAskGate;

impl AskGate for UnavailableAskGate {
    fn ask(
        &self,
        _question: String,
        _options: Option<Vec<String>>,
    ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> {
        Box::pin(async {
            Err(
                "the ask tool requires the interactive UI (not available in print mode)"
                    .to_string(),
            )
        })
    }
}

/// Context passed to a tool execution.
#[derive(Clone)]
pub struct ToolContext {
    /// Working directory the tool operates in.
    pub cwd: PathBuf,
    /// The ask gate for the `ask` tool (None → ask fails).
    pub ask_gate: Option<Arc<dyn AskGate>>,
}

impl fmt::Debug for ToolContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ToolContext")
            .field("cwd", &self.cwd)
            .finish_non_exhaustive()
    }
}

/// A tool invocation the model requested.
#[derive(Debug, Clone)]
pub struct ToolCallInfo {
    pub tool_call_id: String,
    pub name: String,
    pub arguments: Value,
}

/// The result of a tool execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutcome {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutcome {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

/// A tool execution failure (as opposed to a tool-reported error result).
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("tool `{0}` is not registered")]
    NotFound(String),

    #[error("tool `{name}` failed: {message}")]
    Failed { name: String, message: String },
}

impl ToolError {
    pub fn failed(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Failed {
            name: name.into(),
            message: message.into(),
        }
    }
}

/// A built-in or extension tool the model may call.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Stable tool name, e.g. `read`.
    fn name(&self) -> &str;

    /// Human-readable description for the model.
    fn description(&self) -> String;

    /// JSON Schema describing `arguments`.
    fn parameters(&self) -> Value;

    /// Execute the tool with parsed arguments.
    async fn execute(
        &self,
        arguments: Value,
        context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError>;
}

/// A tool's wire spec (deterministic — see the ai crate's serializer).
pub fn tool_spec(tool: &dyn Tool) -> agent_m_ai::ToolSpec {
    agent_m_ai::ToolSpec {
        name: tool.name().to_string(),
        description: tool.description(),
        parameters: tool.parameters(),
    }
}

/// The outcome of a permission check for one tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Permission {
    Allowed,
    Denied(String),
}

/// Decides whether a tool call may run. The TUI always wraps
/// SelectiveAskGate in ReadOnlyAutoApproveGate, `--yes` or not: read-only
/// tools never prompt, risky calls always do (ECC GateGuard), everything
/// else — including a benign shell command like `ls`/`cat` run via `bash`
/// — auto-approves, because a human is present either way. Print mode and
/// flows have no human to ask, so `--yes` still matters there: without it
/// they deny all (including reads); with it they use DangerousCommandGate
/// (denies risky, allows benign). `--no-tools` is the strongest boundary:
/// no tools registered, nothing to authorize.
#[async_trait]
pub trait PermissionGate: Send + Sync {
    async fn authorize(&self, tool_call: &ToolCallInfo) -> Permission;
}

/// A permission gate that asks the user (via the wrapped closure) ONLY for
/// risky commands and auto-approves everything else (`--yes` in the TUI).
pub struct SelectiveAskGate<F>
where
    F: Fn(ToolCallInfo) -> Pin<Box<dyn Future<Output = Permission> + Send>> + Send + Sync,
{
    policy: Arc<crate::risk::RiskPolicy>,
    ask: F,
}

impl<F> SelectiveAskGate<F>
where
    F: Fn(ToolCallInfo) -> Pin<Box<dyn Future<Output = Permission> + Send>> + Send + Sync,
{
    pub fn new(policy: Arc<crate::risk::RiskPolicy>, ask: F) -> Self {
        Self { policy, ask }
    }
}

#[async_trait]
impl<F> PermissionGate for SelectiveAskGate<F>
where
    F: Fn(ToolCallInfo) -> Pin<Box<dyn Future<Output = Permission> + Send>> + Send + Sync,
{
    async fn authorize(&self, tool_call: &ToolCallInfo) -> Permission {
        if self.policy.risk(tool_call).is_some() {
            (self.ask)(tool_call.clone()).await
        } else {
            Permission::Allowed
        }
    }
}

/// Wraps another gate and denies risky commands outright even when the
/// inner gate would auto-approve them (print mode: no UI to ask).
pub struct DangerousCommandGate<G> {
    policy: Arc<crate::risk::RiskPolicy>,
    inner: G,
}

impl<G> DangerousCommandGate<G> {
    pub fn new(policy: Arc<crate::risk::RiskPolicy>, inner: G) -> Self {
        Self { policy, inner }
    }
}

#[async_trait]
impl<G: PermissionGate + Send + Sync> PermissionGate for DangerousCommandGate<G> {
    async fn authorize(&self, tool_call: &ToolCallInfo) -> Permission {
        if let Some(reason) = self.policy.risk(tool_call) {
            match self.inner.authorize(tool_call).await {
                Permission::Allowed => Permission::Denied(format!(
                    "risky command requires explicit approval: {reason}"
                )),
                other => other,
            }
        } else {
            self.inner.authorize(tool_call).await
        }
    }
}

/// Wraps another gate so read-only tools (no side effects: `read`, `grep`,
/// `find`, `ls`, `ask`, `search` — the same set plan mode allows) never need
/// approval; everything else defers to `inner`. Used by the interactive TUI
/// so browsing a codebase doesn't mean a y/n prompt per file. Not applied to
/// print mode / flows: their "disabled unless --yes" default has no UI to
/// bypass anything through, so it stays as-is.
pub struct ReadOnlyAutoApproveGate<G> {
    inner: G,
}

impl<G> ReadOnlyAutoApproveGate<G> {
    pub fn new(inner: G) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<G: PermissionGate + Send + Sync> PermissionGate for ReadOnlyAutoApproveGate<G> {
    async fn authorize(&self, tool_call: &ToolCallInfo) -> Permission {
        if crate::agent::PLAN_TOOLS.contains(&tool_call.name.as_str()) {
            Permission::Allowed
        } else {
            self.inner.authorize(tool_call).await
        }
    }
}

/// Auto-approve every tool call (`--yes`).
pub struct AlwaysAllowGate;

#[async_trait]
impl PermissionGate for AlwaysAllowGate {
    async fn authorize(&self, _tool_call: &ToolCallInfo) -> Permission {
        Permission::Allowed
    }
}

/// Refuses everything, with a message the model can act on. The gate to use
/// when there is no human available to approve (print mode without `--yes`).
pub struct DenyAllGate;

#[async_trait]
impl PermissionGate for DenyAllGate {
    async fn authorize(&self, _tool_call: &ToolCallInfo) -> Permission {
        Permission::Denied(
            "tools are disabled here (no interactive approval available); \
             re-run with --yes to enable them"
                .to_string(),
        )
    }
}

/// A gate backed by an arbitrary async closure, letting the UI route the
/// decision through its own event loop.
pub struct ClosureGate<F> {
    authorize_fn: F,
}

impl<F> ClosureGate<F>
where
    F: Fn(&ToolCallInfo) -> Pin<Box<dyn Future<Output = Permission> + Send>> + Send + Sync,
{
    pub fn new(authorize_fn: F) -> Self {
        Self { authorize_fn }
    }
}

#[async_trait]
impl<F> PermissionGate for ClosureGate<F>
where
    F: Fn(&ToolCallInfo) -> Pin<Box<dyn Future<Output = Permission> + Send>> + Send + Sync,
{
    async fn authorize(&self, tool_call: &ToolCallInfo) -> Permission {
        (self.authorize_fn)(tool_call).await
    }
}

/// Convenience for tests and simple callers: gate on a `bool` closure.
pub struct BoolGate<F> {
    allow_fn: F,
}

impl<F> BoolGate<F>
where
    F: Fn(&ToolCallInfo) -> bool + Send + Sync,
{
    pub fn new(allow_fn: F) -> Self {
        Self { allow_fn }
    }
}

#[async_trait]
impl<F> PermissionGate for BoolGate<F>
where
    F: Fn(&ToolCallInfo) -> bool + Send + Sync,
{
    async fn authorize(&self, tool_call: &ToolCallInfo) -> Permission {
        if (self.allow_fn)(tool_call) {
            Permission::Allowed
        } else {
            Permission::Denied("blocked by permission gate".to_string())
        }
    }
}

/// A registry of tools keyed by name, with allow/deny filtering.
#[derive(Default)]
pub struct ToolRegistry {
    tools: std::collections::BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.values().cloned().collect()
    }

    /// Filter to the allowed set. When `allow` is empty, everything not
    /// excluded is kept.
    pub fn filter(&self, allow: &[String], exclude: &[String]) -> ToolRegistry {
        let mut registry = ToolRegistry::default();
        for (name, tool) in &self.tools {
            if exclude.iter().any(|excluded| excluded == name) {
                continue;
            }
            if !allow.is_empty() && !allow.iter().any(|allowed| allowed == name) {
                continue;
            }
            registry.register(tool.clone());
        }
        registry
    }
}

impl fmt::Debug for ToolRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolRegistry")
            .field("tools", &self.names())
            .finish()
    }
}
