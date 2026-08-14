use agent_m_agent::{Permission, RiskPolicy, ToolCallInfo};
use rustyline::DefaultEditor;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Prompt a human to approve a High/Critical tool call (the interactive
/// gate). Used as the ask closure for `LevelGate`/`TierGate`, which always
/// route High/Critical here — even under `--yes` (check.md principle 6).
/// Headless modes (daemon) have no TTY: the readline fails and the call is
/// denied, which is the safe default.
pub fn ask_human(
    policy: Arc<RiskPolicy>,
    tool_call: ToolCallInfo,
) -> Pin<Box<dyn Future<Output = Permission> + Send>> {
    Box::pin(async move {
        let risk = policy.risk(&tool_call);
        let consequence = policy.consequence(&tool_call);

        let args_str = serde_json::to_string(&tool_call.arguments).unwrap_or_default();
        let prompt_text = format!(
            "\n⚠️  [Security Gate] Tool Execution Requested:\n    Tool: {}\n    Args: {}\n    Risk Level: {}\n    Consequence: {}\n\n[y] Approve  [n] Deny > ",
            tool_call.name,
            args_str,
            risk.as_deref().unwrap_or("High"),
            consequence.unwrap_or_default()
        );

        let response = tokio::task::block_in_place(|| {
            let mut rl = DefaultEditor::new().ok()?;
            rl.readline(&prompt_text).ok()
        });

        match response.as_deref().map(str::trim).map(str::to_lowercase).as_deref() {
            Some("y") | Some("yes") => Permission::Allowed,
            _ => Permission::Denied("User denied execution via security gate.".to_string()),
        }
    })
}
