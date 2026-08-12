use agent_m_agent::{Permission, PermissionGate, RiskPolicy, ToolCallInfo};
use async_trait::async_trait;
use rustyline::DefaultEditor;
use std::sync::Arc;

pub struct CliPromptGate {
    policy: Arc<RiskPolicy>,
    yes: bool,
}

impl CliPromptGate {
    pub fn new(policy: Arc<RiskPolicy>, yes: bool) -> Self {
        Self { policy, yes }
    }
}

#[async_trait]
impl PermissionGate for CliPromptGate {
    async fn authorize(&self, tool_call: &ToolCallInfo) -> Permission {
        if self.yes {
            return Permission::Allowed;
        }

        let risk = self.policy.risk(tool_call);
        if risk.is_none() {
            return Permission::Allowed;
        }

        let consequence = self.policy.consequence(tool_call);

        let args_str = serde_json::to_string(&tool_call.arguments).unwrap_or_default();
        let prompt_text = format!(
            "\n⚠️  [Security Gate] Tool Execution Requested:\n    Tool: {}\n    Args: {}\n    Risk Level: {:?}\n    Consequence: {}\n\n[y] Approve  [n] Deny > ",
            tool_call.name,
            args_str,
            risk.as_deref().unwrap_or("Low"),
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
    }
}
