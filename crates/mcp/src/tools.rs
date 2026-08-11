//! Adapt MCP server tools to the agent `Tool` trait.

use std::sync::Arc;

use agent_m_agent::{Tool, ToolCallInfo, ToolContext, ToolError, ToolOutcome, tool_spec};
use serde_json::Value;

use crate::client::{McpClient, McpToolDef};

/// An MCP server's tool, callable through the agent tool loop. The client is
/// shared (Mutex) because `Tool::execute` takes `&self` while the underlying
/// MCP client is sequential per request id.
#[derive(Clone)]
pub struct McpTool {
    /// Qualified name `server__tool`, so two servers can expose the same tool.
    pub name: String,
    pub description: String,
    pub parameters: Value,
    client: Arc<tokio::sync::Mutex<McpClient>>,
}

impl McpTool {
    /// Build a `Tool`-trait adapter for one tool advertised by an MCP server.
    pub fn new(server: &str, client: Arc<tokio::sync::Mutex<McpClient>>, def: McpToolDef) -> Self {
        McpTool {
            name: format!("{server}__{}", def.name),
            description: format!(
                "MCP tool from server `{server}`: {}",
                def.description.trim()
            ),
            parameters: def.input_schema,
            client,
        }
    }

    /// Human label without the server prefix (for the tool registry).
    pub fn bare_name(&self) -> &str {
        self.name.split("__").last().unwrap_or(&self.name)
    }
}

#[async_trait::async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    async fn execute(
        &self,
        arguments: Value,
        _context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let tool_name = self.bare_name().to_string();
        // tokio's Mutex::lock returns the guard directly (no poisoning).
        let mut client = self.client.lock().await;
        match client.call_tool(&tool_name, arguments).await {
            Ok(content) => Ok(ToolOutcome {
                content,
                is_error: false,
            }),
            Err(error) => Ok(ToolOutcome {
                content: format!("MCP tool failed: {error}"),
                is_error: true,
            }),
        }
    }
}

/// Build `Tool` adapters for every tool of a connected server. Returns the
/// tools and a handle that keeps the shared client alive.
pub async fn connect_tools(
    server: &str,
    mut client: McpClient,
) -> Result<(Vec<Arc<dyn Tool>>, Arc<tokio::sync::Mutex<McpClient>>), String> {
    let defs = client.list_tools().await.map_err(|e| e.to_string())?;
    let shared = Arc::new(tokio::sync::Mutex::new(client));
    let tools = defs
        .into_iter()
        .map(|def| Arc::new(McpTool::new(server, shared.clone(), def)) as Arc<dyn Tool>)
        .collect::<Vec<_>>();
    Ok((tools, shared))
}

/// Convenience: the agent-side `tool_spec` for an MCP tool (used by tests and
/// the registry to surface specs without a live server).
pub fn mcp_tool_spec(tool: &McpTool) -> agent_m_ai::ToolSpec {
    let arc: Arc<dyn Tool> = Arc::new(tool.clone());
    tool_spec(arc.as_ref())
}

/// Test/diagnostic helper: summarize a tool call info for approval messages.
pub fn describe_call(call: &ToolCallInfo) -> String {
    format!("{} {}", call.name, call.arguments)
}

impl From<McpTool> for Arc<dyn Tool> {
    fn from(tool: McpTool) -> Self {
        Arc::new(tool)
    }
}
