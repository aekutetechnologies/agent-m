//! The `write` tool: write a file, creating parent directories.

use agent_m_agent::{Tool, ToolContext, ToolError, ToolOutcome};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::path::Path;

/// Writes (overwrites) a file, creating parent directories as needed.
pub struct WriteTool;

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> String {
        "Write content to a file, creating parent directories if needed. Overwrites existing files. Returns the number of bytes written.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file, relative to the session cwd or absolute" },
                "content": { "type": "string", "description": "The full content to write" }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(
        &self,
        arguments: Value,
        context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let path = arguments
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::failed("write", "missing string argument `path`"))?;
        let content = arguments
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::failed("write", "missing string argument `content`"))?;

        let path = crate::resolve_path(&context.cwd, path);
        if let Some(parent) = Path::new(&path).parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|error| {
                ToolError::failed(
                    "write",
                    format!("cannot create {}: {error}", parent.display()),
                )
            })?;
        }
        tokio::fs::write(&path, content.as_bytes())
            .await
            .map_err(|error| {
                ToolError::failed("write", format!("cannot write {}: {error}", path.display()))
            })?;
        Ok(ToolOutcome::success(format!(
            "Successfully wrote {} bytes to {}",
            content.len(),
            path.display()
        )))
    }
}
