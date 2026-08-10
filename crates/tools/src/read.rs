//! The `read` tool: read a file with offset/limit and truncation notices.

use agent_m_agent::{Tool, ToolContext, ToolError, ToolOutcome};
use async_trait::async_trait;
use serde_json::{Value, json};

/// Reads a text file, optionally from an offset line with a line limit.
pub struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> String {
        "Read a text file. Supports `offset` (0-based line) and `limit` (max lines). Output is truncated to 2000 lines / 50 KB with a continuation notice.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file, relative to the session cwd or absolute" },
                "offset": { "type": "number", "description": "0-based starting line" },
                "limit": { "type": "number", "description": "Maximum number of lines to return" }
            },
            "required": ["path"]
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
            .ok_or_else(|| ToolError::failed("read", "missing string argument `path`"))?;
        let offset = arguments.get("offset").and_then(Value::as_u64).unwrap_or(0) as usize;
        let limit = arguments
            .get("limit")
            .and_then(Value::as_u64)
            .map(|v| v as usize);

        let path = crate::resolve_path("read", &context.cwd, path)?;
        // Bound memory and time: reject huge or non-regular files up front and
        // cap the read duration (security review MEDIUM).
        const MAX_READ_BYTES: u64 = 10 * 1024 * 1024;
        const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
        let metadata = tokio::fs::metadata(&path).await.map_err(|error| {
            ToolError::failed("read", format!("cannot stat {}: {error}", path.display()))
        })?;
        if !metadata.is_file() {
            return Ok(ToolOutcome::error(format!(
                "{} is a directory, not a file — use the `ls` tool to list it, or pass a file path to `read`",
                path.display()
            )));
        }
        if metadata.len() > MAX_READ_BYTES {
            return Ok(ToolOutcome::error(format!(
                "file {} is {} MB; the read tool is limited to 10 MB (use grep or bash to inspect it)",
                path.display(),
                metadata.len() / (1024 * 1024)
            )));
        }
        let contents = tokio::time::timeout(READ_TIMEOUT, tokio::fs::read_to_string(&path))
            .await
            .map_err(|_| {
                ToolError::failed(
                    "read",
                    format!("timed out reading {} (10s limit)", path.display()),
                )
            })?
            .map_err(|error| {
                ToolError::failed("read", format!("cannot read {}: {error}", path.display()))
            })?;

        let lines: Vec<&str> = contents.lines().collect();
        let total = lines.len();
        let start = offset.min(total);
        let end = limit.map_or(total, |requested| {
            start.saturating_add(requested).min(total)
        });
        let selected = lines[start..end].join("\n");

        let (kept, truncation_notice) = crate::truncate::truncate_output(&selected);
        let mut result = kept;
        if let Some(notice) = truncation_notice {
            result.push('\n');
            result.push_str(&notice);
        }
        if end < total {
            result.push_str(&format!(
                "\n(file has {total} lines; use offset={end} to continue)"
            ));
        }
        if result.is_empty() {
            result = "(empty file or no lines in range)".to_string();
        }
        Ok(ToolOutcome::success(result))
    }
}
