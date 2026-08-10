//! The `ls` tool: list directory entries, sorted case-insensitively.

use agent_m_agent::{Tool, ToolContext, ToolError, ToolOutcome};
use async_trait::async_trait;
use serde_json::{Value, json};

/// Default maximum number of entries returned.
const DEFAULT_LIMIT: usize = 500;

/// Lists directory entries, mirroring pi's `ls` tool (case-insensitive sort,
/// `/` suffix for directories, includes dotfiles, capped at 500 entries).
pub struct LsTool;

#[async_trait]
impl Tool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> String {
        "List directory entries. Directories get a `/` suffix. Sorted case-insensitively, includes dotfiles. Capped at 500 entries.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory to list (default: session cwd)" },
                "limit": { "type": "number", "description": "Maximum entries (default 500)" }
            }
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
            .map(|path| crate::resolve_path(&context.cwd, path))
            .unwrap_or_else(|| context.cwd.clone());
        let limit = arguments
            .get("limit")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_LIMIT);

        let mut entries = tokio::fs::read_dir(&path).await.map_err(|error| {
            ToolError::failed("ls", format!("cannot list {}: {error}", path.display()))
        })?;

        let mut names: Vec<(String, bool)> = Vec::new();
        while let Some(entry) = entries.next_entry().await.map_err(|error| {
            ToolError::failed("ls", format!("cannot read {}: {error}", path.display()))
        })? {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_dir = entry
                .file_type()
                .await
                .map(|file_type| file_type.is_dir())
                .unwrap_or(false);
            names.push((name, is_dir));
        }
        names.sort_by_key(|(name, _)| name.to_lowercase());

        let total = names.len();
        let truncated = names.len() > limit;
        names.truncate(limit);

        let mut result = names
            .iter()
            .map(|(name, is_dir)| {
                if *is_dir {
                    format!("{name}/")
                } else {
                    name.clone()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        if result.is_empty() {
            result = "(empty directory)".to_string();
        }
        if truncated {
            result.push_str(&format!("\n(… {limit} of {total} entries shown)"));
        }
        Ok(ToolOutcome::success(result))
    }
}
