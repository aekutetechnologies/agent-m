//! The `edit` tool: exact-text multi-edit against the original file, with a
//! unified diff in the result.

use agent_m_agent::{Tool, ToolContext, ToolError, ToolOutcome};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use similar::TextDiff;

/// One exact-text replacement, mirroring pi's edit tool schema.
#[derive(Debug, Deserialize)]
struct EditOperation {
    #[serde(rename = "oldText")]
    old_text: String,
    #[serde(rename = "newText")]
    new_text: String,
}

/// Applies a batch of exact-text edits to a file and returns the unified diff.
/// Each `oldText` must appear exactly once in the file at the time it is
/// applied (edits are applied against the evolving content, so overlapping
/// edits fail loudly instead of being silently dropped).
pub struct EditTool;

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn description(&self) -> String {
        "Apply exact-text edits to a file. `edits` is an array of {oldText, newText} where each oldText must appear exactly once in the file at the time it is applied. Returns the unified diff of the changes.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file" },
                "edits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": { "type": "string" },
                            "newText": { "type": "string" }
                        },
                        "required": ["oldText", "newText"]
                    }
                }
            },
            "required": ["path", "edits"]
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
            .ok_or_else(|| ToolError::failed("edit", "missing string argument `path`"))?;
        let operations: Vec<EditOperation> = serde_json::from_value(
            arguments
                .get("edits")
                .cloned()
                .unwrap_or(Value::Array(vec![])),
        )
        .map_err(|error| ToolError::failed("edit", format!("invalid `edits`: {error}")))?;
        if operations.is_empty() {
            return Err(ToolError::failed("edit", "`edits` must not be empty"));
        }

        let path = crate::resolve_path(&context.cwd, path);
        let original = tokio::fs::read_to_string(&path).await.map_err(|error| {
            ToolError::failed("edit", format!("cannot read {}: {error}", path.display()))
        })?;

        // Apply each edit against the evolving content. Each oldText must
        // appear exactly once in the file *at the time it is applied*, so an
        // edit whose target was overwritten by an earlier edit fails loudly
        // instead of being silently skipped.
        let mut edited = original.clone();
        let mut applied = 0usize;
        for operation in &operations {
            if operation.old_text.is_empty() {
                return Err(ToolError::failed(
                    "edit",
                    "oldText must not be empty (each edit must be an exact-text replacement)",
                ));
            }
            let occurrences = edited.matches(&operation.old_text).count();
            if occurrences != 1 {
                return Err(ToolError::failed(
                    "edit",
                    format!(
                        "oldText `{}` appears {occurrences} times in {}; expected exactly once",
                        preview(&operation.old_text, 60),
                        path.display()
                    ),
                ));
            }
            edited = edited.replacen(&operation.old_text, &operation.new_text, 1);
            applied += 1;
        }

        if edited == original {
            return Ok(ToolOutcome::success(
                "No changes made (edits produced identical content)",
            ));
        }

        let diff = TextDiff::from_lines(&original, &edited);
        let patch: String = diff
            .unified_diff()
            .context_radius(3)
            .header(&path.display().to_string(), &path.display().to_string())
            .to_string();

        tokio::fs::write(&path, edited.as_bytes())
            .await
            .map_err(|error| {
                ToolError::failed("edit", format!("cannot write {}: {error}", path.display()))
            })?;

        Ok(ToolOutcome::success(format!(
            "Applied {} edit(s) to {}\n{}",
            applied,
            path.display(),
            patch
        )))
    }
}

fn preview(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let head: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}
