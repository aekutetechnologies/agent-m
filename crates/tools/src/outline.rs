use agent_m_agent::{Tool, ToolContext, ToolError, ToolOutcome};
use async_trait::async_trait;
use serde_json::{Value, json};

pub struct ViewOutlineTool;

#[async_trait]
impl Tool for ViewOutlineTool {
    fn name(&self) -> &str {
        "view_outline"
    }

    fn description(&self) -> String {
        "Parse a source file and return its structural outline (classes, functions, methods, traits) with line numbers. Use this before reading large files to locate specific components to save token budget.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file, relative to the session cwd or absolute" }
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
            .ok_or_else(|| ToolError::failed("view_outline", "missing string argument `path`"))?;

        let path = crate::resolve_path("view_outline", &context.cwd, path)?;
        
        let metadata = tokio::fs::metadata(&path).await.map_err(|error| {
            ToolError::failed("view_outline", format!("cannot stat {}: {error}", path.display()))
        })?;
        
        if !metadata.is_file() {
            return Ok(ToolOutcome::error(format!(
                "{} is a directory, not a file",
                path.display()
            )));
        }

        let contents = tokio::fs::read_to_string(&path)
            .await
            .map_err(|error| {
                ToolError::failed("view_outline", format!("cannot read {}: {error}", path.display()))
            })?;

        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let mut symbols = crate::index::extract_symbols(extension, &contents);
        symbols.sort_by_key(|s| s.line);
        
        if symbols.is_empty() {
            return Ok(ToolOutcome::success(format!("No major structural symbols found in {}", path.display())));
        }

        let lines: Vec<&str> = contents.lines().collect();
        let mut output = String::new();
        output.push_str(&format!("Outline for {}:\n", path.display()));
        
        for symbol in symbols {
            if symbol.line > 0 && symbol.line <= lines.len() {
                let line_text = lines[symbol.line - 1].trim();
                output.push_str(&format!("Line {}: {}\n", symbol.line, line_text));
            } else {
                output.push_str(&format!("Line {}: {} {}\n", symbol.line, symbol.kind, symbol.name));
            }
        }

        Ok(ToolOutcome::success(output))
    }
}
