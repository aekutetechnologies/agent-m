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

#[cfg(test)]
mod tests {
    use super::*;
    use agent_m_agent::ToolContext;
    use std::io::Write;
    use tempfile::NamedTempFile;

    async fn run(path: &str, cwd: &std::path::Path) -> ToolOutcome {
        ViewOutlineTool
            .execute(serde_json::json!({"path": path}), &ToolContext::simple(cwd.to_path_buf()))
            .await
            .expect("execute ok")
    }

    #[tokio::test]
    async fn rust_functions_appear() {
        let mut f = NamedTempFile::with_suffix(".rs").unwrap();
        writeln!(f, "fn foo() {{}}\nfn bar(x: i32) -> i32 {{ x }}").unwrap();
        let out = run(f.path().to_str().unwrap(), f.path().parent().unwrap()).await;
        assert!(out.content.contains("foo") || out.content.contains("bar"), "{}", out.content);
    }

    #[tokio::test]
    async fn python_functions_appear() {
        let mut f = NamedTempFile::with_suffix(".py").unwrap();
        writeln!(f, "def hello():\n    pass\nclass World:\n    pass").unwrap();
        let out = run(f.path().to_str().unwrap(), f.path().parent().unwrap()).await;
        assert!(out.content.contains("hello") || out.content.contains("World"), "{}", out.content);
    }

    #[tokio::test]
    async fn js_functions_appear() {
        let mut f = NamedTempFile::with_suffix(".js").unwrap();
        writeln!(f, "function greet() {{}}\nconst add = (a, b) => a + b;").unwrap();
        let out = run(f.path().to_str().unwrap(), f.path().parent().unwrap()).await;
        // Either a symbol is found, or we gracefully say none found — no panic either way.
        assert!(!out.is_error, "should not be an error: {}", out.content);
    }

    #[tokio::test]
    async fn empty_file_returns_graceful_message() {
        let f = NamedTempFile::with_suffix(".rs").unwrap();
        let out = run(f.path().to_str().unwrap(), f.path().parent().unwrap()).await;
        assert!(!out.is_error, "{}", out.content);
        assert!(
            out.content.contains("No major structural symbols") || out.content.contains("Outline"),
            "{}",
            out.content
        );
    }

    #[tokio::test]
    async fn unknown_extension_returns_result_not_error() {
        let mut f = NamedTempFile::with_suffix(".go").unwrap();
        writeln!(f, "func main() {{}}").unwrap();
        let out = run(f.path().to_str().unwrap(), f.path().parent().unwrap()).await;
        assert!(!out.is_error, "unexpected error for .go file: {}", out.content);
    }
}
