//! The `find` tool: find files by glob pattern, with a pure-Rust walk.

use agent_m_agent::{Tool, ToolContext, ToolError, ToolOutcome};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};

/// Default maximum number of results returned.
const DEFAULT_LIMIT: usize = 1000;

/// Finds files matching a glob pattern under a path.
pub struct FindTool;

#[async_trait]
impl Tool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> String {
        "Find files by glob pattern (supports *, ?, **). `path` defaults to the session cwd. Returns paths relative to the search root, capped at 1000 results.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern, e.g. \"**/*.rs\" or \"src/main.rs\"" },
                "path": { "type": "string", "description": "Search root (default: session cwd)" },
                "limit": { "type": "number", "description": "Maximum results (default 1000)" }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(
        &self,
        arguments: Value,
        context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let pattern = arguments
            .get("pattern")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::failed("find", "missing string argument `pattern`"))?;
        let root = crate::resolve_path(
            "find",
            &context.cwd,
            arguments.get("path").and_then(Value::as_str).unwrap_or("."),
        )?;
        let limit = arguments
            .get("limit")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_LIMIT)
            .clamp(1, 10_000); // clamp LLM-controlled limits (security review LOW)
        let regex = glob_regex(pattern)
            .map_err(|error| ToolError::failed("find", format!("invalid pattern: {error}")))?;

        let mut results: Vec<String> = Vec::new();
        walk(&root, &regex, &root, &mut results, limit)
            .map_err(|error| ToolError::failed("find", format!("failed to walk: {error}")))?;

        let mut result = results.join("\n");
        if result.is_empty() {
            result = "(no files found)".to_string();
        }
        if results.len() >= limit {
            result.push_str(&format!("\n(results capped at {limit})"));
        }
        Ok(ToolOutcome::success(result))
    }
}

/// Translate a glob (`*`, `?`, `**`) into an anchored regex over the
/// root-relative path.
fn glob_regex(pattern: &str) -> Result<Regex, regex::Error> {
    let mut expression = String::new();
    expression.push('^');
    let mut chars = pattern.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '*' => {
                if chars.peek() == Some(&'*') {
                    chars.next();
                    expression.push_str(".*");
                } else {
                    expression.push_str("[^/]*");
                }
            }
            '?' => expression.push_str("[^/]"),
            other => expression.push_str(&regex::escape(&other.to_string())),
        }
    }
    expression.push('$');
    Regex::new(&expression)
}

fn walk(
    dir: &std::path::Path,
    regex: &Regex,
    root: &std::path::Path,
    results: &mut Vec<String>,
    limit: usize,
) -> Result<(), std::io::Error> {
    if results.len() >= limit {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        if results.len() >= limit {
            return Ok(());
        }
        let entry = entry?;
        let file_type = entry.file_type()?;
        // Skip symlinks so a `ln -s . loop` cannot recurse forever.
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        if file_type.is_dir() {
            walk(&path, regex, root, results, limit)?;
        } else if file_type.is_file() {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();
            if regex.is_match(&relative) {
                results.push(relative);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_patterns_match_expected_paths() {
        let any_rust = glob_regex("**/*.rs").unwrap();
        assert!(any_rust.is_match("src/main.rs"));
        assert!(any_rust.is_match("crates/ai/src/lib.rs"));
        assert!(!any_rust.is_match("src/main.py"));

        let exact = glob_regex("src/main.rs").unwrap();
        assert!(exact.is_match("src/main.rs"));
        assert!(!exact.is_match("src/other.rs"));

        let single = glob_regex("*.rs").unwrap();
        assert!(single.is_match("main.rs"));
        assert!(!single.is_match("src/main.rs"));
    }
}
