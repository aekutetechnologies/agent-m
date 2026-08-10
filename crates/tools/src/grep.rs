//! The `grep` tool: search files for a regex, with a pure-Rust fallback when
//! the `rg` binary is not available.

use agent_m_agent::{Tool, ToolContext, ToolError, ToolOutcome};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// Default maximum number of matches returned.
const DEFAULT_LIMIT: usize = 100;
/// Maximum line length reported per match.
const MAX_LINE_CHARS: usize = 500;
/// Skip files larger than this when reading fallback matches.
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// Searches files for a pattern. Uses ripgrep when available, otherwise a
/// pure-Rust directory walk.
pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> String {
        "Search files for a regex pattern. `path` defaults to the session cwd. Returns path:line:match lines, capped at 100 matches, respecting .gitignore when ripgrep is available.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regular expression to search for" },
                "path": { "type": "string", "description": "Directory or file to search (default: session cwd)" },
                "glob": { "type": "string", "description": "Only search files matching this glob" },
                "ignoreCase": { "type": "boolean", "description": "Case-insensitive search" },
                "literal": { "type": "boolean", "description": "Treat pattern as a literal string" },
                "context": { "type": "number", "description": "Lines of context around each match" },
                "limit": { "type": "number", "description": "Maximum matches (default 100)" }
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
            .ok_or_else(|| ToolError::failed("grep", "missing string argument `pattern`"))?;
        let root = crate::resolve_path(
            "grep",
            &context.cwd,
            arguments.get("path").and_then(Value::as_str).unwrap_or("."),
        )?;
        let glob = arguments.get("glob").and_then(Value::as_str);
        let ignore_case = arguments
            .get("ignoreCase")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let literal = arguments
            .get("literal")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let context_lines = arguments
            .get("context")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        let limit = arguments
            .get("limit")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(DEFAULT_LIMIT);

        // Compile the pattern.
        let pattern_source = if literal {
            regex::escape(pattern)
        } else {
            pattern.to_string()
        };
        let regex = regex::RegexBuilder::new(&pattern_source)
            .case_insensitive(ignore_case)
            .build()
            .map_err(|error| ToolError::failed("grep", format!("invalid pattern: {error}")))?;

        let use_ripgrep = is_ripgrep_available().await;
        let (kept, notices) = if use_ripgrep {
            grep_with_ripgrep(
                &root,
                &pattern_source,
                ignore_case,
                literal,
                glob,
                context_lines,
                limit,
            )
            .await
        } else {
            grep_with_walk(&root, &regex, glob, context_lines, limit)
        }
        .map_err(|error| ToolError::failed("grep", error))?;

        let mut result = kept;
        for notice in notices {
            result.push('\n');
            result.push_str(&notice);
        }
        if result.is_empty() {
            result = "(no matches)".to_string();
        }
        Ok(ToolOutcome::success(result))
    }
}

async fn is_ripgrep_available() -> bool {
    tokio::process::Command::new("rg")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .map(|status| status.success())
        .unwrap_or(false)
}

async fn grep_with_ripgrep(
    root: &Path,
    pattern: &str,
    ignore_case: bool,
    literal: bool,
    glob: Option<&str>,
    context_lines: usize,
    limit: usize,
) -> Result<(String, Vec<String>), String> {
    let mut command = tokio::process::Command::new("rg");
    command
        .arg("--no-heading")
        .arg("--line-number")
        .arg("--hidden")
        .arg("--color")
        .arg("never");
    if ignore_case {
        command.arg("-i");
    }
    if literal {
        command.arg("-F");
    }
    if context_lines > 0 {
        command.arg(format!("-C{context_lines}"));
    }
    command.arg("--max-count").arg(limit.to_string());
    if let Some(glob) = glob {
        command.arg("-g").arg(glob);
    }
    // Exclude sensitive files (after user globs: last match wins).
    for exclude in crate::paths::sensitive_globs() {
        command.arg("-g").arg(exclude);
    }
    // `--` ends option parsing so a pattern that looks like a flag (e.g.
    // `--no-ignore`) cannot flip rg's options (security review MEDIUM).
    command.arg("--").arg(pattern).arg(root);

    let output = command
        .output()
        .await
        .map_err(|error| format!("failed to run rg: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let (kept, truncation_notice) = crate::truncate::truncate_output(&stdout);
    let mut notices = Vec::new();
    if let Some(notice) = truncation_notice {
        notices.push(notice);
    }
    if !output.status.success() && !stdout.is_empty() {
        notices.push(format!(
            "rg exited with code {}",
            output.status.code().unwrap_or(-1)
        ));
    }
    Ok((kept, notices))
}

fn grep_with_walk(
    root: &Path,
    regex: &Regex,
    glob: Option<&str>,
    context_lines: usize,
    limit: usize,
) -> Result<(String, Vec<String>), String> {
    let mut matches: Vec<String> = Vec::new();
    let mut files_searched = 0usize;
    walk_files(root, &mut |path| {
        files_searched += 1;
        if matches.len() >= limit {
            return;
        }
        if let Some(glob) = glob
            && !glob_matches(glob, path)
        {
            return;
        }
        let Ok(contents) = std::fs::read(path) else {
            return;
        };
        if contents.len() as u64 > MAX_FILE_BYTES {
            return;
        }
        let Ok(contents) = String::from_utf8(contents) else {
            return;
        };
        let lines: Vec<&str> = contents.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            if matches.len() >= limit {
                return;
            }
            if regex.is_match(line) {
                let line_text = truncate_line(line);
                let start = index.saturating_sub(context_lines);
                let end = (index + 1 + context_lines).min(lines.len());
                if context_lines > 0 {
                    for (context_index, context_line) in
                        lines.iter().enumerate().take(end).skip(start)
                    {
                        let marker = if context_index == index { ':' } else { '-' };
                        matches.push(format!(
                            "{}:{}{} {}",
                            path.display(),
                            context_index + 1,
                            marker,
                            truncate_line(context_line)
                        ));
                    }
                    matches.push(String::new());
                } else {
                    matches.push(format!("{}:{}:{}", path.display(), index + 1, line_text));
                }
            }
        }
    })
    .map_err(|error| format!("failed to walk {}: {error}", root.display()))?;

    let mut notices = Vec::new();
    if files_searched == 0 {
        notices.push(format!("no files found under {}", root.display()));
    }
    if matches.len() >= limit {
        notices.push(format!(
            "(results capped at {limit}; narrowing the pattern is recommended)"
        ));
    }
    let (kept, truncation_notice) = crate::truncate::truncate_output(&matches.join("\n"));
    if let Some(notice) = truncation_notice {
        notices.push(notice);
    }
    Ok((kept, notices))
}

fn truncate_line(line: &str) -> String {
    let mut chars = line.chars();
    let head: String = chars.by_ref().take(MAX_LINE_CHARS).collect();
    if chars.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}

fn walk_files(root: &Path, on_file: &mut impl FnMut(&PathBuf)) -> Result<(), std::io::Error> {
    if root.is_file() {
        on_file(&root.to_path_buf());
        return Ok(());
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            // Skip symlinks so a `ln -s . loop` cannot loop forever.
            if file_type.is_symlink() {
                continue;
            }
            let path = entry.path();
            if crate::paths::is_sensitive(&path) {
                continue;
            }
            if file_type.is_dir() {
                stack.push(path);
            } else if file_type.is_file() {
                on_file(&path);
            }
        }
    }
    Ok(())
}

/// Very small glob matcher supporting `*`, `?` and `**` (translated to a regex
/// matched against the full path).
fn glob_matches(glob: &str, path: &Path) -> bool {
    let mut expression = String::new();
    expression.push('^');
    let mut chars = glob.chars().peekable();
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
        .map(|regex| regex.is_match(&path.to_string_lossy()))
        .unwrap_or(false)
}
