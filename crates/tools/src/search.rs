//! The `search` tool: local symbol/keyword lookup against the per-project
//! index. Read-only, so it is allowed in plan mode. Lets the model find where
//! things are defined (symbols, identifiers, file names) without grepping
//! blindly or reading whole files.

use agent_m_agent::{Tool, ToolContext, ToolError, ToolOutcome};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::path::Path;

use crate::index::{IndexedFile, SymbolHit};

/// Maximum number of matches returned.
const MAX_RESULTS: usize = 20;

/// One scored match ready for display.
struct Match {
    score: u32,
    path: String,
    hits: Vec<SymbolHit>,
}

pub struct SearchTool;

#[async_trait]
impl Tool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }

    fn description(&self) -> String {
        "Search the codebase for symbols, identifiers, and file names using a local index (read-only). Returns the top matches with file, line, and a snippet. Use this to locate definitions before reading whole files.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol, identifier, or file-name fragment to look up"
                },
                "path": {
                    "type": "string",
                    "description": "Project root to search (default: session cwd)"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(
        &self,
        arguments: Value,
        context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let query = arguments
            .get("query")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::failed("search", "missing string argument `query`"))?
            .to_lowercase();
        let root = arguments
            .get("path")
            .and_then(Value::as_str)
            .map(|path| crate::resolve_path(&context.cwd, path))
            .unwrap_or_else(|| context.cwd.clone());
        if !root.is_dir() {
            return Ok(ToolOutcome::error(format!(
                "`{}` is not a directory; search works on a project root",
                root.display()
            )));
        }
        let index = crate::index::load_or_build(&root);

        let mut matches: Vec<Match> = Vec::new();
        for file in &index.files {
            if let Some(matched) = score_file(file, &query) {
                matches.push(matched);
            }
        }
        matches.sort_by_key(|item| std::cmp::Reverse(item.score));
        matches.truncate(MAX_RESULTS);

        if matches.is_empty() {
            return Ok(ToolOutcome::success(format!(
                "no matches for `{query}` in {} ({} files indexed)",
                root.display(),
                index.files.len()
            )));
        }

        let mut out = format!(
            "{} match(es) for `{query}` ({} files indexed):\n",
            matches.len(),
            index.files.len()
        );
        for matched in &matches {
            if matched.hits.is_empty() {
                out.push_str(&format!("  {} (file name match)\n", matched.path));
                continue;
            }
            for hit in &matched.hits {
                let snippet = snippet_at(&root, &matched.path, hit.line);
                out.push_str(&format!(
                    "  {}:{} ({} {}){}",
                    matched.path,
                    hit.line,
                    hit.kind,
                    hit.name,
                    snippet.map(|s| format!(" — {s}")).unwrap_or_default(),
                ));
                out.push('\n');
            }
        }
        Ok(ToolOutcome::success(out))
    }
}

/// Score one indexed file against the query; `None` if nothing matches.
fn score_file(file: &IndexedFile, query: &str) -> Option<Match> {
    let mut score = 0u32;
    let mut hits: Vec<SymbolHit> = Vec::new();
    for symbol in &file.symbols {
        let name = symbol.name.to_lowercase();
        if name == *query {
            score = score.max(100);
            hits.push(symbol.clone());
        } else if name.starts_with(query) {
            score = score.max(60);
            hits.push(symbol.clone());
        } else if name.contains(query) {
            score = score.max(40);
        }
    }
    if score < 50 {
        for identifier in &file.identifiers {
            if identifier == query {
                score = score.max(50);
            } else if identifier.starts_with(query) {
                score = score.max(30);
            }
        }
    }
    if score < 40 {
        let basename = file
            .path
            .rsplit('/')
            .next()
            .unwrap_or(&file.path)
            .to_lowercase();
        if basename.contains(query) {
            score = score.max(25);
        }
    }
    if score == 0 {
        // Cheap substring fallback on everything we indexed.
        let haystack = format!(
            "{}{}",
            file.symbols
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            file.identifiers.join(" ")
        )
        .to_lowercase();
        if haystack.contains(query) {
            score = 10;
        }
    }
    if score == 0 {
        return None;
    }
    Some(Match {
        score,
        path: file.path.clone(),
        hits,
    })
}

/// One-line snippet around `line` (1-based), truncated.
fn snippet_at(root: &Path, relative: &str, line: usize) -> Option<String> {
    let content = std::fs::read_to_string(root.join(relative)).ok()?;
    let target = content.lines().nth(line.saturating_sub(1))?;
    let mut chars = target.trim().chars();
    let head: String = chars.by_ref().take(120).collect();
    let mut snippet = head;
    if chars.next().is_some() {
        snippet.push('…');
    }
    Some(snippet)
}
