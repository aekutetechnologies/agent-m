//! Bounded store of recent tool outputs for the REPL. Full outputs are kept so
//! `/tool-output last|<n>` can reprint them; the live transcript only shows a
//! one-line summary.

#[derive(Clone, Debug)]
pub struct StoredOutput {
    pub name: String,
    pub full: String,
}

#[derive(Default)]
pub struct ToolStore {
    entries: Vec<StoredOutput>,
}

const MAX_STORED: usize = 20;
const MAX_SUMMARY_CHARS: usize = 120;

impl ToolStore {
    pub fn push(&mut self, name: &str, full: String) {
        self.entries.push(StoredOutput {
            name: name.to_string(),
            full,
        });
        if self.entries.len() > MAX_STORED {
            let excess = self.entries.len() - MAX_STORED;
            self.entries.drain(0..excess);
        }
    }

    /// `last` / `latest`, or a 1-based index where `1` is the most recent.
    pub fn get(&self, spec: &str) -> Option<StoredOutput> {
        match spec.trim() {
            "" | "last" | "latest" => self.entries.last().cloned(),
            other => {
                let n: usize = other.parse().ok()?;
                if n == 0 || n > self.entries.len() {
                    return None;
                }
                Some(self.entries[self.entries.len() - n].clone())
            }
        }
    }

    pub fn list(&self) -> String {
        if self.entries.is_empty() {
            return "No tool outputs recorded this session.".to_string();
        }
        let len = self.entries.len();
        let mut out = String::new();
        for (i, e) in self.entries.iter().enumerate() {
            out.push_str(&format!("  {}. [{}]\n", len - i, e.name));
        }
        out
    }
}

/// A one-line digest of a tool result: first non-empty line … last non-empty
/// line (when different) plus a count of remaining lines.
pub fn summarize(content: &str) -> (String, usize) {
    let non_empty: Vec<&str> = content
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .collect();

    let (first, last) = match non_empty.as_slice() {
        [] => return (String::new(), 0),
        [only] => (*only, None),
        [first, .., last] => (*first, Some(*last)),
    };

    const HALF: usize = MAX_SUMMARY_CHARS / 2;
    let trunc = |s: &str, n: usize| -> String {
        let t: String = s.chars().take(n).collect();
        if s.chars().count() > n { format!("{t}…") } else { t }
    };

    let text = if let Some(last) = last {
        format!("{} … {}", trunc(first, HALF), trunc(last, HALF))
    } else {
        trunc(first, MAX_SUMMARY_CHARS)
    };

    let extra_lines = content.lines().count().saturating_sub(1);
    (text, extra_lines)
}

const MAX_HUMANIZE_CHARS: usize = 80;

fn quoted(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

/// Human-readable description of a tool call, e.g. `reading "src/main.jsx"`.
/// Known tools map to a verb + primary arg key; MCP/plugin tools fall back to
/// generic keys (`query`/`url`/`path`/`command`); anything else keeps the raw
/// JSON args so the call stays identifiable.
pub fn humanize(name: &str, args: &serde_json::Value) -> String {
    match name {
        "read" => render("reading", "path", args),
        "outline" => render("outlining", "path", args),
        "write" => render("writing", "path", args),
        "edit" => render("editing", "path", args),
        "ls" => render("listing", "path", args),
        "find" => render("finding", "pattern", args),
        "grep" => render("searching for", "pattern", args),
        "bash" => render("running", "command", args),
        "search" => render("searching", "query", args),
        "web_search" => render("searching the web for", "query", args),
        "web_fetch" => render("fetching", "url", args),
        "ask" => render("asking", "question", args),
        "delegate" => render("delegating a task", "prompt", args),
        _ => {
            let generic = ["query", "url", "path", "command"]
                .into_iter()
                .find(|k| args.get(k).and_then(serde_json::Value::as_str).is_some());
            if let Some(key) = generic {
                let bare = name.rsplit("__").next().unwrap_or(name);
                render(bare, key, args)
            } else {
                let dump = args.to_string();
                let short: String = dump.chars().take(MAX_HUMANIZE_CHARS * 2).collect();
                format!("{} {}", name, short)
            }
        }
    }
}

fn render(verb: &str, key: &str, args: &serde_json::Value) -> String {
    let val = args
        .get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let out = if val.is_empty() {
        format!("{verb} {}", args)
    } else {
        format!("{verb} {}", quoted(val))
    };
    out.chars().take(MAX_HUMANIZE_CHARS).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_shows_first_and_last() {
        let (text, extra) = summarize("line one\nline two\n\nline four");
        assert!(text.contains("line one"), "{text:?}");
        assert!(text.contains("line four"), "{text:?}");
        assert!(text.contains('…'), "{text:?}");
        assert_eq!(extra, 3);
    }

    #[test]
    fn summarize_single_line_no_separator() {
        let (text, extra) = summarize("only line");
        assert_eq!(text, "only line");
        assert_eq!(extra, 0);
    }

    #[test]
    fn summarize_empty() {
        let (text, extra) = summarize("");
        assert_eq!(text, "");
        assert_eq!(extra, 0);
    }

    #[test]
    fn summarize_truncates_long_lines() {
        let long = "x".repeat(300);
        let (text, _) = summarize(&long);
        // single-line: truncated to HALF + ellipsis? No — single-line uses MAX_SUMMARY_CHARS.
        assert!(text.chars().count() <= MAX_SUMMARY_CHARS + 1);
        assert!(text.ends_with('…'));
    }

    #[test]
    fn store_indexing_is_one_based_from_most_recent() {
        let mut store = ToolStore::default();
        store.push("first", "1".to_string());
        store.push("second", "2".to_string());
        assert_eq!(store.get("last").unwrap().full, "2");
        assert_eq!(store.get("latest").unwrap().full, "2");
        assert_eq!(store.get("1").unwrap().full, "2");
        assert_eq!(store.get("2").unwrap().full, "1");
        assert!(store.get("3").is_none());
        assert!(store.get("0").is_none());
    }

    #[test]
    fn store_bounds_size() {
        let mut store = ToolStore::default();
        for i in 0..MAX_STORED + 10 {
            store.push(&format!("t{i}"), format!("{i}"));
        }
        assert_eq!(store.list().lines().count(), MAX_STORED);
    }

    #[test]
    fn humanize_known_tools() {
        use serde_json::json;
        assert_eq!(
            humanize("find", &json!({"pattern": "**/*.config.*"})),
            "finding \"**/*.config.*\""
        );
        assert_eq!(
            humanize("read", &json!({"path": "src/main.jsx"})),
            "reading \"src/main.jsx\""
        );
        assert_eq!(
            humanize("bash", &json!({"command": "ls -la"})),
            "running \"ls -la\""
        );
        assert_eq!(
            humanize("ask", &json!({"question": "backend or wasm?"})),
            "asking \"backend or wasm?\""
        );
        assert_eq!(
            humanize("web_search", &json!({"query": "rust mcp"})),
            "searching the web for \"rust mcp\""
        );
    }

    #[test]
    fn humanize_escapes_quotes() {
        use serde_json::json;
        assert_eq!(
            humanize("grep", &json!({"pattern": "say \"hi\""})),
            "searching for \"say \\\"hi\\\"\""
        );
    }

    #[test]
    fn humanize_plugin_tools_use_generic_keys() {
        use serde_json::json;
        assert_eq!(
            humanize("jira", &json!({"query": "agent-m"})),
            "jira \"agent-m\""
        );
        assert_eq!(
            humanize("github-repo-info", &json!({"path": "."})),
            "github-repo-info \".\""
        );
    }

    #[test]
    fn humanize_unknown_falls_back_to_json() {
        use serde_json::json;
        assert_eq!(
            humanize("mystery-tool", &json!({"a": 1, "b": "x"})),
            "mystery-tool {\"a\":1,\"b\":\"x\"}"
        );
    }

    #[test]
    fn humanize_truncates_long_values() {
        use serde_json::json;
        let long = "x".repeat(300);
        let out = humanize("bash", &json!({"command": long}));
        assert!(out.chars().count() <= MAX_HUMANIZE_CHARS);
    }
}
