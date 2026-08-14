//! MCP server registry: where servers are configured and how they load.
//!
//! Uses the cross-tool convention (`mcpServers`), read from, in order:
//! `~/.agent-m/agent/mcp.json` then `settings.json` — same file family as the
//! rest of agent-m's config. A server is either a spawned command
//! (`command` + `args` + `env`) or a Streamable HTTP URL (`url`).

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Advisory match of a bare tool name against `readOnlyTools` patterns:
/// exact name or `*` wildcard (e.g. `get-*`, `search-*`).
pub fn matches_patterns(name: &str, patterns: &[String]) -> bool {
    patterns.iter().any(|pattern| {
        if pattern == "*" {
            return true;
        }
        if let Some(split) = pattern.split_once('*') {
            let (prefix, suffix) = split;
            return (prefix.is_empty() || name.starts_with(prefix))
                && (suffix.is_empty() || name.ends_with(suffix))
                && name.len() >= prefix.len() + suffix.len();
        }
        pattern == name
    })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Spawned-process server (stdio transport).
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Remote Streamable HTTP server.
    #[serde(default)]
    pub url: Option<String>,
    /// Advisory patterns (exact or `*` wildcard) on bare tool names treated as
    /// read-only, so they auto-approve even when the server doesn't advertise
    /// `annotations.readOnlyHint`. Not a security boundary.
    #[serde(default)]
    pub read_only_tools: Vec<String>,
}

/// All configured servers, keyed by name.
pub type McpServers = BTreeMap<String, McpServerConfig>;

/// Load `mcpServers` from `mcp.json` (preferred) or `settings.json` under the
/// given agent dir. Returns an empty map when neither file has servers.
pub fn load_servers(agent_dir: &Path) -> McpServers {
    let mut servers = McpServers::new();
    for file in ["mcp.json", "settings.json"] {
        let Ok(text) = std::fs::read_to_string(agent_dir.join(file)) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if let Some(servers_value) = value.get("mcpServers").and_then(Value::as_object) {
            for (name, config) in servers_value {
                if let Ok(config) = serde_json::from_value::<McpServerConfig>(config.clone()) {
                    servers.insert(name.clone(), config);
                }
            }
        }
        if !servers.is_empty() {
            break;
        }
    }
    servers
}

/// Probe for a binary in the usual install locations, in order, falling back to
/// the bare name (so the user can fix their PATH).
fn find_bin(name: &str) -> String {
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cargo"));
    let candidates = [
        cargo_home.join("bin").join(name),
        std::path::PathBuf::from("/opt/homebrew/bin").join(name),
        std::path::PathBuf::from("/usr/local/bin").join(name),
    ];
    for candidate in candidates {
        if candidate.is_file() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    name.to_string()
}

/// Create a starter mcp.json template when launching agent-m for the first time.
pub fn ensure_default_mcp(agent_dir: &Path) -> std::io::Result<()> {
    let _ = std::fs::create_dir_all(agent_dir);
    let path = agent_dir.join("mcp.json");
    if path.exists() {
        return Ok(());
    }
    let github = find_bin("github-mcp-server");
    let filesystem = find_bin("rust-mcp-filesystem");
    let jira = find_bin("jira-mcp-rs");
    let postgres = find_bin("mcp-postgres");
    let template = serde_json::json!({
        "mcpServers": {
            "github": {
                "command": github,
                "args": ["stdio"],
                "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "" },
                "readOnlyTools": ["get-*", "search-*", "list-*"]
            },
            "filesystem": {
                "command": filesystem,
                "args": ["."],
                "readOnlyTools": ["read-*", "list-*"]
            },
            "jira": {
                "command": jira,
                "args": [],
                "env": {
                    "JIRA_BASE_URL": "",
                    "JIRA_EMAIL": "",
                    "JIRA_API_TOKEN": ""
                },
                "readOnlyTools": ["get-*", "search-*"]
            },
            "postgres": {
                "command": postgres,
                "args": [],
                "env": { "DATABASE_URL": "" }
            }
        }
    });
    let pretty = serde_json::to_string_pretty(&template)?;
    std::fs::write(&path, format!("{pretty}\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn loads_servers_from_mcp_json_and_settings() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("mcp.json"),
            r#"{"mcpServers": {"filesystem": {"command": "npx", "args": ["-y", "server-fs"]}}}"#,
        )
        .unwrap();
        let servers = load_servers(dir.path());
        assert_eq!(servers.len(), 1);
        assert_eq!(servers["filesystem"].command.as_deref(), Some("npx"));
        assert_eq!(servers["filesystem"].args, vec!["-y", "server-fs"]);
    }

    #[test]
    fn settings_json_fallback_and_http_url() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join("settings.json"),
            r#"{"mcpServers": {"remote": {"url": "https://mcp.example.com/mcp"}}}"#,
        )
        .unwrap();
        let servers = load_servers(dir.path());
        assert_eq!(servers.len(), 1);
        assert_eq!(
            servers["remote"].url.as_deref(),
            Some("https://mcp.example.com/mcp")
        );
        // mcp.json wins over settings.json.
        std::fs::write(
            dir.path().join("mcp.json"),
            r#"{"mcpServers": {"local": {"command": "echo"}}}"#,
        )
        .unwrap();
        let servers = load_servers(dir.path());
        assert_eq!(servers.len(), 1);
        assert!(servers.contains_key("local"));
    }

    #[test]
    fn missing_config_yields_empty() {
        let dir = tempdir().unwrap();
        assert!(load_servers(dir.path()).is_empty());
    }

    #[test]
    fn matches_exact_and_wildcard_patterns() {
        let patterns = vec!["get-*".to_string(), "search_repos".to_string(), "*".to_string()];
        assert!(matches_patterns("get-issue", &patterns));
        assert!(matches_patterns("search_repos", &patterns));
        assert!(matches_patterns("anything", &patterns)); // leading `*` too
        let only_wild = vec!["read_*".to_string(), "list_*".to_string()];
        assert!(matches_patterns("read_file", &only_wild));
        assert!(matches_patterns("list_tools", &only_wild));
        assert!(!matches_patterns("write_file", &only_wild));
    }

    #[test]
    fn ensure_default_mcp_uses_rust_servers() {
        let dir = tempdir().unwrap();
        ensure_default_mcp(dir.path()).unwrap();
        let text = std::fs::read_to_string(dir.path().join("mcp.json")).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        for config in value["mcpServers"]
            .as_object()
            .unwrap()
            .values()
        {
            let command = config["command"].as_str().unwrap();
            assert!(
                !command.contains("npx"),
                "command should not be npx-based: {command}"
            );
        }
    }
}
