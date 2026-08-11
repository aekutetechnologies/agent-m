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
}
