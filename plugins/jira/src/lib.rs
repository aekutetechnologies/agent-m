//! Jira plugin (separate-repo shape): search issues, comment, transition.
//! Auth: JIRA_URL + JIRA_TOKEN env vars (agent-m keys pattern).

use agent_m_plugin_sdk::PluginEntry;
use agent_m_plugin_sdk::tools::{ToolDef, entry};
use std::sync::OnceLock;

fn jira_base() -> Result<String, String> {
    std::env::var("JIRA_URL")
        .map_err(|_| "set JIRA_URL (e.g. https://your.atlassian.net)".to_string())
}
fn jira_token() -> Result<String, String> {
    std::env::var("JIRA_TOKEN").map_err(|_| "set JIRA_TOKEN".to_string())
}

/// Resolve auth: explicit `base`/`token` in the arguments win over the env
/// vars (tests pass their mock server per call to avoid env races).
fn base_from(arguments: &serde_json::Value) -> Result<String, String> {
    arguments
        .get("base")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .map(Ok)
        .unwrap_or_else(jira_base)
}
fn token_from(arguments: &serde_json::Value) -> Result<String, String> {
    arguments
        .get("token")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .map(Ok)
        .unwrap_or_else(jira_token)
}

fn client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .build()
        .map_err(|error| format!("http client: {error}"))
}

fn search(arguments: &str, _cwd: &str) -> Result<String, String> {
    let query: serde_json::Value = serde_json::from_str(arguments).map_err(|e| e.to_string())?;
    let jql = query
        .get("query")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let base = base_from(&query)?;
    let token = token_from(&query)?;
    let response = client()?
        .get(format!("{base}/rest/api/3/search"))
        .query(&[("jql", jql), ("maxResults", "5")])
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|error| format!("jira search failed: {error}"))?;
    let status = response.status();
    let body: serde_json::Value = response.json().map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("jira search HTTP {status}: {body}"));
    }
    let issues = body
        .get("issues")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = String::new();
    for issue in issues {
        let key = issue
            .get("key")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        let summary = issue
            .pointer("/fields/summary")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        out.push_str(&format!("{key}: {summary}\n"));
    }
    Ok(out.trim_end().to_string())
}

fn comment(arguments: &str, _cwd: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(arguments).map_err(|e| e.to_string())?;
    let key = value
        .get("key")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `key`")?;
    let body = value
        .get("body")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `body`")?;
    let base = base_from(&value)?;
    let token = token_from(&value)?;
    let response = client()?
        .post(format!("{base}/rest/api/3/issue/{key}/comment"))
        .header("Authorization", format!("Bearer {token}"))
        .json(&serde_json::json!({ "body": body }))
        .send()
        .map_err(|error| format!("jira comment failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("jira comment HTTP {status}"));
    }
    Ok(format!("commented on {key}"))
}

fn transition(arguments: &str, _cwd: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(arguments).map_err(|e| e.to_string())?;
    let key = value
        .get("key")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `key`")?;
    let id = value
        .get("transitionId")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `transitionId`")?;
    let base = base_from(&value)?;
    let token = token_from(&value)?;
    let response = client()?
        .post(format!("{base}/rest/api/3/issue/{key}/transitions"))
        .header("Authorization", format!("Bearer {token}"))
        .json(&serde_json::json!({ "transition": { "id": id } }))
        .send()
        .map_err(|error| format!("jira transition failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("jira transition HTTP {status}"));
    }
    Ok(format!("transitioned {key}"))
}

static DEFS: &[ToolDef] = &[
    ToolDef {
        name: "jira-search",
        description: "Search Jira issues (JQL)",
        parameters: r#"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#,
        execute: search,
    },
    ToolDef {
        name: "jira-comment",
        description: "Add a comment to a Jira issue",
        parameters: r#"{"type":"object","properties":{"key":{"type":"string"},"body":{"type":"string"}},"required":["key","body"]}"#,
        execute: comment,
    },
    ToolDef {
        name: "jira-transition",
        description: "Transition a Jira issue (e.g. to Done)",
        parameters: r#"{"type":"object","properties":{"key":{"type":"string"},"transitionId":{"type":"string"}},"required":["key","transitionId"]}"#,
        execute: transition,
    },
];

struct EntryHolder(*const PluginEntry);
// SAFETY: the entry points into leaked, immutable plugin state.
unsafe impl Send for EntryHolder {}
unsafe impl Sync for EntryHolder {}

static ENTRY: OnceLock<EntryHolder> = OnceLock::new();

#[unsafe(no_mangle)]
pub extern "C" fn agent_m_plugin_entry() -> *const PluginEntry {
    ENTRY
        .get_or_init(|| EntryHolder(Box::leak(Box::new(entry("jira", "0.1.0", DEFS)))))
        .0
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn search_parses_issues() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let server = runtime.block_on(MockServer::start());
        runtime.block_on(
            Mock::given(method("GET"))
                .and(path("/rest/api/3/search"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "issues": [
                        { "key": "PROJ-1", "fields": { "summary": "Fix login" } },
                        { "key": "PROJ-2", "fields": { "summary": "Add tests" } }
                    ]
                })))
                .mount(&server),
        );
        let out = search(
            format!(
                r#"{{"query":"project = PROJ","base":"{}","token":"fake"}}"#,
                server.uri()
            )
            .as_str(),
            ".",
        )
        .expect("search");
        assert!(out.contains("PROJ-1: Fix login"), "got: {out}");
        assert!(out.contains("PROJ-2: Add tests"), "got: {out}");
    }

    #[test]
    fn transition_posts() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let server = runtime.block_on(MockServer::start());
        runtime.block_on(
            Mock::given(method("POST"))
                .and(path("/rest/api/3/issue/PROJ-1/transitions"))
                .respond_with(ResponseTemplate::new(204))
                .mount(&server),
        );
        let out = transition(
            format!(
                r#"{{"key":"PROJ-1","transitionId":"31","base":"{}","token":"fake"}}"#,
                server.uri()
            )
            .as_str(),
            ".",
        )
        .expect("transition");
        assert!(out.contains("PROJ-1"), "got: {out}");
    }
}
