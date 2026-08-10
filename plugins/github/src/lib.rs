//! GitHub plugin (separate-repo shape): repo info + create pull request.
//! Auth: GITHUB_TOKEN env var.

use agent_m_plugin_sdk::PluginEntry;
use agent_m_plugin_sdk::tools::{ToolDef, entry};
use std::sync::OnceLock;

/// The GitHub API base; overridable via `GITHUB_API` (tests point it at a
/// wiremock server).
fn api() -> String {
    std::env::var("GITHUB_API").unwrap_or_else(|_| "https://api.github.com".to_string())
}

fn token() -> Result<String, String> {
    std::env::var("GITHUB_TOKEN").map_err(|_| "set GITHUB_TOKEN".to_string())
}

fn client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .user_agent("agent-m")
        .build()
        .map_err(|error| format!("http client: {error}"))
}

fn repo_info(arguments: &str, _cwd: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(arguments).map_err(|e| e.to_string())?;
    let repo = value
        .get("repo")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `repo` (owner/name)")?;
    let response = client()?
        .get(format!("{}/repos/{repo}", api()))
        .header("Authorization", format!("Bearer {}", token()?))
        .send()
        .map_err(|error| format!("github repo failed: {error}"))?;
    let status = response.status();
    let body: serde_json::Value = response.json().map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("github repo HTTP {status}: {body}"));
    }
    Ok(format!(
        "{} ({}): {} — default branch {}",
        body.get("full_name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(repo),
        body.get("language")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?"),
        body.get("description")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(""),
        body.get("default_branch")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("main"),
    ))
}

fn create_pr(arguments: &str, _cwd: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(arguments).map_err(|e| e.to_string())?;
    let repo = value
        .get("repo")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `repo` (owner/name)")?;
    let title = value
        .get("title")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `title`")?;
    let head = value
        .get("head")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `head` (branch)")?;
    let base = value
        .get("base")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("main");
    let body = value
        .get("body")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let response = client()?
        .post(format!("{}/repos/{repo}/pulls", api()))
        .header("Authorization", format!("Bearer {}", token()?))
        .json(&serde_json::json!({
            "title": title,
            "head": head,
            "base": base,
            "body": body
        }))
        .send()
        .map_err(|error| format!("github pr failed: {error}"))?;
    let status = response.status();
    let body: serde_json::Value = response.json().map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("github pr HTTP {status}: {body}"));
    }
    Ok(format!(
        "PR created: {} (#{})",
        body.get("html_url")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(""),
        body.get("number")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    ))
}

static DEFS: &[ToolDef] = &[
    ToolDef {
        name: "github-repo-info",
        description: "Get repository metadata (owner/name)",
        parameters: r#"{"type":"object","properties":{"repo":{"type":"string"}},"required":["repo"]}"#,
        execute: repo_info,
    },
    ToolDef {
        name: "github-create-pr",
        description: "Create a pull request",
        parameters: r#"{"type":"object","properties":{"repo":{"type":"string"},"title":{"type":"string"},"head":{"type":"string"},"base":{"type":"string"},"body":{"type":"string"}},"required":["repo","title","head"]}"#,
        execute: create_pr,
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
        .get_or_init(|| EntryHolder(Box::leak(Box::new(entry("github", "0.1.0", DEFS)))))
        .0
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn create_pr_posts() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let server = runtime.block_on(MockServer::start());
        runtime.block_on(
            Mock::given(method("POST"))
                .and(path("/repos/acme/app/pulls"))
                .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
                    "html_url": "https://github.com/acme/app/pull/7",
                    "number": 7
                })))
                .mount(&server),
        );
        // SAFETY: tests are single-threaded here.
        unsafe { std::env::set_var("GITHUB_API", server.uri()) };
        // SAFETY: tests are single-threaded here.
        unsafe { std::env::set_var("GITHUB_TOKEN", "fake-token") };
        let out = create_pr(
            r#"{"repo":"acme/app","title":"t","head":"b","base":"main"}"#,
            ".",
        )
        .expect("create pr");
        assert!(out.contains("acme/app/pull/7"), "got: {out}");
    }
}
