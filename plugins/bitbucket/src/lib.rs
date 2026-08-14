//! Bitbucket plugin (separate-repo shape): issue search, comment, transition,
//! and create pull request against the Bitbucket Cloud API v2
//! (`https://api.bitbucket.org/2.0`). Auth: BITBUCKET_TOKEN env var
//! (app password for Cloud, PAT for Server — the plugin sends it as a
//! `Bearer` token, mirroring the jira plugin; for Bitbucket Cloud app
//! passwords your base URL/proxy may need to swap to Basic auth).

use agent_m_plugin_sdk::PluginEntry;
use agent_m_plugin_sdk::tools::{ToolDef, entry};
use std::sync::OnceLock;

fn bitbucket_base() -> Result<String, String> {
    std::env::var("BITBUCKET_BASE")
        .ok()
        .or_else(|| std::env::var("BITBUCKET_URL").ok())
        .filter(|value| !value.is_empty())
        .map(Ok)
        .unwrap_or_else(|| Ok("https://api.bitbucket.org".to_string()))
}
fn bitbucket_token() -> Result<String, String> {
    std::env::var("BITBUCKET_TOKEN").map_err(|_| "set BITBUCKET_TOKEN".to_string())
}

/// Resolve the API base: `base_url` wins over `base` (create-pr uses `base`
/// for the destination branch), then the env var. Tests pass their mock
/// server per call to avoid env races.
fn base_from(arguments: &serde_json::Value) -> Result<String, String> {
    arguments
        .get("base_url")
        .or_else(|| arguments.get("base"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .map(Ok)
        .unwrap_or_else(bitbucket_base)
}
fn token_from(arguments: &serde_json::Value) -> Result<String, String> {
    arguments
        .get("token")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .map(Ok)
        .unwrap_or_else(bitbucket_token)
}
fn workspace_from(arguments: &serde_json::Value) -> Result<String, String> {
    arguments
        .get("workspace")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .map(Ok)
        .unwrap_or_else(|| {
            std::env::var("BITBUCKET_WORKSPACE")
                .map_err(|_| "missing `workspace` (or set BITBUCKET_WORKSPACE)".to_string())
        })
}
fn repo_from(arguments: &serde_json::Value) -> Result<String, String> {
    arguments
        .get("repo")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "missing `repo` (slug)".to_string())
}
fn issue_id_from(arguments: &serde_json::Value) -> Result<String, String> {
    arguments
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| "missing `id` (issue number)".to_string())
}

fn client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .user_agent("agent-m")
        .build()
        .map_err(|error| format!("http client: {error}"))
}

/// Search issues in a repository: `GET /2.0/repositories/{ws}/{repo}/issues`.
/// Returns `#{id}: {title} ({state})` lines.
fn search(arguments: &str, _cwd: &str) -> Result<String, String> {
    let query: serde_json::Value = serde_json::from_str(arguments).map_err(|e| e.to_string())?;
    let workspace = workspace_from(&query)?;
    let repo = repo_from(&query)?;
    let q = query
        .get("q")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("(state = \"new\" OR state = \"open\")");
    let base = base_from(&query)?;
    let token = token_from(&query)?;
    let response = client()?
        .get(format!(
            "{base}/2.0/repositories/{workspace}/{repo}/issues"
        ))
        .query(&[("q", q), ("pagelen", "25")])
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|error| format!("bitbucket search failed: {error}"))?;
    let status = response.status();
    let body: serde_json::Value = response.json().map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("bitbucket search HTTP {status}: {body}"));
    }
    let issues = body
        .get("values")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = String::new();
    for issue in issues {
        let id = issue
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let title = issue
            .get("title")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("");
        let state = issue
            .get("state")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("?");
        out.push_str(&format!("#{id}: {title} ({state})\n"));
    }
    Ok(out.trim_end().to_string())
}

/// Comment on an issue: `POST /2.0/repositories/{ws}/{repo}/issues/{id}/comments`.
fn comment(arguments: &str, _cwd: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(arguments).map_err(|e| e.to_string())?;
    let workspace = workspace_from(&value)?;
    let repo = repo_from(&value)?;
    let id = issue_id_from(&value)?;
    let body = value
        .get("body")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `body`")?;
    let base = base_from(&value)?;
    let token = token_from(&value)?;
    let response = client()?
        .post(format!(
            "{base}/2.0/repositories/{workspace}/{repo}/issues/{id}/comments"
        ))
        .header("Authorization", format!("Bearer {token}"))
        .json(&serde_json::json!({ "content": { "raw": body } }))
        .send()
        .map_err(|error| format!("bitbucket comment failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("bitbucket comment HTTP {status}"));
    }
    Ok(format!("commented on #{id} in {workspace}/{repo}"))
}

/// Transition an issue: `POST /2.0/repositories/{ws}/{repo}/issues/{id}/changes`.
/// `status` is the target state ("new", "open", "resolved", "on hold", …).
fn transition(arguments: &str, _cwd: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(arguments).map_err(|e| e.to_string())?;
    let workspace = workspace_from(&value)?;
    let repo = repo_from(&value)?;
    let id = issue_id_from(&value)?;
    let status = value
        .get("status")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `status` (target state)")?;
    let base = base_from(&value)?;
    let token = token_from(&value)?;
    let response = client()?
        .post(format!(
            "{base}/2.0/repositories/{workspace}/{repo}/issues/{id}/changes"
        ))
        .header("Authorization", format!("Bearer {token}"))
        .json(&serde_json::json!({ "changes": { "status": { "new": status } } }))
        .send()
        .map_err(|error| format!("bitbucket transition failed: {error}"))?;
    let status_code = response.status();
    if !status_code.is_success() {
        return Err(format!("bitbucket transition HTTP {status_code}"));
    }
    Ok(format!("transitioned #{id} -> {status}"))
}

/// Create a pull request: `POST /2.0/repositories/{ws}/{repo}/pullrequests`.
/// `source`/`destination` are branch names (defaults: source = current branch
/// via `head`, destination = `main`).
fn create_pr(arguments: &str, _cwd: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(arguments).map_err(|e| e.to_string())?;
    let workspace = workspace_from(&value)?;
    let repo = repo_from(&value)?;
    let title = value
        .get("title")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `title`")?;
    let head = value
        .get("head")
        .and_then(serde_json::Value::as_str)
        .ok_or("missing `head` (source branch)")?;
    let base_branch = value
        .get("base")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("main");
    let description = value
        .get("description")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let base = base_from(&value)?;
    let token = token_from(&value)?;
    let response = client()?
        .post(format!(
            "{base}/2.0/repositories/{workspace}/{repo}/pullrequests"
        ))
        .header("Authorization", format!("Bearer {token}"))
        .json(&serde_json::json!({
            "title": title,
            "description": description,
            "source": { "branch": { "name": head } },
            "destination": { "branch": { "name": base_branch } }
        }))
        .send()
        .map_err(|error| format!("bitbucket pr failed: {error}"))?;
    let status = response.status();
    let body: serde_json::Value = response.json().map_err(|e| e.to_string())?;
    if !status.is_success() {
        return Err(format!("bitbucket pr HTTP {status}: {body}"));
    }
    Ok(format!(
        "PR created: {} (#{})",
        body.get("links")
            .and_then(|links| links.get("html"))
            .and_then(|html| html.get("href"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or(""),
        body.get("id").and_then(serde_json::Value::as_u64).unwrap_or(0)
    ))
}

static DEFS: &[ToolDef] = &[
    ToolDef {
        name: "bitbucket-search",
        description: "Search Bitbucket issues in a repository",
        parameters: r#"{"type":"object","properties":{"workspace":{"type":"string"},"repo":{"type":"string"},"q":{"type":"string"},"base":{"type":"string"},"token":{"type":"string"}},"required":["workspace","repo"]}"#,
        execute: search,
    },
    ToolDef {
        name: "bitbucket-comment",
        description: "Add a comment to a Bitbucket issue",
        parameters: r#"{"type":"object","properties":{"workspace":{"type":"string"},"repo":{"type":"string"},"id":{"type":"string"},"body":{"type":"string"},"base":{"type":"string"},"token":{"type":"string"}},"required":["workspace","repo","id","body"]}"#,
        execute: comment,
    },
    ToolDef {
        name: "bitbucket-transition",
        description: "Transition a Bitbucket issue (change status)",
        parameters: r#"{"type":"object","properties":{"workspace":{"type":"string"},"repo":{"type":"string"},"id":{"type":"string"},"status":{"type":"string"},"base":{"type":"string"},"token":{"type":"string"}},"required":["workspace","repo","id","status"]}"#,
        execute: transition,
    },
    ToolDef {
        name: "bitbucket-create-pr",
        description: "Create a pull request",
        parameters: r#"{"type":"object","properties":{"workspace":{"type":"string"},"repo":{"type":"string"},"title":{"type":"string"},"head":{"type":"string"},"base":{"type":"string"},"description":{"type":"string"},"base_url":{"type":"string"},"token":{"type":"string"}},"required":["workspace","repo","title","head"]}"#,
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
        .get_or_init(|| EntryHolder(Box::leak(Box::new(entry("bitbucket", "0.1.0", DEFS)))))
        .0
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn base_uri(server: &MockServer) -> String {
        format!(
            r#"{{"workspace":"acme","repo":"app","base":"{}","token":"fake"}}"#,
            server.uri()
        )
    }

    #[test]
    fn search_parses_issues() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let server = runtime.block_on(MockServer::start());
        runtime.block_on(
            Mock::given(method("GET"))
                .and(path("/2.0/repositories/acme/app/issues"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "values": [
                        { "id": 7, "title": "Fix login", "state": "open" },
                        { "id": 9, "title": "Add tests", "state": "new" }
                    ]
                })))
                .mount(&server),
        );
        let out = search(base_uri(&server).as_str(), ".").expect("search");
        assert!(out.contains("#7: Fix login (open)"), "got: {out}");
        assert!(out.contains("#9: Add tests (new)"), "got: {out}");
    }

    #[test]
    fn comment_posts() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let server = runtime.block_on(MockServer::start());
        runtime.block_on(
            Mock::given(method("POST"))
                .and(path("/2.0/repositories/acme/app/issues/7/comments"))
                .respond_with(ResponseTemplate::new(201))
                .mount(&server),
        );
        let args = format!(
            r#"{{"workspace":"acme","repo":"app","id":"7","body":"done","base":"{}","token":"fake"}}"#,
            server.uri()
        );
        let out = comment(args.as_str(), ".").expect("comment");
        assert!(out.contains("#7"), "got: {out}");
    }

    #[test]
    fn transition_posts() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let server = runtime.block_on(MockServer::start());
        runtime.block_on(
            Mock::given(method("POST"))
                .and(path("/2.0/repositories/acme/app/issues/7/changes"))
                .respond_with(ResponseTemplate::new(200))
                .mount(&server),
        );
        let args = format!(
            r#"{{"workspace":"acme","repo":"app","id":"7","status":"resolved","base":"{}","token":"fake"}}"#,
            server.uri()
        );
        let out = transition(args.as_str(), ".").expect("transition");
        assert!(out.contains("resolved"), "got: {out}");
    }

    #[test]
    fn create_pr_posts() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let server = runtime.block_on(MockServer::start());
        runtime.block_on(
            Mock::given(method("POST"))
                .and(path("/2.0/repositories/acme/app/pullrequests"))
                .respond_with(ResponseTemplate::new(201).set_body_json(serde_json::json!({
                    "id": 42,
                    "links": { "html": { "href": "https://bitbucket.org/acme/app/pull-requests/42" } }
                })))
                .mount(&server),
        );
        let args = format!(
            r#"{{"workspace":"acme","repo":"app","title":"t","head":"fix/login","base":"main","base_url":"{}","token":"fake"}}"#,
            server.uri()
        );
        let out = create_pr(args.as_str(), ".").expect("create pr");
        assert!(out.contains("pull-requests/42"), "got: {out}");
    }
}
