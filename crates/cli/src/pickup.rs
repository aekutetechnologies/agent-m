//! `agent-m pickup`: auto-pick the next assigned open Jira ticket, resolve
//! the repo to work in, and hand it to the flow — the first step of the
//! autonomous SDLC loop.
//!
//! - Ticket source: Jira (`JIRA_URL` + `JIRA_TOKEN` env vars, same as the
//!   jira plugin). Default JQL: assigned to the current user, not closed.
//! - Repo mapping: `~/.agent-m/agent/repos.json` maps a project key to a git
//!   URL (`{ "PROJ": "https://github.com/acme/app.git" }`); a `--repo`
//!   override wins, then the map, then an error.
//! - Isolation: a fresh git worktree is created (reusing
//!   `create_worktree`), the flow runs inside it with `${ticket}` and
//!   `${repo}` seeded, and the ticket is moved to In Progress first.
//! - `--dry-run` prints the pick without doing anything.

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The repo-mapping file name under the agent dir.
pub const REPOS_FILE: &str = "repos.json";

/// Project key → git URL mapping.
pub type RepoMap = HashMap<String, String>;

/// What pickup decided to work on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PickedTicket {
    pub key: String,
    pub summary: String,
    /// git URL or local path of the repo the flow will work in.
    pub repo: String,
}

pub fn repos_path(agent_dir: &Path) -> PathBuf {
    agent_dir.join(REPOS_FILE)
}

/// Load the project→repo map. A missing or unparsable file yields an empty
/// map (resolution then fails with a clear message unless `--repo` is given).
pub fn load_repo_map(agent_dir: &Path) -> RepoMap {
    std::fs::read_to_string(repos_path(agent_dir))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

/// `"PROJ-42"` → `"PROJ"` (project key = prefix before the first `-`).
pub fn project_key(ticket: &str) -> &str {
    ticket.split('-').next().unwrap_or(ticket)
}

/// Resolve the repo for a ticket: explicit override → map lookup → error.
pub fn resolve_repo(
    ticket: &str,
    repos: &RepoMap,
    override_repo: Option<&str>,
) -> Result<String> {
    if let Some(repo) = override_repo.filter(|repo| !repo.is_empty()) {
        return Ok(repo.to_string());
    }
    let project = project_key(ticket);
    repos
        .get(project)
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "no repo mapping for project `{project}` (ticket {ticket}); add it to \
                 {REPOS_FILE} or pass --repo"
            )
        })
}

/// Default JQL: my open, not-yet-in-progress tickets, most recently updated
/// first. Excluding `In Progress` keeps a failed flow's ticket from being
/// re-picked forever and lets concurrent workers never race on the same
/// ticket (the worker transitions it to In Progress before it can be
/// re-picked).
pub fn default_jql() -> String {
    "assignee = currentUser() AND status not in (Done, Closed, Canceled, In Progress) \
     ORDER BY updated DESC"
        .to_string()
}

fn jira_base() -> Result<String> {
    std::env::var("JIRA_URL")
        .map_err(|_| anyhow!("pickup needs JIRA_URL (e.g. https://your.atlassian.net)"))
}
fn jira_token() -> Result<String> {
    std::env::var("JIRA_TOKEN")
        .map_err(|_| anyhow!("pickup needs JIRA_TOKEN (same env as the jira plugin)"))
}

/// Fetch the first matching ticket: `(key, summary)`, or `None` when nothing
/// is assigned to you and open.
pub async fn query_ticket(base: &str, token: &str, jql: &str) -> Result<Option<(String, String)>> {
    let response = reqwest::Client::new()
        .get(format!("{base}/rest/api/3/search"))
        .query(&[("jql", jql), ("fields", "summary"), ("maxResults", "1")])
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .await
        .context("jira search failed")?;
    let status = response.status();
    let body: Value = response.json().await.context("jira search returned bad JSON")?;
    if !status.is_success() {
        return Err(anyhow!("jira search HTTP {status}: {body}"));
    }
    let issue = body
        .get("issues")
        .and_then(Value::as_array)
        .and_then(|issues| issues.first());
    let Some(issue) = issue else {
        return Ok(None);
    };
    let key = issue
        .get("key")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let summary = issue
        .pointer("/fields/summary")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    Ok(Some((key, summary)))
}

/// Move a ticket to a new status (`POST /rest/api/3/issue/{key}/transitions`).
pub async fn transition_ticket(
    base: &str,
    token: &str,
    key: &str,
    transition_id: &str,
) -> Result<()> {
    let response = reqwest::Client::new()
        .post(format!("{base}/rest/api/3/issue/{key}/transitions"))
        .header("Authorization", format!("Bearer {token}"))
        .json(&serde_json::json!({ "transition": { "id": transition_id } }))
        .send()
        .await
        .context("jira transition failed")?;
    let status = response.status();
    if !status.is_success() {
        let body: Value = response.json().await.unwrap_or_default();
        return Err(anyhow!("jira transition HTTP {status}: {body}"));
    }
    Ok(())
}

/// The inputs for one pickup decision.
pub struct PickInputs<'a> {
    pub agent_dir: &'a Path,
    /// Explicit ticket key (skips the Jira query).
    pub ticket: Option<&'a str>,
    /// Explicit repo (skips the map lookup).
    pub repo: Option<&'a str>,
    /// JQL override; defaults to `default_jql()`.
    pub jql: Option<&'a str>,
}

/// Resolve the next ticket and the repo to work in. Pure decision — no side
/// effects — so dry-run and tests share it.
pub async fn pick(inputs: PickInputs<'_>) -> Result<PickedTicket> {
    if let Some(key) = inputs.ticket.filter(|key| !key.is_empty()) {
        let repos = load_repo_map(inputs.agent_dir);
        let repo = resolve_repo(key, &repos, inputs.repo)?;
        return Ok(PickedTicket {
            key: key.to_string(),
            summary: "(explicit --ticket)".to_string(),
            repo,
        });
    }
    let base = jira_base()?;
    let token = jira_token()?;
    let jql = inputs.jql.unwrap_or_default();
    let jql = if jql.is_empty() {
        default_jql()
    } else {
        jql.to_string()
    };
    let Some((key, summary)) = query_ticket(&base, &token, &jql).await? else {
        return Err(anyhow!("no open tickets assigned to you (JQL: {jql})"));
    };
    let repos = load_repo_map(inputs.agent_dir);
    let repo = resolve_repo(&key, &repos, inputs.repo)?;
    Ok(PickedTicket {
        key,
        summary,
        repo,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_key_splits_prefix() {
        assert_eq!(project_key("PROJ-42"), "PROJ");
        assert_eq!(project_key("TEAM-PROJ-7"), "TEAM");
        assert_eq!(project_key("noproject"), "noproject");
    }

    #[test]
    fn default_jql_excludes_in_progress() {
        let jql = default_jql();
        assert!(jql.contains("In Progress"), "{jql}");
        assert!(jql.contains("Done"), "{jql}");
        assert!(jql.contains("ORDER BY updated DESC"), "{jql}");
    }

    #[test]
    fn resolve_repo_override_wins_then_map() {
        let mut repos = RepoMap::new();
        repos.insert("PROJ".to_string(), "https://github.com/acme/app.git".to_string());
        // Override wins even when the map has an entry.
        assert_eq!(
            resolve_repo("PROJ-1", &repos, Some("/local/repo")).unwrap(),
            "/local/repo"
        );
        // Map lookup by project prefix.
        assert_eq!(
            resolve_repo("PROJ-2", &repos, None).unwrap(),
            "https://github.com/acme/app.git"
        );
        // Unknown project → clear error naming the file.
        let err = resolve_repo("OTHER-3", &repos, None).unwrap_err();
        assert!(err.to_string().contains("no repo mapping for project `OTHER`"), "{err}");
        assert!(err.to_string().contains(REPOS_FILE), "{err}");
    }

    #[test]
    fn missing_repo_file_loads_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_repo_map(dir.path()).is_empty());
    }

    #[test]
    fn loads_repo_map_from_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join(REPOS_FILE),
            r#"{ "PROJ": "https://github.com/acme/app.git" }"#,
        )
        .unwrap();
        let repos = load_repo_map(dir.path());
        assert_eq!(
            repos.get("PROJ").map(String::as_str),
            Some("https://github.com/acme/app.git")
        );
    }

    #[tokio::test]
    async fn query_ticket_parses_first_issue() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/rest/api/3/search"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "issues": [
                    { "key": "PROJ-7", "fields": { "summary": "Fix login" } },
                    { "key": "PROJ-9", "fields": { "summary": "Add tests" } }
                ]
            })))
            .mount(&server)
            .await;
        let got = query_ticket(&server.uri(), "fake", "assignee = currentUser()")
            .await
            .expect("query");
        assert_eq!(got, Some(("PROJ-7".to_string(), "Fix login".to_string())));
    }

    #[tokio::test]
    async fn query_ticket_empty_when_no_issues() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .and(wiremock::matchers::path("/rest/api/3/search"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "issues": []
            })))
            .mount(&server)
            .await;
        let got = query_ticket(&server.uri(), "fake", "jql")
            .await
            .expect("query");
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn transition_posts_transition_id() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/rest/api/3/issue/PROJ-7/transitions"))
            .respond_with(wiremock::ResponseTemplate::new(204))
            .mount(&server)
            .await;
        transition_ticket(&server.uri(), "fake", "PROJ-7", "11")
            .await
            .expect("transition");
    }

    /// Local git fixture: init a repo with one commit (needed before
    /// `git worktree add` can branch from it).
    fn init_git_fixture(dir: &Path) {
        for args in [
            vec!["init", "-q", "-b", "main"],
            vec!["config", "user.email", "test@example.com"],
            vec!["config", "user.name", "Test"],
        ] {
            let status = std::process::Command::new("git")
                .args(&args)
                .current_dir(dir)
                .status()
                .expect("git command runs");
            assert!(status.success(), "git {args:?} failed");
        }
        std::fs::write(dir.join("README.md"), "# fixture\n").unwrap();
        let status = std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir)
            .status()
            .unwrap();
        assert!(status.success());
        let status = std::process::Command::new("git")
            .args(["commit", "-q", "-m", "init"])
            .current_dir(dir)
            .status()
            .unwrap();
        assert!(status.success());
    }

    /// End-to-end shape of `agent-m pickup --ticket PROJ-1` with a local git
    /// fixture: resolve the repo from the mapping, then create the isolated
    /// worktree branch the flow will run inside.
    #[tokio::test]
    async fn pick_and_worktree_with_local_git_fixture() {
        let fixture = tempfile::tempdir().unwrap();
        init_git_fixture(fixture.path());

        let agent_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            agent_dir.path().join(REPOS_FILE),
            serde_json::json!({ "PROJ": fixture.path().to_string_lossy() }).to_string(),
        )
        .unwrap();

        // 1. Pick: explicit ticket → mapping resolves the local fixture.
        let picked = pick(PickInputs {
            agent_dir: agent_dir.path(),
            ticket: Some("PROJ-1"),
            repo: None,
            jql: None,
        })
        .await
        .expect("pick");
        assert_eq!(picked.key, "PROJ-1");
        // The mapping stores the fixture path as-is (like a git URL).
        assert_eq!(Path::new(&picked.repo), fixture.path());

        // 2. Worktree: fresh branch checkout under the agent dir — this is
        //    the checkout the flow runs inside (clone step skipped).
        let worktree =
            agent_m_agent::create_worktree(Path::new(&picked.repo), agent_dir.path(), Some(&picked.key))
                .expect("worktree");
        assert!(worktree.is_dir());
        assert!(
            worktree.starts_with(agent_dir.path().canonicalize().unwrap()),
            "worktree under agent dir: {worktree:?}"
        );
        let branch = std::process::Command::new("git")
            .args(["branch", "--show-current"])
            .current_dir(&worktree)
            .output()
            .unwrap();
        let branch = String::from_utf8_lossy(&branch.stdout);
        assert!(
            branch.trim().starts_with("agent-m/PROJ-1-"),
            "branch was {branch:?}"
        );
        // The worktree shares the fixture's content (the repo checkout).
        assert!(worktree.join("README.md").exists());
    }
}
