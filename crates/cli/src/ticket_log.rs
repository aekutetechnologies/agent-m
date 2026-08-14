//! Per-ticket daemon reports (Phase 10): every ticket run by the concurrent
//! pickup supervisor is an independent `agent-m ticket-run` process, and each
//! one appends a JSONL report to `<agent_dir>/tickets/<KEY>.jsonl`. The
//! supervisor survives a ticket's crash (real process isolation) and
//! `agent-m ticket-log <KEY>` tails the report while the ticket runs.
//!
//! Line kinds:
//! - `pickup` — ticket picked, transition + worktree decided
//! - `step`   — one top-level flow step started/ended (live via on_progress)
//! - `verdict`— final result: OK/FAILED, fix rounds, PR link when present

use anyhow::{Context, Result};
use serde_json::Value;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// The report file for one ticket: `<agent_dir>/tickets/<KEY>.jsonl`.
pub fn report_path(agent_dir: &Path, key: &str) -> PathBuf {
    agent_dir.join("tickets").join(format!("{key}.jsonl"))
}

fn tickets_dir(agent_dir: &Path) -> Result<PathBuf> {
    let dir = agent_dir.join("tickets");
    std::fs::create_dir_all(&dir).with_context(|| {
        format!(
            "cannot create ticket reports dir {}",
            dir.display()
        )
    })?;
    #[cfg(unix)]
    {
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    }
    Ok(dir)
}

/// Append one JSON line to the ticket's report (creating the file on first
/// write). Locked down like session logs: reports can contain secrets the
/// model read, so 0700 dir + 0600 file.
pub fn append(agent_dir: &Path, key: &str, value: &Value) -> Result<()> {
    let dir = tickets_dir(agent_dir)?;
    let path = dir.join(format!("{key}.jsonl"));
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("cannot open ticket report {}", path.display()))?;
    #[cfg(unix)]
    {
        let _ = file.set_permissions(std::fs::Permissions::from_mode(0o600));
    }
    let mut line = serde_json::to_string(value)?;
    line.push('\n');
    file.write_all(line.as_bytes())
        .with_context(|| format!("cannot write ticket report {}", path.display()))
}

/// All report lines for a ticket, in append order (empty if no report yet).
pub fn read_lines(agent_dir: &Path, key: &str) -> Result<Vec<String>> {
    let path = report_path(agent_dir, key);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("no ticket report at {}", path.display()))?;
    Ok(raw.lines().map(str::to_string).collect())
}

/// One human-readable line for `agent-m ticket-log` output.
pub fn render_line(line: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(line) else {
        return line.to_string();
    };
    let ts = value.get("ts").and_then(Value::as_str).unwrap_or("");
    let kind = value.get("kind").and_then(Value::as_str).unwrap_or("?");
    let step = value.get("step").and_then(Value::as_str).unwrap_or("");
    match kind {
        "pickup" => {
            let ticket = value.get("ticket").and_then(Value::as_str).unwrap_or("");
            let summary = value.get("summary").and_then(Value::as_str).unwrap_or("");
            format!("{ts} pickup {ticket} — {summary}")
        }
        "step" => {
            let status = value.get("status").and_then(Value::as_str).unwrap_or("");
            format!("{ts} [{status}] {step}")
        }
        "verdict" => {
            let status = value.get("status").and_then(Value::as_str).unwrap_or("");
            let rounds = value.get("fix_rounds").and_then(Value::as_u64).unwrap_or(0);
            let mut out = format!("{ts} verdict {status} (fix rounds: {rounds})");
            if let Some(pr) = value.get("pr").and_then(Value::as_str)
                && !pr.is_empty()
            {
                out.push_str(&format!(" 🔗 {pr}"));
            }
            out
        }
        _ => line.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn append_and_read_round_trip() {
        let dir = TempDir::new().expect("tempdir");
        append(
            dir.path(),
            "PROJ-42",
            &serde_json::json!({ "ts": "t1", "kind": "pickup", "ticket": "PROJ-42", "summary": "Fix the bug" }),
        )
        .expect("append");
        append(
            dir.path(),
            "PROJ-42",
            &serde_json::json!({ "ts": "t2", "kind": "step", "step": "verify", "status": "running" }),
        )
        .expect("append");
        let lines = read_lines(dir.path(), "PROJ-42").expect("read");
        assert_eq!(lines.len(), 2, "both lines must round-trip");
        assert!(lines[0].contains("\"kind\":\"pickup\""), "got: {}", lines[0]);
        assert!(lines[1].contains("\"status\":\"running\""), "got: {}", lines[1]);
    }

    #[test]
    fn read_lines_missing_report_errors() {
        let dir = TempDir::new().expect("tempdir");
        let err = read_lines(dir.path(), "PROJ-1").expect_err("no report yet");
        assert!(
            err.to_string().contains("no ticket report"),
            "got: {err}"
        );
    }

    #[test]
    fn render_verdict_includes_pr_when_present() {
        let line = serde_json::json!({
            "ts": "t3",
            "kind": "verdict",
            "status": "OK",
            "fix_rounds": 2,
            "pr": "https://github.com/o/r/pull/7"
        });
        let rendered = render_line(&line.to_string());
        assert!(rendered.contains("verdict OK"), "got: {rendered}");
        assert!(rendered.contains("fix rounds: 2"), "got: {rendered}");
        assert!(rendered.contains("pull/7"), "got: {rendered}");
    }

    #[test]
    fn render_step_line_shows_status() {
        let line = serde_json::json!({
            "ts": "t4",
            "kind": "step",
            "step": "github-create-pr",
            "status": "failed"
        });
        let rendered = render_line(&line.to_string());
        assert!(rendered.contains("[failed] github-create-pr"), "got: {rendered}");
    }
}
