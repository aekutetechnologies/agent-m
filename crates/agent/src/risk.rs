//! Risk hints: cheap, best-effort recognition of tool calls a human should
//! look at. NOT a security boundary. A shell string can hide anything
//! (`eval "$(base64 -d …)"`), so every check here is an accident-catcher for
//! a cooperative model — never containment. The real boundaries are: no tool
//! registered, a human reading the call, and the OS user we run as.
// ponytail: heuristic hints only; the real fix is an OS sandbox around the
// bash tool (sandbox-exec / bubblewrap / container). Do not grow this list
// into an arms race — it cannot be won.

use crate::tool::ToolCallInfo;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// What the host knows about the risk of a call. One instance per session.
#[derive(Debug, Clone, Default)]
pub struct RiskPolicy {
    /// Session working directory: writes resolving outside it are flagged.
    pub cwd: PathBuf,
    /// Tool names the host cannot inspect (plugin tools not marked trusted
    /// at install time). Always flagged.
    pub opaque_tools: Vec<String>,
}

/// The four trust tiers of check.md principle 5 (risk-based permissions).
/// The harness — never the model — decides the tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    /// Read files, search logs: no approval.
    Low,
    /// Workspace writes, benign commands: optional approval.
    Medium,
    /// Destructive-ish changes: approval required.
    High,
    /// Delete data, device writes, opaque tools: always required.
    Critical,
}

impl RiskLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            RiskLevel::Low => "low",
            RiskLevel::Medium => "medium",
            RiskLevel::High => "high",
            RiskLevel::Critical => "critical",
        }
    }
}

/// A risk assessment: the tier plus the human-readable consequence reason.
#[derive(Debug, Clone)]
pub struct RiskAssessment {
    pub level: RiskLevel,
    pub reason: Option<String>,
}

/// A consequence-framing sentence for the approval box ("This will …").
fn consequence(level: RiskLevel, reason: Option<&str>) -> Option<String> {
    let base = match level {
        RiskLevel::Low | RiskLevel::Medium => None,
        RiskLevel::High => Some("This could discard or rewrite work.".to_string()),
        RiskLevel::Critical => Some("This can destroy data or change the host.".to_string()),
    };
    match (base, reason) {
        (Some(mut text), Some(reason)) => {
            text.push_str(&format!(" ({reason})"));
            Some(text)
        }
        (Some(text), None) => Some(text),
        (None, _) => None,
    }
}

impl RiskPolicy {
    /// 4-tier assessment (check.md principle 5). Opaque plugin tools and
    /// destructive shell commands are Critical; workspace-hostile changes are
    /// High; workspace writes and ordinary commands are Medium; reads are Low.
    pub fn assess(&self, call: &ToolCallInfo) -> RiskAssessment {
        if self.opaque_tools.iter().any(|n| n == &call.name) {
            return RiskAssessment {
                level: RiskLevel::Critical,
                reason: Some(format!(
                    "plugin tool `{}` — the host cannot inspect what it does",
                    call.name
                )),
            };
        }
        // Read-only tools are always Low.
        if crate::agent::PLAN_TOOLS.contains(&call.name.as_str()) && call.name != "ask" {
            return RiskAssessment {
                level: RiskLevel::Low,
                reason: None,
            };
        }
        // Any tool with a `command` string is a shell, whatever it is named:
        // bash, the test-runner plugin's `run-tests`, future tools.
        if let Some(command) = call.arguments.get("command").and_then(Value::as_str) {
            if command_is_read_only(command) {
                return RiskAssessment {
                    level: RiskLevel::Low,
                    reason: None,
                };
            }
            let (level, reason) = command_risk(command);
            return RiskAssessment {
                level,
                reason: reason.map(str::to_string),
            };
        }
        // write/edit: the target path, not the tool name.
        if matches!(call.name.as_str(), "write" | "edit")
            && let Some(path) = call.arguments.get("path").and_then(Value::as_str)
        {
            return self.path_assessment(path);
        }
        RiskAssessment {
            level: RiskLevel::Medium,
            reason: None,
        }
    }

    /// Backwards-compatible binary flag: Some(reason) for High/Critical.
    pub fn risk(&self, call: &ToolCallInfo) -> Option<String> {
        let assessment = self.assess(call);
        match assessment.level {
            RiskLevel::High | RiskLevel::Critical => assessment.reason,
            _ => None,
        }
    }

    /// Consequence framing for the approval box (principle 6).
    pub fn consequence(&self, call: &ToolCallInfo) -> Option<String> {
        let assessment = self.assess(call);
        consequence(assessment.level, assessment.reason.as_deref())
    }

    fn path_assessment(&self, path: &str) -> RiskAssessment {
        let target = normalize(&if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            self.cwd.join(path)
        });
        if !target.starts_with(&self.cwd) {
            return RiskAssessment {
                level: RiskLevel::High,
                reason: Some(format!(
                    "writes outside the workspace: {}",
                    target.display()
                )),
            };
        }
        if target.components().any(|c| c.as_os_str() == ".git") {
            return RiskAssessment {
                level: RiskLevel::High,
                reason: Some("writes inside .git (hooks/config run on your next command)".into()),
            };
        }
        RiskAssessment {
            level: RiskLevel::Medium,
            reason: None,
        }
    }
}

/// Lexical .. normalization for path risk checking.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

/// Approximate segmentation: quoting is ignored on purpose (hint, not parser).
fn segments(command: &str) -> impl Iterator<Item = &str> {
    command
        .split(['\n', ';', '|', '&', '(', ')', '`'])
        .map(str::trim)
}

/// Read-only shell commands: auto-approve (Low tier).
fn command_is_read_only(command: &str) -> bool {
    segments(command).all(|segment| {
        let Some(head) = head(segment) else {
            return true;
        };
        match head {
            "ls" | "cat" | "pwd" | "echo" | "true" | "false" | "head" | "tail" | "wc" | "grep"
            | "rg" | "env" | "which" | "printf" | "jq" | "sed" | "awk"
                if !segment.contains(">") =>
            {
                true
            }
            // git reads only (status/log/diff/show) are fine.
            "git" => {
                let rest: Vec<&str> = segment.split_whitespace().skip(1).collect();
                rest.first().is_some_and(|sub| {
                    matches!(
                        *sub,
                        "status" | "log" | "diff" | "show" | "branch" | "remote"
                    )
                })
            }
            _ => false,
        }
    })
}

fn command_risk(command: &str) -> (RiskLevel, Option<&'static str>) {
    // Pipe-to-shell: fetch + a shell interpreter anywhere in the pipeline.
    let heads: Vec<&str> = segments(command).filter_map(head).collect();
    let fetches = heads
        .iter()
        .any(|h| matches!(*h, "curl" | "wget" | "base64"));
    let shells = heads
        .iter()
        .any(|h| matches!(*h, "sh" | "bash" | "zsh" | "dash"));
    if fetches && shells {
        return (
            RiskLevel::High,
            Some("pipes downloaded content into a shell"),
        );
    }

    for segment in segments(command) {
        let mut words = segment
            .split_whitespace()
            .skip_while(|w| w.contains('=') && !w.starts_with('-')); // VAR=x prefixes
        let Some(raw_head) = words.next() else {
            continue;
        };
        // /bin/rm -> rm ; flag obfuscation via expansion is unreadable
        if raw_head.contains('$') {
            return (
                RiskLevel::High,
                Some("the command name is built from a shell expansion"),
            );
        }
        let head = raw_head.rsplit('/').next().unwrap_or(raw_head);
        let rest: Vec<&str> = words.collect();
        let flag = |long: &str, short: char| {
            rest.iter().any(|w| {
                *w == long || (w.starts_with('-') && !w.starts_with("--") && w.contains(short))
            })
        };
        match head {
            "sudo" | "doas" => {
                return (RiskLevel::Critical, Some("runs as another user (sudo)"));
            }
            "eval" => {
                return (
                    RiskLevel::High,
                    Some("eval: the command cannot be inspected"),
                );
            }
            "rm" if flag("--recursive", 'r') || flag("--recursive", 'R') => {
                return (RiskLevel::Critical, Some("recursive delete"));
            }
            "git" => {
                // Subcommand token, not adjacency: catches `git -C /repo reset --hard`.
                if rest.contains(&"reset") && rest.contains(&"--hard") {
                    return (
                        RiskLevel::High,
                        Some("git reset --hard (discards uncommitted work)"),
                    );
                }
                if rest.contains(&"clean") && (flag("--force", 'f') || flag("--force", 'd')) {
                    return (
                        RiskLevel::High,
                        Some("git clean -f (deletes untracked files)"),
                    );
                }
                if rest.contains(&"checkout") && flag("--force", 'f') {
                    return (
                        RiskLevel::High,
                        Some("git checkout --force (discards local changes)"),
                    );
                }
                if rest.contains(&"push")
                    && (flag("--force", 'f') || rest.iter().any(|w| w.starts_with("--force")))
                {
                    return (
                        RiskLevel::High,
                        Some("git push --force (rewrites remote history)"),
                    );
                }
            }
            "chmod" | "chown" if flag("--recursive", 'R') => {
                return (RiskLevel::High, Some("recursive permission change"));
            }
            "find"
                if rest
                    .iter()
                    .any(|w| w.starts_with("-exec") || *w == "-delete") =>
            {
                return (RiskLevel::High, Some("find -exec/-delete"));
            }
            "mkfs" | "fdisk" | "dd" | "shred" | "diskutil" => {
                return (RiskLevel::Critical, Some("disk/device level command"));
            }
            "crontab" | "launchctl" | "systemctl" => {
                return (
                    RiskLevel::High,
                    Some("installs or changes a background job"),
                );
            }
            _ => {}
        }
        // Any device target, redirect or argument, minus the boring ones.
        if segment
            .split(|c: char| c.is_whitespace() || c == '>' || c == '<')
            .any(|w| {
                w.starts_with("/dev/")
                    && !matches!(
                        w,
                        "/dev/null"
                            | "/dev/stdout"
                            | "/dev/stderr"
                            | "/dev/tty"
                            | "/dev/urandom"
                            | "/dev/random"
                            | "/dev/zero"
                    )
            })
        {
            return (RiskLevel::Critical, Some("writes to a device file"));
        }
    }
    (RiskLevel::Medium, None)
}

fn head(segment: &str) -> Option<&str> {
    segment
        .split_whitespace()
        .next()
        .map(|w| w.rsplit('/').next().unwrap_or(w))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(name: &str, arguments: Value) -> ToolCallInfo {
        ToolCallInfo {
            tool_call_id: "t".to_string(),
            name: name.to_string(),
            arguments,
        }
    }

    fn policy() -> RiskPolicy {
        RiskPolicy {
            cwd: PathBuf::from("/work"),
            opaque_tools: vec!["jira-search".to_string()],
        }
    }

    #[test]
    fn four_tier_classification() {
        let policy = policy();
        // Low: read-only tools and read-only shell commands.
        assert_eq!(
            policy
                .assess(&call("read", json!({ "path": "/work/a.rs" })))
                .level,
            RiskLevel::Low
        );
        assert_eq!(
            policy
                .assess(&call("bash", json!({ "command": "ls -la" })))
                .level,
            RiskLevel::Low
        );
        assert_eq!(
            policy
                .assess(&call("bash", json!({ "command": "git status" })))
                .level,
            RiskLevel::Low
        );
        // Medium: workspace writes and ordinary commands.
        assert_eq!(
            policy
                .assess(&call("edit", json!({ "path": "/work/a.rs" })))
                .level,
            RiskLevel::Medium
        );
        assert_eq!(
            policy
                .assess(&call("bash", json!({ "command": "cargo test" })))
                .level,
            RiskLevel::Medium
        );
        // High: workspace-hostile changes.
        assert_eq!(
            policy
                .assess(&call("edit", json!({ "path": "/elsewhere/b.rs" })))
                .level,
            RiskLevel::High
        );
        assert_eq!(
            policy
                .assess(&call("bash", json!({ "command": "git reset --hard HEAD" })))
                .level,
            RiskLevel::High
        );
        // Critical: destructive commands + opaque plugin tools.
        assert_eq!(
            policy
                .assess(&call("bash", json!({ "command": "rm -rf /work/tmp" })))
                .level,
            RiskLevel::Critical
        );
        assert_eq!(
            policy
                .assess(&call("bash", json!({ "command": "sudo rm -f x" })))
                .level,
            RiskLevel::Critical
        );
        assert_eq!(
            policy.assess(&call("jira-search", json!({}))).level,
            RiskLevel::Critical
        );
    }

    #[test]
    fn consequence_framing_exists_for_high_and_critical() {
        let policy = policy();
        let high = policy
            .consequence(&call("bash", json!({ "command": "git reset --hard" })))
            .expect("high consequence");
        assert!(high.contains("discard or rewrite"), "got: {high}");
        let critical = policy
            .consequence(&call("bash", json!({ "command": "rm -rf /work" })))
            .expect("critical consequence");
        assert!(critical.contains("destroy"), "got: {critical}");
        let low = policy.consequence(&call("read", json!({ "path": "/work/a.rs" })));
        assert!(low.is_none(), "low has no consequence framing");
    }
}
