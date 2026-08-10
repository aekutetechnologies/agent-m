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

impl RiskPolicy {
    pub fn risk(&self, call: &ToolCallInfo) -> Option<String> {
        if self.opaque_tools.iter().any(|n| n == &call.name) {
            return Some(format!(
                "plugin tool `{}` — the host cannot inspect what it does",
                call.name
            ));
        }
        // Any tool with a `command` string is a shell, whatever it is named:
        // bash, the test-runner plugin's `run-tests`, future tools.
        if let Some(command) = call.arguments.get("command").and_then(Value::as_str)
            && let Some(reason) = command_risk(command)
        {
            return Some(reason.to_string());
        }
        // write/edit: the target path, not the tool name.
        if matches!(call.name.as_str(), "write" | "edit")
            && let Some(path) = call.arguments.get("path").and_then(Value::as_str)
        {
            return self.path_risk(path);
        }
        None
    }

    fn path_risk(&self, path: &str) -> Option<String> {
        let target = normalize(&if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            self.cwd.join(path)
        });
        if !target.starts_with(&self.cwd) {
            return Some(format!(
                "writes outside the workspace: {}",
                target.display()
            ));
        }
        if target.components().any(|c| c.as_os_str() == ".git") {
            return Some("writes inside .git (hooks/config run on your next command)".into());
        }
        None
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

fn command_risk(command: &str) -> Option<&'static str> {
    // Pipe-to-shell: fetch + a shell interpreter anywhere in the pipeline.
    let heads: Vec<&str> = segments(command).filter_map(head).collect();
    let fetches = heads
        .iter()
        .any(|h| matches!(*h, "curl" | "wget" | "base64"));
    let shells = heads
        .iter()
        .any(|h| matches!(*h, "sh" | "bash" | "zsh" | "dash"));
    if fetches && shells {
        return Some("pipes downloaded content into a shell");
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
            return Some("the command name is built from a shell expansion");
        }
        let head = raw_head.rsplit('/').next().unwrap_or(raw_head);
        let rest: Vec<&str> = words.collect();
        let flag = |long: &str, short: char| {
            rest.iter().any(|w| {
                *w == long || (w.starts_with('-') && !w.starts_with("--") && w.contains(short))
            })
        };
        match head {
            "sudo" | "doas" => return Some("runs as another user (sudo)"),
            "eval" => return Some("eval: the command cannot be inspected"),
            "rm" if flag("--recursive", 'r') || flag("--recursive", 'R') => {
                return Some("recursive delete");
            }
            "git" => {
                // Subcommand token, not adjacency: catches `git -C /repo reset --hard`.
                if rest.contains(&"reset") && rest.contains(&"--hard") {
                    return Some("git reset --hard (discards uncommitted work)");
                }
                if rest.contains(&"clean") && (flag("--force", 'f') || flag("--force", 'd')) {
                    return Some("git clean -f (deletes untracked files)");
                }
                if rest.contains(&"checkout") && flag("--force", 'f') {
                    return Some("git checkout --force (discards local changes)");
                }
                if rest.contains(&"push")
                    && (flag("--force", 'f') || rest.iter().any(|w| w.starts_with("--force")))
                {
                    return Some("git push --force (rewrites remote history)");
                }
            }
            "chmod" | "chown" if flag("--recursive", 'R') => {
                return Some("recursive permission change");
            }
            "find"
                if rest
                    .iter()
                    .any(|w| w.starts_with("-exec") || *w == "-delete") =>
            {
                return Some("find -exec/-delete");
            }
            "mkfs" | "fdisk" | "dd" | "shred" | "diskutil" => {
                return Some("disk/device level command");
            }
            "crontab" | "launchctl" | "systemctl" => {
                return Some("installs or changes a background job");
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
            return Some("writes to a device file");
        }
    }
    None
}

fn head(segment: &str) -> Option<&str> {
    segment
        .split_whitespace()
        .next()
        .map(|w| w.rsplit('/').next().unwrap_or(w))
}
