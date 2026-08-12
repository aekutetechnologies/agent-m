//! Git worktree isolation for parallel sessions (Xirp-style).
//!
//! Each `--worktree` session gets its own `agent-m/<slug>-<ts>` branch and a
//! detached checkout under `<agent_dir>/worktrees/`, so many agent sessions
//! can work on the same repository without stepping on each other. All
//! operations are plain `git worktree` calls — no state of the main checkout
//! is touched.

use std::path::{Path, PathBuf};
use std::process::Command;

fn run(cwd: &Path, args: &[&str]) -> Result<std::process::Output, String> {
    Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .map_err(|error| format!("git {args:?} failed: {error}"))
}

/// Create a fresh worktree + branch for a new session. Returns the new
/// directory, or a clear error when `cwd` is not a git repository.
pub fn create_worktree(
    cwd: &Path,
    agent_dir: &Path,
    name: Option<&str>,
) -> Result<PathBuf, String> {
    if !crate::checkpoint::is_git_repo(cwd) {
        return Err("--worktree requires a git repository".to_string());
    }
    let slug = name
        .map(sanitize_slug)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| repo_slug(cwd));
    let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let branch = format!("agent-m/{slug}-{timestamp}");
    let dir = agent_dir
        .join("worktrees")
        .join(format!("{slug}-{timestamp}"));
    let output = run(
        cwd,
        &[
            "worktree",
            "add",
            "-b",
            &branch,
            dir.to_str().unwrap_or_default(),
        ],
    )?;
    if !output.status.success() {
        return Err(format!(
            "git worktree add failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    // `git worktree list` reports canonical paths (e.g. /private/var/… on
    // macOS, where /var is a symlink) — return the same form so callers can
    // compare and cd into it consistently.
    Ok(dir.canonicalize().unwrap_or(dir))
}

/// The absolute paths of every worktree in the repository containing `cwd`
/// (porcelain: `worktree <path>` lines).
pub fn list_worktrees(cwd: &Path) -> Vec<String> {
    run(cwd, &["worktree", "list", "--porcelain"])
        .ok()
        .map(|output| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter(|line| line.starts_with("worktree "))
                .map(|line| line.trim_start_matches("worktree ").to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// Remove the worktree `cwd` is inside. Refuses to remove the main checkout.
pub fn remove_worktree(cwd: &Path) -> Result<String, String> {
    let top = String::from_utf8_lossy(&run(cwd, &["rev-parse", "--show-toplevel"])?.stdout)
        .trim()
        .to_string();
    let worktrees = list_worktrees(cwd);
    let is_main = worktrees
        .first()
        .map(|first| *first == top)
        .unwrap_or(false);
    if is_main || worktrees.len() == 1 {
        return Err(
            "cannot remove the main worktree — use `/sessions` or plain git instead".to_string(),
        );
    }
    let output = run(cwd, &["worktree", "remove", &top])?;
    if !output.status.success() {
        return Err(format!(
            "git worktree remove failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(top)
}

fn sanitize_slug(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect()
}

fn repo_slug(cwd: &Path) -> String {
    let top = run(cwd, &["rev-parse", "--show-toplevel"])
        .ok()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_default();
    top.rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("repo")
        .to_string()
}

#[cfg(test)]
mod worktree_tests {
    use super::*;

    fn init_repo(dir: &Path) {
        Command::new("git")
            .args(["init", "-q", "-b", "main"])
            .current_dir(dir)
            .status()
            .unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@example.com"])
            .current_dir(dir)
            .status()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir)
            .status()
            .unwrap();
        std::fs::write(dir.join("README.md"), "# repo\n").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(dir)
            .status()
            .unwrap();
        Command::new("git")
            .args(["commit", "-q", "-m", "init"])
            .current_dir(dir)
            .status()
            .unwrap();
    }

    #[test]
    fn create_list_and_remove_worktree() {
        let repo = tempfile::tempdir().unwrap();
        let agent = tempfile::tempdir().unwrap();
        init_repo(repo.path());

        let dir = create_worktree(repo.path(), agent.path(), Some("fix-auth"))
            .expect("worktree creation must succeed");
        assert!(dir.is_dir());
        assert!(
            dir.starts_with(agent.path().canonicalize().unwrap()),
            "worktree under agent dir: {dir:?}"
        );

        // The worktree is on its own agent-m/<slug>-<ts> branch.
        let branch = Command::new("git")
            .args(["branch", "--show-current"])
            .current_dir(&dir)
            .output()
            .unwrap();
        let branch = String::from_utf8_lossy(&branch.stdout);
        assert!(
            branch.trim().starts_with("agent-m/fix-auth-"),
            "branch was {branch:?}"
        );

        // The main checkout keeps its own branch untouched.
        let main_branch = Command::new("git")
            .args(["branch", "--show-current"])
            .current_dir(repo.path())
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&main_branch.stdout).trim(), "main");

        let listed = list_worktrees(repo.path());
        assert!(listed.len() >= 2, "main + worktree listed: {listed:?}");
        assert!(
            listed.iter().any(|p| Path::new(p) == dir),
            "worktree {dir:?} not in list {listed:?}"
        );

        let removed = remove_worktree(&dir).expect("remove must succeed");
        assert_eq!(Path::new(&removed), dir);
        assert!(!dir.exists(), "worktree dir removed");
    }

    #[test]
    fn refuses_non_repo_and_main_worktree() {
        let not_a_repo = tempfile::tempdir().unwrap();
        let agent = tempfile::tempdir().unwrap();
        let error = create_worktree(not_a_repo.path(), agent.path(), None).unwrap_err();
        assert!(error.contains("requires a git repository"), "{error}");

        let repo = tempfile::tempdir().unwrap();
        init_repo(repo.path());
        let error = remove_worktree(repo.path()).unwrap_err();
        assert!(error.contains("main worktree"), "{error}");
    }
}
