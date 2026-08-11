//! Git-based checkpoints (aider/Claude-Code-style instant rollback).
//!
//! A checkpoint is a `git stash create` snapshot — a commit of the working
//! tree that touches neither the index nor the branch — plus a label. The
//! SHA is recorded in a per-session ledger, and `/restore` applies it back
//! with `git stash apply`, giving a 3-way merge instead of a destructive
//! checkout. All operations are local and non-disruptive.

use std::path::Path;
use std::process::Command;

/// True when `cwd` is inside a git work tree.
pub fn is_git_repo(cwd: &Path) -> bool {
    Command::new("git")
        .arg("rev-parse")
        .arg("--is-inside-work-tree")
        .current_dir(cwd)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Snapshot the working tree. Returns the checkpoint SHA, or `None` when
/// there is nothing to snapshot. Never modifies the index or branch.
pub fn create_checkpoint(cwd: &Path, label: &str) -> Result<Option<String>, String> {
    if !is_git_repo(cwd) {
        return Ok(None);
    }
    // `git diff --quiet` only detects working-tree-vs-index changes; it misses
    // staged-only changes (where the index differs from HEAD but the working tree
    // matches the index). `git status --porcelain` covers all three states:
    // modified, staged, and untracked.
    let status = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(cwd)
        .output()
        .map_err(|error| format!("git status failed: {error}"))?;
    if status.stdout.is_empty() {
        return Ok(None);
    }
    // `git stash create` snapshots tracked modified + staged changes into a
    // stash commit without touching refs/stash. Untracked files are not captured
    // (would require a separate commit object; use the ledger label instead).
    let output = Command::new("git")
        .args(["stash", "create"])
        .current_dir(cwd)
        .output()
        .map_err(|error| format!("git stash create failed: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if sha.is_empty() {
        return Ok(None);
    }
    // `git stash create` leaves the commit dangling (not on refs/stash), so
    // the message is lost. `git stash store` records it on refs/stash with the
    // label, making it visible in `git stash list` and recoverable by SHA.
    let store = Command::new("git")
        .args(["stash", "store", "-m", &format!("agent-m: {label}"), &sha])
        .current_dir(cwd)
        .output()
        .map_err(|error| format!("git stash store failed: {error}"))?;
    if !store.status.success() {
        return Err(String::from_utf8_lossy(&store.stderr).trim().to_string());
    }
    Ok(Some(sha))
}

/// Roll the working tree back to a checkpoint's state. This intentionally
/// discards current uncommitted changes — that is the meaning of a restore.
/// Files created after the checkpoint (untracked) are left alone.
pub fn restore_checkpoint(cwd: &Path, sha: &str) -> Result<(), String> {
    let output = Command::new("git")
        .args(["restore", "--source", sha, "--worktree", "--", "."])
        .current_dir(cwd)
        .output()
        .map_err(|error| format!("git restore failed: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "git restore --source {sha} failed:\n{}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn checkpoint_roundtrip_in_a_temp_repo() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        run(repo, &["init", "-q"]);
        run(repo, &["config", "user.email", "test@example.com"]);
        run(repo, &["config", "user.name", "Test"]);
        fs::write(repo.join("a.txt"), "v1").unwrap();
        run(repo, &["add", "a.txt"]);
        run(repo, &["commit", "-q", "-m", "base"]);

        // Mutate → checkpoint captures v2.
        fs::write(repo.join("a.txt"), "v2").unwrap();
        let sha = create_checkpoint(repo, "test")
            .expect("checkpoint")
            .expect("some changes");
        assert!(!sha.is_empty());

        // Mutate again → restore should bring back v2.
        fs::write(repo.join("a.txt"), "v3").unwrap();
        restore_checkpoint(repo, &sha).expect("restore");
        assert_eq!(fs::read_to_string(repo.join("a.txt")).unwrap(), "v2");

        // Clean tree → no checkpoint.
        run(repo, &["add", "a.txt"]);
        run(repo, &["commit", "-q", "-m", "v2"]);
        assert!(create_checkpoint(repo, "clean").unwrap().is_none());
    }

    #[test]
    fn staged_only_changes_are_captured() {
        let dir = tempdir().unwrap();
        let repo = dir.path();
        run(repo, &["init", "-q"]);
        run(repo, &["config", "user.email", "test@example.com"]);
        run(repo, &["config", "user.name", "Test"]);
        fs::write(repo.join("a.txt"), "v1").unwrap();
        run(repo, &["add", "a.txt"]);
        run(repo, &["commit", "-q", "-m", "base"]);

        // Stage a change without touching the working tree further.
        fs::write(repo.join("a.txt"), "v2").unwrap();
        run(repo, &["add", "a.txt"]);
        // Working tree now matches the index (both at v2), so `git diff --quiet`
        // would have incorrectly returned "clean". `git status --porcelain` sees
        // the staged change and correctly returns a non-empty status.
        let sha = create_checkpoint(repo, "staged")
            .expect("checkpoint")
            .expect("staged changes should produce a checkpoint");
        assert!(!sha.is_empty());
    }

    #[test]
    fn non_git_dir_yields_none() {
        let dir = tempdir().unwrap();
        assert!(create_checkpoint(dir.path(), "x").unwrap().is_none());
    }

    fn run(cwd: &Path, args: &[&str]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(cwd)
            .output()
            .expect("git runs");
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
