//! OS-level sandbox wrappers for the bash tool.
//!
//! Opt-in via `AGENT_M_SANDBOX=1`. Adds a kernel-enforced filesystem write
//! boundary around shell commands: writes outside `cwd` are denied at the OS
//! level, complementing the heuristic risk gate in `agent-m-agent::risk`.
//!
//! Supported platforms:
//! - macOS: `sandbox-exec -p <profile>` (Seatbelt / SBPL). Deprecated but
//!   present through macOS 15. No extra deps.
//! - Linux: Landlock via the `pre_exec` hook + `landlock` crate (kernel ≥ 5.13).
//!   Falls back to `bwrap` (bubblewrap) when Landlock is unsupported.
//! - Other: falls through to no sandbox.

use std::path::Path;

const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";
#[cfg(target_os = "linux")]
const BWRAP: &str = "bwrap";

/// Returns `true` when the caller requested OS-level sandboxing.
pub fn sandbox_enabled() -> bool {
    matches!(
        std::env::var("AGENT_M_SANDBOX").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// Build the command that will run `shell -c command`, optionally wrapped in an
/// OS sandbox. The returned command has no `current_dir`, `stdout`, or `stderr`
/// set — callers configure those as usual.
pub fn sandboxed_command(cwd: &Path, shell: &str, command: &str) -> tokio::process::Command {
    #[cfg(target_os = "macos")]
    if sandbox_enabled()
        && let Some(cmd) = macos_sandbox_exec(cwd, shell, command)
    {
        return cmd;
    }
    #[cfg(target_os = "linux")]
    if sandbox_enabled() {
        if let Some(cmd) = linux_landlock(cwd, shell, command) {
            return cmd;
        }
        if let Some(cmd) = bubblewrap(cwd, shell, command) {
            return cmd;
        }
    }
    let mut cmd = tokio::process::Command::new(shell);
    cmd.arg("-c").arg(command);
    cmd
}

/// Wrap the command in macOS `sandbox-exec` with an SBPL profile that:
/// - allows all actions by default (reads, network, process execution),
/// - denies all file writes outside `cwd` and `/tmp`,
/// - additionally denies writes into `cwd/.git`.
///
/// Returns `None` when `sandbox-exec` is not present (e.g. non-Apple build).
#[cfg(target_os = "macos")]
fn macos_sandbox_exec(cwd: &Path, shell: &str, command: &str) -> Option<tokio::process::Command> {
    if !Path::new(SANDBOX_EXEC).exists() {
        return None;
    }
    // Canonicalize to resolve symlinks (e.g. /tmp → /private/tmp on macOS).
    let real_cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    let cwd_str = real_cwd.to_string_lossy();

    // A `"`, `\` or newline in the path would break the SBPL profile string
    // (fail-closed: sandbox-exec fails to spawn — but reject explicitly so
    // the caller gets a clear error instead of silently no sandbox).
    if cwd_str.contains(['"', '\\', '\n']) {
        return None;
    }

    // SBPL: most-specific subpath match wins, so the .git deny overrides the
    // parent cwd allow even though it appears after it.
    let profile = format!(
        "(version 1)\n\
         (allow default)\n\
         (deny file-write*)\n\
         (allow file-write* (subpath \"/private/tmp\"))\n\
         (allow file-write* (subpath \"/var/folders\"))\n\
         (allow file-write* (subpath \"{cwd_str}\"))\n\
         (deny file-write* (subpath \"{cwd_str}/.git\"))"
    );

    let mut cmd = tokio::process::Command::new(SANDBOX_EXEC);
    cmd.args(["-p", &profile, shell, "-c", command]);
    Some(cmd)
}

/// Wrap the command in a Linux Landlock sandbox via the `pre_exec` hook. The
/// closure runs in the child after `fork()` and before `exec()`, where it
/// builds a ruleset that:
/// - handles all filesystem access rights,
/// - allows read access to the whole filesystem (so the shell and tools work),
/// - allows write access only under `cwd` (and `/tmp`),
/// - denies writes into `cwd/.git`.
///
/// Returns `None` when Landlock is unsupported (kernel < 5.13) or the ruleset
/// cannot be built, so the caller can fall back to bubblewrap.
#[cfg(target_os = "linux")]
fn linux_landlock(cwd: &Path, shell: &str, command: &str) -> Option<tokio::process::Command> {
    use std::os::unix::process::CommandExt;

    // Detect support up front; skip gracefully on older kernels.
    let abi = landlock::ABI::V1::try_new().ok()?;
    let real_cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    let cwd_str = real_cwd.to_string_lossy().into_owned();

    let mut cmd = tokio::process::Command::new(shell);
    cmd.arg("-c").arg(command);
    // Safety: the closure runs in the child after fork(), before exec(), in a
    // single-threaded context. It only calls Landlock syscalls via the crate.
    unsafe {
        cmd.pre_exec(move || {
            let read = landlock::AccessFs::from_read(abi);
            let write = landlock::AccessFs::from_write(abi);
            let status = landlock::Ruleset::default()
                .handle_access(read | write)
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .create()
                .map_err(|e| std::io::Error::other(e.to_string()))?
                // Read access to the whole filesystem.
                .add_rule(landlock::PathBeneath::new(
                    landlock::PathFd::new("/").map_err(|e| std::io::Error::other(e.to_string()))?,
                    read,
                ))
                .map_err(|e| std::io::Error::other(e.to_string()))?
                // Write access under cwd and /tmp.
                .add_rule(landlock::PathBeneath::new(
                    landlock::PathFd::new(&cwd_str)
                        .map_err(|e| std::io::Error::other(e.to_string()))?,
                    write,
                ))
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .add_rule(landlock::PathBeneath::new(
                    landlock::PathFd::new("/tmp")
                        .map_err(|e| std::io::Error::other(e.to_string()))?,
                    write,
                ))
                .map_err(|e| std::io::Error::other(e.to_string()))?
                .restrict_self()
                .map_err(|e| std::io::Error::other(e.to_string()))?;
            // If Landlock wasn't actually enforced, refuse to run unsandboxed.
            if !matches!(status.ruleset, landlock::RulesetStatus::FullyEnforced) {
                return Err(std::io::Error::other(
                    "Landlock not fully enforced; refusing to run unsandboxed",
                ));
            }
            Ok(())
        });
    }
    Some(cmd)
}

/// Wrap the command in bubblewrap (`bwrap`), a user-namespace sandbox. Used as
/// a fallback when Landlock is unsupported. Returns `None` when `bwrap` is not
/// on `PATH`.
#[cfg(target_os = "linux")]
fn bubblewrap(cwd: &Path, shell: &str, command: &str) -> Option<tokio::process::Command> {
    let real_cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    let cwd_str = real_cwd.to_string_lossy().into_owned();
    let mut cmd = tokio::process::Command::new(BWRAP);
    cmd.args([
        "--bind",
        &cwd_str,
        &cwd_str,
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--",
        shell,
        "-c",
        command,
    ]);
    Some(cmd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_flag_parsing() {
        // These run sequentially within the test binary; no parallel interference.
        unsafe { std::env::set_var("AGENT_M_SANDBOX", "1") };
        assert!(sandbox_enabled());
        unsafe { std::env::set_var("AGENT_M_SANDBOX", "true") };
        assert!(sandbox_enabled());
        unsafe { std::env::set_var("AGENT_M_SANDBOX", "0") };
        assert!(!sandbox_enabled());
        unsafe { std::env::set_var("AGENT_M_SANDBOX", "false") };
        assert!(!sandbox_enabled());
        unsafe { std::env::remove_var("AGENT_M_SANDBOX") };
    }
}
