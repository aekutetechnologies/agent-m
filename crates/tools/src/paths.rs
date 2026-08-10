//! Filesystem containment for the file tools.
//!
//! resolve_path: one chokepoint for all file tools, proving a path is inside
//! an allowed root before any tool opens it. Lexical .. normalization +
//! canonicalizing the deepest existing ancestor means containment works even
//! for files that don't exist yet (write/edit).

use agent_m_agent::ToolError;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

/// Extra roots the user explicitly approved (`--allow-path`, `settings.allowedPaths`).
static ALLOWED: OnceLock<Vec<PathBuf>> = OnceLock::new();

/// Called once at startup by the CLI. Unresolvable paths are dropped.
pub fn set_allowed_paths(paths: impl IntoIterator<Item = PathBuf>) {
    let _ = ALLOWED.set(
        paths
            .into_iter()
            .filter_map(|p| p.canonicalize().ok())
            .collect(),
    );
}

/// cwd plus user-approved roots, all canonical. `AGENT_M_ALLOW_PATH` (`:`-separated)
/// is read per call so subprocesses and tests need no wiring.
fn roots(cwd: &Path) -> Vec<PathBuf> {
    let mut roots = vec![cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf())];
    roots.extend(ALLOWED.get().cloned().unwrap_or_default());
    if let Ok(list) = std::env::var("AGENT_M_ALLOW_PATH") {
        roots.extend(
            list.split(':')
                .filter(|s| !s.is_empty())
                .filter_map(|s| PathBuf::from(s).canonicalize().ok()),
        );
    }
    roots
}

/// Resolve a tool argument path and prove it is inside an allowed root.
/// The returned path is symlink-free, so callers may open it directly.
pub fn resolve_path(tool: &str, cwd: &Path, arg: &str) -> Result<PathBuf, ToolError> {
    let roots = roots(cwd);
    // roots[0] is the canonical cwd, so `lexical` is always absolute below.
    let joined = if Path::new(arg).is_absolute() {
        PathBuf::from(arg)
    } else {
        roots[0].join(arg)
    };
    let real = canonical_with_tail(&lexical_normalize(&joined));

    if !roots.iter().any(|root| real.starts_with(root)) {
        // Never echo `real`: it would confirm the symlink target / home layout.
        return Err(ToolError::failed(
            tool,
            format!(
                "path `{arg}` is outside the working directory ({}); agent-m's file tools only \
                 access paths inside it (the user can pass --allow-path <dir> to permit another)",
                roots[0].display()
            ),
        ));
    }
    if let Some(rule) = sensitive_rule(&real, &roots) {
        return Err(ToolError::failed(
            tool,
            format!(
                "path `{arg}` is blocked by agent-m's sensitive-path rules ({rule}); \
                 the user must pass --allow-path for that exact path to permit it"
            ),
        ));
    }
    Ok(real)
}

/// Resolve `.` and `..` textually, without touching the filesystem, so a
/// symlinked parent cannot make `..` mean something other than it reads as.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            } // no-op at root: `/..` == `/`
            other => out.push(other),
        }
    }
    out
}

/// Canonicalize the deepest existing ancestor and re-attach the missing tail,
/// so a file that does not exist yet (`write`) is still containment-checked.
/// After `lexical_normalize` every tail component is Normal, so nothing can
/// re-enter the path with `..` after the canonical prefix.
fn canonical_with_tail(lexical: &Path) -> PathBuf {
    let mut tail: Vec<&std::ffi::OsStr> = Vec::new();
    let mut probe: &Path = lexical;
    loop {
        if let Ok(mut real) = probe.canonicalize() {
            for part in tail.iter().rev() {
                real.push(part);
            }
            return real;
        }
        match (probe.parent(), probe.file_name()) {
            (Some(parent), Some(name)) => {
                tail.push(name);
                probe = parent;
            }
            // Nothing canonicalizes: return the lexical path, which fails the
            // containment check below (fail closed).
            _ => return lexical.to_path_buf(),
        }
    }
}

// ponytail: canonicalize-then-open has a TOCTOU window. Tool calls run
// sequentially and `bash` needs no race, so there is no in-design attacker;
// upgrade to openat2(RESOLVE_BENEATH)/O_NOFOLLOW only if tools ever run
// concurrently or bash gets sandboxed.

/// Denied even inside an allowed root. Matched against the canonical path.
const SENSITIVE_DIRS: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".aws",
    ".kube",
    ".docker",
    ".password-store",
];
const SENSITIVE_NAMES: &[&str] = &[
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    ".htpasswd",
    "credentials",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
];
const SENSITIVE_SUFFIXES: &[&str] = &[".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"];

/// `Some(rule)` when `real` must not be touched. An `--allow-path` entry naming
/// the path *exactly* is explicit user consent and lifts every rule except the
/// agent's own data directory, which holds the API key.
fn sensitive_rule(real: &Path, roots: &[PathBuf]) -> Option<&'static str> {
    // Agent's own directory is non-overridable.
    let agent = agent_dir();
    for dir in [
        agent.canonicalize().ok(),
        Some(agent.clone()),
        dirs::home_dir().map(|h| h.join(".agent-m")),
    ]
    .into_iter()
    .flatten()
    {
        if real.starts_with(&dir) {
            return Some("agent data directory (contains the API key)");
        }
    }

    // Exact --allow-path opt-in lifts the remaining rules.
    if roots.iter().any(|root| root == real) {
        return None;
    }

    let name = real.file_name()?.to_string_lossy().to_ascii_lowercase();
    if name.starts_with(".env") {
        return Some("env file");
    } // .env, .env.local, …
    if SENSITIVE_NAMES.contains(&name.as_str()) {
        return Some("credential file");
    }
    if SENSITIVE_SUFFIXES.iter().any(|s| name.ends_with(s)) {
        return Some("private key file");
    }
    if name == "config" && real.parent()?.file_name()? == ".git" {
        return Some("git config (may embed access tokens)");
    }
    if real
        .components()
        .any(|c| SENSITIVE_DIRS.contains(&&*c.as_os_str().to_string_lossy()))
    {
        return Some("credential directory");
    }
    None
}

/// Same rules for directory walks (no allow-path override — a walk never has
/// an exact user opt-in for the file it stumbled onto).
pub(crate) fn is_sensitive(path: &Path) -> bool {
    sensitive_rule(path, &[]).is_some()
}

/// ripgrep exclude globs mirroring the denylist, derived from the same lists.
pub(crate) fn sensitive_globs() -> Vec<String> {
    let mut globs = vec!["!.env*".into(), "!.git/config".into()];
    globs.extend(SENSITIVE_NAMES.iter().map(|n| format!("!{n}")));
    globs.extend(SENSITIVE_SUFFIXES.iter().map(|s| format!("!*{s}")));
    globs.extend(SENSITIVE_DIRS.iter().map(|d| format!("!{d}/**")));
    globs
}

fn agent_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("AGENT_M_DIR") {
        return PathBuf::from(dir);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".agent-m")
}
