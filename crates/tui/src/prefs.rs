//! check.md principle 11: learn user preferences from behavior and reflect
//! them back to the model as a static "Known preferences" prompt block.
//!
//! Signals (all observed by the harness, never inferred by the model):
//! - `!command` shell usage: the command family (first token) frequency.
//! - `/undo` usage: the user is reverting the agent's edits.
//!
//! The prompt block is rebuilt only when the file changes, so the byte-stable
//! prefix cache is preserved within a session.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Preferences {
    /// First-token counts for `!command` invocations (e.g. git, cargo, bun).
    #[serde(default)]
    pub command_usage: BTreeMap<String, u64>,
    /// How many times the user reverted a file with `/undo`.
    #[serde(default)]
    pub undos: u64,
    #[serde(default)]
    pub updated_at: Option<String>,
}

/// The path of the preferences file (`<agent_dir>/preferences.json`).
pub fn preferences_path(agent_dir: &Path) -> PathBuf {
    agent_dir.join("preferences.json")
}

pub fn load(agent_dir: &Path) -> Preferences {
    std::fs::read_to_string(preferences_path(agent_dir))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn save(agent_dir: &Path, prefs: &Preferences) {
    if let Ok(text) = serde_json::to_string(prefs) {
        let _ = std::fs::write(preferences_path(agent_dir), text);
    }
}

/// Record a `!command` invocation, splitting the first token as the family.
pub fn record_command(agent_dir: &Path, command: &str) {
    let family = command.split_whitespace().next().unwrap_or("?").to_string();
    let mut prefs = load(agent_dir);
    *prefs.command_usage.entry(family).or_insert(0) += 1;
    prefs.updated_at = Some(crate::sessions::now_iso());
    save(agent_dir, &prefs);
}

/// Record an `/undo`.
pub fn record_undo(agent_dir: &Path) {
    let mut prefs = load(agent_dir);
    prefs.undos += 1;
    prefs.updated_at = Some(crate::sessions::now_iso());
    save(agent_dir, &prefs);
}

/// The static system-prompt block. Empty when there is nothing worth
/// stating. Rebuilt only when preferences change (byte-stable in-session).
pub fn prompt_block(prefs: &Preferences) -> String {
    let total: u64 = prefs.command_usage.values().sum();
    let mut parts: Vec<String> = Vec::new();
    if total >= 2 {
        let top: Vec<String> = prefs
            .command_usage
            .iter()
            .filter(|(_, count)| **count >= 2)
            .map(|(family, count)| format!("{family} ({count}×)"))
            .collect();
        if !top.is_empty() {
            parts.push(format!(
                "The user frequently runs `!command`: {} — prefer those workflows.",
                top.join(", ")
            ));
        }
    }
    if prefs.undos >= 1 {
        parts.push(
            "The user has reverted edits with /undo before — be conservative with edits; propose before mutating."
                .to_string(),
        );
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("\n\nKnown user preferences:\n- {}", parts.join("\n- "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn records_command_families_and_builds_prompt() {
        let dir = tempdir().unwrap();
        record_command(dir.path(), "cargo test");
        record_command(dir.path(), "git status");
        record_command(dir.path(), "cargo build");
        let prefs = load(dir.path());
        assert_eq!(prefs.command_usage.get("cargo"), Some(&2));
        assert_eq!(prefs.command_usage.get("git"), Some(&1));
        let block = prompt_block(&prefs);
        assert!(block.contains("cargo (2×)"), "top family: {block}");
        assert!(
            !block.contains("git"),
            "git used once, not highlighted: {block}"
        );
        // A single use produces no block.
        let dir2 = tempdir().unwrap();
        record_command(dir2.path(), "ls");
        assert!(prompt_block(&load(dir2.path())).is_empty());
    }

    #[test]
    fn undo_signal_softens_edit_instructions() {
        let dir = tempdir().unwrap();
        record_undo(dir.path());
        record_undo(dir.path());
        let block = prompt_block(&load(dir.path()));
        assert!(block.contains("reverted edits"), "got: {block}");
        assert!(block.contains("propose before mutating"));
    }
}
