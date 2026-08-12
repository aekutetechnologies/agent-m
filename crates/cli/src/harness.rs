//! Continual-harness state (Prime Agent `/refine` style): a persistent,
//! model-editable layer of memories, prompt notes, and skills that sits on
//! top of the immutable base system prompt. Every change is an auditable
//! `RefineOp` and can be rolled back by id.
//!
//! The prompt block is rebuilt only when the file changes, so the byte-stable
//! prefix cache is preserved within a session (one cache miss at apply).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntryKind {
    Memory,
    Note,
    Skill,
}

impl EntryKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            EntryKind::Memory => "memory",
            EntryKind::Note => "note",
            EntryKind::Skill => "skill",
        }
    }
}

/// One durable harness entry. `version` bumps on each update; `source` is the
/// id of the `RefineOp` that created/last touched it (audit trail).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessEntry {
    pub id: String,
    pub kind: EntryKind,
    pub text: String,
    pub created_at: String,
    pub updated_at: String,
    pub version: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// One applied (or rolled-back) refinement. `before`/`after` hold the entry
/// text at the time of the op so a rollback can restore exactly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineOp {
    pub id: String,
    /// What triggered the refinement: "manual" or "auto".
    pub trigger: String,
    /// create | update | delete
    pub action: String,
    /// memory | note | skill
    pub kind: String,
    pub entry_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
    pub created_at: String,
}

/// The full harness state. Serialized with a fixed key order and sorted
/// entries so an unchanged harness round-trips byte-identically (prefix-cache
/// friendly).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Harness {
    #[serde(default)]
    pub entries: Vec<HarnessEntry>,
    #[serde(default)]
    pub refine_log: Vec<RefineOp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
}

pub fn harness_path(agent_dir: &Path) -> PathBuf {
    agent_dir.join("harness.json")
}

pub fn load(agent_dir: &Path) -> Harness {
    std::fs::read_to_string(harness_path(agent_dir))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

pub fn save(agent_dir: &Path, harness: &Harness) {
    if let Some(parent) = harness_path(agent_dir).parent()
        && let Ok(text) = serde_json::to_string(harness)
    {
        let _ = std::fs::create_dir_all(parent);
        // Atomic-ish write: temp + rename avoids a torn file on crash.
        let tmp = parent.join("harness.json.tmp");
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, harness_path(agent_dir));
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(
                    harness_path(agent_dir),
                    std::fs::Permissions::from_mode(0o600),
                );
            }
        }
    }
}

pub fn new_entry_id(kind: &EntryKind, created_at: &str) -> String {
    // Stable, readable ids: kind + timestamp (no uuid dependency needed).
    let stamp: String = created_at
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect();
    format!("{}-{}", kind.as_str(), stamp)
}

/// Upsert an entry by id: creates with version 1, or bumps version + updates
/// text on an existing one. Returns the applied op.
pub fn upsert_entry(
    harness: &mut Harness,
    id: String,
    kind: EntryKind,
    text: String,
    trigger: &str,
    now: &str,
) -> RefineOp {
    let op_id = format!(
        "op-{}",
        now.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
    );
    if let Some(existing) = harness.entries.iter_mut().find(|entry| entry.id == id) {
        let before = existing.text.clone();
        existing.kind = kind.clone();
        existing.text = text.clone();
        existing.updated_at = now.to_string();
        existing.version += 1;
        existing.source = Some(op_id.clone());
        harness.updated_at = Some(now.to_string());
        let op = RefineOp {
            id: op_id,
            trigger: trigger.to_string(),
            action: "update".to_string(),
            kind: kind.as_str().to_string(),
            entry_id: id,
            before: Some(before),
            after: Some(text),
            created_at: now.to_string(),
        };
        harness.refine_log.push(op.clone());
        return op;
    }
    harness.entries.push(HarnessEntry {
        id: id.clone(),
        kind: kind.clone(),
        text: text.clone(),
        created_at: now.to_string(),
        updated_at: now.to_string(),
        version: 1,
        source: Some(op_id.clone()),
    });
    harness.updated_at = Some(now.to_string());
    let op = RefineOp {
        id: op_id,
        trigger: trigger.to_string(),
        action: "create".to_string(),
        kind: kind.as_str().to_string(),
        entry_id: id,
        before: None,
        after: Some(text),
        created_at: now.to_string(),
    };
    harness.refine_log.push(op.clone());
    op
}

/// Delete an entry by id. Returns the op (with `before` set) or None when the
/// id does not exist.
pub fn delete_entry(harness: &mut Harness, id: &str, trigger: &str, now: &str) -> Option<RefineOp> {
    let position = harness.entries.iter().position(|entry| entry.id == id)?;
    let removed = harness.entries.remove(position);
    let op_id = format!(
        "op-{}",
        now.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
    );
    harness.updated_at = Some(now.to_string());
    let op = RefineOp {
        id: op_id,
        trigger: trigger.to_string(),
        action: "delete".to_string(),
        kind: removed.kind.as_str().to_string(),
        entry_id: id.to_string(),
        before: Some(removed.text),
        after: None,
        created_at: now.to_string(),
    };
    harness.refine_log.push(op.clone());
    Some(op)
}

/// The static system-prompt block for the harness layer. Empty when there is
/// nothing worth stating; rebuilt only when the state changes.
pub fn harness_block(harness: &Harness, prefs_block: &str) -> String {
    if harness.entries.is_empty() && prefs_block.trim().is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::new();
    // Deterministic order: kind, then id.
    let mut entries: Vec<&HarnessEntry> = harness.entries.iter().collect();
    entries.sort_by(|a, b| (a.kind.as_str(), &a.id).cmp(&(b.kind.as_str(), &b.id)));
    for entry in entries {
        let text = entry.text.replace("</harness_entry>", "< / harness_entry>");
        parts.push(format!(
            "<harness_entry kind=\"{}\">\n{}\n</harness_entry>",
            entry.kind.as_str(),
            text
        ));
    }
    let prefs = prefs_block.trim();
    if !prefs.is_empty() {
        parts.push(format!(
            "<harness_entry kind=\"preference\">\n{}\n</harness_entry>",
            prefs.replace("</harness_entry>", "< / harness_entry>")
        ));
    }
    format!(
        "\n\n<harness_state>\n{}\n</harness_state>",
        parts.join("\n")
    )
}

/// Invert a refine op by id: `create` removes the entry, `delete` re-creates
/// it from `before`, `update` restores `before`. Removes the op from the log
/// so the rollback itself is audited. Returns the op, or None for an unknown
/// id.
pub fn rollback_op(harness: &mut Harness, op_id: &str) -> Option<RefineOp> {
    let position = harness.refine_log.iter().position(|op| op.id == op_id)?;
    let op = harness.refine_log.remove(position);
    match op.action.as_str() {
        "create" => {
            if let Some(index) = harness.entries.iter().position(|e| e.id == op.entry_id) {
                harness.entries.remove(index);
            }
        }
        "delete" | "update" => {
            let kind = match op.kind.as_str() {
                "memory" => EntryKind::Memory,
                "note" => EntryKind::Note,
                "skill" => EntryKind::Skill,
                _ => return Some(op),
            };
            if let Some(text) = &op.before {
                upsert_entry(
                    harness,
                    op.entry_id.clone(),
                    kind,
                    text.clone(),
                    "rollback",
                    &op.created_at,
                );
            }
        }
        _ => {}
    }
    Some(op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn round_trips_byte_identically() {
        let dir = tempdir().unwrap();
        let mut harness = Harness::default();
        upsert_entry(
            &mut harness,
            "memory-retry".to_string(),
            EntryKind::Memory,
            "flaky tests: retry three times".to_string(),
            "manual",
            "2026-08-12T10-00-00.000Z",
        );
        upsert_entry(
            &mut harness,
            "skill-rebase".to_string(),
            EntryKind::Skill,
            "prefer interactive rebase over merge".to_string(),
            "manual",
            "2026-08-12T10-00-01.000Z",
        );
        save(dir.path(), &harness);
        let loaded = load(dir.path());
        assert_eq!(loaded.entries.len(), 2);
        assert_eq!(loaded.refine_log.len(), 2);
        // Serialization is deterministic: save(load(x)) == save(x).
        let mut again = loaded.clone();
        again
            .entries
            .sort_by(|a, b| (a.kind.as_str(), &a.id).cmp(&(b.kind.as_str(), &b.id)));
        again.refine_log.sort_by(|a, b| a.id.cmp(&b.id));
        let first = serde_json::to_string(&harness).unwrap();
        let second = serde_json::to_string(&again).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn tolerant_of_missing_and_malformed_files() {
        let dir = tempdir().unwrap();
        assert!(load(dir.path()).entries.is_empty());
        std::fs::write(dir.path().join("harness.json"), "not json").unwrap();
        assert!(load(dir.path()).entries.is_empty());
    }

    #[test]
    fn upsert_updates_in_place_and_bumps_version() {
        let mut harness = Harness::default();
        let create = upsert_entry(
            &mut harness,
            "memory-x".to_string(),
            EntryKind::Memory,
            "v1".to_string(),
            "manual",
            "2026-08-12T10-00-00.000Z",
        );
        assert_eq!(create.action, "create");
        assert_eq!(create.before, None);
        let update = upsert_entry(
            &mut harness,
            "memory-x".to_string(),
            EntryKind::Memory,
            "v2".to_string(),
            "auto",
            "2026-08-12T10-00-05.000Z",
        );
        assert_eq!(update.action, "update");
        assert_eq!(update.before.as_deref(), Some("v1"));
        assert_eq!(update.after.as_deref(), Some("v2"));
        assert_eq!(harness.entries.len(), 1);
        assert_eq!(harness.entries[0].version, 2);
        assert_eq!(harness.refine_log.len(), 2);
    }

    #[test]
    fn delete_removes_and_records_before() {
        let mut harness = Harness::default();
        upsert_entry(
            &mut harness,
            "note-n".to_string(),
            EntryKind::Note,
            "keep it simple".to_string(),
            "manual",
            "2026-08-12T10-00-00.000Z",
        );
        let op =
            delete_entry(&mut harness, "note-n", "manual", "2026-08-12T10-00-10.000Z").unwrap();
        assert_eq!(op.action, "delete");
        assert_eq!(op.before.as_deref(), Some("keep it simple"));
        assert!(harness.entries.is_empty());
        assert!(
            delete_entry(
                &mut harness,
                "missing",
                "manual",
                "2026-08-12T10-00-10.000Z"
            )
            .is_none()
        );
    }

    #[test]
    fn rollback_inverts_create_update_delete() {
        let mut harness = Harness::default();
        // create → rollback removes the entry
        let create = upsert_entry(
            &mut harness,
            "memory-m".to_string(),
            EntryKind::Memory,
            "learned".to_string(),
            "manual",
            "2026-08-12T10-00-00.000Z",
        );
        let op = rollback_op(&mut harness, &create.id).unwrap();
        assert_eq!(op.action, "create");
        assert!(harness.entries.is_empty());
        assert!(!harness.refine_log.iter().any(|o| o.id == create.id));

        // update → rollback restores the old text
        upsert_entry(
            &mut harness,
            "memory-m".to_string(),
            EntryKind::Memory,
            "v1".to_string(),
            "manual",
            "2026-08-12T10-00-00.000Z",
        );
        let update = upsert_entry(
            &mut harness,
            "memory-m".to_string(),
            EntryKind::Memory,
            "v2".to_string(),
            "manual",
            "2026-08-12T10-00-05.000Z",
        );
        rollback_op(&mut harness, &update.id).unwrap();
        assert_eq!(harness.entries[0].text, "v1");
        assert_eq!(
            harness.entries[0].version, 3,
            "rollback is a new op, version bumps"
        );

        // delete → rollback re-creates from `before`
        let delete = delete_entry(
            &mut harness,
            "memory-m",
            "manual",
            "2026-08-12T10-00-10.000Z",
        )
        .unwrap();
        rollback_op(&mut harness, &delete.id).unwrap();
        assert_eq!(harness.entries.len(), 1);
        assert_eq!(harness.entries[0].text, "v1");

        // unknown id → None
        assert!(rollback_op(&mut harness, "op-nope").is_none());
    }

    #[test]
    fn block_is_empty_without_state_and_deterministic() {
        let _dir = tempdir().unwrap();
        let harness = Harness::default();
        assert_eq!(harness_block(&harness, ""), "");
        let mut harness = Harness::default();
        upsert_entry(
            &mut harness,
            "memory-a".to_string(),
            EntryKind::Memory,
            "alpha".to_string(),
            "manual",
            "2026-08-12T10-00-00.000Z",
        );
        upsert_entry(
            &mut harness,
            "memory-b".to_string(),
            EntryKind::Memory,
            "beta".to_string(),
            "manual",
            "2026-08-12T10-00-01.000Z",
        );
        let block = harness_block(&harness, "");
        assert!(block.contains("<harness_entry kind=\"memory\">\nalpha\n</harness_entry>"));
        assert!(block.contains("<harness_entry kind=\"memory\">\nbeta\n</harness_entry>"));
        assert!(block.starts_with("\n\n<harness_state>"));
        // Same state → same block (deterministic).
        let mut copy = harness.clone();
        copy.entries.reverse();
        assert_eq!(harness_block(&copy, ""), block);
    }
}
