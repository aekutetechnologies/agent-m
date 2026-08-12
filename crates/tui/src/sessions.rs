//! JSONL session persistence, mirroring pi's format: a header line followed by
//! one message entry per line, under `~/.agent-m/agent/sessions/--<cwd>--/`.

use agent_m_agent::SessionMessage;
use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Write a file atomically and ensure 0600 permissions.
pub fn atomic_save(path: &Path, content: impl AsRef<[u8]>) -> std::io::Result<()> {
    let tmp = path.with_extension(format!("tmp-{}", uuid::Uuid::now_v7().simple()));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    {
        let mut file = options.open(&tmp)?;
        file.write_all(content.as_ref())?;
        file.sync_all()?;
    }
    std::fs::rename(tmp, path)
}

const SESSION_VERSION: u64 = 1;

/// Append-only store for one session.
pub struct SessionStore {
    path: PathBuf,
    file: File,
}

impl SessionStore {
    /// Create a new session file in `<agent_dir>/sessions/--<cwd>--/`.
    pub fn create(agent_dir: &Path, cwd: &Path) -> Result<SessionStore> {
        let dir = session_dir(agent_dir, cwd);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("cannot create session dir {}", dir.display()))?;
        // Session transcripts can contain secrets the model read, so lock the
        // directory and file down (security review MEDIUM).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
        }
        let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%S%.3fZ");
        let id = uuid::Uuid::now_v7().simple();
        let path = dir.join(format!("{timestamp}-{id}.jsonl"));
        let mut file = File::create(&path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        }
        let header = json!({
            "type": "session",
            "version": SESSION_VERSION,
            "id": uuid::Uuid::now_v7().to_string(),
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "cwd": cwd.to_string_lossy(),
        });
        writeln!(file, "{header}")?;
        Ok(SessionStore { path, file })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one message entry.
    pub fn append(&mut self, message: &SessionMessage) -> Result<()> {
        let mut entry = message_to_entry(message)?;
        if entry.get("ts").is_none() {
            entry["ts"] = json!(now_iso());
        }
        writeln!(self.file, "{entry}")?;
        self.file.flush()?;
        Ok(())
    }
}

/// Current local time as an RFC3339-style ISO string for the audit trail.
pub fn now_iso() -> String {
    chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Load the most recent session for `cwd`, if any, as messages.
pub fn resume(agent_dir: &Path, cwd: &Path) -> Result<Vec<SessionMessage>> {
    let dir = session_dir(agent_dir, cwd);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    let Some(latest) = files.pop() else {
        return Ok(Vec::new());
    };
    load_entries(&latest)
}

fn session_dir(agent_dir: &Path, cwd: &Path) -> PathBuf {
    let sanitized: String = cwd
        .to_string_lossy()
        .chars()
        .map(|ch| match ch {
            '/' | ':' | '\\' | ' ' => '_',
            other => other,
        })
        .collect();
    agent_dir.join("sessions").join(format!("--{sanitized}--"))
}

fn message_to_entry(message: &SessionMessage) -> Result<Value> {
    Ok(match message {
        SessionMessage::User { content, images } => {
            let mut entry = json!({ "type": "message", "kind": "user", "content": content });
            if !images.is_empty() {
                entry["images"] = json!(images);
            }
            entry
        }
        SessionMessage::Assistant {
            content,
            usage,
            stop_reason,
            model,
            trust,
        } => json!({
            "type": "message",
            "kind": "assistant",
            "content": content,
            "usage": usage,
            "trust": trust,
            "stopReason": stop_reason,
            "model": model
        }),
        SessionMessage::ToolResult {
            tool_call_id,
            name,
            content,
            is_error,
        } => json!({
            "type": "message",
            "kind": "toolResult",
            "toolCallId": tool_call_id,
            "name": name,
            "content": content,
            "isError": is_error
        }),
        SessionMessage::Summary { text } => json!({
            "type": "message",
            "kind": "summary",
            "content": text
        }),
    })
}

fn entry_to_message(value: &Value) -> Result<SessionMessage> {
    let kind = value.get("kind").and_then(Value::as_str).unwrap_or("");
    Ok(match kind {
        "user" => SessionMessage::User {
            content: value
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            images: serde_json::from_value(
                value.get("images").cloned().unwrap_or(Value::Array(vec![])),
            )
            .unwrap_or_default(),
        },
        "assistant" => SessionMessage::Assistant {
            content: serde_json::from_value(
                value
                    .get("content")
                    .cloned()
                    .unwrap_or(Value::Array(vec![])),
            )?,
            usage: serde_json::from_value(value.get("usage").cloned().unwrap_or(Value::Null))?,
            stop_reason: serde_json::from_value(
                value
                    .get("stopReason")
                    .cloned()
                    .unwrap_or(Value::String("stop".to_string())),
            )?,
            model: value
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            trust: serde_json::from_value(
                value
                    .get("trust")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default())),
            )
            .unwrap_or_default(),
        },
        "toolResult" => SessionMessage::ToolResult {
            tool_call_id: value
                .get("toolCallId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            name: value
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            content: value
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            is_error: value
                .get("isError")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        },
        "summary" => SessionMessage::Summary {
            text: value
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        },
        _ => return Err(anyhow!("unknown session entry kind `{kind}`")),
    })
}

fn load_entries(path: &Path) -> Result<Vec<SessionMessage>> {
    let file =
        File::open(path).with_context(|| format!("cannot open session {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut messages = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)?;
        if value.get("type").and_then(Value::as_str) == Some("session") {
            continue;
        }
        messages.push(entry_to_message(&value).map_err(|error| {
            anyhow!(
                "invalid entry on line {} of {}: {error}",
                index + 1,
                path.display()
            )
        })?);
    }
    Ok(messages)
}

/// A single audit-journal row (principle 7): timestamp + a one-line summary.
#[derive(Debug, Clone)]
pub struct JournalEntry {
    pub time: String,
    pub kind: String,
    pub text: String,
}

/// Read the session file into narrated journal rows (time + kind + summary).
pub fn journal(agent_dir: &Path, cwd: &Path) -> Vec<JournalEntry> {
    let dir = session_dir(agent_dir, cwd);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    let Some(latest) = files.pop() else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(&latest) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        // Skip the session header (no `kind`) — it is not an event.
        let Some(kind) = value.get("kind").and_then(Value::as_str) else {
            continue;
        };
        let kind = kind.to_string();
        let time = value
            .get("ts")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let text = match kind.as_str() {
            "user" => value
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            "assistant" => value
                .get("content")
                .and_then(Value::as_array)
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default(),
            "toolResult" => {
                let name = value.get("name").and_then(Value::as_str).unwrap_or("tool");
                let content = value.get("content").and_then(Value::as_str).unwrap_or("");
                let mut line = format!("{name}: {content}");
                line.truncate(120);
                line
            }
            _ => String::new(),
        };
        rows.push(JournalEntry { time, kind, text });
    }
    rows
}

/// One undoable file snapshot (check.md principle 8: reversible actions).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UndoEntry {
    pub path: String,
    /// The file's content before the mutation; None = the file did not exist.
    pub before: Option<String>,
}

/// One git checkpoint entry (check.md-aligned instant rollback).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointEntry {
    pub sha: String,
    pub label: String,
    pub ts: String,
}

/// Persist the checkpoint ledger (`<agent_dir>/checkpoints/<stem>.json`).
pub fn save_checkpoints(
    agent_dir: &Path,
    session_stem: &str,
    entries: &[CheckpointEntry],
) -> Result<()> {
    let dir = agent_dir.join("checkpoints");
    std::fs::create_dir_all(&dir)?;
    let text = serde_json::to_string(entries)?;
    atomic_save(&dir.join(format!("{session_stem}.json")), text)?;
    Ok(())
}

/// Load the checkpoint ledger (empty when absent or corrupt).
pub fn load_checkpoints(agent_dir: &Path, session_stem: &str) -> Vec<CheckpointEntry> {
    let path = agent_dir
        .join("checkpoints")
        .join(format!("{session_stem}.json"));
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str::<Vec<CheckpointEntry>>(&text).ok())
        .unwrap_or_default()
}

/// Persist the undo ledger for a session (`<agent_dir>/undo/<stem>.json`).
pub fn save_undo(agent_dir: &Path, session_stem: &str, entries: &[UndoEntry]) -> Result<()> {
    let dir = agent_dir.join("undo");
    std::fs::create_dir_all(&dir)?;
    let text = serde_json::to_string(entries)?;
    atomic_save(&dir.join(format!("{session_stem}.json")), text)?;
    Ok(())
}

/// Apply one undo entry: restore the before-content or delete the file when
/// it did not exist before. Returns the action performed for messaging.
pub fn apply_undo(entry: &UndoEntry, cwd: &Path) -> std::io::Result<&'static str> {
    let target = std::path::Path::new(&entry.path);

    // Containment check: prevent /undo from writing outside the workspace
    let canonical_cwd = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());

    // To check containment safely without requiring the file to exist yet,
    // we resolve the parent directory.
    let parent = target.parent().unwrap_or(std::path::Path::new(""));
    let canonical_parent = if parent.exists() {
        parent
            .canonicalize()
            .unwrap_or_else(|_| parent.to_path_buf())
    } else {
        // If parent doesn't exist, we'll try checking prefix directly.
        // In a real containment we'd create the dir safely or reject it.
        // For undo, the parent directory must exist because undo replaces files.
        parent.to_path_buf()
    };

    if !canonical_parent.starts_with(&canonical_cwd) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "undo path escapes workspace boundary",
        ));
    }

    match &entry.before {
        Some(content) => {
            std::fs::write(target, content)?;
            Ok("restored")
        }
        None => {
            if target.exists() {
                std::fs::remove_file(target)?;
            }
            Ok("deleted")
        }
    }
}

/// Load the undo ledger for a session (empty when absent or corrupt).
pub fn load_undo(agent_dir: &Path, session_stem: &str) -> Vec<UndoEntry> {
    let path = agent_dir.join("undo").join(format!("{session_stem}.json"));
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str::<Vec<UndoEntry>>(&text).ok())
        .unwrap_or_default()
}

/// Persist the current task plan for a session as JSON under
/// `<agent_dir>/tasks/<session-stem>.json` (survives restarts and compaction).
pub fn save_todos(
    agent_dir: &Path,
    session_stem: &str,
    todos: &[crate::plan::TodoItem],
) -> Result<()> {
    let dir = agent_dir.join("tasks");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("cannot create tasks dir {}", dir.display()))?;
    let items: Vec<Value> = todos
        .iter()
        .map(|todo| json!({ "step": todo.step, "text": todo.text, "completed": todo.completed }))
        .collect();
    let path = dir.join(format!("{session_stem}.json"));
    atomic_save(
        &path,
        serde_json::to_string_pretty(&json!({ "todos": items }))?,
    )
    .with_context(|| format!("cannot write {}", path.display()))?;
    Ok(())
}

/// Load the persisted plan for a session, if any.
pub fn load_todos(agent_dir: &Path, session_stem: &str) -> Vec<crate::plan::TodoItem> {
    let path = agent_dir.join("tasks").join(format!("{session_stem}.json"));
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return Vec::new();
    };
    value
        .get("todos")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    Some(crate::plan::TodoItem {
                        step: item.get("step")?.as_u64()? as usize,
                        text: item.get("text")?.as_str()?.to_string(),
                        completed: item
                            .get("completed")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_m_ai::{ContentPart, StopReason, Usage};
    use tempfile::tempdir;

    #[test]
    fn undo_ledger_roundtrip() {
        let dir = tempdir().unwrap();
        let entries = vec![
            UndoEntry {
                path: "src/a.rs".to_string(),
                before: Some("old content".to_string()),
            },
            UndoEntry {
                path: "new.txt".to_string(),
                before: None,
            },
        ];
        save_undo(dir.path(), "sess", &entries).unwrap();
        let loaded = load_undo(dir.path(), "sess");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].before.as_deref(), Some("old content"));
        assert!(loaded[1].before.is_none(), "None before survives");
        // Missing/corrupt ledger loads empty.
        assert!(load_undo(dir.path(), "missing").is_empty());
    }

    #[test]
    fn apply_undo_restores_content_and_deletes_new_files() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("a.txt");
        std::fs::write(&path, "current").unwrap();
        let entry = UndoEntry {
            path: path.to_string_lossy().to_string(),
            before: Some("old content".to_string()),
        };
        assert_eq!(apply_undo(&entry, dir.path()).unwrap(), "restored");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "old content");
        // File that did not exist before → deleted.
        let new_path = dir.path().join("new.txt");
        std::fs::write(&new_path, "x").unwrap();
        let entry = UndoEntry {
            path: new_path.to_string_lossy().to_string(),
            before: None,
        };
        assert_eq!(apply_undo(&entry, dir.path()).unwrap(), "deleted");
        assert!(!new_path.exists());
    }

    #[test]
    fn apply_undo_refuses_paths_outside_the_workspace() {
        let dir = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let victim = outside.path().join("victim.txt");
        std::fs::write(&victim, "precious").unwrap();
        // Absolute path outside the workspace → refused.
        let entry = UndoEntry {
            path: victim.to_string_lossy().to_string(),
            before: Some("evil".to_string()),
        };
        assert!(apply_undo(&entry, dir.path()).is_err());
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "precious");
        // A `..`-based relative path that would escape the workspace → refused.
        let escape = UndoEntry {
            path: "../victim.txt".to_string(),
            before: Some("evil".to_string()),
        };
        assert!(apply_undo(&escape, dir.path()).is_err());
    }

    #[test]
    fn appended_messages_carry_timestamps_and_journal_reads_them() {
        let dir = tempdir().unwrap();
        let mut store = SessionStore::create(dir.path(), dir.path()).unwrap();
        store
            .append(&SessionMessage::User {
                content: "hello".to_string(),
                images: Vec::new(),
            })
            .unwrap();
        let rows = journal(dir.path(), dir.path());
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].kind, "user");
        assert_eq!(rows[0].text, "hello");
        assert!(!rows[0].time.is_empty(), "ts recorded: {}", rows[0].time);
        // ISO-ish: contains a T separator.
        assert!(rows[0].time.contains('T'), "got: {}", rows[0].time);
    }

    #[test]
    fn todos_roundtrip() {
        let dir = tempdir().unwrap();
        let todos = vec![
            crate::plan::TodoItem {
                step: 1,
                text: "read the client".to_string(),
                completed: false,
            },
            crate::plan::TodoItem {
                step: 2,
                text: "change color".to_string(),
                completed: true,
            },
        ];
        save_todos(dir.path(), "session-1", &todos).unwrap();
        assert_eq!(load_todos(dir.path(), "session-1"), todos);
        assert!(load_todos(dir.path(), "missing").is_empty());
    }

    fn sample_messages() -> Vec<SessionMessage> {
        vec![
            SessionMessage::User {
                content: "hi".to_string(),
                images: Vec::new(),
            },
            SessionMessage::Assistant {
                content: vec![ContentPart::Text {
                    text: "hello there".to_string(),
                }],
                usage: Some(Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read_tokens: 8,
                    cache_creation_tokens: 2,
                    total_tokens: 15,
                    cost: 0.0,
                }),
                stop_reason: StopReason::Stop,
                model: "deepseek-chat".to_string(),
                trust: Default::default(),
            },
            SessionMessage::ToolResult {
                tool_call_id: "call_1".to_string(),
                name: "bash".to_string(),
                content: "ok".to_string(),
                is_error: false,
            },
        ]
    }

    #[test]
    fn roundtrips_messages() {
        let dir = tempdir().unwrap();
        let mut store = SessionStore::create(dir.path(), dir.path()).unwrap();
        for message in sample_messages() {
            store.append(&message).unwrap();
        }
        drop(store);

        let loaded = resume(dir.path(), dir.path()).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0], sample_messages()[0]);
        assert_eq!(loaded[1], sample_messages()[1]);
        assert_eq!(loaded[2], sample_messages()[2]);
    }

    #[test]
    fn resume_with_no_sessions_is_empty() {
        let dir = tempdir().unwrap();
        let loaded = resume(dir.path(), dir.path()).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn list_sessions_metadata_across_cwds() {
        let dir = tempdir().unwrap();
        // Two cwd dirs, one session file each (with a header + a user msg +
        // an assistant msg with a model).
        for cwd in ["--_tmp_proj_a--", "--_tmp_proj_b--"] {
            let cwd_dir = dir.path().join("sessions").join(cwd);
            std::fs::create_dir_all(&cwd_dir).unwrap();
            let path = cwd_dir.join("2026-08-11T05-09-02.438Z-abc123.jsonl");
            let body = format!(
                "{}\n{}\n{}\n",
                json!({"type": "session", "version": 1}),
                json!({"type": "message", "kind": "user", "content": "explain this repo"}),
                json!({"type": "message", "kind": "assistant", "model": "deepseek-chat", "content": "ok"}),
            );
            std::fs::write(&path, body).unwrap();
        }
        let metas = list_sessions(dir.path());
        assert_eq!(metas.len(), 2);
        for meta in &metas {
            assert!(
                meta.cwd.starts_with("/tmp/proj"),
                "cwd decoded: {}",
                meta.cwd
            );
            assert_eq!(meta.messages, 2);
            assert_eq!(meta.first_prompt, "explain this repo");
            assert_eq!(meta.model, "deepseek-chat");
            assert!(meta.stem.starts_with("2026-08-11T05-09-02.438Z-"));
        }
        // Sorted newest-first.
        assert!(metas[0].updated >= metas[1].updated);
    }

    #[test]
    fn session_store_at_path_appends_to_existing_file() {
        let dir = tempdir().unwrap();
        let cwd_dir = dir.path().join("sessions").join("--_tmp_proj--");
        std::fs::create_dir_all(&cwd_dir).unwrap();
        let path = cwd_dir.join("2026-08-11T05-09-02.438Z-abc.jsonl");
        std::fs::write(
            &path,
            json!({"type": "session", "version": 1}).to_string() + "\n",
        )
        .unwrap();
        let mut store = SessionStore::at_path(&path).unwrap();
        store
            .append(&SessionMessage::User {
                content: "hi".to_string(),
                images: Vec::new(),
            })
            .unwrap();
        let loaded = load_session_file(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(store.path(), path.as_path());
    }
}

/// Load a user-defined slash-command template from
/// `<agent_dir>/commands/<name>.md` (no extension in the command name).
/// Supports `${cwd}` and `${input}` placeholders.
pub fn load_custom_command(agent_dir: &Path, name: &str) -> Option<String> {
    let safe: String = name
        .trim_start_matches('/')
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '_')
        .collect();
    if safe.is_empty() {
        return None;
    }
    let path = agent_dir.join("commands").join(format!("{safe}.md"));
    std::fs::read_to_string(&path).ok()
}

/// List the names of all user-defined slash commands.
pub fn list_custom_commands(agent_dir: &Path) -> Vec<String> {
    let dir = agent_dir.join("commands");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut names = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
        .filter_map(|path| {
            path.file_stem()
                .map(|stem| stem.to_string_lossy().to_string())
        })
        .collect::<Vec<_>>();
    names.sort();
    names
}

#[cfg(test)]
mod custom_command_tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn loads_and_sanitizes_command_names() {
        let dir = tempdir().unwrap();
        let commands = dir.path().join("commands");
        std::fs::create_dir_all(&commands).unwrap();
        std::fs::write(
            commands.join("review.md"),
            "Review ${cwd} focusing on ${input}",
        )
        .unwrap();
        std::fs::write(commands.join("ship.md"), "Ship it").unwrap();
        assert_eq!(
            load_custom_command(dir.path(), "/review").as_deref(),
            Some("Review ${cwd} focusing on ${input}")
        );
        assert!(
            load_custom_command(dir.path(), "../evil").is_none(),
            "path escape blocked"
        );
        let names = list_custom_commands(dir.path());
        assert_eq!(names, vec!["review", "ship"]);
    }
}

/// Sidebar sections the user collapsed, from settings.json
/// (`collapsedSidebarSections`). Unknown/absent → empty (all expanded).
pub fn load_collapsed_sections(agent_dir: &Path) -> Vec<String> {
    let path = agent_dir.join("settings.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&text) else {
        return Vec::new();
    };
    value
        .get("collapsedSidebarSections")
        .and_then(|entry| entry.as_array())
        .map(|array| {
            array
                .iter()
                .filter_map(|entry| entry.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Persist the collapsed sidebar sections, preserving every other key in
/// settings.json (providers, preferences, …).
pub fn save_collapsed_sections(agent_dir: &Path, sections: &[String]) {
    let path = agent_dir.join("settings.json");
    let mut value = std::fs::read_to_string(&path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .unwrap_or_else(|| json!({}));
    if let Some(map) = value.as_object_mut() {
        map.insert("collapsedSidebarSections".to_string(), json!(sections));
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(
        &path,
        serde_json::to_string_pretty(&value).unwrap_or_default(),
    );
}

/// Metadata for one stored session, for the `/sessions` picker.
#[derive(Debug, Clone)]
pub struct SessionMeta {
    /// Session file stem — the key for the todos/undo/checkpoint ledgers.
    pub stem: String,
    /// Full path to the session `.jsonl`.
    pub path: PathBuf,
    /// Human-readable cwd the session ran in (from the directory name).
    pub cwd: String,
    /// Timestamp from the filename (`2026-08-11T05-09-02.438Z-…`), newest
    /// first.
    pub updated: String,
    /// Number of message entries (excluding the header line).
    pub messages: usize,
    /// First user prompt, truncated for display.
    pub first_prompt: String,
    /// Model of the last assistant message, when known.
    pub model: String,
}

/// List every stored session across all cwd directories, newest first.
/// Reads only the first + last lines of each file — cheap even with
/// hundreds of sessions.
pub fn list_sessions(agent_dir: &Path) -> Vec<SessionMeta> {
    let root = agent_dir.join("sessions");
    let mut metas = Vec::new();
    let Ok(dirs) = std::fs::read_dir(&root) else {
        return metas;
    };
    for dir in dirs.flatten() {
        let dir_path = dir.path();
        let dir_name = dir.file_name().to_string_lossy().to_string();
        let cwd = dir_name
            .strip_prefix("--")
            .and_then(|name| name.strip_suffix("--"))
            .unwrap_or(&dir_name)
            .replace('_', "/");
        let Ok(files) = std::fs::read_dir(&dir_path) else {
            continue;
        };
        for file in files.flatten() {
            let path = file.path();
            if path.extension().is_none_or(|ext| ext != "jsonl") {
                continue;
            }
            let Some(_file_name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let file_stem = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            // Filename is `<ISO timestamp with dashes>-<32-hex uuid>`; the
            // uuid is the LAST dash-separated component, so take everything
            // before it as the sortable timestamp.
            let updated = file_stem
                .rfind('-')
                .map(|dash| file_stem[..dash].to_string())
                .unwrap_or_else(|| file_stem.clone());
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let mut messages = 0usize;
            let mut first_prompt = String::new();
            let mut model = String::new();
            for (index, line) in text.lines().enumerate() {
                if index == 0 {
                    continue; // session header
                }
                messages += 1;
                if let Ok(value) = serde_json::from_str::<Value>(line) {
                    if first_prompt.is_empty() && value.get("kind") == Some(&json!("user")) {
                        first_prompt = value
                            .get("content")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .chars()
                            .take(80)
                            .collect();
                    }
                    if value.get("kind") == Some(&json!("assistant"))
                        && let Some(m) = value.get("model").and_then(Value::as_str)
                    {
                        model = m.to_string();
                    }
                }
            }
            metas.push(SessionMeta {
                stem: file_stem,
                path,
                cwd: cwd.clone(),
                updated,
                messages,
                first_prompt,
                model,
            });
        }
    }
    metas.sort_by(|a, b| b.updated.cmp(&a.updated));
    metas
}

impl SessionStore {
    /// Open an existing session file for appending (the `/sessions` resume
    /// path re-points the store at the resumed session so follow-ups stay in
    /// that session's transcript + ledger).
    pub fn at_path(path: &Path) -> Result<SessionStore> {
        let file = std::fs::OpenOptions::new()
            .append(true)
            .open(path)
            .with_context(|| format!("cannot open session file {}", path.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = file.set_permissions(std::fs::Permissions::from_mode(0o600));
        }
        Ok(SessionStore {
            path: path.to_path_buf(),
            file,
        })
    }
}

/// Load every message entry from a session file (the `/sessions` resume
/// path).
pub fn load_session_file(path: &Path) -> Result<Vec<SessionMessage>> {
    load_entries(path)
}
