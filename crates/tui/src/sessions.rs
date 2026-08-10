//! JSONL session persistence, mirroring pi's format: a header line followed by
//! one message entry per line, under `~/.agent-m/agent/sessions/--<cwd>--/`.

use agent_m_agent::SessionMessage;
use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

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
        let entry = message_to_entry(message)?;
        writeln!(self.file, "{entry}")?;
        self.file.flush()?;
        Ok(())
    }
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
        SessionMessage::User { content } => {
            json!({ "type": "message", "kind": "user", "content": content })
        }
        SessionMessage::Assistant {
            content,
            usage,
            stop_reason,
            model,
        } => json!({
            "type": "message",
            "kind": "assistant",
            "content": content,
            "usage": usage,
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
    std::fs::write(
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
}
