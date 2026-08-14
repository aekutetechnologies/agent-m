//! Background daemon sessions (check.md principle 6, interruption handling):
//! a persistent agent runs for the life of the daemon; clients attach over a
//! local unix socket, send `prompt` commands, and receive live `EVENT` lines.
//! The conversation stays in memory across clients, so attach/detach is free.
//!
//! Wire protocol (line-based, documented in docs/usage/sessions-worktrees.mdx):
//! ```text
//! → <token>\n
//! ← OK: Authenticated\n
//! → prompt <text>\n            run one prompt; results/events stream
//! ← EVENT <json>\n  (0..n)     live AgentEvent (same shape as --stream-json)
//! ← RESULT ok\n | RESULT error <msg>\n
//! → resume\n                   load the session file for cwd into the agent
//! ← RESULT ok restored N messages\n
//! → status\n
//! ← RESULT ok messages=N tokens=T/W\n
//! → QUIT\n
//! ```

use agent_m_agent::{Agent, AgentOptions, PermissionGate, SessionMessage};
use agent_m_ai::Provider;
use anyhow::Result;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader as AsyncBufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, broadcast, mpsc};

pub fn get_sockets_dir(agent_dir: &Path) -> Result<PathBuf> {
    let dir = agent_dir.join("sockets");
    std::fs::create_dir_all(&dir)?;
    #[cfg(unix)]
    {
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700));
    }
    Ok(dir)
}

pub fn get_socket_path(agent_dir: &Path, session_id: &str) -> Result<PathBuf> {
    let dir = get_sockets_dir(agent_dir)?;
    Ok(dir.join(format!("{session_id}.sock")))
}

pub fn get_token_path(agent_dir: &Path, session_id: &str) -> Result<PathBuf> {
    let dir = get_sockets_dir(agent_dir)?;
    Ok(dir.join(format!("{session_id}.token")))
}

pub async fn run_daemon(
    session_id: String,
    provider: Arc<dyn Provider>,
    agent_options: AgentOptions,
    gate: Arc<dyn PermissionGate>,
    agent_dir: PathBuf,
    cwd: PathBuf,
) -> Result<()> {
    let socket_path = get_socket_path(&agent_dir, &session_id)?;
    let token_path = get_token_path(&agent_dir, &session_id)?;

    if socket_path.exists() {
        let _ = std::fs::remove_file(&socket_path);
    }

    let token = uuid::Uuid::now_v7().simple().to_string();
    std::fs::write(&token_path, &token)?;
    #[cfg(unix)]
    {
        let _ = std::fs::set_permissions(&token_path, std::fs::Permissions::from_mode(0o600));
    }

    let listener = UnixListener::bind(&socket_path)?;
    #[cfg(unix)]
    {
        let _ = std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600));
    }

    println!("Daemon `{session_id}` running on socket: {}", socket_path.display());

    // The persistent agent: one long-horizon conversation, N attached
    // clients. `prompt` serializes through the lock.
    let agent = Arc::new(Mutex::new(Agent::new(provider, agent_options, gate)));

    // Live events go to every attached client as `EVENT <json>` lines.
    let (event_tx, _) = broadcast::channel(512);

    // Session log (jsonl under agent_dir/sessions) so `resume` works across
    // daemon restarts. Appended from events; the format matches the REPL.
    let store = Arc::new(Mutex::new(crate::sessions::SessionStore::create(
        &agent_dir, &cwd,
    )?));
    {
        let mut agent_guard = agent.lock().await;
        let tx = event_tx.clone();
        let store = store.clone();
        // tool_call_id → tool name, to reconstruct ToolResult entries. The
        // subscribe listener is `Fn`, so interior mutability via Arc<Mutex>.
        let tool_names: Arc<std::sync::Mutex<std::collections::HashMap<String, String>>> =
            Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
        agent_guard.subscribe(move |event| {
            use agent_m_agent::AgentEvent as E;
            match event {
                E::MessageEnd { message } => {
                    if let Ok(mut store) = store.try_lock() {
                        let _ = store.append(message);
                    }
                    let _ = tx.send(format!("EVENT {}", crate::event_to_json(event)));
                }
                E::ToolExecutionStart {
                    tool_call_id, name, ..
                } => {
                    if let Ok(mut names) = tool_names.lock() {
                        names.insert(tool_call_id.clone(), name.clone());
                    }
                    let _ = tx.send(format!("EVENT {}", crate::event_to_json(event)));
                }
                E::ToolExecutionEnd {
                    tool_call_id, outcome,
                } => {
                    let name = tool_names
                        .lock()
                        .ok()
                        .and_then(|mut names| names.remove(tool_call_id));
                    if let Some(name) = name
                        && let Ok(mut store) = store.try_lock()
                    {
                        let _ = store.append(&SessionMessage::ToolResult {
                            tool_call_id: tool_call_id.clone(),
                            name,
                            content: outcome.content.clone(),
                            is_error: outcome.is_error,
                        });
                    }
                    let _ = tx.send(format!("EVENT {}", crate::event_to_json(event)));
                }
                _ => {
                    let _ = tx.send(format!("EVENT {}", crate::event_to_json(event)));
                }
            }
        });
    }

    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let token_expected = token.clone();
                let agent = agent.clone();
                let event_rx = event_tx.subscribe();
                let agent_dir = agent_dir.clone();
                let cwd = cwd.clone();
                tokio::spawn(async move {
                    if let Err(e) =
                        handle_client(stream, &token_expected, agent, event_rx, agent_dir, cwd)
                            .await
                    {
                        eprintln!("Client handler error: {e}");
                    }
                });
            }
            Err(e) => {
                eprintln!("Socket accept error: {e}");
                break;
            }
        }
    }

    let _ = std::fs::remove_file(&socket_path);
    let _ = std::fs::remove_file(&token_path);
    Ok(())
}

async fn handle_client(
    stream: UnixStream,
    token_expected: &str,
    agent: Arc<Mutex<Agent>>,
    mut event_rx: broadcast::Receiver<String>,
    agent_dir: PathBuf,
    cwd: PathBuf,
) -> Result<()> {
    let (reader, writer) = stream.into_split();
    let mut buf_reader = AsyncBufReader::new(reader);

    // One writer task owns the socket; everything else sends lines into it.
    let (tx, mut rx) = mpsc::channel::<String>(64);
    let writer_task = tokio::spawn(async move {
        let mut writer = writer;
        while let Some(line) = rx.recv().await {
            if writer.write_all(line.as_bytes()).await.is_err() {
                break;
            }
            if writer.write_all(b"\n").await.is_err() {
                break;
            }
        }
    });

    let mut auth_line = String::new();
    buf_reader.read_line(&mut auth_line).await?;
    if auth_line.trim() != token_expected {
        tx.send("ERROR: Invalid session token".to_string()).await.ok();
        drop(tx);
        let _ = writer_task.await;
        return Ok(());
    }
    tx.send("OK: Authenticated".to_string()).await.ok();

    let mut line = String::new();
    loop {
        tokio::select! {
            // `biased` keeps commands ahead of a fast event stream.
            biased;
            // Live agent events → the client.
            ev = event_rx.recv() => {
                match ev {
                    Ok(ev) => { if tx.send(ev).await.is_err() { break; } }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            // Client commands.
            read = buf_reader.read_line(&mut line) => {
                let n = read?;
                if n == 0 { break; }
                let msg = line.trim().to_string();
                line.clear();
                if msg == "QUIT" { break; }

                if let Some(text) = msg.strip_prefix("prompt ") {
                    let text = text.to_string();
                    let agent = agent.clone();
                    let tx = tx.clone();
                    tokio::spawn(async move {
                        let outcome = match agent.try_lock() {
                            Ok(mut a) => match a.prompt(text).await {
                                Ok(()) => "RESULT ok".to_string(),
                                Err(e) => format!("RESULT error {e}"),
                            },
                            Err(_) => "RESULT error daemon busy (a prompt is already running)".to_string(),
                        };
                        let _ = tx.send(outcome).await;
                    });
                } else if msg == "resume" {
                    let messages = crate::sessions::resume(&agent_dir, &cwd)?;
                    let mut a = agent.lock().await;
                    a.restore_messages(messages.clone());
                    drop(a);
                    tx.send(format!("RESULT ok restored {} messages", messages.len())).await.ok();
                } else if msg == "status" {
                    let a = agent.lock().await;
                    let (tokens, window) = a.context_usage();
                    let count = a.messages().len();
                    let window = window.map_or_else(|| "?".to_string(), |w| w.to_string());
                    drop(a);
                    tx.send(format!("RESULT ok messages={count} tokens={tokens}/{window}")).await.ok();
                } else {
                    tx.send(format!("RESULT error unknown command: {msg}")).await.ok();
                }
            }
        }
    }

    tx.send("QUIT".to_string()).await.ok();
    drop(tx);
    let _ = writer_task.await;
    Ok(())
}

