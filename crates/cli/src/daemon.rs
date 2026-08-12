use agent_m_agent::{Agent, AgentOptions, PermissionGate};
use agent_m_ai::Provider;
use anyhow::Result;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader as AsyncBufReader};
use tokio::net::{UnixListener, UnixStream};

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
    _cwd: PathBuf,
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

    let _agent = Agent::new(provider, agent_options, gate);

    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let token_expected = token.clone();
                tokio::spawn(async move {
                    if let Err(e) = handle_client(stream, &token_expected).await {
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

async fn handle_client(stream: UnixStream, token_expected: &str) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut buf_reader = AsyncBufReader::new(reader);

    let mut auth_line = String::new();
    buf_reader.read_line(&mut auth_line).await?;

    let client_token = auth_line.trim();
    if client_token != token_expected {
        writer.write_all(b"ERROR: Invalid session token\n").await?;
        return Ok(());
    }

    writer.write_all(b"OK: Authenticated\n").await?;

    let mut line = String::new();
    while buf_reader.read_line(&mut line).await? > 0 {
        let msg = line.trim();
        if msg == "QUIT" {
            break;
        }
        writer.write_all(format!("ACK: {}\n", msg).as_bytes()).await?;
        line.clear();
    }

    Ok(())
}
