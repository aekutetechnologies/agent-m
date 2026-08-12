use anyhow::{Context, Result, anyhow};
use std::path::Path;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

pub async fn run_attach(session_id: &str, agent_dir: &Path) -> Result<()> {
    let socket_path = crate::daemon::get_socket_path(agent_dir, session_id)?;
    let token_path = crate::daemon::get_token_path(agent_dir, session_id)?;

    if !socket_path.exists() {
        return Err(anyhow!(
            "No active daemon found for session `{}` at {}",
            session_id,
            socket_path.display()
        ));
    }

    let token = std::fs::read_to_string(&token_path)
        .with_context(|| format!("cannot read session token {}", token_path.display()))?;

    let stream = UnixStream::connect(&socket_path)
        .await
        .with_context(|| format!("cannot connect to socket {}", socket_path.display()))?;

    let (reader, mut writer) = stream.into_split();
    let mut buf_reader = BufReader::new(reader);

    // Token authentication handshake
    writer
        .write_all(format!("{}\n", token.trim()).as_bytes())
        .await?;

    let mut response = String::new();
    buf_reader.read_line(&mut response).await?;

    if !response.starts_with("OK:") {
        return Err(anyhow!("Authentication failed: {}", response.trim()));
    }

    println!("Connected to daemon `{session_id}`.");

    let mut rl = rustyline::DefaultEditor::new()?;
    loop {
        let readline = tokio::task::block_in_place(|| rl.readline(&format!("attach ({session_id}) > ")));
        match readline {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed == "/detach" || trimmed == "/exit" {
                    let _ = writer.write_all(b"QUIT\n").await;
                    println!("Detached from daemon `{session_id}`.");
                    break;
                }
                writer.write_all(format!("{}\n", trimmed).as_bytes()).await?;
                let mut ack = String::new();
                buf_reader.read_line(&mut ack).await?;
                println!("{}", ack.trim());
            }
            Err(_) => {
                let _ = writer.write_all(b"QUIT\n").await;
                println!("Detached.");
                break;
            }
        }
    }

    Ok(())
}
