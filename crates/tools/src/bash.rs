//! The `bash` tool: run a shell command with a timeout and truncation.

use agent_m_agent::{Tool, ToolContext, ToolError, ToolOutcome};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::time::Duration;

/// Default timeout for a bash command, in seconds.
const DEFAULT_TIMEOUT_SECONDS: u64 = 120;

/// Runs a shell command in the session working directory.
pub struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> String {
        "Run a shell command in the session working directory. Returns stdout and stderr merged, truncated to 2000 lines / 50 KB. Use a timeout for long-running commands.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The shell command to run" },
                "timeout": { "type": "number", "description": "Timeout in seconds (default 120)" }
            },
            "required": ["command"]
        })
    }

    async fn execute(
        &self,
        arguments: Value,
        context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let command = arguments
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::failed("bash", "missing string argument `command`"))?;
        let timeout_seconds = arguments
            .get("timeout")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_TIMEOUT_SECONDS);

        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
        let mut process = crate::sandbox::sandboxed_command(&context.cwd, &shell, command);
        process
            .current_dir(&context.cwd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        // Run the command in its own process group so a timeout can kill the
        // whole tree (shell + any grandchildren), not just the shell.
        #[cfg(unix)]
        {
            process.process_group(0);
        }
        let mut child = process
            .spawn()
            .map_err(|error| ToolError::failed("bash", format!("failed to spawn: {error}")))?;

        // Drain the pipes concurrently with `wait()`, storing at most the cap
        // and discarding the rest, so verbose output cannot deadlock `wait()`
        // and memory stays bounded (review: pipe-drain ordering).
        let mut stdout_pipe = child.stdout.take();
        let mut stderr_pipe = child.stderr.take();
        let cap = crate::truncate::MAX_BYTES + 8192;
        let stdout_task = tokio::spawn(async move { read_capped(&mut stdout_pipe, cap).await });
        let stderr_task = tokio::spawn(async move { read_capped(&mut stderr_pipe, cap).await });
        let run = async {
            let status = child
                .wait()
                .await
                .map_err(|error| ToolError::failed("bash", format!("failed to wait: {error}")))?;
            let stdout = stdout_task
                .await
                .map_err(|error| ToolError::failed("bash", format!("read task failed: {error}")))?
                .map_err(|error| ToolError::failed("bash", format!("read failed: {error}")))?;
            let stderr = stderr_task
                .await
                .map_err(|error| ToolError::failed("bash", format!("read task failed: {error}")))?
                .map_err(|error| ToolError::failed("bash", format!("read failed: {error}")))?;
            Ok::<_, ToolError>((status, stdout, stderr))
        };

        let timeout = Duration::from_secs(timeout_seconds.max(1));
        let (stdout, stderr, status) = match tokio::time::timeout(timeout, run).await {
            Ok(result) => {
                let (status, stdout, stderr) = result?;
                (stdout, stderr, status)
            }
            Err(_) => {
                // Kill the whole process group, then the direct child as a
                // fallback, so backgrounded grandchildren cannot linger.
                #[cfg(unix)]
                if let Some(id) = child.id() {
                    unsafe {
                        libc::kill(-(id as i32), libc::SIGKILL);
                    }
                }
                let _ = child.kill().await;
                return Ok(ToolOutcome::error(format!(
                    "command timed out after {timeout_seconds}s"
                )));
            }
        };

        let mut combined = stdout;
        if !stderr.is_empty() {
            if !combined.is_empty() && !combined.ends_with('\n') {
                combined.push('\n');
            }
            combined.push_str(&stderr);
        }
        let combined = combined.trim_end().to_string();

        let output =
            crate::truncate::offload_or_truncate(&combined, "bash", context.output_dir.as_deref());

        let mut result = if status.success() {
            let text = if output.is_empty() {
                "command succeeded with no output".to_string()
            } else {
                output
            };
            ToolOutcome::success(text)
        } else {
            let mut text = format!("command exited with code {}", status.code().unwrap_or(-1));
            if !output.is_empty() {
                text.push_str(":\n");
                text.push_str(&output);
            }
            ToolOutcome::error(text)
        };

        // Surface the exit code to the model explicitly.
        if let Some(code) = status.code()
            && code != 0
        {
            result.content = format!("(exit code {code})\n{}", result.content);
        }
        Ok(result)
    }
}

/// Read a pipe to EOF, storing at most `cap` bytes and discarding the rest, so
/// the child never blocks on a full pipe and memory stays bounded.
async fn read_capped<P>(pipe: &mut Option<P>, cap: usize) -> Result<String, std::io::Error>
where
    P: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::AsyncReadExt;
    let Some(mut handle) = pipe.take() else {
        return Ok(String::new());
    };
    let mut stored = String::new();
    let mut chunk = [0u8; 8192];
    loop {
        let read = handle.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        if stored.len() < cap {
            let take = (cap - stored.len()).min(read);
            stored.push_str(&String::from_utf8_lossy(&chunk[..take]));
        }
    }
    Ok(stored)
}
