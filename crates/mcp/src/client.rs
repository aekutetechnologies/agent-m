//! MCP clients: stdio (spawned process) and Streamable HTTP transports.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

use crate::jsonrpc::{frame, parse_response, request, unframe};

const CLIENT_NAME: &str = "agent-m";
const CLIENT_VERSION: &str = "0.1.0";
const PROTOCOL_VERSION: &str = "2025-06-18";

/// A connected MCP server. `tools/call` results are plain text/JSON content
/// blocks (the common case for tool servers).
pub struct McpClient {
    /// JSON-RPC request id counter.
    next_id: u64,
    transport: Transport,
}

enum Transport {
    Stdio(StdioTransport),
    Http(HttpTransport),
}

struct StdioTransport {
    stdin: tokio::process::ChildStdin,
    responses: mpsc::UnboundedReceiver<(u64, Result<Value, String>)>,
    _child: Child,
}

struct HttpTransport {
    client: reqwest::Client,
    url: String,
}

impl McpClient {
    /// Connect to a stdio MCP server (e.g. `npx -y @modelcontextprotocol/server-…`).
    pub async fn connect_stdio(
        command: &str,
        args: &[String],
        env: &[(String, String)],
    ) -> Result<Self> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit());
        for (key, value) in env {
            cmd.env(key, value);
        }
        let mut child = cmd.spawn().context("failed to spawn MCP server")?;
        let stdin = child.stdin.take().context("MCP server has no stdin")?;
        let stdout = child.stdout.take().context("MCP server has no stdout")?;
        let (tx, rx) = mpsc::unbounded_channel();
        // Reader task: parse Content-Length frames and route by id.
        let mut reader = BufReader::new(stdout);
        tokio::spawn(async move {
            let mut buf: Vec<u8> = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                let read = match reader.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                buf.extend_from_slice(&chunk[..read]);
                // A peer that keeps sending without a parseable frame is
                // broken — stop buffering instead of growing unboundedly.
                if !crate::jsonrpc::buffer_is_recoverable(&buf) {
                    break;
                }
                while let Some((message, consumed)) = unframe(&buf) {
                    buf.drain(..consumed);
                    if let Some(id) = message.get("id").and_then(Value::as_u64) {
                        let _ = tx.send((id, parse_response(&message)));
                    }
                }
            }
        });
        let mut client = McpClient {
            next_id: 1,
            transport: Transport::Stdio(StdioTransport {
                stdin,
                responses: rx,
                _child: child,
            }),
        };
        client.initialize().await?;
        Ok(client)
    }

    /// Connect to a Streamable HTTP MCP server (POST JSON-RPC; SSE responses
    /// are consumed until the response object arrives).
    pub async fn connect_http(url: &str) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .context("failed to build HTTP client")?;
        let mut mcp = McpClient {
            next_id: 1,
            transport: Transport::Http(HttpTransport {
                client,
                url: url.to_string(),
            }),
        };
        mcp.initialize().await?;
        Ok(mcp)
    }

    async fn initialize(&mut self) -> Result<()> {
        let result = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": { "name": CLIENT_NAME, "version": CLIENT_VERSION },
                }),
            )
            .await?;
        let protocol = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .unwrap_or(PROTOCOL_VERSION);
        // notify initialized (best-effort; some transports require it)
        let _ = self.notify("notifications/initialized", json!({})).await;
        tracing::debug!(%protocol, "MCP initialize handshake complete");
        Ok(())
    }

    /// `tools/list` → the server's tool table.
    pub async fn list_tools(&mut self) -> Result<Vec<McpToolDef>> {
        let result = self.request("tools/list", json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        Ok(tools
            .iter()
            .filter_map(|tool| {
                let name = tool.get("name").and_then(Value::as_str)?.to_string();
                let description = tool
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let input_schema = tool
                    .get("inputSchema")
                    .cloned()
                    .unwrap_or_else(|| json!({ "type": "object" }));
                Some(McpToolDef {
                    name,
                    description,
                    input_schema,
                })
            })
            .collect())
    }

    /// `tools/call` → the text content of the result.
    pub async fn call_tool(&mut self, name: &str, arguments: Value) -> Result<String> {
        let result = self
            .request(
                "tools/call",
                json!({ "name": name, "arguments": arguments }),
            )
            .await?;
        if result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            let text = extract_text(&result);
            bail!("MCP tool {name} errored: {text}");
        }
        Ok(extract_text(&result))
    }

    /// Issue a request and wait for the response with the matching id.
    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let message = request(method, params, id);
        match &mut self.transport {
            Transport::Stdio(transport) => {
                transport
                    .stdin
                    .write_all(&frame(&message))
                    .await
                    .context("MCP stdio write failed")?;
                transport.stdin.flush().await.ok();
                let outcome = tokio::time::timeout(std::time::Duration::from_secs(30), async {
                    loop {
                        let (response_id, response) = transport
                            .responses
                            .recv()
                            .await
                            .ok_or_else(|| "MCP server closed the stream".to_string())?;
                        if response_id == id {
                            return response;
                        }
                    }
                })
                .await
                .context("MCP request timed out")?
                .map_err(anyhow::Error::msg)?;
                Ok(outcome)
            }
            Transport::Http(transport) => {
                let response = transport
                    .client
                    .post(&transport.url)
                    .header("content-type", "application/json")
                    .header("accept", "application/json, text/event-stream")
                    .json(&message)
                    .send()
                    .await
                    .context("MCP HTTP request failed")?;
                let content_type = response
                    .headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("")
                    .to_string();
                let bytes = response.bytes().await.context("MCP HTTP read failed")?;
                let body = if content_type.contains("text/event-stream") {
                    parse_sse(&bytes).unwrap_or(Value::Null)
                } else {
                    serde_json::from_slice(&bytes).unwrap_or(Value::Null)
                };
                if body == Value::Null {
                    bail!("MCP HTTP returned an unparseable response");
                }
                parse_response(&body).map_err(anyhow::Error::msg)
            }
        }
    }

    /// Fire-and-forget notification (no response expected).
    async fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        let message = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        match &mut self.transport {
            Transport::Stdio(transport) => {
                transport
                    .stdin
                    .write_all(&frame(&message))
                    .await
                    .context("MCP stdio write failed")?;
                transport.stdin.flush().await.ok();
            }
            Transport::Http(transport) => {
                let _ = transport
                    .client
                    .post(&transport.url)
                    .json(&message)
                    .send()
                    .await;
            }
        }
        Ok(())
    }
}

/// A tool definition advertised by an MCP server.
#[derive(Debug, Clone)]
pub struct McpToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

fn extract_text(result: &Value) -> String {
    result
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(Value::as_str) == Some("text") {
                        block.get("text").and_then(Value::as_str)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// Accumulate `data: …` lines from an SSE body into the last JSON object.
/// Accumulate `data:` lines from an SSE body into JSON-RPC objects. The
/// response is the last object that carries an `id` (notifications don't);
/// multi-line `data:` continuations are joined.
fn parse_sse(bytes: &[u8]) -> Option<Value> {
    let text = std::str::from_utf8(bytes).ok()?;
    let mut data_lines: Vec<String> = Vec::new();
    let mut last_with_id: Option<Value> = None;
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim().to_string());
        } else {
            flush_sse(&mut data_lines, &mut last_with_id);
        }
    }
    flush_sse(&mut data_lines, &mut last_with_id);
    last_with_id
}

fn flush_sse(data_lines: &mut Vec<String>, last_with_id: &mut Option<Value>) {
    if data_lines.is_empty() {
        return;
    }
    let joined = data_lines.join("");
    data_lines.clear();
    if let Ok(value) = serde_json::from_str::<Value>(&joined)
        && value.get("id").is_some()
    {
        *last_with_id = Some(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_picks_last_object_with_an_id_and_joins_multiline_data() {
        // A notification (no id) then the response; the response wins.
        let body = b"event: message\ndata: {\"jsonrpc\":\"2.0\",\"method\":\"ping\"}\n\n\
                     event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\n\
                     data: \"result\":{\"tools\":[]}}\n\n";
        let parsed = parse_sse(body).expect("parsed");
        assert_eq!(parsed["result"]["tools"], json!([]));
        assert_eq!(parsed["id"], 1);
    }
}
