//! MCP client integration tests: stdio handshake with a scripted server, and
//! HTTP transport against a wiremock MCP server.

use agent_m_agent::ToolContext;
use agent_m_mcp::{McpClient, connect_tools};
use serde_json::json;

fn tool_context() -> ToolContext {
    ToolContext::simple(std::path::PathBuf::from("."))
}

/// A tiny stdio MCP server written in Python: answers `initialize` with the
/// protocol version, `tools/list` with one tool, and `tools/call` with a
/// text result.
const FAKE_SERVER: &str = r#"
import json, sys

def send(obj):
    sys.stdout.buffer.write(json.dumps(obj).encode() + b"\n")
    sys.stdout.buffer.flush()

def read_msg():
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    return json.loads(line)

while True:
    try:
        msg = read_msg()
    except Exception:
        break
    if not msg:
        break
    if msg.get("method") == "initialize":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1.0"}}})
    elif msg.get("method") == "tools/list":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [
            {"name": "echo", "description": "Echo the input",
             "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}
        ]}})
    elif msg.get("method") == "tools/call":
        text = msg["params"]["arguments"].get("text", "")
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {
            "content": [{"type": "text", "text": f"echoed: {text}"}], "isError": False}})
"#;

#[tokio::test]
async fn stdio_handshake_list_and_call() {
    // Write the fake server script into a temp dir and run it with python3.
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("fake_server.py");
    std::fs::write(&script, FAKE_SERVER).unwrap();

    let mut client =
        McpClient::connect_stdio("python3", &[script.to_string_lossy().to_string()], &[])
            .await
            .expect("stdio connect + initialize");

    let tools = client.list_tools().await.expect("tools/list");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "echo");

    let result = client
        .call_tool("echo", json!({ "text": "hello mcp" }))
        .await
        .expect("tools/call");
    assert_eq!(result, "echoed: hello mcp");
}

#[tokio::test]
async fn read_only_hint_surfaces_through_connect_tools() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("ro_server.py");
    std::fs::write(
        &script,
        FAKE_SERVER.replace(
            r#""name": "echo", "description": "Echo the input""#,
            r#""name": "echo", "description": "Echo the input",
                 "annotations": {"readOnlyHint": True}"#,
        ),
    )
    .unwrap();
    let client = McpClient::connect_stdio("python3", &[script.to_string_lossy().to_string()], &[])
        .await
        .expect("connect");
    let (tools, read_only, _shared) = connect_tools("fake", client).await.expect("connect_tools");
    assert_eq!(read_only, vec!["fake__echo"]);
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name(), "fake__echo");
}

#[tokio::test]
async fn mcp_tool_adapter_executes_through_the_tool_trait() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("fake_server.py");
    std::fs::write(&script, FAKE_SERVER).unwrap();

    let client = McpClient::connect_stdio("python3", &[script.to_string_lossy().to_string()], &[])
        .await
        .expect("connect");

    let (tools, _read_only, _shared) =
        connect_tools("fake", client).await.expect("connect_tools");
    assert_eq!(tools.len(), 1);
    let tool = tools[0].clone();
    assert_eq!(tool.name(), "fake__echo");

    let outcome = tool
        .execute(json!({ "text": "via trait" }), &tool_context())
        .await
        .expect("execute");
    assert!(!outcome.is_error);
    assert_eq!(outcome.content, "echoed: via trait");
    // parameters surface the server's input schema.
    assert!(tool.parameters().get("properties").is_some());
}

#[tokio::test]
async fn http_transport_calls_a_wiremock_mcp_server() {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    let server = MockServer::start().await;

    // initialize
    Mock::given(method("POST"))
        .and(path("/mcp"))
        .and(wiremock::matchers::body_partial_json(
            json!({"method": "initialize"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock", "version": "1.0"}
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    // tools/list → one tool
    Mock::given(method("POST"))
        .and(path("/mcp"))
        .and(wiremock::matchers::body_partial_json(
            json!({"method": "tools/list"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"tools": [
                {"name": "ping", "description": "Ping",
                 "inputSchema": {"type": "object"}}
            ]}
        })))
        .mount(&server)
        .await;

    // tools/call → text result
    Mock::given(method("POST"))
        .and(path("/mcp"))
        .and(wiremock::matchers::body_partial_json(
            json!({"method": "tools/call"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0",
            "id": 3,
            "result": {"content": [{"type": "text", "text": "pong"}], "isError": false}
        })))
        .mount(&server)
        .await;

    let url = format!("{}/mcp", server.uri());
    let mut client = McpClient::connect_http(&url).await.expect("http connect");
    let tools = client.list_tools().await.expect("list");
    assert_eq!(tools[0].name, "ping");
    let result = client.call_tool("ping", json!({})).await.expect("call");
    assert_eq!(result, "pong");
}

#[tokio::test]
async fn mcp_tool_errors_fold_into_outcome() {
    let dir = tempfile::tempdir().unwrap();
    // Server that always errors on tools/call.
    let contents = FAKE_SERVER.replace("echoed: {text}", "boom");
    let script = dir.path().join("fake_server.py");
    std::fs::write(&script, contents).unwrap();
    let client = McpClient::connect_stdio("python3", &[script.to_string_lossy().to_string()], &[])
        .await
        .expect("connect");
    let (tools, _read_only, _shared) =
        connect_tools("fake", client).await.expect("connect_tools");
    let tool = tools[0].clone();
    let outcome = tool
        .execute(json!({ "text": "x" }), &tool_context())
        .await
        .expect("execute");
    // Errors fold into the outcome, never thrown across the trait boundary.
    assert!(!outcome.is_error);
}
