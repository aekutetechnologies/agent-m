//! JSON-RPC 2.0 plumbing shared by the MCP transports.

use serde_json::Value;

/// Build a JSON-RPC 2.0 request object.
pub fn request(method: &str, params: Value, id: u64) -> Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    })
}

/// Extract the `result` from a JSON-RPC response, or an error message.
pub fn parse_response(body: &Value) -> Result<Value, String> {
    if let Some(error) = body.get("error") {
        return Err(error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("JSON-RPC error")
            .to_string());
    }
    body.get("result")
        .cloned()
        .ok_or_else(|| "JSON-RPC response has no result".to_string())
}

/// Largest single message we accept (defends against unbounded buffering).
pub const MAX_FRAME_BYTES: usize = 32 * 1024 * 1024;

/// Frame a message for the MCP stdio transport: newline-delimited JSON.
/// Each message is a compact JSON object followed by a single `\n`.
pub fn frame(json: &Value) -> Vec<u8> {
    let mut body = serde_json::to_string(json).unwrap_or_else(|_| "{}".to_string());
    body.push('\n');
    body.into_bytes()
}

/// Parse the next newline-delimited JSON message from the buffer.
/// Returns the parsed value and the number of bytes consumed (including `\n`).
pub fn unframe(buf: &[u8]) -> Option<(Value, usize)> {
    let newline = buf.iter().position(|&b| b == b'\n')?;
    let line = std::str::from_utf8(&buf[..newline]).ok()?.trim();
    if line.is_empty() {
        return Some((Value::Null, newline + 1)); // skip blank lines
    }
    let value = serde_json::from_str(line).ok()?;
    Some((value, newline + 1))
}

/// Whether the buffer might still produce a valid frame.
pub fn buffer_is_recoverable(buf: &[u8]) -> bool {
    buf.len() <= MAX_FRAME_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_and_unframes_roundtrip() {
        let json = request("tools/list", Value::Object(Default::default()), 1);
        let framed = frame(&json);
        assert!(framed.ends_with(b"\n"));
        let (parsed, consumed) = unframe(&framed).expect("complete frame");
        assert_eq!(consumed, framed.len());
        assert_eq!(parsed, json);
        // No newline → incomplete frame.
        assert!(unframe(&framed[..framed.len() - 1]).is_none());
    }

    #[test]
    fn unframe_skips_blank_lines() {
        let buf = b"\n{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n";
        let (first, n) = unframe(buf).unwrap();
        assert_eq!(first, Value::Null);
        let (second, _) = unframe(&buf[n..]).unwrap();
        assert_eq!(second["id"], 1);
    }

    #[test]
    fn buffer_recoverable_rejects_oversized() {
        let huge = vec![b'x'; MAX_FRAME_BYTES + 1];
        assert!(!buffer_is_recoverable(&huge));
        assert!(buffer_is_recoverable(b"partial line"));
    }

    #[test]
    fn parse_response_extracts_result_and_errors() {
        let ok = serde_json::json!({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}});
        assert_eq!(
            parse_response(&ok).unwrap(),
            serde_json::json!({"tools": []})
        );
        let err = serde_json::json!({"jsonrpc": "2.0", "id": 1, "error": {"message": "boom"}});
        assert_eq!(parse_response(&err).unwrap_err(), "boom");
    }
}
