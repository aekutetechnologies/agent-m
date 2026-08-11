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

/// Frame a message for the stdio transport (LSP-style).
pub fn frame(json: &Value) -> Vec<u8> {
    let body = serde_json::to_string(json).unwrap_or_else(|_| "{}".to_string());
    let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
    out.extend_from_slice(body.as_bytes());
    out
}

/// Parse the next framed message from a byte buffer, returning the message
/// and the bytes consumed. Returns None when the buffer holds an incomplete
/// frame.
/// Largest frame we accept from an MCP peer (defends against a peer that
/// declares a huge `Content-Length` — either to overflow arithmetic or to
/// grow our buffer unboundedly).
pub const MAX_FRAME_BYTES: usize = 32 * 1024 * 1024;

pub fn unframe(buf: &[u8]) -> Option<(Value, usize)> {
    let header_end = find_subslice(buf, b"\r\n\r\n")?;
    let header = std::str::from_utf8(&buf[..header_end]).ok()?;
    let length = header.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case("content-length")
            .then(|| value.trim().parse::<usize>().ok())
            .flatten()
    })?;
    if length > MAX_FRAME_BYTES {
        return None; // caller treats this as a protocol failure, never a slice
    }
    let body_start = header_end.checked_add(4)?;
    let body_end = body_start.checked_add(length)?;
    if buf.len() < body_end {
        return None;
    }
    let body = std::str::from_utf8(&buf[body_start..body_end]).ok()?;
    let value = serde_json::from_str(body).ok()?;
    Some((value, body_end))
}

/// Whether the buffer can ever yield a frame (used to bound how much the
/// reader accumulates before treating the stream as broken).
pub fn buffer_is_recoverable(buf: &[u8]) -> bool {
    if buf.len() > MAX_FRAME_BYTES + 256 {
        return false;
    }
    if let Some(header_end) = find_subslice(buf, b"\r\n\r\n") {
        let header = std::str::from_utf8(&buf[..header_end]).unwrap_or("");
        if let Some(length) = header.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim()
                .eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        }) {
            return length <= MAX_FRAME_BYTES;
        }
    }
    // No complete header yet — could still be an in-flight header.
    buf.len() < 16_384
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unframe_rejects_oversized_declared_length() {
        let huge = "Content-Length: 999999999999\r\n\r\n{}";
        assert!(unframe(huge.as_bytes()).is_none());
        assert!(!buffer_is_recoverable(huge.as_bytes()));
        // A sane-but-incomplete frame stays recoverable.
        let partial = "Content-Length: 50\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":\"abc";
        assert!(unframe(partial.as_bytes()).is_none());
        assert!(buffer_is_recoverable(partial.as_bytes()));
    }

    #[test]
    fn frames_and_unframes_roundtrip() {
        let json = request("tools/list", Value::Object(Default::default()), 1);
        let framed = frame(&json);
        let (parsed, consumed) = unframe(&framed).expect("complete frame");
        assert_eq!(consumed, framed.len());
        assert_eq!(parsed, json);
        // Partial frames return None without consuming.
        assert!(unframe(&framed[..framed.len() - 5]).is_none());
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
