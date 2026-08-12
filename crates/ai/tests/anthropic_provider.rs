//! End-to-end tests of the Anthropic provider against a wiremock stub:
//! text streaming, thinking blocks, tool-call assembly, and HTTP errors.

use agent_m_ai::{
    AnthropicProvider, ChatRequest, ContentPart, LlmMessage, Provider, StopReason, StreamEvent,
};
use futures_util::StreamExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn provider(base_url: &str) -> AnthropicProvider {
    AnthropicProvider::new(
        "anthropic",
        "Anthropic",
        base_url,
        Some("test-key".to_string()),
        vec![agent_m_ai::ModelSpec::new("claude-3-5-sonnet-20241022")],
    )
}

fn request() -> ChatRequest {
    ChatRequest {
        model: "claude-3-5-sonnet-20241022".to_string(),
        system: "You are a test assistant.".to_string(),
        messages: vec![LlmMessage::User {
            content: "Hello".to_string(),
            images: vec![],
        }],
        tools: vec![],
        temperature: None,
        variant: None,
    }
}

async fn mount_sse(server: &MockServer, sse_body: &'static str) {
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse_body, "text/event-stream"),
        )
        .mount(server)
        .await;
}

async fn collect(provider: &AnthropicProvider, req: ChatRequest) -> Vec<StreamEvent> {
    provider
        .stream_chat(req)
        .await
        .expect("stream_chat should not fail")
        .collect::<Vec<_>>()
        .await
}

#[tokio::test]
async fn streams_text_delta_and_done() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10,\"cache_read_input_tokens\":4,\"cache_creation_input_tokens\":0,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"message\":{\"usage\":{\"output_tokens\":2}}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ),
    )
    .await;

    let events = collect(&provider(&server.uri()), request()).await;

    let text_deltas: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::TextDelta { delta } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text_deltas, vec!["Hello", " world"]);

    let done = events.iter().find_map(|e| match e {
        StreamEvent::Done { message } => Some(message),
        _ => None,
    });
    let done = done.expect("Done event must be present");
    assert_eq!(done.stop_reason, StopReason::Stop);
    let usage = done.usage.as_ref().expect("usage in Done");
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.cache_read_tokens, 4);
}

#[tokio::test]
async fn streams_thinking_delta() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Let me think...\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Answer\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"message\":{\"usage\":{\"output_tokens\":3}}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ),
    )
    .await;

    let events = collect(&provider(&server.uri()), request()).await;

    let thinking: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::ThinkingDelta { delta } => Some(delta.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(thinking, vec!["Let me think..."]);

    let done = events.iter().find_map(|e| match e {
        StreamEvent::Done { message } => Some(message),
        _ => None,
    });
    let content = &done.expect("Done").content;
    let has_thinking = content.iter().any(|p| matches!(p, ContentPart::Thinking { .. }));
    assert!(has_thinking, "thinking should be in Done content");
}

#[tokio::test]
async fn assembles_tool_call_from_chunks() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":20,\"cache_read_input_tokens\":0,\"cache_creation_input_tokens\":0,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"read\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"src/main.rs\\\"\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"}\"}}\n\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"message\":{\"usage\":{\"output_tokens\":10}}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ),
    )
    .await;

    let events = collect(&provider(&server.uri()), request()).await;

    let done = events.iter().find_map(|e| match e {
        StreamEvent::Done { message } => Some(message),
        _ => None,
    });
    let content = &done.expect("Done").content;
    let tool_call = content.iter().find_map(|p| match p {
        ContentPart::ToolCall { id, name, arguments } => Some((id, name, arguments)),
        _ => None,
    });
    let (id, name, args) = tool_call.expect("tool call in Done content");
    assert_eq!(id, "call_1");
    assert_eq!(name, "read");
    assert_eq!(args.get("path").and_then(|v| v.as_str()), Some("src/main.rs"));
}

#[tokio::test]
async fn http_error_returns_ai_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .respond_with(ResponseTemplate::new(429).set_body_string(
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"Rate limited"}}"#,
        ))
        .mount(&server)
        .await;

    let msg = match provider(&server.uri()).stream_chat(request()).await {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected Err, got Ok"),
    };
    assert!(
        msg.contains("429") || msg.contains("rate") || msg.contains("Rate") || msg.contains("API"),
        "unexpected error: {msg}"
    );
}
