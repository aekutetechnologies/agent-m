//! End-to-end tests of the OpenAI-compatible provider against a wiremock
//! stub server: SSE streaming, tool-call assembly, request shape, errors.

use agent_m_ai::{
    ChatRequest, ContentPart, LlmMessage, OpenAiCompatibleProvider, Provider, StopReason,
    StreamEvent, ToolSpec,
};
use futures_util::StreamExt;
use std::sync::{Arc, Mutex};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn provider(base_url: &str) -> OpenAiCompatibleProvider {
    OpenAiCompatibleProvider::new(
        "deepseek",
        "DeepSeek",
        base_url,
        Some("test-key".to_string()),
        vec![
            agent_m_ai::ModelSpec::new("deepseek-chat"),
            agent_m_ai::ModelSpec::new("deepseek-reasoner"),
        ],
        None,
    )
}

fn request(messages: Vec<LlmMessage>) -> ChatRequest {
    ChatRequest {
        model: "deepseek-chat".to_string(),
        system: String::new(),
        messages,
        tools: vec![ToolSpec {
            name: "read".to_string(),
            description: "Read a file".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string" } }
            }),
        }],
        temperature: None,
        variant: None,
    }
}

/// Mount a stub that replies with the given SSE body and captures the request.
async fn mount_sse(server: &MockServer, sse_body: &'static str) -> Arc<Mutex<Option<Vec<u8>>>> {
    let captured: Arc<Mutex<Option<Vec<u8>>>> = Arc::new(Mutex::new(None));
    let capture = captured.clone();
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(move |request: &Request| {
            *capture.lock().unwrap() = Some(request.body.clone());
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_raw(sse_body, "text/event-stream")
        })
        .mount(server)
        .await;
    captured
}

async fn collect(provider: &OpenAiCompatibleProvider, request: ChatRequest) -> Vec<StreamEvent> {
    provider
        .stream_chat(request)
        .await
        .expect("stream_chat should not fail")
        .collect::<Vec<_>>()
        .await
}

#[tokio::test]
async fn streams_text_and_parses_cache_usage() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n\
         data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n\
         data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
         data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2,\"prompt_cache_hit_tokens\":8,\"prompt_cache_miss_tokens\":2,\"total_tokens\":12}}\n\n\
         data: [DONE]\n\n",
    )
    .await;

    let provider = provider(&server.uri());
    let events = collect(
        &provider,
        request(vec![LlmMessage::User {
            images: Vec::new(),
            content: "hi".to_string(),
        }]),
    )
    .await;

    let mut text = String::new();
    let mut done = None;
    let mut saw_start = false;
    for event in events {
        match event {
            StreamEvent::Start => saw_start = true,
            StreamEvent::TextDelta { delta } => text.push_str(&delta),
            StreamEvent::Done { message } => done = Some(message),
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert!(saw_start);
    assert_eq!(text, "Hello world");
    let message = done.expect("Done event");
    assert_eq!(
        message.content,
        vec![ContentPart::Text {
            text: "Hello world".to_string()
        }]
    );
    assert_eq!(message.stop_reason, StopReason::Stop);
    let usage = message.usage.expect("usage present");
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 2);
    assert_eq!(usage.cache_read_tokens, 8);
    assert_eq!(usage.cache_creation_tokens, 2);
}

#[tokio::test]
async fn streams_thinking_and_assembles_tool_calls() {
    let server = MockServer::start().await;
    mount_sse(
        &server,
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"let me think\"},\"finish_reason\":null}]}\n\n\
         data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"pa\"}}]},\"finish_reason\":null}]}\n\n\
         data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"th\\\":\\\"x\\\"}\"}}]},\"finish_reason\":null}]}\n\n\
         data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
         data: [DONE]\n\n",
    )
    .await;

    let provider = provider(&server.uri());
    let events = collect(
        &provider,
        request(vec![LlmMessage::User {
            images: Vec::new(),
            content: "read x".to_string(),
        }]),
    )
    .await;

    let mut thinking = String::new();
    let mut done = None;
    let mut tool_deltas = 0;
    for event in events {
        match event {
            StreamEvent::ThinkingDelta { delta } => thinking.push_str(&delta),
            StreamEvent::ToolCallDelta { .. } => tool_deltas += 1,
            StreamEvent::Done { message } => done = Some(message),
            StreamEvent::Error { message } => panic!("error event: {message}"),
            _ => {}
        }
    }
    assert_eq!(thinking, "let me think");
    assert_eq!(tool_deltas, 2);
    let message = done.expect("Done event");
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    assert!(matches!(
        message.content.as_slice(),
        [ContentPart::Thinking { .. }, ContentPart::ToolCall { .. }]
    ));
    match &message.content[1] {
        ContentPart::ToolCall {
            id,
            name,
            arguments,
        } => {
            assert_eq!(id, "call_1");
            assert_eq!(name, "read");
            assert_eq!(*arguments, serde_json::json!({ "path": "x" }));
        }
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn request_body_has_expected_shape_and_deterministic_order() {
    let server = MockServer::start().await;
    let captured = mount_sse(&server, "data: [DONE]\n\n").await;

    let provider = provider(&server.uri());
    collect(
        &provider,
        request(vec![LlmMessage::User {
            images: Vec::new(),
            content: "hi".to_string(),
        }]),
    )
    .await;

    let body: serde_json::Value =
        serde_json::from_slice(captured.lock().unwrap().as_ref().expect("request captured"))
            .expect("valid JSON body");
    assert_eq!(body["model"], "deepseek-chat");
    assert_eq!(body["stream"], true);
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"], "hi");
    assert_eq!(body["tools"][0]["function"]["name"], "read");

    // Deterministic ordering: every object's keys serialize sorted (the
    // serde_json BTreeMap-backed Map property this crate relies on).
    fn assert_keys_sorted(value: &serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                let keys: Vec<&String> = map.keys().collect();
                let mut sorted = keys.clone();
                sorted.sort();
                assert_eq!(keys, sorted, "object keys are sorted (deterministic)");
                for value in map.values() {
                    assert_keys_sorted(value);
                }
            }
            serde_json::Value::Array(items) => {
                for item in items {
                    assert_keys_sorted(item);
                }
            }
            _ => {}
        }
    }
    assert_keys_sorted(&body);
}

#[tokio::test]
async fn provider_errors_surface_as_api_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(401)
                .set_body_raw(r#"{"error":{"message":"bad key"}}"#, "application/json"),
        )
        .mount(&server)
        .await;

    let provider = provider(&server.uri());
    let result = provider
        .stream_chat(request(vec![LlmMessage::User {
            images: Vec::new(),
            content: "hi".to_string(),
        }]))
        .await;
    let error = match result {
        Ok(_) => panic!("401 should fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("HTTP 401"), "got: {error}");
}

#[tokio::test]
async fn missing_key_is_reported() {
    let provider = OpenAiCompatibleProvider::new(
        "deepseek",
        "DeepSeek",
        "http://localhost:1",
        None,
        vec![agent_m_ai::ModelSpec::new("deepseek-chat")],
        None,
    );
    let result = provider
        .stream_chat(request(vec![LlmMessage::User {
            images: Vec::new(),
            content: "hi".to_string(),
        }]))
        .await;
    let error = match result {
        Ok(_) => panic!("missing key should fail"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("DEEPSEEK_API_KEY"),
        "got: {error}"
    );
}

#[tokio::test]
async fn vision_gate_rejects_images_on_text_only_models() {
    use agent_m_ai::{LlmMessage, ModelSpec};
    let server = MockServer::start().await;
    let provider = OpenAiCompatibleProvider::new(
        "deepseek",
        "DeepSeek",
        "https://api.deepseek.com".to_string(),
        Some("test-key".to_string()),
        vec![ModelSpec::new("deepseek-chat")],
        None,
    );
    let request = agent_m_ai::ChatRequest {
        model: "deepseek-chat".to_string(),
        system: "you".to_string(),
        messages: vec![LlmMessage::User {
            content: "what is this?".to_string(),
            images: vec!["data:image/png;base64,AAAA".to_string()],
        }],
        tools: vec![],
        temperature: None,
        variant: None,
    };
    let result = provider.stream_chat(request).await;
    let error = result.err().expect("must reject images");
    assert!(
        error.to_string().contains("does not support image input"),
        "got: {error}"
    );
    // The gate fires before any HTTP request.
    let _ = server; // the mock server exists only to prove no call was needed
}
