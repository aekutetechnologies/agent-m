//! Byte-stable request serialization.
//!
//! Every object is built as `serde_json::Value`, whose `Map` is a `BTreeMap`
//! (sorted keys) unless the `preserve_order` feature is enabled. Combined with
//! field-stable message types, that makes request bodies deterministic: two
//! requests sharing a conversation prefix produce byte-identical JSON for that
//! prefix, which is what lets providers serve it from their prefix cache.

use serde_json::{Value, json};

use crate::types::{ChatRequest, ContentPart, LlmMessage, ToolSpec};

/// Serialize a slice of messages into the wire `messages` array.
///
/// Thinking parts are not sent back to OpenAI-compatible providers; tool calls
/// are translated into `tool_calls`, and tool results into `role: "tool"`.
pub fn serialize_messages(messages: &[LlmMessage]) -> Value {
    Value::Array(messages.iter().map(wire_message).collect())
}

/// Serialize a tool spec into the wire `tools` entry.
pub fn serialize_tool(tool: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
}

/// Build the full `/chat/completions` request body deterministically.
/// The system prompt is prepended as `messages[0]` so the cacheable prefix
/// (system + history) stays byte-identical across turns.
pub fn build_chat_request_body(request: &ChatRequest) -> Value {
    let mut messages: Vec<Value> = Vec::with_capacity(request.messages.len() + 1);
    if !request.system.is_empty() {
        messages.push(json!({ "role": "system", "content": request.system }));
    }
    if let Value::Array(rest) = serialize_messages(&request.messages) {
        messages.extend(rest);
    }
    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "stream": true,
    });
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(temperature);
    }
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.iter().map(serialize_tool).collect());
    }
    body
}

fn wire_message(message: &LlmMessage) -> Value {
    match message {
        LlmMessage::System { content } => {
            json!({ "role": "system", "content": content })
        }
        LlmMessage::User { content } => {
            json!({ "role": "user", "content": content })
        }
        LlmMessage::Assistant { content, .. } => {
            let mut text = String::new();
            let mut tool_calls = Vec::new();
            for part in content {
                match part {
                    ContentPart::Text { text: t } => text.push_str(t),
                    // Reasoning is not sent back to OpenAI-compatible providers.
                    ContentPart::Thinking { .. } => {}
                    ContentPart::ToolCall {
                        id,
                        name,
                        arguments,
                    } => tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::to_string(arguments)
                                .unwrap_or_else(|_| "{}".to_string()),
                        }
                    })),
                }
            }
            let mut message = json!({ "role": "assistant", "content": text });
            if !tool_calls.is_empty() {
                message["tool_calls"] = Value::Array(tool_calls);
            }
            message
        }
        LlmMessage::Tool {
            tool_call_id,
            name,
            content,
        } => {
            json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{StopReason, Usage};

    fn sample_messages() -> Vec<LlmMessage> {
        vec![
            LlmMessage::System {
                content: "You are a coding agent.".to_string(),
            },
            LlmMessage::User {
                content: "Read the file".to_string(),
            },
            LlmMessage::Assistant {
                content: vec![
                    ContentPart::Text {
                        text: "Let me look.".to_string(),
                    },
                    ContentPart::ToolCall {
                        id: "call_1".to_string(),
                        name: "read".to_string(),
                        arguments: json!({ "path": "src/main.rs" }),
                    },
                ],
                usage: Some(Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read_tokens: 0,
                    cache_creation_tokens: 10,
                    total_tokens: 15,
                    cost: 0.0,
                }),
                stop_reason: Some(StopReason::ToolUse),
            },
            LlmMessage::Tool {
                tool_call_id: "call_1".to_string(),
                name: "read".to_string(),
                content: "fn main() {}".to_string(),
            },
        ]
    }

    #[test]
    fn serialization_is_deterministic() {
        let messages = sample_messages();
        let a = serde_json::to_vec(&serialize_messages(&messages)).unwrap();
        let b = serde_json::to_vec(&serialize_messages(&messages)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn shared_prefix_is_byte_identical() {
        let base = sample_messages();
        let first = [
            base.clone(),
            vec![LlmMessage::User {
                content: "First follow-up".to_string(),
            }],
        ]
        .concat();
        let second = [
            base.clone(),
            vec![
                LlmMessage::User {
                    content: "First follow-up".to_string(),
                },
                LlmMessage::User {
                    content: "Second follow-up".to_string(),
                },
            ],
        ]
        .concat();

        let body_first = serde_json::to_vec(&build_chat_request_body(&ChatRequest {
            model: "deepseek-chat".to_string(),
            system: String::new(),
            messages: first.clone(),
            tools: vec![],
            temperature: None,
        }))
        .unwrap();
        let body_second = serde_json::to_vec(&build_chat_request_body(&ChatRequest {
            model: "deepseek-chat".to_string(),
            system: String::new(),
            messages: second.clone(),
            tools: vec![],
            temperature: None,
        }))
        .unwrap();

        // The cacheable unit is the message array: the second request's array
        // is the first's with entries appended, so its serialized bytes share
        // the first's bytes up to the first array's closing bracket.
        let messages_first = serde_json::to_vec(&serialize_messages(&first)).unwrap();
        let messages_second = serde_json::to_vec(&serialize_messages(&second)).unwrap();
        assert!(messages_second.starts_with(&messages_first[..messages_first.len() - 1]));

        // Structural equality of the shared prefix, independent of serialization.
        assert_eq!(first[..5], second[..5]);

        // The whole bodies also agree up to the end of the shared message list.
        let shared_prefix_len = body_first
            .iter()
            .position(|byte| *byte == b']')
            .expect("messages array close")
            + 1;
        assert_eq!(
            body_first[..shared_prefix_len],
            body_second[..shared_prefix_len]
        );
    }

    #[test]
    fn system_prompt_is_prepended_as_first_message() {
        let body = build_chat_request_body(&ChatRequest {
            model: "deepseek-chat".to_string(),
            system: "You are a coding agent.".to_string(),
            messages: vec![LlmMessage::User {
                content: "hi".to_string(),
            }],
            tools: vec![],
            temperature: None,
        });
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "You are a coding agent.");
        assert_eq!(body["messages"][1]["role"], "user");
        // An empty system prompt emits no system message (prefix stays stable).
        let no_system = build_chat_request_body(&ChatRequest {
            model: "deepseek-chat".to_string(),
            system: String::new(),
            messages: vec![LlmMessage::User {
                content: "hi".to_string(),
            }],
            tools: vec![],
            temperature: None,
        });
        assert_eq!(no_system["messages"][0]["role"], "user");
    }

    #[test]
    fn tool_call_wire_shape() {
        let messages = sample_messages();
        let wire = serialize_messages(&messages);
        let assistant = &wire[2];
        assert_eq!(assistant["role"], "assistant");
        assert_eq!(assistant["tool_calls"][0]["id"], "call_1");
        assert_eq!(assistant["tool_calls"][0]["function"]["name"], "read");
        assert_eq!(
            assistant["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"src/main.rs\"}"
        );
        let tool_result = &wire[3];
        assert_eq!(tool_result["role"], "tool");
        assert_eq!(tool_result["tool_call_id"], "call_1");
    }
}
