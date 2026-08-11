//! The `ask` tool: the model asks the user a question and waits for the
//! answer. The answer is returned as the tool result so the model can
//! continue. Requires the interactive UI (an `AskGate`); in print mode the
//! gate is absent and `ask` fails with a clear message.

use agent_m_agent::{Tool, ToolContext, ToolError, ToolOutcome};
use async_trait::async_trait;
use serde_json::{Value, json};

pub struct AskTool;

#[async_trait]
impl Tool for AskTool {
    fn name(&self) -> &str {
        "ask"
    }

    fn description(&self) -> String {
        "Ask the user a clarifying question and wait for their answer. Use this when the task is ambiguous and you need input before continuing.".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user"
                },
                "options": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional suggested answers shown as a picker"
                },
                "multi_select": {
                    "type": "boolean",
                    "description": "Allow the user to select multiple options (requires options)"
                }
            },
            "required": ["question"]
        })
    }

    async fn execute(
        &self,
        arguments: Value,
        context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let question = arguments
            .get("question")
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::failed("ask", "missing string argument `question`"))?
            .to_string();
        let options = arguments
            .get("options")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            });
        let multi_select = arguments
            .get("multi_select")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let Some(gate) = &context.ask_gate else {
            return Ok(ToolOutcome::error(
                "the ask tool requires the interactive UI (not available in print mode)",
            ));
        };
        match gate.ask(question, options, multi_select).await {
            Ok(answer) => Ok(ToolOutcome::success(format!("User answer: {answer}"))),
            Err(message) => Ok(ToolOutcome::error(format!("ask cancelled: {message}"))),
        }
    }
}
