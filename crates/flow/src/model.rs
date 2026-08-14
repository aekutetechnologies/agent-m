//! Flow model: a YAML-defined pipeline of steps, plus the
//! shared context with `${step.output}` reference substitution.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A full flow definition (parsed from YAML).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flow {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// Cap on the TOTAL number of fix rounds across all `verify` steps in one
    /// run. When exhausted, the verify step stops-and-reports instead of
    /// burning more model calls. `None` = unlimited (per-step
    /// `max_fix_rounds` still applies).
    #[serde(default)]
    pub error_budget: Option<usize>,
    #[serde(default)]
    pub steps: Vec<FlowStep>,
}

/// One step in a flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum FlowStep {
    /// Run an agent prompt (plan-mode or build-mode). `message` may contain
    /// `${ref}` substitutions from the flow context.
    Prompt {
        name: String,
        #[serde(default)]
        mode: Option<String>,
        #[serde(default)]
        message: Option<String>,
        #[serde(default)]
        model: Option<String>,
    },
    /// Ask the user a question (human review gate). Rejecting aborts the
    /// flow unless `on_reject: continue`.
    Ask {
        name: String,
        question: String,
        #[serde(default)]
        options: Option<Vec<String>>,
        #[serde(default = "default_stop")]
        on_reject: String,
    },
    /// Run one registered tool (built-in or plugin) directly.
    Tool {
        name: String,
        tool: String,
        #[serde(default)]
        arguments: Option<Value>,
    },
    /// Branch on a context value: `if: "${steps.plan.output} == true"`.
    Condition {
        name: String,
        #[serde(rename = "if")]
        if_condition: String,
        #[serde(default)]
        then: Vec<FlowStep>,
        #[serde(default)]
        else_: Option<Vec<FlowStep>>,
    },
    /// GSD-style fresh-context phase: runs `steps` in a fresh sub-agent with
    /// a compact state handoff (see the phase step implementation).
    Phase {
        name: String,
        #[serde(default)]
        prompt: Option<String>,
        #[serde(default)]
        steps: Vec<FlowStep>,
    },
    /// GSD-style verify: run a test/check command and loop fixes for a
    /// bounded number of rounds.
    Verify {
        name: String,
        #[serde(default)]
        command: Option<String>,
        #[serde(default = "default_fix_rounds")]
        max_fix_rounds: usize,
    },
    /// Run a fresh structured sub-agent (the delegate protocol): the
    /// `prompt` runs in a brand-new context window that MUST reply with a
    /// single JSON object (conforming to `schema` when given). The parsed
    /// JSON is stored at `${steps.<name>.output.json}` and the
    /// pretty-printed text at `${steps.<name>.output.content}`, so
    /// downstream steps can reference real fields
    /// (`${steps.<name>.output.json.risk}`). Invalid JSON fails the step
    /// unless `on_invalid: continue`.
    Delegate {
        name: String,
        prompt: String,
        #[serde(default)]
        schema: Option<Value>,
        #[serde(default)]
        tools: Option<Vec<String>>,
        #[serde(default = "default_delegate_turns")]
        max_turns: usize,
        #[serde(default = "default_stop")]
        on_invalid: String,
    },
}

fn default_stop() -> String {
    "stop".to_string()
}
fn default_fix_rounds() -> usize {
    3
}
fn default_delegate_turns() -> usize {
    4
}

/// Shared, serializable flow state. Step outputs are stored under
/// `steps.<name>.output`, statuses under `steps.<name>.status`; arbitrary
/// inputs can be pre-seeded and are expanded with `${...}`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlowContext {
    pub values: serde_json::Map<String, Value>,
}

impl FlowContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        self.values.insert(key.into(), value);
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        // Exact flat key first, then a dotted-path walk.
        if let Some(value) = self.values.get(key) {
            return Some(value);
        }
        let mut current = None;
        for (index, segment) in key.split('.').enumerate() {
            if index == 0 {
                current = self.values.get(segment);
            } else {
                current = current.and_then(|value| value.get(segment));
            }
            current?;
        }
        current
    }

    /// Replace every `${path}` occurrence with the resolved value (as a
    /// string for scalars, JSON for objects/arrays).
    pub fn expand(&self, template: &str) -> String {
        let re = regex::Regex::new(r"\$\{([a-zA-Z0-9_.-]+)\}").expect("static regex");
        re.replace_all(template, |caps: &regex::Captures| {
            let path = &caps[1];
            match self.get(path) {
                Some(Value::String(text)) => text.clone(),
                Some(other) => serde_json::to_string(other).unwrap_or_default(),
                None => caps[0].to_string(),
            }
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_yaml_flow() {
        let yaml = r#"
name: agentic-dev
description: issue to PR
steps:
  - type: tool
    name: jira
    tool: jira-search
    arguments: { query: "login" }
  - type: prompt
    name: plan
    mode: plan
    message: "Create a plan for ${steps.jira.output}"
  - type: condition
    name: check
    if: "${steps.jira.output} != empty"
    then:
      - type: ask
        name: review
        question: "Approve the plan?"
"#;
        let flow: Flow = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(flow.name, "agentic-dev");
        assert_eq!(flow.steps.len(), 3);
        assert!(matches!(flow.steps[0], FlowStep::Tool { .. }));
        assert!(matches!(flow.steps[1], FlowStep::Prompt { .. }));
        match &flow.steps[2] {
            FlowStep::Condition { then, .. } => assert!(matches!(then[0], FlowStep::Ask { .. })),
            _ => panic!("expected condition"),
        }
    }

    #[test]
    fn parses_delegate_step() {
        let yaml = r#"
name: structured
steps:
  - type: delegate
    name: analysis
    prompt: "Map entrypoints and rate the risk"
    schema:
      type: object
      properties:
        risk: { type: string }
    tools: [read, search]
    max_turns: 6
    on_invalid: continue
"#;
        let flow: Flow = serde_yml::from_str(yaml).expect("parse");
        assert_eq!(flow.steps.len(), 1);
        match &flow.steps[0] {
            FlowStep::Delegate {
                name,
                prompt,
                schema,
                tools,
                max_turns,
                on_invalid,
            } => {
                assert_eq!(name, "analysis");
                assert_eq!(prompt, "Map entrypoints and rate the risk");
                assert_eq!(
                    schema.as_ref().and_then(|s| s.get("type")),
                    Some(&Value::String("object".to_string()))
                );
                assert_eq!(
                    tools.as_ref().unwrap(),
                    &vec!["read".to_string(), "search".to_string()]
                );
                assert_eq!(*max_turns, 6);
                assert_eq!(on_invalid, "continue");
            }
            _ => panic!("expected delegate step"),
        }
        // Defaults: no schema/tools, max_turns 4, on_invalid stop.
        let minimal: Flow =
            serde_yml::from_str("name: x\nsteps:\n  - type: delegate\n    name: d\n    prompt: p\n")
                .expect("parse");
        match &minimal.steps[0] {
            FlowStep::Delegate {
                schema, tools, max_turns, on_invalid, ..
            } => {
                assert!(schema.is_none());
                assert!(tools.is_none());
                assert_eq!(*max_turns, 4);
                assert_eq!(on_invalid, "stop");
            }
            _ => panic!("expected delegate step"),
        }
    }

    #[test]
    fn expands_references() {
        let mut context = FlowContext::new();
        context.set("steps.jira.output", Value::String("ticket ABC".to_string()));
        context.set("steps.plan.status", Value::String("succeeded".to_string()));
        assert_eq!(
            context.expand("got ${steps.jira.output} with ${steps.plan.status}"),
            "got ticket ABC with succeeded"
        );
        // Missing refs stay literal.
        assert_eq!(context.expand("x ${missing.path}"), "x ${missing.path}");
    }
}
