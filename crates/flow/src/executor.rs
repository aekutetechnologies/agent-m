//! Flow executor: runs steps sequentially, records per-step status, stores
//! outputs in the shared context, and aborts on failure.

use agent_m_agent::{
    Agent, AgentOptions, AskGate, Mode, PermissionGate, SessionMessage, Tool, ToolContext,
};
use agent_m_ai::Provider;
use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Arc;

use crate::model::{Flow, FlowContext, FlowStep};

/// Dependencies the executor needs to run prompt/tool/ask steps.
pub struct FlowDeps {
    pub provider: Arc<dyn Provider>,
    /// Template agent options: model, system_prompt, tools, cwd, context window.
    /// No gate — permission_gate is passed separately to avoid stale clones.
    pub agent_options: AgentOptions,
    /// All registered tools (built-in + plugin). `tool` steps resolve here.
    pub tools: Vec<Arc<dyn Tool>>,
    /// Permission gate for every tool call in this flow: tool steps, verify
    /// commands, and the model's own calls inside prompt/phase/verify-fix steps.
    pub permission_gate: Arc<dyn PermissionGate>,
    /// Ask gate for `ask` steps (None → ask steps fail with a clear message).
    pub ask_gate: Option<Arc<dyn AskGate>>,
    /// Where per-flow state artifacts live (`<dir>/<flow>/STATE.md`,
    /// `CONTEXT.json`). None disables artifact writing.
    pub state_dir: Option<PathBuf>,
    /// Live progress callback: fired with `Running` before each top-level
    /// step executes and with the final status when it completes.
    pub on_progress: Option<Arc<dyn Fn(FlowProgress) + Send + Sync>>,
}

/// A live progress notification for one top-level flow step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowProgress {
    pub step_index: usize,
    pub step_name: String,
    pub status: StepStatus,
}

/// Per-step status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Skipped,
}

impl StepStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            StepStatus::Pending => "pending",
            StepStatus::Running => "running",
            StepStatus::Succeeded => "succeeded",
            StepStatus::Failed => "failed",
            StepStatus::Skipped => "skipped",
        }
    }
}

/// The recorded result of one step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub name: String,
    pub step_type: String,
    pub status: StepStatus,
    pub output: Option<Value>,
    pub error: Option<String>,
}

/// The full result of a flow run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowRun {
    pub flow_name: String,
    pub steps: Vec<StepRecord>,
    pub context: FlowContext,
}

/// Run a flow. Steps execute in order; a failed step aborts the flow (unless
/// it is an `ask` step with `on_reject: continue`).
pub async fn run_flow(flow: &Flow, context: &mut FlowContext, deps: &FlowDeps) -> Result<FlowRun> {
    let mut records = Vec::new();
    run_steps(flow, &flow.steps, 0, context, deps, &mut records).await?;
    Ok(FlowRun {
        flow_name: flow.name.clone(),
        steps: records,
        context: context.clone(),
    })
}

fn run_steps<'a>(
    flow: &'a Flow,
    steps: &'a [FlowStep],
    depth: usize,
    context: &'a mut FlowContext,
    deps: &'a FlowDeps,
    records: &'a mut Vec<StepRecord>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
    Box::pin(async move {
        for (index, step) in steps.iter().enumerate() {
            // Progress is reported for top-level steps only (the sidebar maps
            // them by index; nested branch/phase steps are their concern).
            if depth == 0
                && let Some(on_progress) = &deps.on_progress
            {
                on_progress(FlowProgress {
                    step_index: index,
                    step_name: step.name().to_string(),
                    status: StepStatus::Running,
                });
            }
            let record = run_step(flow, step, depth, context, deps).await?;
            if depth == 0
                && let Some(on_progress) = &deps.on_progress
            {
                on_progress(FlowProgress {
                    step_index: index,
                    step_name: step.name().to_string(),
                    status: record.status,
                });
            }
            let failed = record.status == StepStatus::Failed;
            let continued =
                matches!(step, FlowStep::Ask { on_reject, .. } if on_reject == "continue");
            // Flatten condition branches into the top-level record list so the
            // UI/CLI shows the steps that actually ran.
            let branches: Vec<StepRecord> = if matches!(step, FlowStep::Condition { .. }) {
                record
                    .output
                    .as_ref()
                    .and_then(|output| output.get("branches"))
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(|item| {
                                serde_json::from_value::<StepRecord>(item.clone()).ok()
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let step_error = record.error.clone();
            records.push(record);
            records.extend(branches);
            write_state(deps, flow, records, context);
            if failed && !continued {
                bail!(
                    "flow aborted at step `{}`: {}",
                    step.name(),
                    step_error.as_deref().unwrap_or("step failed")
                );
            }
        }
        Ok(())
    })
}

/// ECC-style strategic compaction: compact at turn boundaries once usage
/// passes `threshold` (a fraction of the window) — never mid-run.
pub fn should_compact(tokens: u64, window: u64, threshold: f64) -> bool {
    window > 0 && tokens as f64 > window as f64 * threshold
}

/// Persist GSD-style state artifacts for the flow run (STATE.md + the full
/// context as CONTEXT.json), so a flow's progress survives restarts.
fn write_state(deps: &FlowDeps, flow: &Flow, records: &[StepRecord], context: &FlowContext) {
    let Some(root) = &deps.state_dir else {
        return;
    };
    let dir = root.join(&flow.name);
    let _ = std::fs::create_dir_all(&dir);
    let mut state = String::from(
        "# STATE.md — flow navigation

",
    );
    state.push_str(&format!(
        "## Flow: {}

",
        flow.name
    ));
    for record in records {
        state.push_str(&format!(
            "- [{}] **{}** ({})
",
            match record.status {
                StepStatus::Succeeded => "x",
                StepStatus::Failed => "!",
                StepStatus::Skipped => "~",
                _ => " ",
            },
            record.name,
            record.step_type
        ));
    }
    let _ = std::fs::write(dir.join("STATE.md"), state);
    let _ = std::fs::write(
        dir.join("CONTEXT.json"),
        serde_json::to_string_pretty(&context).unwrap_or_default(),
    );
}

fn run_step<'a>(
    flow: &'a Flow,
    step: &'a FlowStep,
    depth: usize,
    context: &'a mut FlowContext,
    deps: &'a FlowDeps,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<StepRecord>> + Send + 'a>> {
    Box::pin(async move { run_step_inner(flow, step, depth, context, deps).await })
}

async fn run_step_inner(
    flow: &Flow,
    step: &FlowStep,
    depth: usize,
    context: &mut FlowContext,
    deps: &FlowDeps,
) -> Result<StepRecord> {
    tracing::info!("flow `{}`: running step `{}`", flow.name, step.name());
    match step {
        FlowStep::Prompt {
            name,
            mode,
            message,
            model,
        } => {
            let prompt = context.expand(message.as_deref().unwrap_or(""));
            let mode = match mode.as_deref().unwrap_or("build") {
                "plan" => Mode::Plan,
                _ => Mode::Build,
            };
            let mut options = deps.agent_options.clone();
            options.mode = mode;
            options.ask_gate = deps.ask_gate.clone();
            if let Some(model) = model {
                options.model = model.clone();
            }
            let mut agent =
                Agent::new(deps.provider.clone(), options, deps.permission_gate.clone());
            let deltas = Arc::new(std::sync::Mutex::new(String::new()));
            let capture = deltas.clone();
            agent.subscribe(move |event| {
                if let agent_m_agent::AgentEvent::MessageUpdate {
                    delta: agent_m_ai::StreamEvent::TextDelta { delta },
                } = event
                {
                    capture.lock().unwrap().push_str(delta);
                }
            });
            match agent.prompt(prompt).await {
                Ok(()) => {
                    let output = agent
                        .messages()
                        .iter()
                        .rev()
                        .find_map(|message| match message {
                            SessionMessage::Assistant { content, .. } => Some(
                                content
                                    .iter()
                                    .filter_map(|part| match part {
                                        agent_m_ai::ContentPart::Text { text } => {
                                            Some(text.as_str())
                                        }
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            ),
                            _ => None,
                        })
                        .unwrap_or_else(|| deltas.lock().unwrap().clone());
                    set_step_output(context, name, json!({ "output": output }));
                    Ok(StepRecord {
                        name: name.clone(),
                        step_type: "prompt".to_string(),
                        status: StepStatus::Succeeded,
                        output: Some(Value::String(output)),
                        error: None,
                    })
                }
                Err(error) => {
                    set_step_output(context, name, json!({ "status": "failed" }));
                    Ok(StepRecord {
                        name: name.clone(),
                        step_type: "prompt".to_string(),
                        status: StepStatus::Failed,
                        output: None,
                        error: Some(error.to_string()),
                    })
                }
            }
        }
        FlowStep::Ask {
            name,
            question,
            options,
            on_reject: _,
        } => {
            let question = context.expand(question);
            let Some(gate) = &deps.ask_gate else {
                return Ok(StepRecord {
                    name: name.clone(),
                    step_type: "ask".to_string(),
                    status: StepStatus::Failed,
                    output: None,
                    error: Some(
                        "ask steps require the interactive UI (not available in print mode)"
                            .to_string(),
                    ),
                });
            };
            match gate.ask(question, options.clone()).await {
                Ok(answer) => {
                    set_step_output(context, name, json!({ "answer": answer }));
                    Ok(StepRecord {
                        name: name.clone(),
                        step_type: "ask".to_string(),
                        status: StepStatus::Succeeded,
                        output: Some(json!({ "answer": answer })),
                        error: None,
                    })
                }
                Err(message) => {
                    set_step_output(context, name, json!({ "status": "rejected" }));
                    Ok(StepRecord {
                        name: name.clone(),
                        step_type: "ask".to_string(),
                        status: StepStatus::Failed,
                        output: None,
                        error: Some(message),
                        // `on_reject: continue` is handled by the caller.
                    })
                }
            }
        }
        FlowStep::Tool {
            name,
            tool,
            arguments,
        } => {
            let expanded = arguments
                .as_ref()
                .map(|value| expand_json(value, context))
                .unwrap_or(Value::Object(Default::default()));
            let Some(tool_impl) = deps.tools.iter().find(|t| t.name() == tool) else {
                return Ok(StepRecord {
                    name: name.clone(),
                    step_type: "tool".to_string(),
                    status: StepStatus::Failed,
                    output: None,
                    error: Some(format!("unknown tool `{tool}`")),
                });
            };
            // Tools run through the permission gate (approvals, destructive
            // command denial) exactly like agent tool calls.
            let call = agent_m_agent::ToolCallInfo {
                tool_call_id: name.clone(),
                name: tool.clone(),
                arguments: expanded.clone(),
            };
            if let agent_m_agent::Permission::Denied(reason) =
                deps.permission_gate.authorize(&call).await
            {
                return Ok(StepRecord {
                    name: name.clone(),
                    step_type: "tool".to_string(),
                    status: StepStatus::Failed,
                    output: None,
                    error: Some(format!("permission denied: {reason}")),
                });
            }
            let tool_context = ToolContext {
                cwd: deps.agent_options.cwd.clone(),
                ask_gate: deps.ask_gate.clone(),
            };
            match tool_impl.execute(expanded, &tool_context).await {
                Ok(outcome) => {
                    let output = json!({ "content": outcome.content, "isError": outcome.is_error });
                    set_step_output(context, name, output.clone());
                    let status = if outcome.is_error {
                        StepStatus::Failed
                    } else {
                        StepStatus::Succeeded
                    };
                    Ok(StepRecord {
                        name: name.clone(),
                        step_type: "tool".to_string(),
                        status,
                        output: Some(output),
                        error: outcome.is_error.then_some(outcome.content),
                    })
                }
                Err(error) => Ok(StepRecord {
                    name: name.clone(),
                    step_type: "tool".to_string(),
                    status: StepStatus::Failed,
                    output: None,
                    error: Some(error.to_string()),
                }),
            }
        }
        FlowStep::Condition {
            name,
            if_condition,
            then,
            else_,
        } => {
            let expanded = context.expand(if_condition);
            let truthy = evaluate_condition(&expanded);
            let branch = if truthy {
                then
            } else {
                else_.as_deref().unwrap_or(&[])
            };
            let mut branch_records = Vec::new();
            run_steps(flow, branch, depth + 1, context, deps, &mut branch_records).await?;
            let skipped = branch.is_empty();
            Ok(StepRecord {
                name: name.clone(),
                step_type: "condition".to_string(),
                status: if skipped {
                    StepStatus::Skipped
                } else {
                    StepStatus::Succeeded
                },
                output: Some(json!({ "branches": branch_records })),
                error: None,
            })
        }
        FlowStep::Phase {
            name,
            prompt,
            steps,
        } => {
            // A phase is a named group. If it has a prompt, run it first as a
            // build-mode agent step, then the nested steps.
            let mut nested = Vec::new();
            if let Some(prompt) = prompt {
                let record = run_step(
                    flow,
                    &FlowStep::Prompt {
                        name: format!("{name}:prompt"),
                        mode: None,
                        message: Some(prompt.clone()),
                        model: None,
                    },
                    depth,
                    context,
                    deps,
                )
                .await?;
                nested.push(record);
                if nested[0].status == StepStatus::Failed {
                    return Ok(StepRecord {
                        name: name.clone(),
                        step_type: "phase".to_string(),
                        status: StepStatus::Failed,
                        output: None,
                        error: nested[0].error.clone(),
                    });
                }
            }
            run_steps(flow, steps, depth + 1, context, deps, &mut nested).await?;
            Ok(StepRecord {
                name: name.clone(),
                step_type: "phase".to_string(),
                status: StepStatus::Succeeded,
                output: Some(json!({ "steps": nested })),
                error: None,
            })
        }
        FlowStep::Verify {
            name,
            command,
            max_fix_rounds,
        } => run_verify(name, command.as_deref(), *max_fix_rounds, context, deps).await,
    }
}

/// GSD-style verify: run the check command; on failure, loop a bounded number
/// of fix rounds (each: a build-mode agent prompt seeded with the failure,
/// then re-run the command) until it passes.
async fn run_verify(
    name: &str,
    command: Option<&str>,
    max_fix_rounds: usize,
    context: &mut FlowContext,
    deps: &FlowDeps,
) -> Result<StepRecord> {
    let command = command.unwrap_or("cargo test");
    let bash = deps.tools.iter().find(|t| t.name() == "bash");
    let tool_context = ToolContext {
        cwd: deps.agent_options.cwd.clone(),
        ask_gate: deps.ask_gate.clone(),
    };
    let run_once = move |cmd: String| {
        let tool_context = tool_context.clone();
        let bash = bash;
        let permission_gate = deps.permission_gate.clone();
        async move {
            let Some(bash) = bash else {
                return Err(anyhow!("verify needs the bash tool"));
            };
            let call = agent_m_agent::ToolCallInfo {
                tool_call_id: format!("verify-{name}"),
                name: "bash".to_string(),
                arguments: json!({ "command": cmd.clone() }),
            };
            if let agent_m_agent::Permission::Denied(reason) =
                permission_gate.authorize(&call).await
            {
                return Err(anyhow!("permission denied: {reason}"));
            }
            match bash.execute(json!({ "command": cmd }), &tool_context).await {
                Ok(outcome) => Ok(outcome),
                Err(error) => Err(anyhow!(error.to_string())),
            }
        }
    };

    let mut output = run_once(context.expand(command)).await?;
    let mut rounds = 0;
    while output.is_error && rounds < max_fix_rounds {
        rounds += 1;
        tracing::info!("verify `{name}`: fix round {rounds}");
        let mut options = deps.agent_options.clone();
        options.mode = Mode::Build;
        options.ask_gate = deps.ask_gate.clone();
        let mut agent = Agent::new(deps.provider.clone(), options, deps.permission_gate.clone());
        let deltas = Arc::new(std::sync::Mutex::new(String::new()));
        let capture = deltas.clone();
        agent.subscribe(move |event| {
            if let agent_m_agent::AgentEvent::MessageUpdate {
                delta: agent_m_ai::StreamEvent::TextDelta { delta },
            } = event
            {
                capture.lock().unwrap().push_str(delta);
            }
        });
        let fix_prompt = format!(
            "The verification command failed. Fix the root cause.\n\nCommand: {command}\n\nOutput:\n{}\n\nReply with the fix summary when done.",
            output.content
        );
        if let Err(error) = agent.prompt(fix_prompt).await {
            let _ = run_once;
            return Ok(StepRecord {
                name: name.to_string(),
                step_type: "verify".to_string(),
                status: StepStatus::Failed,
                output: None,
                error: Some(error.to_string()),
            });
        }
        output = run_once(context.expand(command)).await?;
    }
    let status = if output.is_error {
        StepStatus::Failed
    } else {
        StepStatus::Succeeded
    };
    let result =
        json!({ "output": output.content, "isError": output.is_error, "fix_rounds": rounds });
    set_step_output(context, name, result.clone());
    Ok(StepRecord {
        name: name.to_string(),
        step_type: "verify".to_string(),
        status,
        output: Some(result),
        error: output.is_error.then_some(output.content),
    })
}

/// Evaluate a simple condition: `"${a} == true"`, `"${a} != \"x\""`, or a
/// bare truthy value.
fn evaluate_condition(expanded: &str) -> bool {
    let trimmed = expanded.trim();
    for (operator, negate) in [("!=", true), ("==", false)] {
        if let Some(index) = trimmed.find(operator) {
            let left = trimmed[..index].trim().trim_matches('"');
            let right = trimmed[index + operator.len()..].trim().trim_matches('"');
            let equal = left == right;
            return if negate { !equal } else { equal };
        }
    }
    // Bare value: truthiness.
    match trimmed {
        "true" | "yes" | "1" => true,
        "false" | "no" | "0" | "" => false,
        other => !other.is_empty() && other != "empty",
    }
}

/// Store `value` at `steps.<name>.output` and set the status (nested form so
/// dotted paths like `${steps.jira.output.content}` resolve).
fn set_step_output(context: &mut FlowContext, name: &str, value: Value) {
    let steps = context
        .values
        .entry("steps".to_string())
        .or_insert_with(|| json!({}));
    let step = steps
        .as_object_mut()
        .expect("steps context entry is an object")
        .entry(name.to_string())
        .or_insert_with(|| json!({}));
    let entry = step.as_object_mut().expect("step entry is an object");
    entry.insert("output".to_string(), value);
    entry.insert("status".to_string(), Value::String("done".to_string()));
}

/// Recursively expand `${ref}` strings inside a JSON value.
fn expand_json(value: &Value, context: &FlowContext) -> Value {
    match value {
        Value::String(text) => Value::String(context.expand(text)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| expand_json(item, context))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), expand_json(value, context)))
                .collect(),
        ),
        other => other.clone(),
    }
}

impl FlowStep {
    /// The step's name (used for progress display and logs).
    pub fn name(&self) -> &str {
        match self {
            FlowStep::Prompt { name, .. }
            | FlowStep::Ask { name, .. }
            | FlowStep::Tool { name, .. }
            | FlowStep::Condition { name, .. }
            | FlowStep::Phase { name, .. }
            | FlowStep::Verify { name, .. } => name,
        }
    }
}

impl std::fmt::Display for StepStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Load a flow from a YAML file.
pub fn load_flow(path: &PathBuf) -> Result<Flow> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| anyhow!("cannot read flow {}: {error}", path.display()))?;
    serde_yml::from_str(&text).map_err(|error| anyhow!("cannot parse flow: {error}"))
}
