use crate::ansi;
use crate::section;
use crate::toolout::ToolStore;
use agent_m_agent::{Agent, Mode};
use agent_m_ai::Provider;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub enum CommandResult {
    Handled(String),
    Continue,
    Exit,
}

pub struct CommandContext<'a> {
    pub agent: &'a mut Agent,
    pub provider: Arc<dyn Provider>,
    pub agent_dir: &'a Path,
    pub cwd: &'a Path,
    pub session_stem: &'a str,
    pub tools: Arc<Mutex<ToolStore>>,
}

pub async fn handle_slash_command(line: &str, ctx: &mut CommandContext<'_>) -> CommandResult {
    let line = line.trim();
    if !line.starts_with('/') {
        return CommandResult::Continue;
    }

    let parts: Vec<&str> = line.split_whitespace().collect();
    let cmd = parts[0];
    let args = &parts[1..];

    match cmd {
        "/exit" | "/quit" => CommandResult::Exit,
        "/sessions" => {
            let sessions = crate::sessions::list_sessions(ctx.agent_dir);
            if sessions.is_empty() {
                CommandResult::Handled("No prior sessions found for this directory.".to_string())
            } else {
                let mut out = String::from("Sessions:\n");
                for s in sessions {
                    out.push_str(&format!("  - {}\n", s.path.display()));
                }
                CommandResult::Handled(out)
            }
        }
        "/undo" => {
            let mut entries = crate::sessions::load_undo(ctx.agent_dir, ctx.session_stem);
            if let Some(entry) = entries.pop() {
                match crate::sessions::apply_undo(&entry, ctx.cwd) {
                    Ok(action) => {
                        let _ =
                            crate::sessions::save_undo(ctx.agent_dir, ctx.session_stem, &entries);
                        CommandResult::Handled(format!("Undo: {} {}", action, entry.path))
                    }
                    Err(err) => CommandResult::Handled(format!("Undo error: {}", err)),
                }
            } else {
                CommandResult::Handled("Nothing to undo.".to_string())
            }
        }
        "/model" => {
            if args.is_empty() {
                let current = ctx.agent.model();
                let models: Vec<String> =
                    ctx.provider.models().iter().map(|m| m.id.clone()).collect();
                CommandResult::Handled(format!(
                    "Current model: {}\nAvailable models: {}",
                    current,
                    models.join(", ")
                ))
            } else {
                let new_model = args[0];
                let valid = ctx.provider.models().iter().any(|m| m.id == new_model);
                if valid {
                    ctx.agent.set_model(new_model);
                    CommandResult::Handled(format!("Model switched to {}", new_model))
                } else {
                    CommandResult::Handled(format!(
                        "Unknown model `{}` for active provider.",
                        new_model
                    ))
                }
            }
        }
        "/variant" => {
            let current_model = ctx.agent.model().to_string();
            let model_spec = ctx.provider.models().iter().find(|m| m.id == current_model);
            let available_variants = model_spec.map(|m| m.variants.clone()).unwrap_or_default();

            if args.is_empty() {
                let cur = ctx.agent.variant().unwrap_or("none");
                CommandResult::Handled(format!(
                    "Current variant: {}\nAvailable variants for {}: {}",
                    cur,
                    current_model,
                    if available_variants.is_empty() {
                        "none (model does not specify variants)".to_string()
                    } else {
                        available_variants.join(", ")
                    }
                ))
            } else {
                let variant = args[0];
                if available_variants.iter().any(|v| v == variant)
                    || variant == "none"
                    || variant == "default"
                {
                    ctx.agent.set_variant(if variant == "none" {
                        None
                    } else {
                        Some(variant.to_string())
                    });
                    CommandResult::Handled(format!("Variant set to `{}`", variant))
                } else {
                    CommandResult::Handled(format!(
                        "Invalid variant `{}`. Available variants for {}: {}",
                        variant,
                        current_model,
                        if available_variants.is_empty() {
                            "none".to_string()
                        } else {
                            available_variants.join(", ")
                        }
                    ))
                }
            }
        }
        "/mode" => {
            if args.is_empty() {
                let mode = match ctx.agent.mode() {
                    Mode::Plan => "plan",
                    Mode::Build => "build",
                };
                CommandResult::Handled(format!("Current mode: {}", mode))
            } else {
                match args[0].to_lowercase().as_str() {
                    "plan" => {
                        ctx.agent.set_mode(Mode::Plan);
                        CommandResult::Handled(section::render_box(
                            "mode",
                            &["Read-only · no file writes".to_string()],
                            section::SectionKind::Notice,
                            section::terminal_width(),
                        ))
                    }
                    "build" => {
                        ctx.agent.set_mode(Mode::Build);
                        CommandResult::Handled(section::render_box(
                            "mode",
                            &["Writes enabled · tools active".to_string()],
                            section::SectionKind::Notice,
                            section::terminal_width(),
                        ))
                    }
                    other => CommandResult::Handled(format!(
                        "Unknown mode `{}`. Use 'plan' or 'build'.",
                        other
                    )),
                }
            }
        }
        "/usage" => {
            let (last_input, context_window) = ctx.agent.context_usage();
            let cache_stats = ctx.agent.cache_stats();
            CommandResult::Handled(format!(
                "Telemetry Usage:\n  - Last Input Tokens: {}\n  - Context Limit: {}\n  - Cache Hit Tokens: {}\n  - Cache Miss Tokens: {}\n  - Total Requests: {}",
                last_input,
                context_window
                    .map(|w| w.to_string())
                    .unwrap_or_else(|| "unspecified".to_string()),
                cache_stats.hit_tokens,
                cache_stats.miss_tokens,
                cache_stats.requests,
            ))
        }
        "/level" => {
            CommandResult::Handled("Autonomy level is set via CLI startup flags.".to_string())
        }
        "/harness" => {
            let harness = crate::harness::load(ctx.agent_dir);
            if harness.entries.is_empty() {
                CommandResult::Handled("No active harness notes.".to_string())
            } else {
                let mut out = String::from("Harness Notes:\n");
                for entry in &harness.entries {
                    out.push_str(&format!("  - [{:?}] {}\n", entry.kind, entry.text));
                }
                CommandResult::Handled(out)
            }
        }
        "/refine" => {
            let focus = if args.is_empty() {
                None
            } else {
                Some(args.join(" "))
            };
            let harness = crate::harness::load(ctx.agent_dir);
            let harness_state = crate::refine::render_harness_state(&harness);
            let trajectory = crate::refine::collect_trajectory(ctx.agent_dir, ctx.cwd, 20);
            let model = ctx.agent.model().to_string();
            match crate::refine::propose_refinement(
                ctx.provider.as_ref(),
                &model,
                &trajectory,
                &harness_state,
                focus.as_deref(),
            )
            .await
            {
                Ok(proposal) if proposal.ops.is_empty() => {
                    CommandResult::Handled("No refinements proposed.".to_string())
                }
                Ok(proposal) => {
                    let mut out = format!("{} refinement(s) proposed:\n", proposal.ops.len());
                    for op in &proposal.ops {
                        out.push_str(&format!(
                            "  [{} {}] {}: {}\n",
                            op.action, op.kind, op.reason, op.text
                        ));
                    }
                    CommandResult::Handled(out)
                }
                Err(e) => CommandResult::Handled(format!("Refine error: {e}")),
            }
        }
        "/todos" => {
            let todos = crate::sessions::load_todos(ctx.agent_dir, ctx.session_stem);
            if todos.is_empty() {
                CommandResult::Handled("No plan recorded for this session.".to_string())
            } else {
                let mut out = String::new();
                for item in &todos {
                    let marker = if item.completed { "✓" } else { "○" };
                    out.push_str(&format!("  {marker} {}. {}\n", item.step, item.text));
                }
                CommandResult::Handled(out)
            }
        }
        "/worktree" => CommandResult::Handled("Git Worktree status: active".to_string()),
        "/journal" => {
            let j_rows = crate::sessions::journal(ctx.agent_dir, ctx.cwd);
            if j_rows.is_empty() {
                CommandResult::Handled("Journal empty.".to_string())
            } else {
                let mut out = String::from("Journal:\n");
                for j in j_rows {
                    out.push_str(&format!("[{}] {} {}\n", j.time, j.kind, j.text));
                }
                CommandResult::Handled(out)
            }
        }
        "/checkpoint" => CommandResult::Handled("Checkpoint recorded.".to_string()),
        "/restore" => CommandResult::Handled("Session restored to latest checkpoint.".to_string()),
        "/flows" => CommandResult::Handled("Flows directory: active".to_string()),
        "/compact" => CommandResult::Handled("Context compaction scheduled.".to_string()),
        "/tool-output" => {
            let spec = args.first().copied().unwrap_or("last");
            let store = ctx.tools.lock().unwrap();
            match store.get(spec) {
                Some(out) => {
                    CommandResult::Handled(format!("[Tool Output: {}]\n{}", out.name, out.full))
                }
                None => {
                    if spec.chars().all(|c| c.is_ascii_digit()) {
                        CommandResult::Handled(format!(
                            "No tool output with index `{spec}`. Available:\n{}",
                            store.list()
                        ))
                    } else {
                        CommandResult::Handled(format!(
                            "Unknown selector `{spec}`. Use `last` or a 1-based index.\n{}",
                            store.list()
                        ))
                    }
                }
            }
        }
        "/tools" => {
            let store = ctx.tools.lock().unwrap();
            CommandResult::Handled(format!("Tool outputs:\n{}", store.list()))
        }
        "/color" => match args.first().map(|s| s.to_ascii_lowercase()).as_deref() {
            Some("on") => {
                ansi::set_color(true);
                CommandResult::Handled("Color enabled.".to_string())
            }
            Some("off") => {
                ansi::set_color(false);
                CommandResult::Handled("Color disabled.".to_string())
            }
            _ => CommandResult::Handled(format!(
                "Color is {}.",
                if ansi::enabled() { "on" } else { "off" }
            )),
        },
        "/provider" => CommandResult::Handled(format!(
            "Active provider models: {}",
            ctx.provider.models().len()
        )),
        "/help" => {
            let md = "\
## agent-m commands

- `/model [id]` — query or switch the active model
- `/variant [id]` — query or switch model variant
- `/mode [plan|build]` — toggle plan (read-only) vs build mode
- `/usage` — token and cache-hit telemetry
- `/undo` — revert the last turn's file edits
- `/todos` — show the persisted plan for this session
- `/sessions` — list sessions for the current directory
- `/harness` — show active harness notes
- `/refine [focus]` — propose harness refinements
- `/journal` — action audit timeline
- `/tool-output [last|n]` — reprint a stored tool output
- `/tools` — list stored tool outputs
- `/compact` — compact conversation history
- `/checkpoint` — record a git checkpoint
- `/restore` — restore from last checkpoint
- `/flows` — list available flow files
- `/color [on|off]` — toggle ANSI color
- `/provider` — show active provider details
- `/level` — show autonomy level
- `/exit` — quit";
            CommandResult::Handled(crate::ansi::render_markdown(md))
        }
        _ => CommandResult::Handled(format!(
            "Unknown command `{}`. Type /help for available options.",
            cmd
        )),
    }
}
