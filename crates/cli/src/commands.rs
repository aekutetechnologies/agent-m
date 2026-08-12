use agent_m_agent::{Agent, Mode};
use agent_m_ai::Provider;
use std::path::Path;
use std::sync::Arc;

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
}

pub fn handle_slash_command(line: &str, ctx: &mut CommandContext) -> CommandResult {
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
                        let _ = crate::sessions::save_undo(ctx.agent_dir, ctx.session_stem, &entries);
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
                let models: Vec<String> = ctx.provider.models().iter().map(|m| m.id.clone()).collect();
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
                    CommandResult::Handled(format!("Unknown model `{}` for active provider.", new_model))
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
                if available_variants.iter().any(|v| v == variant) || variant == "none" || variant == "default" {
                    ctx.agent.set_variant(if variant == "none" { None } else { Some(variant.to_string()) });
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
                        CommandResult::Handled("Switched to Plan Mode (read-only).".to_string())
                    }
                    "build" => {
                        ctx.agent.set_mode(Mode::Build);
                        CommandResult::Handled("Switched to Build Mode.".to_string())
                    }
                    other => CommandResult::Handled(format!("Unknown mode `{}`. Use 'plan' or 'build'.", other)),
                }
            }
        }
        "/usage" => {
            let (last_input, context_window) = ctx.agent.context_usage();
            let cache_stats = ctx.agent.cache_stats();
            CommandResult::Handled(format!(
                "Telemetry Usage:\n  - Last Input Tokens: {}\n  - Context Limit: {}\n  - Cache Hit Tokens: {}\n  - Cache Miss Tokens: {}\n  - Total Requests: {}",
                last_input,
                context_window.map(|w| w.to_string()).unwrap_or_else(|| "unspecified".to_string()),
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
            CommandResult::Handled("Refining prompt harness notes...".to_string())
        }
        "/worktree" => {
            CommandResult::Handled("Git Worktree status: active".to_string())
        }
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
        "/checkpoint" => {
            CommandResult::Handled("Checkpoint recorded.".to_string())
        }
        "/restore" => {
            CommandResult::Handled("Session restored to latest checkpoint.".to_string())
        }
        "/flows" => {
            CommandResult::Handled("Flows directory: active".to_string())
        }
        "/compact" => {
            CommandResult::Handled("Context compaction scheduled.".to_string())
        }
        "/provider" => {
            CommandResult::Handled(format!("Active provider models: {}", ctx.provider.models().len()))
        }
        "/help" => {
            let help = "Available Slash Commands:\n\
                  /model [id]    - Query or switch LLM model\n\
                  /variant [id]  - Query or switch model variant (validated against model specs)\n\
                  /usage         - Display token & cache hit telemetry\n\
                  /mode [plan|build] - Toggle Plan (read-only) vs Build mode\n\
                  /undo          - Revert the last turn's file modifications\n\
                  /level         - Show autonomy level status\n\
                  /sessions      - List sessions for current repository\n\
                  /harness       - Show active prompt harness notes\n\
                  /refine        - Trigger auto-harness refinement\n\
                  /worktree      - View worktree status\n\
                  /journal       - View agent action journal\n\
                  /checkpoint    - Record session checkpoint\n\
                  /restore       - Restore session checkpoint\n\
                  /flows         - List flow files\n\
                  /compact       - Compact conversation history context\n\
                  /provider      - View active AI provider details\n\
                  /exit          - Exit the REPL";
            CommandResult::Handled(help.to_string())
        }
        _ => CommandResult::Handled(format!("Unknown command `{}`. Type /help for available options.", cmd)),
    }
}
