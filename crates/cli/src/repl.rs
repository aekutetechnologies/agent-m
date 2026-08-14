use crate::ansi;
use crate::commands::{CommandContext, CommandResult, handle_slash_command};
use crate::toolout::{ToolStore, humanize, summarize};
use agent_m_agent::{Agent, AgentOptions, Mode, PermissionGate, SessionMessage};
use agent_m_ai::{Provider, TrustData, extract_trust_block};
use anyhow::Result;
use crossterm::style::Color;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use std::borrow::Cow;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Default)]
pub struct StreamFilter {
    buffer: String,
}

impl StreamFilter {
    pub fn push(&mut self, delta: &str) -> String {
        self.buffer.push_str(delta);
        if let Some(pos) = self
            .buffer
            .find("<trust>")
            .or_else(|| self.buffer.find("<confidence>"))
        {
            let out = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos..].to_string();
            out
        } else if self.buffer.len() > 12 {
            let flush_len = self.buffer.len() - 12;
            let flush_len = self.buffer.floor_char_boundary(flush_len);
            let out = self.buffer[..flush_len].to_string();
            self.buffer = self.buffer[flush_len..].to_string();
            out
        } else {
            String::new()
        }
    }

    pub fn finish(&mut self) -> String {
        let text = std::mem::take(&mut self.buffer);
        let (_trust, cleaned) = agent_m_ai::extract_trust_block(&text);
        cleaned
    }
}

/// Per-turn "thinking..." indicator: an animated counter shown while the model
/// has not produced any visible output yet, plus a `(thought for Ns)` record
/// when the first visible content arrives.
#[derive(Clone, Default)]
struct Thinking {
    active: Arc<AtomicBool>,
    start: Arc<Mutex<Option<Instant>>>,
}

impl Thinking {
    fn begin(&self) {
        if let Ok(mut s) = self.start.lock() {
            *s = Some(Instant::now());
        }
        self.active.store(true, Ordering::Relaxed);
    }

    /// Stop the indicator if running, clear its line, and print a `(thought for
    /// Ns)` record. Returns the elapsed seconds when it was running.
    fn finish(&self) -> Option<f32> {
        if !self.active.swap(false, Ordering::Relaxed) {
            return None;
        }
        print!("\r\x1b[2K");
        let _ = std::io::stdout().flush();
        let secs = self
            .start
            .lock()
            .ok()
            .and_then(|mut s| s.take())
            .map(|started| started.elapsed().as_secs_f32());
        if let Some(secs) = secs {
            println!("{}", ansi::dim(&format!("(thought for {secs:.1}s)")));
            Some(secs)
        } else {
            println!();
            None
        }
    }

    /// Spawn the background animation task. It self-terminates once `finish`
    /// (or `abort`) clears `active`.
    fn spawn_animation(&self) {
        let active = self.active.clone();
        let start = self.start.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                if !active.load(Ordering::Relaxed) {
                    break;
                }
                let secs = start
                    .lock()
                    .ok()
                    .and_then(|s| *s)
                    .map(|started| started.elapsed().as_secs_f32())
                    .unwrap_or(0.0);
                print!("\r\x1b[2K{} ({secs:.1}s)", ansi::yellow("thinking..."));
                let _ = std::io::stdout().flush();
            }
        });
    }
}

const COMMANDS: &[&str] = &[
    "/exit", "/quit", "/sessions", "/undo", "/model", "/variant",
    "/mode", "/usage", "/level", "/harness", "/refine", "/todos",
    "/worktree", "/journal", "/checkpoint", "/restore", "/flows",
    "/compact", "/tool-output", "/tools", "/color", "/provider", "/tasks", "/help",
];

struct CommandHelper;

impl Completer for CommandHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        if !line.starts_with('/') {
            return Ok((pos, vec![]));
        }
        let prefix = &line[..pos];
        let candidates = COMMANDS
            .iter()
            .filter(|cmd| cmd.starts_with(prefix))
            .map(|cmd| Pair {
                display: cmd.to_string(),
                replacement: cmd.to_string(),
            })
            .collect();
        Ok((0, candidates))
    }
}

impl Hinter for CommandHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        if !line.starts_with('/') || pos != line.len() {
            return None;
        }
        COMMANDS
            .iter()
            .find(|cmd| cmd.starts_with(line))
            .map(|cmd| cmd[line.len()..].to_string())
    }
}

impl Highlighter for CommandHelper {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        Cow::Owned(format!("\x1b[2m{hint}\x1b[0m"))
    }
}
impl Validator for CommandHelper {}
impl rustyline::Helper for CommandHelper {}

pub async fn run_repl(
    provider: Arc<dyn Provider>,
    agent_options: AgentOptions,
    gate: Arc<dyn PermissionGate>,
    agent_dir: PathBuf,
    cwd: PathBuf,
    level_handle: Option<Arc<std::sync::atomic::AtomicU8>>,
) -> Result<()> {
    let config = rustyline::Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut rl =
        rustyline::Editor::<CommandHelper, rustyline::history::DefaultHistory>::with_config(config)?;
    rl.set_helper(Some(CommandHelper));
    let history_file = agent_dir.join("history.txt");
    let _ = rl.load_history(&history_file);

    println!(
        "{}",
        ansi::green("agent-m REPL mode. Type '/exit' or '/help' for commands.")
    );

    let session_stem = "repl-session";
    let provider_clone = provider.clone();
    let mut agent = Agent::new(provider, agent_options, gate);

    let tool_store = Arc::new(Mutex::new(ToolStore::default()));
    let store_listener = tool_store.clone();
    let current_tool = Arc::new(Mutex::new(None::<(String, serde_json::Value)>));
    let current_tool_listener = current_tool.clone();
    let stream_filter = Arc::new(Mutex::new(StreamFilter::default()));
    let filter_listener = stream_filter.clone();
    let thinking = Thinking::default();
    let thinking_listener = thinking.clone();
    // Collects the full assistant text for plan extraction after the turn.
    let turn_text = Arc::new(Mutex::new(String::new()));
    let turn_text_listener = turn_text.clone();
    // Per-turn streaming reply panel.
    let reply_box = Arc::new(Mutex::new(crate::section::ReplyBox::new(
        crate::section::terminal_width(),
    )));
    let reply_box_listener = reply_box.clone();
    // Trust metadata captured from the authoritative TurnEnd event (fallback to
    // re-parsing the raw turn text if the event is missed).
    let turn_trust = Arc::new(Mutex::new(None::<TrustData>));
    let turn_trust_listener = turn_trust.clone();
    // Undo snapshots need agent_dir after the move into the event closure.
    let undo_dir = agent_dir.clone();

    agent.subscribe(move |event| match event {
        agent_m_agent::AgentEvent::MessageUpdate {
            delta: agent_m_ai::StreamEvent::TextDelta { delta },
        } => {
            thinking_listener.finish();
            if let Ok(mut sf) = filter_listener.lock() {
                let to_print = sf.push(delta);
                if !to_print.is_empty()
                    && let Ok(mut rb) = reply_box_listener.lock()
                {
                    rb.push(&to_print);
                }
            }
            if let Ok(mut t) = turn_text_listener.lock() {
                t.push_str(delta);
            }
        }
        agent_m_agent::AgentEvent::ToolExecutionStart {
            name, arguments, ..
        } => {
            // check.md principle 8: snapshot write/edit targets before the
            // tool runs, so /undo can restore them.
            if matches!(name.as_str(), "write" | "edit") {
                if let Some(path) = arguments.get("path").and_then(serde_json::Value::as_str) {
                    let _ = crate::sessions::record_undo_snapshot(
                        &undo_dir,
                        session_stem,
                        std::path::Path::new(path),
                    );
                }
            }
            if let Ok(mut cur) = current_tool_listener.lock() {
                *cur = Some((name.clone(), arguments.clone()));
            }
            // Flush buffered text and close the reply box cleanly before the
            // tool box opens — prevents tool panels from rendering inside an
            // unclosed reply panel.
            if let Ok(mut sf) = filter_listener.lock() {
                let rest = sf.finish();
                if !rest.is_empty() {
                    if let Ok(mut rb) = reply_box_listener.lock() {
                        rb.push(&rest);
                    }
                }
            }
            if let Ok(mut rb) = reply_box_listener.lock() {
                rb.finish();
            }
            thinking_listener.finish();
        }
        agent_m_agent::AgentEvent::ToolExecutionEnd { outcome, .. } => {
            let (tool_name, args) = current_tool_listener
                .lock()
                .ok()
                .and_then(|mut cur| cur.take())
                .unwrap_or_else(|| ("?".to_string(), serde_json::json!({})));
            let content = outcome.content.clone();
            if let Ok(mut store) = store_listener.lock() {
                store.push(&tool_name, content.clone());
            }
            let (summary, extra_lines) = summarize(&content);
            let tail = if extra_lines > 0 {
                ansi::dim(&format!(" … (+{extra_lines} lines)"))
            } else {
                String::new()
            };
            let header = ansi::cyan(&humanize(&tool_name, &args));
            let body_line = if outcome.is_error {
                format!("{} {}", ansi::red("✗"), summary)
            } else {
                format!("↳ {}{}", ansi::dim(&summary), tail)
            };
            let lines = vec![header, body_line];
            let w = crate::section::terminal_width();
            println!(
                "{}",
                crate::section::render_box(
                    "tools",
                    &lines,
                    crate::section::SectionKind::Tools,
                    w
                )
            );
        }
        agent_m_agent::AgentEvent::TurnEnd {
            message: SessionMessage::Assistant { trust, .. },
            ..
        } => {
            if !trust.is_empty()
                && let Ok(mut t) = turn_trust_listener.lock()
            {
                *t = Some(trust.clone());
            }
        }
        agent_m_agent::AgentEvent::Notice { message } => {
            println!("\n{} {}", ansi::yellow("[Notice]"), message);
        }
        _ => {}
    });

    loop {
        let mode_tag = match agent.mode() {
            Mode::Plan => ansi::yellow("plan"),
            Mode::Build => ansi::green("build"),
        };
        let prompt_str = format!(
            "agent-m ({mode_tag} · {}) > ",
            ansi::cyan(agent.model())
        );
        let readline = tokio::task::block_in_place(|| rl.readline(&prompt_str));

        match readline {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(trimmed);
                let _ = rl.save_history(&history_file);

                if trimmed.starts_with('/') {
                    let mut ctx = CommandContext {
                        agent: &mut agent,
                        provider: provider_clone.clone(),
                        agent_dir: &agent_dir,
                        cwd: &cwd,
                        session_stem,
                        tools: tool_store.clone(),
                        level_handle: level_handle.clone(),
                    };
                    match handle_slash_command(trimmed, &mut ctx).await {
                        CommandResult::Exit => break,
                        CommandResult::Handled(msg) => {
                            println!("{msg}");
                            continue;
                        }
                        CommandResult::Continue => {}
                    }
                }

                // Reset per-turn state.
                if let Ok(mut sf) = stream_filter.lock() {
                    *sf = StreamFilter::default();
                }
                if let Ok(mut tt) = turn_trust.lock() {
                    *tt = None;
                }
                if let Ok(mut rb) = reply_box.lock() {
                    *rb = crate::section::ReplyBox::new(crate::section::terminal_width());
                }
                thinking.begin();
                thinking.spawn_animation();
                if let Err(e) = agent.prompt(trimmed.to_string()).await {
                    thinking.finish();
                    println!(
                        "{}",
                        crate::section::render_box(
                            "error",
                            &[format!("{e}")],
                            crate::section::SectionKind::Error,
                            crate::section::terminal_width()
                        )
                    );
                }
                // Flush any remaining streamed text, then close the reply panel.
                if let Ok(mut sf) = stream_filter.lock() {
                    let rest = sf.finish();
                    if !rest.is_empty()
                        && let Ok(mut rb) = reply_box.lock()
                    {
                        rb.push(&rest);
                    }
                }
                if let Ok(mut rb) = reply_box.lock() {
                    rb.finish();
                }
                thinking.finish();

                // Full assistant text for the turn (with the trust block still
                // present; used only as a fallback source for the decision block).
                let full_text = turn_text
                    .lock()
                    .map(|mut t| std::mem::take(&mut *t))
                    .unwrap_or_default();

                // Decision block: prefer the authoritative trust parsed by the
                // agent and emitted on TurnEnd; fall back to re-parsing the raw
                // turn text (e.g. if the event was missed).
                let trust = {
                    let from_event = turn_trust.lock().ok().and_then(|mut t| t.take());
                    from_event
                        .filter(|t| !t.is_empty())
                        .or_else(|| {
                            let (t, _) = extract_trust_block(&full_text);
                            if t.is_empty() {
                                None
                            } else {
                                Some(t)
                            }
                        })
                };
                if let Some(trust) = trust {
                    println!("{}", crate::section::print_decision(&trust));
                    // P9: report evidence citations that point at missing
                    // files or out-of-range lines (advisory at the REPL).
                    let problems =
                        agent_m_agent::check_evidence(&trust, &cwd);
                    if !problems.is_empty() {
                        for problem in problems {
                            println!(
                                "{} {problem}",
                                ansi::yellow("[evidence]")
                            );
                        }
                    }
                }

                // Plan extraction: if the reply contains a Plan: block, render
                // and persist it; apply [DONE:n] markers to any existing todos.
                let mut todos = crate::sessions::load_todos(&agent_dir, session_stem);
                let new_plan = crate::plan::parse_plan(&full_text);
                if !new_plan.is_empty() {
                    todos = new_plan;
                } else {
                    crate::plan::apply_done_markers(&full_text, &mut todos);
                }
                if !todos.is_empty() {
                    crate::sessions::save_todos(&agent_dir, session_stem, &todos).ok();
                    print_plan(&todos, agent.mode() == Mode::Plan);
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
    Ok(())
}

/// Print the plan as a boxed panel with `n/m` progress and a `· read-only`
/// suffix when in plan mode.
pub fn print_plan(todos: &[crate::plan::TodoItem], in_plan_mode: bool) {
    let completed = todos.iter().filter(|t| t.completed).count();
    let total = todos.len();
    let mut title = format!("plan ({completed}/{total})");
    if in_plan_mode {
        title.push_str(" · read-only");
    }
    let lines: Vec<String> = todos
        .iter()
        .map(|item| {
            let marker = if item.completed {
                ansi::fg_only("✓", Color::AnsiValue(ansi::MUTED_GREEN))
            } else {
                ansi::fg_only("○", Color::AnsiValue(ansi::MUTED_GREY))
            };
            let step = ansi::fg_only(&item.step.to_string(), Color::AnsiValue(ansi::MUTED_CYAN));
            let text = ansi::render_inline(&item.text);
            let text = if item.completed { ansi::dim(&text) } else { text };
            format!("{} {}. {}", marker, step, text)
        })
        .collect();
    println!(
        "{}",
        crate::section::render_box(
            &title,
            &lines,
            crate::section::SectionKind::Plan,
            crate::section::terminal_width()
        )
    );
}

#[cfg(test)]
mod tests {
    use super::StreamFilter;

    #[test]
    fn push_handles_multibyte_across_flush_boundary() {
        let mut sf = StreamFilter::default();
        let mut fed = String::new();
        let mut total = String::new();
        for delta in ["aaaaaaaa", "\u{2014}", "bbbbbbbbbbb"] {
            fed.push_str(delta);
            total.push_str(&sf.push(delta));
        }
        total.push_str(&sf.finish());
        assert_eq!(fed, total);
    }
}
