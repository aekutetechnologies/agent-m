use agent_m_agent::{Agent, AgentOptions, PermissionGate};
use agent_m_ai::Provider;
use anyhow::Result;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::io::Write;
use crate::commands::{CommandContext, CommandResult, handle_slash_command};
use crate::progress::TurnProgress;

#[derive(Default)]
pub struct StreamFilter {
    buffer: String,
}

impl StreamFilter {
    pub fn push(&mut self, delta: &str) -> String {
        self.buffer.push_str(delta);
        if let Some(pos) = self.buffer.find("<trust>").or_else(|| self.buffer.find("<confidence>")) {
            let out = self.buffer[..pos].to_string();
            self.buffer = self.buffer[pos..].to_string();
            out
        } else if self.buffer.len() > 12 {
            let flush_len = self.buffer.len() - 12;
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

pub async fn run_repl(
    provider: Arc<dyn Provider>,
    agent_options: AgentOptions,
    gate: Arc<dyn PermissionGate>,
    agent_dir: PathBuf,
    cwd: PathBuf,
) -> Result<()> {
    let mut rl = DefaultEditor::new()?;
    let history_file = agent_dir.join("history.txt");
    let _ = rl.load_history(&history_file);

    println!("agent-m REPL mode. Type '/exit' or '/help' for commands.");

    let session_stem = "repl-session";
    let provider_clone = provider.clone();
    let mut agent = Agent::new(provider, agent_options, gate);

    let progress = Arc::new(Mutex::new(TurnProgress::new()));
    let progress_listener = progress.clone();
    let stream_filter = Arc::new(Mutex::new(StreamFilter::default()));
    let filter_listener = stream_filter.clone();

    agent.subscribe(move |event| match event {
        agent_m_agent::AgentEvent::MessageUpdate {
            delta: agent_m_ai::StreamEvent::TextDelta { delta },
        } => {
            if let Ok(mut sf) = filter_listener.lock() {
                let to_print = sf.push(delta);
                if !to_print.is_empty() {
                    print!("{to_print}");
                    let _ = std::io::stdout().flush();
                }
            }
        }
        agent_m_agent::AgentEvent::ToolExecutionStart {
            name, arguments, ..
        } => {
            let args_summary = serde_json::to_string(&arguments).unwrap_or_default();
            if let Ok(mut p) = progress_listener.lock() {
                p.start_tool(name, &args_summary);
            }
        }
        agent_m_agent::AgentEvent::ToolExecutionEnd { outcome, .. } => {
            if let Ok(mut p) = progress_listener.lock() {
                p.finish_tool(outcome.is_error, &outcome.content);
            }
            if !outcome.is_error {
                println!("[Tool Output] {}", outcome.content);
            }
        }
        agent_m_agent::AgentEvent::Notice { message } => {
            println!("\n[Notice] {message}");
        }
        _ => {}
    });

    loop {
        let prompt_str = format!("agent-m ({}) > ", agent.model());
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
                    };
                    match handle_slash_command(trimmed, &mut ctx) {
                        CommandResult::Exit => break,
                        CommandResult::Handled(msg) => {
                            println!("{msg}");
                            continue;
                        }
                        CommandResult::Continue => {}
                    }
                }

                println!("\n...");
                if let Ok(mut sf) = stream_filter.lock() {
                    *sf = StreamFilter::default();
                }
                if let Err(e) = agent.prompt(trimmed.to_string()).await {
                    println!("[Error] {}", e);
                }
                if let Ok(mut sf) = stream_filter.lock() {
                    let rest = sf.finish();
                    if !rest.is_empty() {
                        print!("{rest}");
                        let _ = std::io::stdout().flush();
                    }
                }
                println!();
            },
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break;
            },
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            },
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }
    Ok(())
}
