//! REPL-side AskGate: prints the model's question to stdout, reads the user's
//! answer from stdin (blocking inside `block_in_place` so the async runtime
//! keeps running), and returns it as the tool result.

use agent_m_agent::ClosureAskGate;
use std::io::{BufRead, Write};

/// Build a `ClosureAskGate` that renders questions interactively in the REPL.
pub fn make_repl_ask_gate() -> ClosureAskGate<impl Fn(String, Option<Vec<String>>, bool)
    -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, String>> + Send>>
    + Send
    + Sync
    + 'static> {
    ClosureAskGate::new(|question, options, multi_select| {
        Box::pin(async move {
            tokio::task::block_in_place(|| {
                let stdout = std::io::stdout();
                let mut out = stdout.lock();

                // Print the question.
                let _ = writeln!(out, "\n\x1b[36m❓ {question}\x1b[0m");

                if let Some(ref opts) = options {
                    for (i, opt) in opts.iter().enumerate() {
                        if multi_select {
                            let _ = writeln!(out, "  [ ] {}. {}", i + 1, opt);
                        } else {
                            let _ = writeln!(out, "  {}. {}", i + 1, opt);
                        }
                    }
                    if multi_select {
                        let _ = writeln!(out, "  Enter numbers separated by spaces (blank = cancel):");
                    } else {
                        let _ = writeln!(out, "  Enter number or type your answer (blank = cancel):");
                    }
                } else {
                    let _ = write!(out, "  > ");
                }
                let _ = out.flush();
                drop(out);

                let stdin = std::io::stdin();
                let mut line = String::new();
                if stdin.lock().read_line(&mut line).is_err() || line.trim().is_empty() {
                    return Err("cancelled".to_string());
                }
                let trimmed = line.trim();

                // Resolve numeric selection(s) against the option list.
                if let Some(ref opts) = options {
                    if multi_select {
                        let selected: Vec<String> = trimmed
                            .split_whitespace()
                            .filter_map(|tok| tok.parse::<usize>().ok())
                            .filter(|&n| n >= 1 && n <= opts.len())
                            .map(|n| opts[n - 1].clone())
                            .collect();
                        return if selected.is_empty() {
                            Err("cancelled".to_string())
                        } else {
                            Ok(selected.join(", "))
                        };
                    }
                    if let Ok(n) = trimmed.parse::<usize>() {
                        if n >= 1 && n <= opts.len() {
                            return Ok(opts[n - 1].clone());
                        }
                    }
                }

                Ok(trimmed.to_string())
            })
        })
    })
}
