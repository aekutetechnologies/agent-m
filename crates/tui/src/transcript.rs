//! Transcript items: the blocks rendered in the scrollable chat area.

use agent_m_agent::ToolOutcome;
use agent_m_ai::{ContentPart, StopReason, TrustData};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::Value;

use crate::markdown::{line_height, render_markdown};
use crate::theme::Theme;

/// Maximum result lines shown for a collapsed tool execution (`ctrl+o` expands).
pub const COLLAPSED_TOOL_LINES: usize = 20;

/// One rendered block in the transcript.
#[derive(Debug, Clone)]
pub enum TranscriptItem {
    User {
        content: String,
    },
    Assistant {
        parts: Vec<ContentPart>,
        stop_reason: StopReason,
        /// Force-show the `Thinking` part in full even if this item is
        /// stale (`ctrl+r`), mirroring `ToolExecution.expanded`.
        thinking_expanded: bool,
        /// Trust metadata from the reply's <trust> block (decision block).
        trust: agent_m_ai::TrustData,
    },
    ToolExecution {
        tool_call_id: String,
        name: String,
        arguments: Value,
        result: Option<ToolOutcome>,
        expanded: bool,
    },
    /// A parsed task plan (rendered as a checkable list).
    Plan {
        todos: Vec<crate::plan::TodoItem>,
    },
    Notice {
        message: String,
    },
}

impl TranscriptItem {
    /// Render the item to styled lines. Paragraphs are wrapped by ratatui at
    /// draw time; use [`TranscriptItem::height`] for layout. `stale` marks an
    /// item from a completed, no-longer-current turn: tool output and
    /// thinking collapse to a compact receipt/summary unless explicitly
    /// expanded (`ctrl+o`/`ctrl+r`), which always wins over `stale`.
    pub fn render(&self, theme: &Theme, _width: usize, stale: bool) -> Vec<Line<'static>> {
        match self {
            TranscriptItem::User { content } => {
                // pi/Claude style: a solid background block with a leading
                // prompt chevron — no borders, one blank line after.
                let mut lines = Vec::new();
                for (index, line) in render_markdown(content, theme).into_iter().enumerate() {
                    let mut styled = line;
                    for span in &mut styled.spans {
                        if span.style.fg.is_none() {
                            span.style = Style::default().fg(theme.user_message_text);
                        }
                        span.style = span.style.bg(theme.user_message_bg);
                    }
                    if index == 0 {
                        styled.spans.insert(
                            0,
                            Span::styled(
                                "❯ ",
                                Style::default().fg(theme.accent).bg(theme.user_message_bg),
                            ),
                        );
                    }
                    lines.push(styled);
                }
                lines.push(Line::default());
                lines
            }
            TranscriptItem::Assistant {
                parts,
                stop_reason,
                thinking_expanded,
                trust,
            } => {
                let mut lines = Vec::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            lines.extend(render_markdown(text, theme));
                        }
                        ContentPart::Thinking { thinking } => {
                            if stale && !*thinking_expanded {
                                lines.push(Line::from(Span::styled(
                                    format!(
                                        "🧠 reasoned ({} lines) — ctrl+r to expand",
                                        thinking.lines().count()
                                    ),
                                    Style::default().fg(theme.warning),
                                )));
                            } else {
                                let style = Style::default()
                                    .fg(theme.thinking_text)
                                    .add_modifier(Modifier::ITALIC);
                                for raw_line in thinking.lines() {
                                    lines.push(Line::from(Span::styled(
                                        format!("… {raw_line}"),
                                        style,
                                    )));
                                }
                            }
                        }
                        ContentPart::ToolCall { .. } => {
                            // Tool calls are rendered as ToolExecution items.
                        }
                    }
                }
                match stop_reason {
                    StopReason::Error => lines.push(Line::from(Span::styled(
                        "(reply failed — see status)".to_string(),
                        Style::default().fg(theme.error),
                    ))),
                    StopReason::Aborted => lines.push(Line::from(Span::styled(
                        "(aborted)".to_string(),
                        Style::default().fg(theme.muted),
                    ))),
                    StopReason::Length => lines.push(Line::from(Span::styled(
                        "(response truncated)".to_string(),
                        Style::default().fg(theme.warning),
                    ))),
                    _ => {}
                }
                lines.extend(render_decision_block(trust, theme));
                lines.push(Line::default());
                lines
            }
            TranscriptItem::ToolExecution {
                name,
                arguments,
                result,
                expanded,
                ..
            } => {
                // pi style: a full-width box tinted by status, bold title
                // line, gray output — no border lines.
                let bg = match result {
                    None => theme.tool_pending_bg,
                    Some(outcome) if outcome.is_error => theme.tool_error_bg,
                    Some(_) => theme.tool_success_bg,
                };
                let mut lines = Vec::new();
                let title = tool_title(name, arguments);
                lines.push(Line::from(Span::styled(
                    title,
                    Style::default()
                        .bg(bg)
                        .fg(theme.user_message_text)
                        .add_modifier(Modifier::BOLD),
                )));

                if let Some(outcome) = result {
                    if stale && !*expanded {
                        // Stale receipt: the title line above is enough.
                    } else {
                        let content_lines: Vec<&str> = outcome.content.lines().collect();
                        let show = if *expanded {
                            content_lines.as_slice()
                        } else if content_lines.len() > COLLAPSED_TOOL_LINES {
                            &content_lines[..COLLAPSED_TOOL_LINES]
                        } else {
                            content_lines.as_slice()
                        };
                        let color = if outcome.is_error {
                            theme.error
                        } else {
                            theme.muted
                        };
                        let style = Style::default().bg(bg).fg(color);
                        for line in show {
                            lines.push(Line::from(Span::styled(line.to_string(), style)));
                        }
                        if content_lines.len() > COLLAPSED_TOOL_LINES && !*expanded {
                            lines.push(Line::from(Span::styled(
                                format!(
                                    "… {} more lines (ctrl+o to expand)",
                                    content_lines.len() - COLLAPSED_TOOL_LINES
                                ),
                                Style::default().bg(bg).fg(theme.warning),
                            )));
                        }
                        if outcome.is_error {
                            lines.push(Line::from(Span::styled(
                                "(tool failed)",
                                Style::default().bg(bg).fg(theme.error),
                            )));
                        }
                    }
                } else {
                    lines.push(Line::from(Span::styled(
                        "Running…",
                        Style::default().bg(bg).fg(theme.muted),
                    )));
                }
                lines.push(Line::default());
                lines
            }
            TranscriptItem::Notice { message } => {
                vec![Line::from(Span::styled(
                    message.clone(),
                    Style::default().fg(theme.warning),
                ))]
            }
            TranscriptItem::Plan { todos } => {
                // A checkable task list, styled like the tool boxes.
                let bg = theme.tool_pending_bg;
                let mut lines = vec![Line::from(Span::styled(
                    format!(
                        "📋 Plan ({}/{})",
                        todos.iter().filter(|t| t.completed).count(),
                        todos.len()
                    ),
                    Style::default()
                        .bg(bg)
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD),
                ))];
                for todo in todos {
                    let marker = if todo.completed { "✓" } else { "○" };
                    let fg = if todo.completed {
                        theme.dim
                    } else {
                        theme.user_message_text
                    };
                    lines.push(Line::from(Span::styled(
                        format!(" {marker} {}. {}", todo.step, todo.text),
                        Style::default().bg(bg).fg(fg),
                    )));
                }
                lines.push(Line::default());
                lines
            }
        }
    }

    /// Total rendered height at the given width (wrap-aware). `stale` must
    /// match whatever was passed to [`TranscriptItem::render`] for the same
    /// item, or the two will disagree on line count.
    pub fn height(&self, theme: &Theme, width: usize, stale: bool) -> usize {
        match self {
            TranscriptItem::User { content } => {
                crate::markdown::markdown_height(content, theme, width) + 1
            }
            TranscriptItem::Assistant {
                parts,
                stop_reason,
                thinking_expanded,
                trust,
            } => {
                let mut height = 0usize;
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            height += crate::markdown::markdown_height(text, theme, width);
                        }
                        ContentPart::Thinking { thinking } => {
                            if stale && !*thinking_expanded {
                                height += 1;
                            } else {
                                height += thinking
                                    .lines()
                                    .map(|line| line_height(&Line::from(line.to_string()), width))
                                    .sum::<usize>();
                            }
                        }
                        ContentPart::ToolCall { .. } => {}
                    }
                }
                if matches!(
                    stop_reason,
                    StopReason::Error | StopReason::Aborted | StopReason::Length
                ) {
                    height += 1;
                }
                height += decision_height(trust, width);
                height += 1; // trailing spacer
                height
            }
            TranscriptItem::ToolExecution {
                result, expanded, ..
            } => {
                let mut height = 1; // title
                if let Some(outcome) = result {
                    if !stale || *expanded {
                        let content_lines: Vec<&str> = outcome.content.lines().collect();
                        let shown = if *expanded {
                            content_lines.len()
                        } else {
                            content_lines.len().min(COLLAPSED_TOOL_LINES)
                        };
                        for line in content_lines.iter().take(shown) {
                            height += line_height(&Line::from(line.to_string()), width);
                        }
                        if content_lines.len() > COLLAPSED_TOOL_LINES && !*expanded {
                            height += 1;
                        }
                        if outcome.is_error {
                            height += 1;
                        }
                    }
                } else {
                    height += 1; // "Running…"
                }
                height += 1; // trailing spacer
                height
            }
            TranscriptItem::Notice { .. } => 1,
            TranscriptItem::Plan { todos } => 1 + todos.len() + 1,
        }
    }
}

/// The bold title line for a tool-execution block, pi-style:
/// `$ <command>` for bash, `read <path>:12-30`, `edit <path>`, …
/// Lines for the trust "decision" block under an assistant reply. Every
/// line is one rendered row (markdown-free) so height is `lines.len()`.
fn render_decision_block(trust: &TrustData, theme: &Theme) -> Vec<Line<'static>> {
    if trust.is_empty() {
        return Vec::new();
    }
    let mut lines = Vec::new();
    let accent = Style::default().fg(theme.accent);
    lines.push(Line::from(Span::styled("── decision ──", accent)));
    if let Some(reason) = &trust.reason {
        lines.push(Line::from(Span::styled(
            format!("why: {reason}"),
            Style::default().fg(theme.dim),
        )));
    }
    if let Some(confidence) = trust.confidence {
        let (tier, _) = trust.confidence_tier().expect("tier for value");
        let (color, label) = match tier {
            "low" => (theme.error, "low"),
            "medium" => (theme.warning, "medium"),
            _ => (Color::Green, "high"),
        };
        let cells = (confidence as usize * 10) / 100;
        let gauge: String = "█".repeat(cells) + &"░".repeat(10 - cells);
        lines.push(Line::from(vec![Span::styled(
            format!("confidence: {gauge} {confidence}% ({label})"),
            color,
        )]));
    }
    for evidence in &trust.evidence {
        let location = match evidence.line {
            Some(line) => format!("{}:{line}", evidence.file),
            None => evidence.file.clone(),
        };
        let note = evidence.note.as_deref().unwrap_or("");
        lines.push(Line::from(vec![
            Span::styled("evidence: ", Style::default().fg(theme.dim)),
            Span::styled(format!("{location}{note}"), Color::Green),
        ]));
    }
    if let Some(outcome) = &trust.expected_outcome {
        lines.push(Line::from(Span::styled(
            format!("expect: {outcome}"),
            Style::default().fg(theme.dim),
        )));
    }
    if let Some(uncertainty) = &trust.uncertainty {
        lines.push(Line::from(Span::styled(
            format!("uncertain: {uncertainty}"),
            Style::default().fg(theme.warning),
        )));
    }
    if !trust.plan.is_empty() {
        let steps: Vec<String> = trust
            .plan
            .iter()
            .enumerate()
            .map(|(index, step)| format!("{}. {step}", index + 1))
            .collect();
        let mut plan_line = format!("plan: {}", steps.join(" → "));
        if let Some(time) = &trust.estimated_time {
            plan_line.push_str(&format!("  (~{time})"));
        }
        lines.push(Line::from(Span::styled(
            plan_line,
            Style::default().fg(theme.accent),
        )));
    }
    lines
}

/// Height contribution of the decision block (one row per rendered line).
fn decision_height(trust: &TrustData, width: usize) -> usize {
    if trust.is_empty() {
        return 0;
    }
    // Wrap long single-row fields (reason, outcome, uncertainty, plan).
    let mut height = 1; // header
    height += trust
        .reason
        .as_ref()
        .map(|r| line_height(&Line::from(r.clone()), width))
        .unwrap_or(0);
    height += usize::from(trust.confidence.is_some());
    height += trust.evidence.len();
    height += trust
        .expected_outcome
        .as_ref()
        .map(|o| line_height(&Line::from(o.clone()), width))
        .unwrap_or(0);
    height += trust
        .uncertainty
        .as_ref()
        .map(|u| line_height(&Line::from(u.clone()), width))
        .unwrap_or(0);
    if !trust.plan.is_empty() {
        let joined: String = trust.plan.join(" → ");
        height += line_height(&Line::from(joined), width);
    }
    height
}

/// Principle 1 (transparency): a deterministic "what is happening" line for
/// the status bar while a tool executes — no model cost.
pub fn narration(name: &str, arguments: &Value) -> String {
    let path = || arguments.get("path").and_then(Value::as_str).unwrap_or("");
    let value = |key: &str| arguments.get(key).and_then(Value::as_str).unwrap_or("");
    let truncate = |s: &str| {
        let head: String = s.chars().take(120).collect();
        if head.chars().count() < s.chars().count() {
            format!("{head}…")
        } else {
            head
        }
    };
    match name {
        "bash" => format!("Running `{}`…", truncate(value("command"))),
        "read" => format!("Reading {}…", path()),
        "write" => format!("Writing {}…", path()),
        "edit" => format!("Editing {}…", path()),
        "grep" => format!("Searching `{}`…", truncate(value("pattern"))),
        "find" => format!("Finding files under {}…", path()),
        "ls" => format!("Listing {}…", path()),
        "search" => format!("Searching `{}`…", truncate(value("query"))),
        "ask" => "Asking you…".to_string(),
        other => format!("Running {other}…"),
    }
}

fn tool_title(name: &str, arguments: &Value) -> String {
    let path = || arguments.get("path").and_then(Value::as_str).unwrap_or("");
    let truncate = |s: &str| {
        let head: String = s.chars().take(160).collect();
        if head.chars().count() < s.chars().count() {
            format!("{head}…")
        } else {
            head
        }
    };
    match name {
        "bash" => format!(
            "$ {}",
            truncate(
                arguments
                    .get("command")
                    .and_then(Value::as_str)
                    .unwrap_or("")
            )
        ),
        "read" => {
            let offset = arguments.get("offset").and_then(Value::as_u64);
            let limit = arguments.get("limit").and_then(Value::as_u64);
            match (offset, limit) {
                (Some(start), Some(len)) => format!(
                    "read {}:{}-{}",
                    path(),
                    start,
                    start.saturating_add(len).saturating_sub(1)
                ),
                (Some(start), None) => format!("read {}:{start}", path()),
                _ => format!("read {}", path()),
            }
        }
        "grep" => {
            let pattern = arguments
                .get("pattern")
                .and_then(Value::as_str)
                .unwrap_or("");
            let root = arguments.get("path").and_then(Value::as_str).unwrap_or(".");
            format!("grep /{pattern}/ in {root}")
        }
        "write" | "edit" | "ls" | "find" => format!("{name} {}", path()),
        _ => format!(
            "{name} {}",
            serde_json::to_string(arguments).unwrap_or_default()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::Theme;

    #[test]
    fn narration_describes_the_active_tool() {
        let args = |key: &str, value: &str| {
            let mut map = serde_json::Map::new();
            map.insert(
                key.to_string(),
                serde_json::Value::String(value.to_string()),
            );
            serde_json::Value::Object(map)
        };
        assert_eq!(
            narration("bash", &args("command", "cargo test")),
            "Running `cargo test`…"
        );
        assert_eq!(
            narration("read", &args("path", "src/main.rs")),
            "Reading src/main.rs…"
        );
        assert_eq!(
            narration("edit", &args("path", "src/app.rs")),
            "Editing src/app.rs…"
        );
        assert_eq!(
            narration("grep", &args("pattern", "TODO")),
            "Searching `TODO`…"
        );
        assert_eq!(narration("ask", &serde_json::json!({})), "Asking you…");
        assert_eq!(
            narration("custom-tool", &serde_json::json!({})),
            "Running custom-tool…"
        );
    }

    #[test]
    fn decision_block_renders_confidence_and_evidence() {
        let trust = TrustData {
            confidence: Some(85),
            reason: Some("Expiry is now validated.".to_string()),
            evidence: vec![agent_m_ai::Evidence {
                file: "auth.ts".to_string(),
                line: Some(83),
                note: Some("no expiry check".to_string()),
            }],
            uncertainty: Some("Not load-tested.".to_string()),
            plan: vec!["Inspect logs".to_string(), "Run tests".to_string()],
            estimated_time: Some("30 seconds".to_string()),
            ..Default::default()
        };
        let theme = Theme::default();
        let lines = render_decision_block(&trust, &theme);
        let text: String = lines
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(
                "
",
            );
        assert!(text.contains("── decision ──"), "header: {text}");
        assert!(text.contains("confidence: ████████"), "gauge: {text}");
        assert!(text.contains("85% (high)"), "tier: {text}");
        assert!(
            text.contains("auth.ts:83no expiry check"),
            "evidence: {text}"
        );
        assert!(
            text.contains("uncertain: Not load-tested."),
            "uncertainty: {text}"
        );
        assert!(
            text.contains("plan: 1. Inspect logs → 2. Run tests  (~30 seconds)"),
            "plan: {text}"
        );
        // Empty trust renders nothing.
        assert!(render_decision_block(&TrustData::default(), &theme).is_empty());
    }

    #[test]
    fn user_block_has_chevron_and_background() {
        let theme = Theme::dark();
        let item = TranscriptItem::User {
            content: "hello".to_string(),
        };
        let lines = item.render(&theme, 40, false);
        assert!(lines.len() >= 2);
        let first = &lines[0].spans;
        assert!(first.iter().any(|span| span.content.starts_with('❯')));
        assert!(
            first
                .iter()
                .all(|span| span.style.bg == Some(theme.user_message_bg))
        );
        // No full-width border lines (pi/Claude style).
        assert!(lines.iter().all(|line| {
            !line
                .spans
                .iter()
                .any(|span| span.content.starts_with('─') && span.content.contains("────"))
        }));
    }

    #[test]
    fn collapsed_tool_output_is_truncated() {
        let theme = Theme::dark();
        let long = (0..50)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let item = TranscriptItem::ToolExecution {
            tool_call_id: "c1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({ "command": "ls" }),
            result: Some(ToolOutcome::success(long)),
            expanded: false,
        };
        let lines = item.render(&theme, 60, false);
        let text: String = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("more lines"), "got: {text}");
    }

    #[test]
    fn expanded_tool_output_is_full() {
        let theme = Theme::dark();
        let long = (0..50)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let item = TranscriptItem::ToolExecution {
            tool_call_id: "c1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({}),
            result: Some(ToolOutcome::success(long)),
            expanded: true,
        };
        let lines = item.render(&theme, 60, false);
        let text: String = lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!text.contains("more lines"), "got: {text}");
        assert!(text.contains("line 49"));
    }

    #[test]
    fn height_matches_rendered_lines_at_wide_width() {
        let theme = Theme::dark();
        let item = TranscriptItem::User {
            content: "hello world".to_string(),
        };
        let rendered = item.render(&theme, 200, false).len();
        assert_eq!(item.height(&theme, 200, false), rendered);
    }

    fn render_text(item: &TranscriptItem, theme: &Theme, width: usize, stale: bool) -> String {
        item.render(theme, width, stale)
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn stale_tool_output_is_receipt_only() {
        let theme = Theme::dark();
        let long = (0..50)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let item = TranscriptItem::ToolExecution {
            tool_call_id: "c1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({ "command": "ls" }),
            result: Some(ToolOutcome::success(long)),
            expanded: false,
        };
        let text = render_text(&item, &theme, 60, true);
        assert!(!text.contains("line 0"), "got: {text}");
        assert!(!text.contains("more lines"), "got: {text}");
        assert!(text.contains("ls"), "title should remain: {text}");
    }

    #[test]
    fn expanded_overrides_stale() {
        let theme = Theme::dark();
        let long = (0..50)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let item = TranscriptItem::ToolExecution {
            tool_call_id: "c1".to_string(),
            name: "bash".to_string(),
            arguments: serde_json::json!({}),
            result: Some(ToolOutcome::success(long)),
            expanded: true,
        };
        let text = render_text(&item, &theme, 60, true);
        assert!(text.contains("line 49"), "got: {text}");
    }

    #[test]
    fn stale_thinking_is_summarized() {
        let theme = Theme::dark();
        let item = TranscriptItem::Assistant {
            parts: vec![ContentPart::Thinking {
                thinking: "step one\nstep two\nstep three".to_string(),
            }],
            stop_reason: StopReason::Stop,
            trust: TrustData::default(),
            thinking_expanded: false,
        };
        let text = render_text(&item, &theme, 60, true);
        assert!(!text.contains("step one"), "got: {text}");
        assert!(text.contains("ctrl+r to expand"), "got: {text}");
    }

    #[test]
    fn thinking_expanded_overrides_stale() {
        let theme = Theme::dark();
        let item = TranscriptItem::Assistant {
            parts: vec![ContentPart::Thinking {
                thinking: "step one\nstep two".to_string(),
            }],
            stop_reason: StopReason::Stop,
            trust: TrustData::default(),
            thinking_expanded: true,
        };
        let text = render_text(&item, &theme, 60, true);
        assert!(text.contains("… step one"), "got: {text}");
        assert!(text.contains("… step two"), "got: {text}");
    }
}
