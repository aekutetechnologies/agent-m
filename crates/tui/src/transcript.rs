//! Transcript items: the blocks rendered in the scrollable chat area.

use agent_m_agent::ToolOutcome;
use agent_m_ai::{ContentPart, StopReason};
use ratatui::style::{Modifier, Style};
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
    /// draw time; use [`TranscriptItem::height`] for layout.
    pub fn render(&self, theme: &Theme, _width: usize) -> Vec<Line<'static>> {
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
            TranscriptItem::Assistant { parts, stop_reason } => {
                let mut lines = Vec::new();
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            lines.extend(render_markdown(text, theme));
                        }
                        ContentPart::Thinking { thinking } => {
                            let style = Style::default()
                                .fg(theme.thinking_text)
                                .add_modifier(Modifier::ITALIC);
                            for raw_line in thinking.lines() {
                                lines
                                    .push(Line::from(Span::styled(format!("… {raw_line}"), style)));
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

    /// Total rendered height at the given width (wrap-aware).
    pub fn height(&self, theme: &Theme, width: usize) -> usize {
        match self {
            TranscriptItem::User { content } => {
                crate::markdown::markdown_height(content, theme, width) + 1
            }
            TranscriptItem::Assistant { parts, stop_reason } => {
                let mut height = 0usize;
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            height += crate::markdown::markdown_height(text, theme, width);
                        }
                        ContentPart::Thinking { thinking } => {
                            height += thinking
                                .lines()
                                .map(|line| line_height(&Line::from(line.to_string()), width))
                                .sum::<usize>();
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
                height += 1; // trailing spacer
                height
            }
            TranscriptItem::ToolExecution {
                result, expanded, ..
            } => {
                let mut height = 1; // title
                if let Some(outcome) = result {
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
    fn user_block_has_chevron_and_background() {
        let theme = Theme::dark();
        let item = TranscriptItem::User {
            content: "hello".to_string(),
        };
        let lines = item.render(&theme, 40);
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
        let lines = item.render(&theme, 60);
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
        let lines = item.render(&theme, 60);
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
        let rendered = item.render(&theme, 200).len();
        assert_eq!(item.height(&theme, 200), rendered);
    }
}
