//! Markdown → styled ratatui lines.
//!
//! Uses pulldown-cmark for parsing and renders block-level structure into
//! styled `Line`s. Lines are NOT pre-wrapped here — ratatui's `Paragraph`
//! wraps at draw time — so [`markdown_height`] estimates the wrapped height
//! for layout and scrolling.

use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::theme::Theme;

/// Render markdown to styled lines (one per source line/block).
pub fn render_markdown(text: &str, theme: &Theme) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TABLES);
    let parser = Parser::new_ext(text, options);

    let mut current: Vec<Span<'static>> = Vec::new();
    let mut current_style = Style::default();
    let mut in_code_block = false;
    let mut code_block_text = String::new();
    let mut list_depth = 0usize;

    let flush = |current: &mut Vec<Span<'static>>, lines: &mut Vec<Line<'static>>| {
        if !current.is_empty() {
            lines.push(Line::from(std::mem::take(current)));
        }
    };

    for event in parser {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                flush(&mut current, &mut lines);
                current_style = heading_style(level, theme);
            }
            Event::End(TagEnd::Heading(_)) => {
                flush(&mut current, &mut lines);
                current_style = Style::default();
            }
            Event::Start(Tag::Paragraph) => {
                flush(&mut current, &mut lines);
                current_style = Style::default();
            }
            Event::End(TagEnd::Paragraph) => {
                flush(&mut current, &mut lines);
            }
            Event::Start(Tag::CodeBlock(_)) => {
                flush(&mut current, &mut lines);
                in_code_block = true;
                code_block_text.clear();
            }
            Event::End(TagEnd::CodeBlock) => {
                in_code_block = false;
                let style = Style::default().fg(theme.md_code).bg(theme.md_code_bg);
                for raw_line in code_block_text.lines() {
                    lines.push(Line::from(Span::styled(raw_line.to_string(), style)));
                }
                lines.push(Line::default());
            }
            Event::Text(text) => {
                if in_code_block {
                    code_block_text.push_str(&text);
                } else {
                    current.push(Span::styled(text.to_string(), current_style));
                }
            }
            Event::Code(text) => {
                current.push(Span::styled(
                    text.to_string(),
                    Style::default().fg(theme.md_code),
                ));
            }
            Event::Start(Tag::Strong) => {
                current_style = current_style.fg(theme.md_bold).add_modifier(Modifier::BOLD);
            }
            Event::End(TagEnd::Strong) => {
                current_style = Style::default();
            }
            Event::Start(Tag::Emphasis) => {
                current_style = current_style
                    .fg(theme.md_italic)
                    .add_modifier(Modifier::ITALIC);
            }
            Event::End(TagEnd::Emphasis) => {
                current_style = Style::default();
            }
            Event::Start(Tag::List(_)) => {
                flush(&mut current, &mut lines);
                list_depth += 1;
            }
            Event::End(TagEnd::List(_)) => {
                flush(&mut current, &mut lines);
                list_depth = list_depth.saturating_sub(1);
            }
            Event::Start(Tag::Item) => {
                flush(&mut current, &mut lines);
                let indent = "  ".repeat(list_depth.saturating_sub(1));
                current.push(Span::styled(
                    format!("{indent}• "),
                    Style::default().fg(theme.dim),
                ));
            }
            Event::End(TagEnd::Item) => {
                flush(&mut current, &mut lines);
            }
            Event::SoftBreak | Event::HardBreak => {
                flush(&mut current, &mut lines);
            }
            Event::Rule => {
                flush(&mut current, &mut lines);
                // Render width isn't known here; a short rule never wraps,
                // unlike the old hardcoded 80 columns on a narrower terminal.
                lines.push(Line::from(Span::styled(
                    "─".repeat(40),
                    Style::default().fg(theme.dim),
                )));
            }
            _ => {}
        }
    }
    flush(&mut current, &mut lines);

    // Drop trailing blank line added after code blocks.
    while lines.last().is_some_and(|line| line.spans.is_empty()) {
        lines.pop();
    }
    lines
}

/// The number of terminal rows this markdown occupies at `width`.
pub fn markdown_height(text: &str, theme: &Theme, width: usize) -> usize {
    render_markdown(text, theme)
        .iter()
        .map(|line| line_height(line, width))
        .sum()
}

/// Wrapped height of one styled line at `width` columns.
pub fn line_height(line: &Line, width: usize) -> usize {
    let text: String = line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect();
    let display_width = UnicodeWidthStr::width(text.as_str());
    if display_width == 0 {
        1
    } else {
        display_width.div_ceil(width.max(1))
    }
}

fn heading_style(level: HeadingLevel, theme: &Theme) -> Style {
    let modifier = match level {
        HeadingLevel::H1 => Modifier::BOLD | Modifier::UNDERLINED,
        _ => Modifier::BOLD,
    };
    Style::default().fg(theme.md_heading).add_modifier(modifier)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_headings_and_paragraphs() {
        let theme = Theme::dark();
        let lines = render_markdown("# Title\n\nSome **bold** and *italic* text.", &theme);
        assert!(lines.len() >= 2);
        assert_eq!(lines[0].spans[0].content, "Title");
        assert_eq!(lines[0].spans[0].style.fg, Some(theme.md_heading));
    }

    #[test]
    fn code_blocks_keep_their_lines() {
        let theme = Theme::dark();
        let lines = render_markdown("```rust\nfn main() {}\n```\n", &theme);
        assert!(lines.iter().any(|line| {
            line.spans
                .iter()
                .any(|span| span.content.contains("fn main()"))
        }));
    }

    #[test]
    fn inline_code_survives_rendering() {
        let theme = Theme::dark();
        let lines = render_markdown("use `serde_json` here", &theme);
        let found = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .find(|span| span.content.as_ref() == "serde_json");
        let span = found.expect("inline code span was dropped");
        assert_eq!(span.style.fg, Some(theme.md_code));
    }

    #[test]
    fn bold_and_italic_get_theme_colors() {
        let theme = Theme::dark();
        let lines = render_markdown("Some **bold** and *italic* text.", &theme);
        let spans: Vec<&Span> = lines.iter().flat_map(|line| line.spans.iter()).collect();
        let bold = spans
            .iter()
            .find(|span| span.content.as_ref() == "bold")
            .expect("bold span missing");
        assert_eq!(bold.style.fg, Some(theme.md_bold));
        assert!(bold.style.add_modifier.contains(Modifier::BOLD));
        let italic = spans
            .iter()
            .find(|span| span.content.as_ref() == "italic")
            .expect("italic span missing");
        assert_eq!(italic.style.fg, Some(theme.md_italic));
        assert!(italic.style.add_modifier.contains(Modifier::ITALIC));
    }

    #[test]
    fn height_accounts_for_wrapping() {
        let theme = Theme::dark();
        let text = "word ".repeat(60);
        let narrow = markdown_height(&text, &theme, 20);
        let wide = markdown_height(&text, &theme, 100);
        assert!(narrow > wide);
        assert!(narrow >= 3);
    }
}
