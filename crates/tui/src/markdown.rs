//! Markdown → styled ratatui lines.
//!
//! Uses pulldown-cmark for parsing and renders block-level structure into
//! styled `Line`s. Lines are NOT pre-wrapped here — ratatui's `Paragraph`
//! wraps at draw time — so [`markdown_height`] estimates the wrapped height
//! for layout and scrolling.

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::theme::Theme;

/// Render markdown to styled lines (one per source line/block). `width` is
/// the terminal column width, used for full-width rules and table separators.
pub fn render_markdown(text: &str, theme: &Theme, width: u16) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TABLES);
    let parser = Parser::new_ext(text, options);

    let mut current: Vec<Span<'static>> = Vec::new();
    let mut current_style = Style::default();
    let mut in_code_block = false;
    let mut code_block_text = String::new();
    // Ordered lists carry their start number (`Some(n)`); unordered are `None`.
    let mut list_stack: Vec<Option<u64>> = Vec::new();
    let mut in_blockquote = false;
    let mut cell_count = 0usize;

    let flush =
        |current: &mut Vec<Span<'static>>, lines: &mut Vec<Line<'static>>, in_blockquote: bool| {
            if !current.is_empty() {
                let mut line = Line::from(std::mem::take(current));
                if in_blockquote {
                    line.spans
                        .insert(0, Span::styled("│ ", Style::default().fg(theme.muted)));
                }
                lines.push(line);
            }
        };

    for event in parser {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                flush(&mut current, &mut lines, in_blockquote);
                current_style = heading_style(level, theme);
            }
            Event::End(TagEnd::Heading(_)) => {
                flush(&mut current, &mut lines, in_blockquote);
                current_style = Style::default();
            }
            Event::Start(Tag::Paragraph) => {
                flush(&mut current, &mut lines, in_blockquote);
                current_style = Style::default();
            }
            Event::End(TagEnd::Paragraph) => {
                flush(&mut current, &mut lines, in_blockquote);
            }
            Event::Start(Tag::BlockQuote(_)) => {
                flush(&mut current, &mut lines, in_blockquote);
                in_blockquote = true;
            }
            Event::End(TagEnd::BlockQuote(_)) => {
                flush(&mut current, &mut lines, in_blockquote);
                in_blockquote = false;
            }
            Event::Start(Tag::CodeBlock(kind)) => {
                flush(&mut current, &mut lines, in_blockquote);
                in_code_block = true;
                code_block_text.clear();
                if let CodeBlockKind::Fenced(lang) = kind
                    && !lang.is_empty()
                {
                    lines.push(Line::from(Span::styled(
                        format!(" {lang} "),
                        Style::default().fg(theme.dim),
                    )));
                }
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
            Event::Start(Tag::Strikethrough) => {
                current_style = current_style.add_modifier(Modifier::CROSSED_OUT);
            }
            Event::End(TagEnd::Strikethrough) => {
                current_style = Style::default();
            }
            Event::Start(Tag::Link { .. }) => {
                current_style = current_style.fg(theme.md_link);
            }
            Event::End(TagEnd::Link) => {
                current_style = Style::default();
            }
            Event::Start(Tag::List(start)) => {
                flush(&mut current, &mut lines, in_blockquote);
                list_stack.push(start);
            }
            Event::End(TagEnd::List(_)) => {
                flush(&mut current, &mut lines, in_blockquote);
                list_stack.pop();
            }
            Event::Start(Tag::Item) => {
                flush(&mut current, &mut lines, in_blockquote);
                let indent = "  ".repeat(list_stack.len().saturating_sub(1));
                match list_stack.last_mut() {
                    Some(Some(number)) => {
                        current.push(Span::styled(
                            format!("{indent}{number}. "),
                            Style::default().fg(theme.dim),
                        ));
                        *number += 1;
                    }
                    _ => {
                        current.push(Span::styled(
                            format!("{indent}• "),
                            Style::default().fg(theme.dim),
                        ));
                    }
                }
            }
            Event::End(TagEnd::Item) => {
                flush(&mut current, &mut lines, in_blockquote);
            }
            Event::Start(Tag::Table(_)) => {
                flush(&mut current, &mut lines, in_blockquote);
            }
            Event::End(TagEnd::Table) => {
                flush(&mut current, &mut lines, in_blockquote);
            }
            Event::Start(Tag::TableHead) => {
                flush(&mut current, &mut lines, in_blockquote);
                cell_count = 0;
            }
            Event::End(TagEnd::TableHead) => {
                flush(&mut current, &mut lines, in_blockquote);
                // Full-width rule under the header row, same as an HR.
                lines.push(Line::from(Span::styled(
                    "─".repeat(width as usize),
                    Style::default().fg(theme.dim),
                )));
            }
            Event::Start(Tag::TableRow) => {
                flush(&mut current, &mut lines, in_blockquote);
                cell_count = 0;
            }
            Event::End(TagEnd::TableRow) => {
                flush(&mut current, &mut lines, in_blockquote);
            }
            Event::Start(Tag::TableCell) => {
                // Separator between cells (not before the first).
                if cell_count > 0 {
                    current.push(Span::styled("  │  ", Style::default().fg(theme.dim)));
                }
                cell_count += 1;
            }
            Event::End(TagEnd::TableCell) => {
                // Cell text accumulates naturally via Event::Text.
            }
            Event::SoftBreak | Event::HardBreak => {
                flush(&mut current, &mut lines, in_blockquote);
            }
            Event::Rule => {
                flush(&mut current, &mut lines, in_blockquote);
                lines.push(Line::from(Span::styled(
                    "─".repeat(width as usize),
                    Style::default().fg(theme.dim),
                )));
            }
            _ => {}
        }
    }
    flush(&mut current, &mut lines, in_blockquote);

    // Drop trailing blank line added after code blocks.
    while lines.last().is_some_and(|line| line.spans.is_empty()) {
        lines.pop();
    }
    lines
}

/// The number of terminal rows this markdown occupies at `width`.
pub fn markdown_height(text: &str, theme: &Theme, width: usize) -> usize {
    render_markdown(text, theme, width as u16)
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
        let lines = render_markdown("# Title\n\nSome **bold** and *italic* text.", &theme, 80);
        assert!(lines.len() >= 2);
        assert_eq!(lines[0].spans[0].content, "Title");
        assert_eq!(lines[0].spans[0].style.fg, Some(theme.md_heading));
    }

    #[test]
    fn code_blocks_keep_their_lines() {
        let theme = Theme::dark();
        let lines = render_markdown("```rust\nfn main() {}\n```\n", &theme, 80);
        assert!(lines.iter().any(|line| {
            line.spans
                .iter()
                .any(|span| span.content.contains("fn main()"))
        }));
    }

    #[test]
    fn inline_code_survives_rendering() {
        let theme = Theme::dark();
        let lines = render_markdown("use `serde_json` here", &theme, 80);
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
        let lines = render_markdown("Some **bold** and *italic* text.", &theme, 80);
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

    #[test]
    fn ordered_lists_use_start_numbers() {
        let theme = Theme::dark();
        let lines = render_markdown("1. first\n2. second\n3. third", &theme, 80);
        let text: String = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect();
        assert!(text.contains("1. first"), "got: {text}");
        assert!(text.contains("2. second"), "got: {text}");
        assert!(text.contains("3. third"), "got: {text}");
        assert!(!text.contains("•"), "bullets should not appear: {text}");
    }

    #[test]
    fn ordered_list_respects_start_attribute() {
        let theme = Theme::dark();
        let lines = render_markdown("3. three\n4. four", &theme, 80);
        let text: String = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect();
        assert!(text.contains("3. three"), "got: {text}");
        assert!(text.contains("4. four"), "got: {text}");
    }

    #[test]
    fn blockquotes_get_prefix() {
        let theme = Theme::dark();
        let lines = render_markdown("> quoted text", &theme, 80);
        let text: String = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect();
        assert!(text.contains("│ "), "blockquote prefix missing: {text}");
        assert!(text.contains("quoted text"), "got: {text}");
    }

    #[test]
    fn strikethrough_gets_crossed_out() {
        let theme = Theme::dark();
        let lines = render_markdown("~~gone~~", &theme, 80);
        let spans: Vec<&Span> = lines.iter().flat_map(|line| line.spans.iter()).collect();
        let span = spans
            .iter()
            .find(|span| span.content.as_ref() == "gone")
            .expect("strikethrough span missing");
        assert!(span.style.add_modifier.contains(Modifier::CROSSED_OUT));
    }

    #[test]
    fn links_use_md_link_color() {
        let theme = Theme::dark();
        let lines = render_markdown("[text](https://example.com)", &theme, 80);
        let spans: Vec<&Span> = lines.iter().flat_map(|line| line.spans.iter()).collect();
        let span = spans
            .iter()
            .find(|span| span.content.as_ref() == "text")
            .expect("link text span missing");
        assert_eq!(span.style.fg, Some(theme.md_link));
    }

    #[test]
    fn tables_render_with_separators_and_rule() {
        let theme = Theme::dark();
        let md = "| a | b |\n|---|---|\n| 1 | 2 |";
        let lines = render_markdown(md, &theme, 80);
        let text: String = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect();
        assert!(text.contains("a"), "header cell missing: {text}");
        assert!(text.contains("b"), "header cell missing: {text}");
        assert!(text.contains("1"), "body cell missing: {text}");
        assert!(text.contains("2"), "body cell missing: {text}");
        assert!(text.contains("│"), "cell separator missing: {text}");
        assert!(text.contains("─"), "header rule missing: {text}");
    }

    #[test]
    fn code_block_language_label() {
        let theme = Theme::dark();
        let lines = render_markdown("```rust\nfn main() {}\n```\n", &theme, 80);
        let text: String = lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect();
        assert!(text.contains("rust"), "language label missing: {text}");
    }

    #[test]
    fn rule_uses_full_width() {
        let theme = Theme::dark();
        let lines = render_markdown("---", &theme, 30);
        let rule = lines
            .iter()
            .find(|line| line.spans.iter().any(|s| s.content.contains('─')))
            .expect("rule line missing");
        let width = rule
            .spans
            .iter()
            .map(|s| UnicodeWidthStr::width(s.content.as_ref()))
            .sum::<usize>();
        assert_eq!(width, 30);
    }
}
