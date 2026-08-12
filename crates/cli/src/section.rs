//! Bordered, color-coded section panels for the REPL (crossterm + Unicode box
//! drawing, no ratatui). Each turn renders tools, the assistant reply, the
//! parsed decision/trust block, and the plan as separate panels so the user can
//! scan a turn at a glance.
//!
//! Color stays off when `ansi::enabled()` is false (`/color off` or `NO_COLOR`):
//! we degrade to a plain `--- title ---` header with an indented body and no
//! box glyphs.

use crate::ansi;
use agent_m_ai::TrustData;
use crossterm::style::Color;
use std::io::Write;

/// Which panel is being drawn. Carries the data needed to pick border color,
/// background tint, and the title suffix (e.g. `· 95% high` for a decision).
#[derive(Clone, Copy)]
pub enum SectionKind {
    Tools,
    Reply,
    Decision { tier: &'static str, confidence: u8 },
    Plan,
    Notice,
    Error,
}

impl SectionKind {
    /// (border foreground color, optional body background tint).
    fn style(&self) -> (Color, Option<Color>) {
        use crate::ansi::{BG_PANEL, MUTED_CYAN, MUTED_GREY, MUTED_RED, MUTED_YELLOW};
        let v = Color::AnsiValue;
        match self {
            SectionKind::Tools => (v(MUTED_CYAN), None),
            SectionKind::Reply => (v(MUTED_GREY), None),
            SectionKind::Decision { tier, .. } => match *tier {
                "low" => (Color::Red, None),
                "medium" => (Color::Yellow, None),
                "high" => (Color::Green, None),
                _ => (v(MUTED_GREY), None),
            },
            SectionKind::Plan => (v(MUTED_YELLOW), Some(v(BG_PANEL))),
            SectionKind::Notice => (v(MUTED_YELLOW), None),
            SectionKind::Error => (v(MUTED_RED), None),
        }
    }

    /// Extra text appended to the title inside the top border, e.g.
    /// `· 95% high` for a high-confidence decision.
    fn title_suffix(&self) -> String {
        match self {
            SectionKind::Decision { tier, confidence } if *confidence > 0 => {
                format!(" · {confidence}% {tier}")
            }
            _ => String::new(),
        }
    }
}

/// Terminal width for panel layout: `crossterm::terminal::size()`, fallback 80,
/// capped at 100 so panels never grow absurdly wide.
pub fn terminal_width() -> usize {
    match crossterm::terminal::size() {
        Ok((w, _)) if w > 0 => (w as usize).min(160),
        _ => 80,
    }
}

/// Render a full boxed panel: top border with `title`, one or more body rows
/// (word-wrapped to the inner width), and a bottom border. Width is clamped to
/// `[24, 100]`; inner width is `width - 4` (for `│ x │`).
pub fn render_box(title: &str, lines: &[String], kind: SectionKind, width: usize) -> String {
    render_box_inner(title, lines, kind, width, ansi::enabled())
}

fn render_box_inner(
    title: &str,
    lines: &[String],
    kind: SectionKind,
    width: usize,
    color: bool,
) -> String {
    let width = width.clamp(24, 160);
    let inner = width.saturating_sub(4);
    let mut out = String::new();
    out.push_str(&draw_top(title, width, kind, color));
    out.push('\n');
    for line in lines {
        for row in wrap_styled_line(line, inner) {
            out.push_str(&draw_body(&row, width, kind, color));
            out.push('\n');
        }
    }
    out.push_str(&draw_bottom(width, kind, color));
    out
}

fn draw_top(title: &str, width: usize, kind: SectionKind, color: bool) -> String {
    let width = width.clamp(24, 160);
    let (border, _bg) = kind.style();
    let full = format!("{}{}", title, kind.title_suffix());
    if !color {
        return format!("--- {} ---", full);
    }
    let title = truncate_visible(&full, width.saturating_sub(6));
    let tv = ansi::visible_width(&title);
    let fill = width.saturating_sub(5 + tv).max(1);
    let left = ansi::fg_only("╭─ ", border);
    let title_c = ansi::fg_only(&title, border);
    let space = ansi::fg_only(" ", border);
    let dashes = ansi::fg_only(&"─".repeat(fill), border);
    let right = ansi::fg_only("╮", border);
    format!("{left}{title_c}{space}{dashes}{right}")
}

fn draw_bottom(width: usize, kind: SectionKind, color: bool) -> String {
    let width = width.clamp(24, 160);
    if !color {
        return String::new();
    }
    let (border, _bg) = kind.style();
    let left = ansi::fg_only("╰─", border);
    let dashes = ansi::fg_only(&"─".repeat(width.saturating_sub(3)), border);
    let right = ansi::fg_only("╯", border);
    format!("{left}{dashes}{right}")
}

fn draw_body(content: &str, width: usize, kind: SectionKind, color: bool) -> String {
    let width = width.clamp(24, 160);
    let inner = width.saturating_sub(4);
    if !color {
        return format!("  {content}");
    }
    let (border, bg) = kind.style();
    let padded = pad_to_visible(content, inner);
    let cell = match bg {
        Some(bg) => ansi::tint_cell(bg, &padded),
        None => padded,
    };
    let vert = ansi::fg_only("│", border);
    format!("{vert}{cell}{vert}")
}

/// Render the parsed decision/trust block as a boxed panel. Returns an empty
/// string when the block is empty (nothing to show).
pub fn print_decision(trust: &TrustData) -> String {
    if trust.is_empty() {
        return String::new();
    }
    let (tier, confidence) = match trust.confidence_tier() {
        Some((t, c)) => (t, c),
        None => ("", 0),
    };
    let kind = SectionKind::Decision { tier, confidence };
    let label = |s: &str| ansi::fg_only(s, Color::AnsiValue(ansi::MUTED_GREY));
    let mut lines: Vec<String> = Vec::new();
    if let Some(reason) = &trust.reason {
        lines.push(format!("{}: {}", label("reason"), reason));
    }
    if let Some(expected) = &trust.expected_outcome {
        lines.push(format!("{}: {}", label("expected"), expected));
    }
    if let Some(uncertainty) = &trust.uncertainty {
        lines.push(format!("{}: {}", label("uncertainty"), uncertainty));
    }
    for ev in &trust.evidence {
        let loc = match ev.line {
            Some(l) => format!("{}:{}", ev.file, l),
            None => ev.file.clone(),
        };
        let note = ev.note.clone().unwrap_or_default();
        lines.push(format!("{}: {} — {}", label("evidence"), loc, note));
    }
    if let Some(et) = &trust.estimated_time {
        lines.push(format!("{}: {}", label("eta"), et));
    }
    render_box("decision", &lines, kind, terminal_width())
}

/// Live, streaming reply panel. Prints the top border on the first visible
/// delta, streams each completed line as a body row, and prints the bottom
/// border when the turn ends — so the reply box fills in as the model speaks
/// instead of appearing all at once.
pub struct ReplyBox {
    active: bool,
    line_buf: String,
    width: usize,
}

impl ReplyBox {
    pub fn new(width: usize) -> Self {
        ReplyBox {
            active: false,
            line_buf: String::new(),
            width: width.clamp(24, 160),
        }
    }

    /// Feed a (already trust-stripped) text delta into the panel.
    pub fn push(&mut self, delta: &str) {
        if !self.active {
            print!("{}", draw_top("reply", self.width, SectionKind::Reply, ansi::enabled()));
            println!();
            self.active = true;
        }
        self.line_buf.push_str(delta);
        while let Some(pos) = self.line_buf.find('\n') {
            let line = self.line_buf[..pos].to_string();
            self.line_buf = self.line_buf[pos + 1..].to_string();
            self.print_line(&line);
        }
        let inner = self.width.saturating_sub(4);
        if ansi::visible_width(&self.line_buf) > inner && !self.line_buf.trim().is_empty() {
            self.print_line(&self.line_buf);
            self.line_buf.clear();
        }
    }

    fn print_line(&self, line: &str) {
        let inner = self.width.saturating_sub(4);
        let rendered = ansi::render_markdown(line.trim_end()).trim_end_matches('\n').to_string();
        for row in wrap_styled_line(&rendered, inner) {
            print!("{}", draw_body(&row, self.width, SectionKind::Reply, ansi::enabled()));
            println!();
        }
    }

    /// Flush any partial line and close the panel.
    pub fn finish(&mut self) {
        if self.active {
            if !self.line_buf.is_empty() {
                self.print_line(&self.line_buf);
                self.line_buf.clear();
            }
            print!("{}", draw_bottom(self.width, SectionKind::Reply, ansi::enabled()));
            println!();
            self.active = false;
        }
        let _ = std::io::stdout().flush();
    }
}

// ---------------------------------------------------------------------------
// Word-wrapping that respects ANSI escapes and UTF-8 char boundaries.
// ---------------------------------------------------------------------------

/// A token from a (possibly styled) line: a raw ANSI control sequence, or a
/// single visible character.
enum Tok<'a> {
    Ctrl(&'a str),
    Ch(char),
}

/// Split a line into [`Tok`]s. ANSI sequences are kept verbatim so their style
/// can be carried onto wrapped continuation lines.
fn tokenize(line: &str) -> Vec<Tok<'_>> {
    let bytes = line.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b {
            if i + 1 < bytes.len() && bytes[i + 1] == b'[' {
                let start = i;
                let mut j = i + 2;
                while j < bytes.len() && !(0x40..=0x7e).contains(&bytes[j]) {
                    j += 1;
                }
                if j < bytes.len() {
                    j += 1; // include the final byte of the sequence
                }
                toks.push(Tok::Ctrl(&line[start..j]));
                i = j;
                continue;
            }
            toks.push(Tok::Ctrl(&line[i..i + 1]));
            i += 1;
            continue;
        }
        let ch = line[i..].chars().next().unwrap();
        toks.push(Tok::Ch(ch));
        i += ch.len_utf8();
    }
    toks
}

/// Wrap a (possibly pre-styled) line to `inner` visible columns, breaking on
/// word boundaries when possible and always on a char boundary (never mid
/// multi-byte UTF-8). Continuation lines re-apply the style prefix that was
/// active at the break, so color survives wrapping.
pub fn wrap_styled_line(line: &str, inner: usize) -> Vec<String> {
    if inner == 0 {
        return vec![String::new()];
    }
    let toks = tokenize(line);
    let mut out: Vec<String> = Vec::new();
    let mut cur: Vec<&Tok<'_>> = Vec::new();
    let mut cur_visible = 0usize;
    let mut last_space: Option<usize> = None;
    let mut prefix = String::new();

    let flush = |cur: &[&Tok<'_>], prefix: &str, first: bool| -> String {
        let mut s = String::new();
        if !first {
            s.push_str(prefix);
        }
        for t in cur {
            match t {
                Tok::Ctrl(c) => s.push_str(c),
                Tok::Ch(ch) => s.push(*ch),
            }
        }
        s
    };

    for tok in &toks {
        match tok {
            Tok::Ctrl(c) => {
                if ansi::is_full_reset(c) {
                    prefix.clear();
                } else {
                    prefix.push_str(c);
                }
                cur.push(tok);
            }
            Tok::Ch(ch) => {
                let w = 1usize; // count every visible char as one cell
                if cur_visible + w > inner && !cur.is_empty() {
                    if let Some(sp) = last_space {
                        let (kept, moved) = cur.split_at(sp + 1);
                        out.push(flush(kept, &prefix, out.is_empty()));
                        cur = moved.to_vec();
                        cur_visible = visible_of(&cur);
                        last_space = find_last_space(&cur);
                    } else {
                        out.push(flush(&cur, &prefix, out.is_empty()));
                        cur = Vec::new();
                        cur_visible = 0;
                        last_space = None;
                    }
                }
                if *ch == ' ' {
                    last_space = Some(cur.len());
                }
                cur.push(tok);
                cur_visible += w;
            }
        }
    }
    if !cur.is_empty() {
        out.push(flush(&cur, &prefix, out.is_empty()));
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

fn visible_of(cur: &[&Tok<'_>]) -> usize {
    cur.iter().filter(|t| matches!(t, Tok::Ch(_))).count()
}

fn find_last_space(cur: &[&Tok<'_>]) -> Option<usize> {
    cur.iter().rposition(|t| matches!(t, Tok::Ch(' ')))
}

/// Right-pad (or leave unchanged) a styled line so its visible width is
/// `inner` columns. Padding spaces sit inside any background tint applied by
/// the caller.
fn pad_to_visible(content: &str, inner: usize) -> String {
    let w = ansi::visible_width(content);
    if w >= inner {
        content.to_string()
    } else {
        format!("{}{}", content, " ".repeat(inner - w))
    }
}

/// Truncate a string to at most `max` visible columns, appending `…` if cut.
fn truncate_visible(s: &str, max: usize) -> String {
    if ansi::visible_width(s) <= max {
        return s.to_string();
    }
    let taken: String = s.chars().take(max).collect();
    format!("{taken}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_long_plain_line_to_inner_width() {
        ansi::init_color();
        ansi::set_color(true);
        let long = "word ".repeat(20);
        let boxed = render_box_inner("tools", &[long], SectionKind::Tools, 40, true);
        for l in boxed.lines() {
            assert!(ansi::visible_width(l) <= 40, "overflow: {l:?}");
        }
        assert!(boxed.contains('╭'));
        assert!(boxed.contains('╰'));
        assert!(boxed.contains('│'));
        ansi::set_color(false);
    }

    #[test]
    fn wraps_on_utf8_em_dash_without_panic() {
        ansi::init_color();
        ansi::set_color(true);
        let long = format!("a{}\u{2014}b ", "x".repeat(5)).repeat(20);
        let rows = wrap_styled_line(&long, 20);
        for r in &rows {
            assert!(ansi::visible_width(r) <= 20, "overflow: {r:?}");
        }
        assert!(!rows.is_empty());
        ansi::set_color(false);
    }

    #[test]
    fn plain_fallback_when_color_off() {
        ansi::set_color(false);
        let out = render_box_inner(
            "tools",
            &["hello world".to_string()],
            SectionKind::Tools,
            40,
            false,
        );
        assert!(out.contains("--- tools ---"), "{out:?}");
        assert!(out.contains("  hello world"), "{out:?}");
        assert!(!out.contains('╭'));
    }

    #[test]
    fn decision_title_includes_confidence() {
        assert_eq!(
            SectionKind::Decision {
                tier: "high",
                confidence: 95
            }
            .title_suffix(),
            " · 95% high"
        );
        assert_eq!(
            SectionKind::Decision {
                tier: "",
                confidence: 0
            }
            .title_suffix(),
            ""
        );
    }

    #[test]
    fn decision_box_renders_fields() {
        ansi::set_color(false);
        let trust = TrustData {
            confidence: Some(95),
            reason: Some("Grep found no matches".into()),
            uncertainty: Some("local files only".into()),
            ..Default::default()
        };
        let out = print_decision(&trust);
        assert!(out.contains("decision"), "{out:?}");
        assert!(out.contains("reason: Grep found no matches"), "{out:?}");
        assert!(out.contains("uncertainty: local files only"), "{out:?}");
    }

    #[test]
    fn empty_trust_renders_nothing() {
        ansi::set_color(false);
        assert_eq!(print_decision(&TrustData::default()), "");
    }

    #[test]
    fn terminal_width_is_bounded() {
        let w = terminal_width();
        assert!(w >= 24 && w <= 160);
    }
}

