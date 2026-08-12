//! ANSI styling for REPL output. Color is enabled when stdout is a TTY and
//! `NO_COLOR` is unset, and can be toggled at runtime via `/color on|off`.

use crossterm::style::{
    Attribute, Color, ResetColor, SetAttribute, SetBackgroundColor, SetForegroundColor,
};
use std::fmt::Write as _;
use std::io::IsTerminal;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

static COLOR_ENABLED: OnceLock<AtomicBool> = OnceLock::new();

pub fn init_color() {
    let on = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    COLOR_ENABLED.get_or_init(|| AtomicBool::new(on));
}

pub fn enabled() -> bool {
    COLOR_ENABLED
        .get()
        .map(|flag| flag.load(Ordering::Relaxed))
        .unwrap_or(false)
}

pub fn set_color(on: bool) {
    if let Some(flag) = COLOR_ENABLED.get() {
        flag.store(on, Ordering::Relaxed);
    }
}

fn styled(text: &str, color: Color, attrs: &[Attribute]) -> String {
    if !enabled() {
        return text.to_string();
    }
    let mut out = String::new();
    for attr in attrs {
        let _ = write!(out, "{}", SetAttribute(*attr));
    }
    let _ = write!(out, "{}{}{}", SetForegroundColor(color), text, ResetColor);
    let _ = write!(out, "{}", SetAttribute(Attribute::Reset));
    out
}

/// Foreground-only color: sets the fg, renders `text`, then drops fg back to
/// default (`\x1b[39m`) — but does NOT emit a full `ResetColor` (`\x1b[0m`).
/// This lets a line keep a background tint (set by the caller around the whole
/// cell) while still coloring sub-spans: a full reset would also clear the bg.
pub fn fg_only(text: &str, color: Color) -> String {
    if !enabled() {
        return text.to_string();
    }
    format!("{}{}{}", SetForegroundColor(color), text, "\x1b[39m")
}

/// Wrap `content` in a background tint (xterm-256 `AnsiValue`). Used for panel
/// body cells; the border characters are drawn separately in the foreground
/// color so they stay readable. When color is off, returns `content` unchanged.
pub fn tint_cell(bg: Color, content: &str) -> String {
    if !enabled() {
        return content.to_string();
    }
    format!("{}{}{}", SetBackgroundColor(bg), content, ResetColor)
}

/// True if an escape sequence is a full style reset (clears fg *and* bg).
/// `fg_only` emits `\x1b[39m` (fg-only reset) which we must NOT treat as a
/// full reset, or the background tint would be lost mid-line.
pub fn is_full_reset(seq: &str) -> bool {
    seq == "\x1b[0m"
}

/// Strip ANSI escape sequences from a string so width math sees only the
/// visible glyphs. Box wrapping needs this because body lines arrive
/// pre-styled (markdown colors) and must wrap on display width, not bytes.
pub fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            // Consume the escape sequence: CSI (ESC [) up to the final byte
            // in the range 0x40–0x7E, or a two-byte sequence (ESC c).
            if chars.peek() == Some(&'[') {
                chars.next();
                for c in chars.by_ref() {
                    if (0x40..=0x7e).contains(&(c as u8)) {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(ch);
    }
    out
}

/// Visible (display) width of a string, ignoring ANSI escapes. Counts each
/// Unicode char as one cell (good enough for the box we draw; wide CJK glyphs
/// will be slightly off but never panic).
pub fn visible_width(text: &str) -> usize {
    strip_ansi(text).chars().count()
}

// Muted xterm-256 palette (indices into the 6x6x6 colour cube), chosen to sit
// calmly on both dark and light terminals instead of the bright defaults.
pub const MUTED_CYAN: u8 = 73; // #5fafaf slate-cyan
pub const MUTED_GREY: u8 = 244; // #808080 soft grey
pub const MUTED_RED: u8 = 131; // #af5f5f brick red
pub const MUTED_YELLOW: u8 = 136; // #af8700 muted amber
pub const MUTED_GREEN: u8 = 71; // #5faf5f muted green

// Subtle xterm-256 background tints for section-panel bodies. Kept very dark
// (near the 234–236 range) so text stays readable on both light and dark
// terminals; the border color remains the primary visual cue.
pub const BG_PANEL: u8 = 236; // #303030 faint neutral panel

pub fn cyan(text: &str) -> String {
    styled(text, Color::AnsiValue(MUTED_CYAN), &[])
}

pub fn dim(text: &str) -> String {
    styled(text, Color::AnsiValue(MUTED_GREY), &[])
}

pub fn red(text: &str) -> String {
    styled(text, Color::AnsiValue(MUTED_RED), &[])
}

pub fn yellow(text: &str) -> String {
    styled(text, Color::AnsiValue(MUTED_YELLOW), &[])
}

pub fn green(text: &str) -> String {
    styled(text, Color::AnsiValue(MUTED_GREEN), &[])
}

pub fn bold(text: &str) -> String {
    if !enabled() {
        return text.to_string();
    }
    let mut out = String::new();
    let _ = write!(out, "{}", SetAttribute(Attribute::Bold));
    let _ = write!(out, "{text}");
    let _ = write!(out, "{}", SetAttribute(Attribute::Reset));
    out
}

/// Lightweight markdown → terminal renderer. Handles the subset that matters
/// for assistant replies: headings, **bold**, `inline code`, bullet lists,
/// fenced code blocks (dim), and `---` rules. No allocations beyond the output
/// string; zero external deps.
pub fn render_markdown(text: &str) -> String {
    let mut out = String::new();
    let mut in_fence = false;
    for line in text.lines() {
        let trimmed = line.trim();
        // Fenced code block toggle.
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            out.push('\n');
            continue;
        }
        if in_fence {
            out.push_str(&dim(line));
            out.push('\n');
            continue;
        }
        // Horizontal rule.
        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            out.push_str(&dim("────────────────────────────────────────"));
            out.push('\n');
            continue;
        }
        // Headings.
        if let Some(rest) = trimmed.strip_prefix("### ") {
            out.push_str(&bold(&cyan(rest)));
            out.push('\n');
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("## ") {
            out.push_str(&bold(&cyan(rest)));
            out.push('\n');
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("# ") {
            out.push_str(&bold(&cyan(rest)));
            out.push('\n');
            continue;
        }
        // Bullet lists.
        if let Some(rest) = trimmed.strip_prefix("- ").or_else(|| trimmed.strip_prefix("* ")) {
            let rendered = render_inline(rest);
            out.push_str(&format!("  • {rendered}"));
            out.push('\n');
            continue;
        }
        // Normal line with inline formatting.
        out.push_str(&render_inline(trimmed));
        out.push('\n');
    }
    out
}

/// Render inline markdown: **bold** and `code` spans.
pub fn render_inline(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while !rest.is_empty() {
        // `inline code`
        if rest.starts_with('`')
            && let Some(end) = rest[1..].find('`').map(|i| i + 1)
        {
            out.push_str(&dim(&rest[1..end]));
            rest = &rest[end + 1..];
            continue;
        }
        // **bold**
        if rest.starts_with("**") {
            if let Some(end) = rest[2..].find("**").map(|i| i + 2) {
                out.push_str(&bold(&rest[2..end]));
                rest = &rest[end + 2..];
                continue;
            }
        }
        // Consume one char.
        let ch = rest.chars().next().unwrap();
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_markdown_heading() {
        let out = render_markdown("## Hello world\nsome text");
        assert!(out.contains("Hello world"), "{out:?}");
        assert!(out.contains("some text"), "{out:?}");
    }

    #[test]
    fn render_markdown_fenced_block() {
        let src = "intro\n```\nfn foo() {}\n```\noutro";
        let out = render_markdown(src);
        assert!(out.contains("fn foo()"), "{out:?}");
        assert!(out.contains("intro"), "{out:?}");
        assert!(out.contains("outro"), "{out:?}");
    }

    #[test]
    fn render_markdown_bullet() {
        let out = render_markdown("- item one\n- item two");
        assert!(out.contains("item one"), "{out:?}");
        assert!(out.contains("item two"), "{out:?}");
        assert!(out.contains('•'), "{out:?}");
    }

    #[test]
    fn render_inline_bold_and_code() {
        let out = render_inline("say **hello** and `world`");
        assert!(out.contains("hello"), "{out:?}");
        assert!(out.contains("world"), "{out:?}");
    }

    #[test]
    fn render_inline_code_followed_by_em_dash() {
        let out = render_inline("`/model [id]` — query or switch the active model");
        assert!(out.contains("/model [id]"), "{out:?}");
        assert!(out.contains("— query or switch the active model"), "{out:?}");
    }
}
