//! Theme support: JSON themes with pi's color keys, dark/light defaults, and a
//! terminal-background heuristic.

use ratatui::style::Color;
use serde_json::Value;
use std::io::{Read, Write};
use std::time::{Duration, Instant};

/// OSC-11 background query: ask the terminal for its background color.
const OSC_11_QUERY: &[u8] = b"\x1b]11;?\x1b\\";
/// OSC-11 response terminator (ST).
const OSC_ST: &[u8] = b"\x1b\\";
/// How long to wait for the terminal to answer before falling back.
const OSC_TIMEOUT: Duration = Duration::from_millis(50);

/// Resolved theme: every color as a concrete ratatui `Color`.
#[derive(Debug, Clone)]
pub struct Theme {
    pub name: String,
    pub user_message_bg: Color,
    pub user_message_text: Color,
    pub md_heading: Color,
    pub md_code: Color,
    pub md_code_bg: Color,
    pub md_bold: Color,
    pub md_italic: Color,
    pub md_link: Color,
    pub thinking_text: Color,
    pub error: Color,
    pub warning: Color,
    pub muted: Color,
    pub dim: Color,
    pub tool_pending_bg: Color,
    pub tool_success_bg: Color,
    pub tool_error_bg: Color,
    pub accent: Color,
    pub scrollbar: Color,
}

/// Build a theme from a partial JSON map (`{ "name": ..., "colors": {...} }`),
/// falling back to the dark defaults for missing keys. Mirrors pi's theme JSON.
pub fn parse_theme(json: &str) -> Result<Theme, String> {
    let value: Value =
        serde_json::from_str(json).map_err(|error| format!("invalid theme JSON: {error}"))?;
    let colors = value
        .get("colors")
        .and_then(Value::as_object)
        .ok_or_else(|| "theme JSON must contain a `colors` object".to_string())?;
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("custom");
    Ok(Theme::from_map(name, colors))
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            user_message_bg: Color::Reset,
            user_message_text: Color::Reset,
            md_heading: Color::Reset,
            md_code: Color::Reset,
            md_code_bg: Color::Reset,
            md_bold: Color::Reset,
            md_italic: Color::Reset,
            md_link: Color::Reset,
            thinking_text: Color::Reset,
            error: Color::Reset,
            warning: Color::Reset,
            muted: Color::Reset,
            dim: Color::Reset,
            tool_pending_bg: Color::Reset,
            tool_success_bg: Color::Reset,
            tool_error_bg: Color::Reset,
            accent: Color::Reset,
            scrollbar: Color::Reset,
        }
    }
}

impl Theme {
    pub fn dark() -> Self {
        Self::from_map("dark", &Self::dark_colors())
    }

    pub fn light() -> Self {
        Self::from_map("light", &Self::light_colors())
    }

    /// Pick dark/light from the terminal environment. Priority: an OSC-11
    /// background query (the terminal's actual background color), then
    /// `COLORFGBG` (rxvt/urxvt/Konsole convention), then macOS system
    /// appearance (a proxy for Terminal.app's default profile), else `dark()`.
    pub fn default_for_terminal() -> Self {
        if let Some(dark) = osc11_background_is_dark() {
            return if dark { Self::dark() } else { Self::light() };
        }
        if let Ok(colorfg) = std::env::var("COLORFGBG") {
            let background = colorfg.split(';').nth(1).unwrap_or("0");
            if let Ok(index) = background.trim().parse::<u8>() {
                if index < 8 {
                    return Self::dark();
                }
                return Self::light();
            }
        }
        if let Some(dark) = macos_dark_mode() {
            return if dark { Self::dark() } else { Self::light() };
        }
        Self::dark()
    }

    /// Downgrade every `Rgb` color to the nearest 256-color `Indexed` unless
    /// the terminal advertises truecolor support. crossterm writes `Rgb` as
    /// the 24-bit SGR form unconditionally (no capability check), and a
    /// terminal that doesn't understand it — Terminal.app is 256-color only —
    /// parses the parameters as independent legacy codes, so components in
    /// 40-47 silently paint a background the theme never asked for.
    pub fn downgrade_for_terminal(self) -> Self {
        if supports_truecolor() {
            return self;
        }
        Self {
            name: self.name,
            user_message_bg: quantize(self.user_message_bg),
            user_message_text: quantize(self.user_message_text),
            md_heading: quantize(self.md_heading),
            md_code: quantize(self.md_code),
            md_code_bg: quantize(self.md_code_bg),
            md_bold: quantize(self.md_bold),
            md_italic: quantize(self.md_italic),
            md_link: quantize(self.md_link),
            thinking_text: quantize(self.thinking_text),
            error: quantize(self.error),
            warning: quantize(self.warning),
            muted: quantize(self.muted),
            dim: quantize(self.dim),
            tool_pending_bg: quantize(self.tool_pending_bg),
            tool_success_bg: quantize(self.tool_success_bg),
            tool_error_bg: quantize(self.tool_error_bg),
            accent: quantize(self.accent),
            scrollbar: quantize(self.scrollbar),
        }
    }

    fn from_map(name: &str, colors: &serde_json::Map<String, Value>) -> Self {
        // Start from the dark defaults, then layer the custom colors on top.
        let mut theme = Self::from_colors_map(&Self::dark_colors());
        theme.name = name.to_string();
        theme.apply(colors);
        theme
    }

    fn from_colors_map(colors: &serde_json::Map<String, Value>) -> Self {
        let mut theme = Theme::default();
        theme.apply(colors);
        theme
    }

    fn apply(&mut self, colors: &serde_json::Map<String, Value>) {
        for (key, value) in colors {
            let parsed = value.as_str().and_then(parse_color).unwrap_or(Color::Reset);
            match key.as_str() {
                "userMessageBg" => self.user_message_bg = parsed,
                "userMessageText" => self.user_message_text = parsed,
                "mdHeading" => self.md_heading = parsed,
                "mdCode" => self.md_code = parsed,
                "mdCodeBg" => self.md_code_bg = parsed,
                "mdBold" => self.md_bold = parsed,
                "mdItalic" => self.md_italic = parsed,
                "mdLink" => self.md_link = parsed,
                "thinkingText" => self.thinking_text = parsed,
                "error" => self.error = parsed,
                "warning" => self.warning = parsed,
                "muted" => self.muted = parsed,
                "dim" => self.dim = parsed,
                "toolPendingBg" => self.tool_pending_bg = parsed,
                "toolSuccessBg" => self.tool_success_bg = parsed,
                "toolErrorBg" => self.tool_error_bg = parsed,
                "accent" => self.accent = parsed,
                "scrollbarThumb" => self.scrollbar = parsed,
                _ => {}
            }
        }
    }

    fn dark_colors() -> serde_json::Map<String, Value> {
        // pi's dark theme palette: warm, low-contrast blocks on the terminal's
        // native background.
        serde_json::from_str(
            r##"{
                "userMessageBg": "#343541",
                "userMessageText": "#d4d4d4",
                "mdHeading": "#f0c674",
                "mdCode": "#8abeb7",
                "mdCodeBg": "#282a2e",
                "mdBold": "#ffffff",
                "mdItalic": "#8b8b8b",
                "mdLink": "#81a2be",
                "thinkingText": "#808080",
                "error": "#f7768e",
                "warning": "#e0af68",
                "muted": "#808080",
                "dim": "#666666",
                "toolPendingBg": "#282832",
                "toolSuccessBg": "#283228",
                "toolErrorBg": "#3c2828",
                "accent": "#8abeb7",
                "scrollbarThumb": "#3a3a4a"
            }"##,
        )
        .unwrap()
    }

    fn light_colors() -> serde_json::Map<String, Value> {
        serde_json::from_str(
            r##"{
                "userMessageBg": "#e8e8ea",
                "userMessageText": "#202020",
                "mdHeading": "#3b5b82",
                "mdCode": "#6c4f00",
                "mdCodeBg": "#f2f0ec",
                "mdBold": "#000000",
                "mdItalic": "#5c5c5c",
                "mdLink": "#1a6ee0",
                "thinkingText": "#6c6c6c",
                "error": "#c0392b",
                "warning": "#a06500",
                "muted": "#8a8a8a",
                "dim": "#b0b0b0",
                "toolPendingBg": "#d5d8dd",
                "toolSuccessBg": "#d8e8d8",
                "toolErrorBg": "#f0d8d8",
                "accent": "#1a6ee0",
                "scrollbarThumb": "#c0c0c0"
            }"##,
        )
        .unwrap()
    }
}

/// `Some(true)` = system Dark Mode active, `Some(false)` = Light Mode,
/// `None` = not macOS or the query failed (caller falls back to `dark()`).
#[cfg(target_os = "macos")]
fn macos_dark_mode() -> Option<bool> {
    std::process::Command::new("defaults")
        .args(["read", "-g", "AppleInterfaceStyle"])
        .output()
        .ok()
        .map(|output| output.status.success())
}

#[cfg(not(target_os = "macos"))]
fn macos_dark_mode() -> Option<bool> {
    None
}

/// Query the terminal's background color via OSC-11 and return whether it is
/// dark. Returns `None` on timeout, parse failure, or when stdin/stdout are
/// not a TTY (so the caller falls back to the heuristic). Raw mode is enabled
/// for the query and restored on every exit path via a `Drop` guard.
fn osc11_background_is_dark() -> Option<bool> {
    use std::os::unix::io::AsRawFd;

    let mut stdout = std::io::stdout();
    let mut stdin = std::io::stdin();
    if !isatty(stdout.as_raw_fd()) || !isatty(stdin.as_raw_fd()) {
        return None;
    }
    // Enable raw mode so the response bytes reach us unprocessed.
    let _guard = RawModeGuard::new()?;
    stdout.write_all(OSC_11_QUERY).ok()?;
    stdout.flush().ok()?;

    // Read until the ST terminator or the timeout, whichever comes first.
    let mut buf = [0u8; 64];
    let mut response = Vec::new();
    let deadline = Instant::now() + OSC_TIMEOUT;
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        // Poll for readability so we don't block past the deadline.
        let mut pollfd = libc::pollfd {
            fd: stdin.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut pollfd, 1, remaining.as_millis() as i32) };
        if ready <= 0 {
            break;
        }
        let n = stdin.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        response.extend_from_slice(&buf[..n]);
        if response.windows(OSC_ST.len()).any(|w| w == OSC_ST) {
            break;
        }
    }
    parse_osc11(&response)
}

/// Parse an OSC-11 response like `\x1b]11;rgb:1e1e/1e1e/1e1e\x1b\\` (or
/// `rgb:1e1e1e/...`) into a dark/light decision. Returns `None` on any
/// malformed input.
fn parse_osc11(response: &[u8]) -> Option<bool> {
    let text = String::from_utf8_lossy(response);
    // Strip the leading CSI/OSC and trailing ST.
    let body = text
        .trim_start_matches('\x1b')
        .trim_start_matches(']')
        .trim_start_matches("11;")
        .trim_end_matches('\x1b')
        .trim_end_matches('\\');
    let rgb = body.strip_prefix("rgb:")?;
    // Components may be 2 or 4 hex digits, separated by '/'.
    let parts: Vec<&str> = rgb.split('/').collect();
    if parts.len() != 3 {
        return None;
    }
    let mut channels = [0u8; 3];
    for (index, part) in parts.iter().enumerate() {
        let hex = &part[..part.len().min(2)];
        channels[index] = u8::from_str_radix(hex, 16).ok()?;
    }
    let (r, g, b) = (channels[0] as f32, channels[1] as f32, channels[2] as f32);
    // Perceived luminance (Rec. 709 coefficients); < 0.5 → dark.
    let luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    Some(luminance < 127.5)
}

/// Enables raw mode on construction and restores it on drop, so the terminal
/// is left in a sane state on every exit path (including panics).
struct RawModeGuard;

impl RawModeGuard {
    fn new() -> Option<Self> {
        crossterm::terminal::enable_raw_mode().ok()?;
        Some(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

/// `isatty(3)` — true when `fd` refers to a terminal.
fn isatty(fd: i32) -> bool {
    unsafe { libc::isatty(fd) == 1 }
}

/// De-facto standard set by iTerm2, Ghostty, WezTerm, kitty, Alacritty and
/// VS Code; Terminal.app leaves it unset. Unknown → assume 256-color, which
/// fails safe (a quantized palette on a truecolor terminal looks nearly
/// identical; the converse is the bug this function exists to avoid).
fn supports_truecolor() -> bool {
    matches!(
        std::env::var("COLORTERM").as_deref(),
        Ok("truecolor") | Ok("24bit")
    )
}

fn quantize(color: Color) -> Color {
    match color {
        Color::Rgb(r, g, b) => Color::Indexed(rgb_to_ansi256(r, g, b)),
        other => other,
    }
}

/// Standard rgb→ansi256 approximation: the 24-step grayscale ramp
/// (232-255) when r≈g≈b, otherwise the 6×6×6 color cube (16-231).
fn rgb_to_ansi256(r: u8, g: u8, b: u8) -> u8 {
    let (rf, gf, bf) = (r as i32, g as i32, b as i32);
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    if max - min < 10 {
        let gray = (rf + gf + bf) / 3;
        if gray < 8 {
            return 16;
        }
        if gray > 248 {
            return 231;
        }
        return (((gray - 8) as f32 / 247.0 * 24.0).round() as u8) + 232;
    }
    let scale = |c: i32| (c as f32 / 255.0 * 5.0).round() as u8;
    16 + 36 * scale(rf) + 6 * scale(gf) + scale(bf)
}

/// Parse "#rgb", "#rrggbb", a named color, or "" (terminal default).
pub fn parse_color(value: &str) -> Option<Color> {
    let value = value.trim();
    if value.is_empty() {
        return Some(Color::Reset);
    }
    if let Some(hex) = value.strip_prefix('#') {
        if hex.len() == 3 {
            let r = u8::from_str_radix(&hex[0..1].repeat(2), 16).ok()?;
            let g = u8::from_str_radix(&hex[1..2].repeat(2), 16).ok()?;
            let b = u8::from_str_radix(&hex[2..3].repeat(2), 16).ok()?;
            return Some(Color::Rgb(r, g, b));
        }
        if hex.len() == 6 {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            return Some(Color::Rgb(r, g, b));
        }
        return None;
    }
    value.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hex_colors() {
        assert_eq!(parse_color("#f7768e"), Some(Color::Rgb(0xf7, 0x76, 0x8e)));
        assert_eq!(parse_color("#f78"), Some(Color::Rgb(0xff, 0x77, 0x88)));
        assert_eq!(parse_color(""), Some(Color::Reset));
        assert_eq!(parse_color("red"), Some(Color::Red));
        assert_eq!(parse_color("not-a-color"), None);
    }

    #[test]
    fn custom_theme_overrides_dark_defaults() {
        let theme = parse_theme(
            r##"{"name": "mine", "colors": {"error": "#ff0000", "mdHeading": "#00ff00"}}"##,
        )
        .unwrap();
        assert_eq!(theme.name, "mine");
        assert_eq!(theme.error, Color::Rgb(0xff, 0, 0));
        assert_eq!(theme.md_heading, Color::Rgb(0, 0xff, 0));
        // Unspecified keys fall back to the dark defaults.
        assert_eq!(theme.user_message_bg, Theme::dark().user_message_bg);
    }

    #[test]
    fn quantize_maps_rgb_to_ansi256() {
        assert_eq!(quantize(Color::Rgb(0x80, 0x80, 0x80)), Color::Indexed(244));
        assert_eq!(quantize(Color::Rgb(0xff, 0, 0)), Color::Indexed(196));
        assert_eq!(quantize(Color::Reset), Color::Reset);
        assert_eq!(quantize(Color::Red), Color::Red);
    }

    #[test]
    fn downgrade_for_terminal_removes_all_rgb_unless_truecolor() {
        unsafe { std::env::remove_var("COLORTERM") };
        let downgraded = Theme::dark().downgrade_for_terminal();
        assert!(!matches!(downgraded.tool_success_bg, Color::Rgb(..)));
        assert!(!matches!(downgraded.muted, Color::Rgb(..)));
        assert!(!matches!(downgraded.user_message_bg, Color::Rgb(..)));

        unsafe { std::env::set_var("COLORTERM", "truecolor") };
        let kept = Theme::dark().downgrade_for_terminal();
        assert!(matches!(kept.tool_success_bg, Color::Rgb(..)));
        unsafe { std::env::remove_var("COLORTERM") };
    }

    #[test]
    fn terminal_heuristic_prefers_dark() {
        unsafe { std::env::set_var("COLORFGBG", "15;0") };
        assert_eq!(Theme::default_for_terminal().name, "dark");
        unsafe { std::env::set_var("COLORFGBG", "0;15") };
        assert_eq!(Theme::default_for_terminal().name, "light");
    }

    #[test]
    fn parses_osc11_dark_and_light() {
        // Dark background (1e1e1e).
        let dark = parse_osc11(b"\x1b]11;rgb:1e1e/1e1e/1e1e\x1b\\");
        assert_eq!(dark, Some(true));
        // Light background (f0f0f0).
        let light = parse_osc11(b"\x1b]11;rgb:f0f0/f0f0/f0f0\x1b\\");
        assert_eq!(light, Some(false));
        // Malformed → None.
        assert_eq!(parse_osc11(b"garbage"), None);
        assert_eq!(parse_osc11(b"\x1b]11;rgb:zz/zz/zz\x1b\\"), None);
    }
}
