//! Output truncation shared by the tools, mirroring pi's `truncate.ts`
//! (2000 lines / 50 KB).

/// Maximum number of output lines kept.
pub const MAX_LINES: usize = 2000;
/// Maximum number of output bytes kept.
pub const MAX_BYTES: usize = 50 * 1024;

/// Truncate tool output and append a notice when something was cut.
pub fn truncate_output(content: &str) -> (String, Option<String>) {
    let mut lines: Vec<&str> = content.lines().collect();
    let mut truncated = false;

    if lines.len() > MAX_LINES {
        lines.truncate(MAX_LINES);
        truncated = true;
    }
    let mut kept = lines.join("\n");
    if kept.len() > MAX_BYTES {
        let mut cut: String = kept.chars().take(MAX_BYTES).collect();
        cut.push_str("\n…");
        kept = cut;
        truncated = true;
    }

    let notice = if truncated {
        Some(format!(
            "(output truncated to {MAX_LINES} lines / {MAX_BYTES} bytes)"
        ))
    } else {
        None
    };
    (kept, notice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_long_output() {
        let content = (0..MAX_LINES + 10)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let (kept, notice) = truncate_output(&content);
        assert!(kept.lines().count() <= MAX_LINES);
        assert!(notice.is_some());
    }

    #[test]
    fn passes_short_output_through() {
        let (kept, notice) = truncate_output("hello\nworld");
        assert_eq!(kept, "hello\nworld");
        assert!(notice.is_none());
    }
}
