//! Output truncation shared by the tools, mirroring pi's `truncate.ts`
//! (2000 lines / 50 KB).
//!
//! For large outputs, `offload_or_truncate` writes the full content to a
//! session output directory and returns a compact preview + path hint.

use std::path::Path;

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

/// Monotonic counter for unique offload filenames within a process lifetime.
static OFFLOAD_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Threshold above which tool output is offloaded to disk (10 KB).
const OFFLOAD_THRESHOLD: usize = 10 * 1024;
/// How much of the offloaded output to keep inline as a preview (2 KB).
const INLINE_PREVIEW_BYTES: usize = 2 * 1024;

/// Return the string to embed in the conversation for `content`.
///
/// - Below `OFFLOAD_THRESHOLD`: inline with normal truncation.
/// - Above threshold and `output_dir` is `Some`: write the full content to
///   `{output_dir}/{filename}`, return a 2 KB head preview + path hint.
/// - Above threshold but no `output_dir`: fall back to plain truncation.
pub fn offload_or_truncate(content: &str, tool_prefix: &str, output_dir: Option<&Path>) -> String {
    let n = OFFLOAD_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let filename = format!("{tool_prefix}_{n:04}.txt");
    if content.len() <= OFFLOAD_THRESHOLD {
        let (kept, notice) = truncate_output(content);
        return match notice {
            Some(n) => format!("{kept}\n{n}"),
            None => kept,
        };
    }
    if let Some(dir) = output_dir
        && std::fs::create_dir_all(dir).is_ok()
    {
        let path = dir.join(filename);
        if std::fs::write(&path, content).is_ok() {
            let preview: String = content.chars().take(INLINE_PREVIEW_BYTES).collect();
            return format!(
                "{preview}\n…\n[Output truncated. Full {} bytes saved to {}.\nUse: read path=\"{}\" to inspect it.]",
                content.len(),
                path.display(),
                path.display(),
            );
        }
    }
    // Fallback: plain truncation (output_dir absent or write failed).
    let (kept, notice) = truncate_output(content);
    match notice {
        Some(n) => format!("{kept}\n{n}"),
        None => kept,
    }
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
