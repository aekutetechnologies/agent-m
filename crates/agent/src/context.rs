//! Context creation: assemble persistent project instructions (AGENTS.md
//! files) into the byte-stable system prompt, pi/Claude-Code style.

use std::path::{Path, PathBuf};

/// A discovered instruction file and its content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstructionFile {
    pub path: PathBuf,
    pub content: String,
}

/// Instruction filenames honored per directory, in precedence order
/// (agent-m's AGENTS.md first, then other agents' conventions). Loading them
/// all keeps agent-m useful in repos written for Claude Code / Cursor / Gemini
/// CLI without fork-specific config.
pub const INSTRUCTION_FILES: &[&str] = &["AGENTS.md", "CLAUDE.md", ".cursorrules", "GEMINI.md"];

/// Discover instructions for `cwd`: walk from `cwd` up to the
/// filesystem root (nearest file first so closer instructions override), then
/// the global `~/.agent-m/agent/AGENTS.md`. Missing files are skipped.
pub fn discover_instructions(cwd: &Path) -> Vec<InstructionFile> {
    let mut files: Vec<InstructionFile> = Vec::new();

    // Ancestors, nearest first (cwd up to root). Stop at home to avoid
    // unrelated parent directories outside the user's projects.
    let home = std::env::var("HOME").unwrap_or_default();
    let home = PathBuf::from(home);
    let mut current = Some(cwd.to_path_buf());
    while let Some(dir) = current {
        for name in INSTRUCTION_FILES {
            let candidate = dir.join(name);
            if candidate.is_file()
                && let Ok(content) = std::fs::read_to_string(&candidate)
            {
                files.push(InstructionFile {
                    path: candidate,
                    content,
                });
            }
        }
        if dir == home || dir.parent().is_none() {
            break;
        }
        current = dir.parent().map(Path::to_path_buf);
    }

    // Global instructions from the agent data directory
    // ($AGENT_M_DIR/agent/AGENTS.md, mirroring the CLI's agent_dir layout).
    if let Ok(agent_dir) = std::env::var("AGENT_M_DIR") {
        let global = PathBuf::from(agent_dir).join("agent").join("AGENTS.md");
        if global.is_file()
            && let Ok(content) = std::fs::read_to_string(&global)
        {
            files.push(InstructionFile {
                path: global,
                content,
            });
        }
    }
    files
}

/// Render the discovered instructions as an XML context block, each wrapped
/// in `<project_instructions path="…">…</project_instructions>`.
pub fn render_instructions(files: &[InstructionFile]) -> String {
    if files.is_empty() {
        return String::new();
    }
    let mut out = String::from("\n\nProject instructions:\n");
    for file in files {
        out.push_str(&format!(
            "<project_instructions path=\"{}\">\n{}\n</project_instructions>\n",
            escape_xml(&file.path.display().to_string()),
            escape_xml(&file.content)
        ));
    }
    out
}

fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn discovers_cross_tool_rules_files_in_order() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "# agent-m rules").unwrap();
        std::fs::write(dir.path().join("CLAUDE.md"), "# claude rules").unwrap();
        std::fs::write(dir.path().join(".cursorrules"), "# cursor rules").unwrap();
        let files = discover_instructions(dir.path());
        let names: Vec<String> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(
            names,
            vec!["AGENTS.md", "CLAUDE.md", ".cursorrules"],
            "precedence order in the same directory: {names:?}"
        );
        // The rendered block is deterministic (byte-stable prefix input).
        let rendered = render_instructions(&files);
        assert!(rendered.contains("claude rules"));
        assert!(rendered.contains("cursor rules"));
        assert_eq!(render_instructions(&files), rendered);
    }

    #[test]
    fn discovers_ancestors_nearest_first() {
        let dir = tempdir().unwrap();
        let inner = dir.path().join("a/b");
        std::fs::create_dir_all(&inner).unwrap();
        std::fs::write(dir.path().join("AGENTS.md"), "root rules").unwrap();
        std::fs::write(dir.path().join("a/b/AGENTS.md"), "inner rules").unwrap();

        let files = discover_instructions(&inner);
        assert_eq!(files.len(), 2);
        assert!(
            files[0].path.ends_with("a/b/AGENTS.md"),
            "nearest first, got {:?}",
            files[0].path
        );
        assert_eq!(files[0].content, "inner rules");
        assert!(files[1].path.ends_with("AGENTS.md"));
        assert_eq!(files[1].content, "root rules");
    }

    #[test]
    fn renders_xml_block_and_escapes() {
        let files = vec![InstructionFile {
            path: PathBuf::from("/tmp/x/AGENTS.md"),
            content: "use <b>bold</b> & keep it simple".to_string(),
        }];
        let rendered = render_instructions(&files);
        assert!(rendered.contains("<project_instructions path=\"/tmp/x/AGENTS.md\">"));
        assert!(rendered.contains("use &lt;b&gt;bold&lt;/b&gt; &amp; keep it simple"));
        assert!(rendered.contains("</project_instructions>"));
    }
}
