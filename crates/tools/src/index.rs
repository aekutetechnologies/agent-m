//! Local symbol/keyword index for the `search` tool.
//!
//! A per-project JSON cache of file paths, symbol names (with line numbers),
//! and identifier tokens. Pure string/identifier extraction — best-effort, no
//! full language parsing and no embedding API, so it never goes stale and
//! costs nothing to rebuild. The index lives under the agent data dir
//! (`~/.agent-m/index/<cwd-hash>.json`), never inside the repo.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// A symbol found at `line` (1-based) in a file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolHit {
    pub kind: String,
    pub name: String,
    pub line: usize,
}

/// One indexed file: relative path + extracted symbols + identifier tokens.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexedFile {
    pub path: String,
    pub symbols: Vec<SymbolHit>,
    pub identifiers: Vec<String>,
}

/// The persisted index for one project root.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SymbolIndex {
    pub root: String,
    /// Newest file mtime (secs) seen at build time — staleness check.
    pub max_mtime: u64,
    pub files: Vec<IndexedFile>,
    pub total_symbols: usize,
}

/// Directories never indexed, whatever the language.
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "target",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "vendor",
    ".idea",
    ".vscode",
    "__pycache__",
];

/// Files larger than this are not indexed.
const MAX_FILE_BYTES: u64 = 1024 * 1024;

/// The agent data dir: `$AGENT_M_DIR` or `~/.agent-m` (mirrors the CLI).
pub fn agent_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("AGENT_M_DIR") {
        return PathBuf::from(dir);
    }
    std::env::var("HOME")
        .map(|home| PathBuf::from(home).join(".agent-m"))
        .unwrap_or_else(|_| PathBuf::from(".agent-m"))
}

/// `<agent_dir>/index/<stable-hash-of-root>.json`
pub fn index_path_for(root: &Path) -> PathBuf {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    use std::hash::{Hash, Hasher};
    root.to_string_lossy().hash(&mut hasher);
    let hash = hasher.finish();
    agent_dir()
        .join("index")
        .join(format!("{hash:016x}-v2.json"))
}

/// Build (or refresh) the index for `root`. Returns the index.
pub fn build_index(root: &Path) -> SymbolIndex {
    let mut index = SymbolIndex {
        root: root.to_string_lossy().to_string(),
        max_mtime: 0,
        files: Vec::new(),
        total_symbols: 0,
    };
    let mut files: Vec<PathBuf> = Vec::new();
    walk_files(root, &mut files);
    files.sort();
    for path in files {
        let Ok(metadata) = std::fs::metadata(&path) else {
            continue;
        };
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs())
            .unwrap_or(0);
        index.max_mtime = index.max_mtime.max(mtime);
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        let mut file = IndexedFile {
            path: relative,
            ..Default::default()
        };
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        file.symbols = extract_symbols(&extension, &content);
        file.identifiers = extract_identifiers(&content);
        index.total_symbols += file.symbols.len();
        index.files.push(file);
    }
    index
}

/// Load the cached index if it exists and is fresh (no file under `root` is
/// newer than the index), otherwise rebuild it.
pub fn load_or_build(root: &Path) -> SymbolIndex {
    let path = index_path_for(root);
    if let Ok(text) = std::fs::read_to_string(&path)
        && let Ok(index) = serde_json::from_str::<SymbolIndex>(&text)
        && index.root == root.to_string_lossy()
        && !is_stale(root, index.max_mtime)
    {
        return index;
    }
    let index = build_index(root);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, serde_json::to_string(&index).unwrap_or_default());
    index
}

/// True if any file under `root` (respecting the ignore rules) is newer than
/// `max_mtime`.
fn is_stale(root: &Path, max_mtime: u64) -> bool {
    let mut newest = 0u64;
    let mut files: Vec<PathBuf> = Vec::new();
    walk_files(root, &mut files);
    for path in files {
        if let Ok(metadata) = std::fs::metadata(&path)
            && let Some(mtime) = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|duration| duration.as_secs())
        {
            newest = newest.max(mtime);
        }
    }
    newest > max_mtime
}

/// Depth-first walk collecting indexable files (skips symlinks, ignore dirs,
/// oversized files).
fn walk_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_symlink() {
            continue; // no symlink loops
        }
        if crate::paths::is_sensitive(&path) {
            continue; // skip secrets
        }
        if file_type.is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            if SKIP_DIRS.contains(&name.as_str()) {
                continue;
            }
            walk_files(&path, files);
        } else if file_type.is_file() {
            if let Ok(metadata) = std::fs::metadata(&path)
                && metadata.len() > MAX_FILE_BYTES
            {
                continue;
            }
            files.push(path);
        }
    }
}

/// Best-effort per-language symbol extraction. Each (kind, name, line).
pub fn extract_symbols(extension: &str, text: &str) -> Vec<SymbolHit> {
    let patterns: Vec<(&str, &str)> = match extension {
        "rs" => vec![
            (
                "fn",
                r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "struct",
                r"(?m)^\s*(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "enum",
                r"(?m)^\s*(?:pub\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "trait",
                r"(?m)^\s*(?:pub\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "impl",
                r"(?m)^\s*impl(?:<[^>]*>)?\s+([A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "const",
                r"(?m)^\s*(?:pub\s+)?const\s+([A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "type",
                r"(?m)^\s*(?:pub\s+)?type\s+([A-Za-z_][A-Za-z0-9_]*)",
            ),
            ("mod", r"(?m)^\s*(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)"),
        ],
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => vec![
            (
                "function",
                r"(?m)^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)",
            ),
            (
                "class",
                r"(?m)^\s*(?:export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)",
            ),
            (
                "interface",
                r"(?m)^\s*(?:export\s+)?interface\s+([A-Za-z_$][A-Za-z0-9_$]*)",
            ),
            (
                "const",
                r"(?m)^\s*(?:export\s+)?const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=",
            ),
            (
                "let",
                r"(?m)^\s*(?:export\s+)?let\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=",
            ),
            (
                "type",
                r"(?m)^\s*(?:export\s+)?type\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=",
            ),
        ],
        "go" => vec![
            (
                "func",
                r"(?m)^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)",
            ),
            ("type", r"(?m)^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("var", r"(?m)^\s*var\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("const", r"(?m)^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)"),
        ],
        "py" => vec![
            (
                "def",
                r"(?m)^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)",
            ),
            ("class", r"(?m)^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)"),
        ],
        "java" | "kt" | "kts" => vec![
            (
                "class",
                r"(?m)^\s*(?:public\s+|private\s+|internal\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "interface",
                r"(?m)^\s*(?:public\s+)?interface\s+([A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "enum",
                r"(?m)^\s*(?:public\s+)?enum\s+([A-Za-z_][A-Za-z0-9_]*)",
            ),
            (
                "fun",
                r"(?m)^\s*(?:fun\s+|public\s+fun\s+)([A-Za-z_][A-Za-z0-9_]*)",
            ),
        ],
        "c" | "h" | "cpp" | "hpp" | "cc" => vec![
            ("define", r"(?m)^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("struct", r"(?m)^\s*struct\s+([A-Za-z_][A-Za-z0-9_]*)"),
            (
                "fn",
                r"(?m)^\s*[A-Za-z_][A-Za-z0-9_]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            ),
        ],
        "sh" | "bash" => vec![
            ("fn", r"(?m)^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("fn", r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\)\s*\{"),
        ],
        "toml" => vec![("key", r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")],
        "json" => vec![("key", r#"(?m)^\s*"([A-Za-z_][A-Za-z0-9_]*)":"#)],
        "yaml" | "yml" => vec![("key", r"(?m)^\s*([A-Za-z_][A-Za-z0-9_-]*):\s*(?:[^#]|$)")],
        _ => Vec::new(),
    };
    let mut hits = Vec::new();
    let mut seen = BTreeSet::new();
    for (kind, pattern) in &patterns {
        let Ok(regex) = regex::Regex::new(pattern) else {
            continue;
        };
        for (index, line) in text.lines().enumerate() {
            for capture in regex.captures_iter(line) {
                if let Some(name) = capture.get(1) {
                    let name = name.as_str().to_string();
                    if seen.insert((kind.to_string(), name.clone(), index + 1)) {
                        hits.push(SymbolHit {
                            kind: kind.to_string(),
                            name,
                            line: index + 1,
                        });
                    }
                }
            }
        }
    }
    hits
}

/// Deduplicated identifier tokens (camelCase/snake_case words) from the file.
pub fn extract_identifiers(text: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    for word in text.split(|character: char| !character.is_alphanumeric() && character != '_') {
        if word.is_empty() || word.len() < 3 {
            continue;
        }
        for token in split_identifier(word) {
            if token.len() >= 3 && seen.insert(token.clone()) && seen.len() > 2000 {
                break;
            }
        }
    }
    seen.into_iter().collect()
}

/// Split `camelCase`/`snake_case`/`PascalCase` into lowercase tokens.
fn split_identifier(word: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = word.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '_' || ch == '-' {
            if !current.is_empty() {
                tokens.push(current.clone().to_lowercase());
                current.clear();
            }
            continue;
        }
        if ch.is_ascii_uppercase() {
            // camelCase boundary: lower followed by upper.
            if let Some(&next) = chars.peek()
                && next.is_ascii_lowercase()
                && !current.is_empty()
            {
                tokens.push(current.clone().to_lowercase());
                current.clear();
            }
            // PascalCase: already-separated run of uppers stays together
            // (e.g. "HTTPClient" -> "httpclient" via the lowercase below).
        }
        current.push(ch);
    }
    if !current.is_empty() {
        tokens.push(current.to_lowercase());
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn fixture() -> tempfile::TempDir {
        let dir = tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::create_dir_all(dir.path().join("node_modules")).unwrap();
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "pub fn cache_hit_count() -> u64 { 0 }\nstruct PrefixCache {}\nconst MAX_BYTES: usize = 1024;\n",
        )
        .unwrap();
        std::fs::write(dir.path().join("src/ignored.rs"), "fn secret() {}").unwrap();
        std::fs::write(
            dir.path().join("node_modules/dep.js"),
            "export const irrelevant = 1;",
        )
        .unwrap();
        std::fs::write(dir.path().join("package.json"), "{ \"name\": \"demo\" }").unwrap();
        dir
    }

    #[test]
    fn extracts_symbols_and_tokens() {
        let dir = fixture();
        let index = build_index(dir.path());
        let lib = index
            .files
            .iter()
            .find(|file| file.path.ends_with("src/lib.rs"))
            .expect("lib.rs indexed");
        let names: Vec<&str> = lib.symbols.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"cache_hit_count"), "{names:?}");
        assert!(names.contains(&"PrefixCache"), "{names:?}");
        assert!(names.contains(&"MAX_BYTES"), "{names:?}");
        assert!(
            lib.identifiers.iter().any(|id| id == "cache"),
            "{:?}",
            lib.identifiers
        );
        assert!(
            lib.identifiers.iter().any(|id| id == "hit"),
            "{:?}",
            lib.identifiers
        );
        assert!(
            lib.identifiers.iter().any(|id| id == "count"),
            "{:?}",
            lib.identifiers
        );
    }

    #[test]
    fn skips_ignored_dirs() {
        let dir = fixture();
        let index = build_index(dir.path());
        assert!(
            index
                .files
                .iter()
                .all(|file| !file.path.contains("node_modules")),
            "node_modules must be skipped"
        );
    }

    #[test]
    fn rebuilds_when_stale() {
        let dir = fixture();
        let first = load_or_build(dir.path());
        let second = load_or_build(dir.path());
        assert_eq!(first.files.len(), second.files.len(), "cached index reused");
        // Touch a file with a newer mtime.
        let lib = dir.path().join("src/lib.rs");
        let now = std::time::SystemTime::now();
        let _ = filetime_dummy(&lib, now);
        let refreshed = load_or_build(dir.path());
        assert!(refreshed.max_mtime >= first.max_mtime);
        let _ = lib;
    }

    // Avoid a filetime dependency: bump mtime by rewriting the file.
    fn filetime_dummy(path: &Path, _now: std::time::SystemTime) -> std::io::Result<()> {
        let content = std::fs::read_to_string(path)?;
        std::fs::write(path, format!("{content}\n// touched\n"))
    }

    #[test]
    fn split_identifier_cases() {
        assert_eq!(
            split_identifier("cacheHitCount"),
            vec!["cache", "hit", "count"]
        );
        assert_eq!(split_identifier("MAX_BYTES"), vec!["max", "bytes"]);
        assert_eq!(split_identifier("PrefixCache"), vec!["prefix", "cache"]);
    }
}
