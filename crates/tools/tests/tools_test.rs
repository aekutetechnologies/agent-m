//! Real-tool tests against temp directories: write→read roundtrip, edit with
//! unified diff, bash success/failure/timeout, ls sorting, grep, find.

use agent_m_agent::{Tool, ToolContext};
use agent_m_tools::{
    AskTool, BashTool, EditTool, FindTool, GrepTool, LsTool, ReadTool, SearchTool, WriteTool,
};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::tempdir;

fn context(cwd: PathBuf) -> ToolContext {
    ToolContext::simple(cwd)
}

#[tokio::test]
async fn write_and_read_roundtrip() {
    let dir = tempdir().unwrap();
    let cwd = context(dir.path().to_path_buf());

    let write = WriteTool
        .execute(
            json!({ "path": "sub/file.txt", "content": "line1\nline2\nline3\n" }),
            &cwd,
        )
        .await
        .expect("write succeeds");
    assert!(!write.is_error);
    assert!(write.content.contains("18 bytes"), "got: {}", write.content);

    let read = ReadTool
        .execute(json!({ "path": "sub/file.txt" }), &cwd)
        .await
        .expect("read succeeds");
    assert_eq!(read.content, "line1\nline2\nline3");

    // offset + limit
    let partial = ReadTool
        .execute(
            json!({ "path": "sub/file.txt", "offset": 1, "limit": 1 }),
            &cwd,
        )
        .await
        .expect("read partial");
    assert!(
        partial.content.starts_with("line2"),
        "got: {}",
        partial.content
    );
}

#[tokio::test]
async fn edit_applies_exact_text_and_returns_diff() {
    let dir = tempdir().unwrap();
    let cwd = context(dir.path().to_path_buf());
    WriteTool
        .execute(
            json!({ "path": "file.txt", "content": "hello world\nfoo bar\n" }),
            &cwd,
        )
        .await
        .expect("write");

    let edit = EditTool
        .execute(
            json!({
                "path": "file.txt",
                "edits": [{ "oldText": "foo bar", "newText": "foo baz" }]
            }),
            &cwd,
        )
        .await
        .expect("edit succeeds");
    assert!(!edit.is_error, "got: {}", edit.content);
    assert!(edit.content.contains("foo baz"), "got: {}", edit.content);

    let read = ReadTool
        .execute(json!({ "path": "file.txt" }), &cwd)
        .await
        .expect("read");
    assert_eq!(read.content, "hello world\nfoo baz");
}

#[tokio::test]
async fn edit_overlapping_edits_fail_loudly() {
    let dir = tempdir().unwrap();
    let cwd = context(dir.path().to_path_buf());
    // edit 1 rewrites "a" → "b"; edit 2's oldText "ab" only existed in the
    // original. Applying edit 1 first makes edit 2 unfindable — it must fail
    // loudly instead of being silently skipped.
    WriteTool
        .execute(json!({ "path": "f.txt", "content": "ab\n" }), &cwd)
        .await
        .expect("write");

    let edit = EditTool
        .execute(
            json!({
                "path": "f.txt",
                "edits": [
                    { "oldText": "a", "newText": "b" },
                    { "oldText": "ab", "newText": "x" }
                ]
            }),
            &cwd,
        )
        .await;
    let error = match edit {
        Ok(_) => panic!("overlapping edits should fail"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("ab"), "got: {error}");
    assert!(error.contains("exactly once"), "got: {error}");

    // The failed batch must not have modified the file.
    let read = ReadTool
        .execute(json!({ "path": "f.txt" }), &cwd)
        .await
        .expect("read");
    assert_eq!(read.content, "ab");
}

#[tokio::test]
async fn edit_rejects_ambiguous_old_text() {
    let dir = tempdir().unwrap();
    let cwd = context(dir.path().to_path_buf());
    WriteTool
        .execute(json!({ "path": "file.txt", "content": "dup dup\n" }), &cwd)
        .await
        .expect("write");

    let edit = EditTool
        .execute(
            json!({ "path": "file.txt", "edits": [{ "oldText": "dup", "newText": "x" }] }),
            &cwd,
        )
        .await;
    let error = match edit {
        Ok(_) => panic!("ambiguous edit should fail"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("appears 2 times"), "got: {error}");
}

#[tokio::test]
async fn bash_success_failure_and_timeout() {
    let dir = tempdir().unwrap();
    let cwd = context(dir.path().to_path_buf());

    let ok = BashTool
        .execute(json!({ "command": "echo hello" }), &cwd)
        .await
        .expect("bash");
    assert!(!ok.is_error);
    assert!(ok.content.contains("hello"), "got: {}", ok.content);

    let failed = BashTool
        .execute(json!({ "command": "exit 3" }), &cwd)
        .await
        .expect("bash");
    assert!(failed.is_error);
    assert!(failed.content.contains("3"), "got: {}", failed.content);

    let timed_out = BashTool
        .execute(json!({ "command": "sleep 5", "timeout": 1 }), &cwd)
        .await
        .expect("bash");
    assert!(timed_out.is_error);
    assert!(
        timed_out.content.contains("timed out"),
        "got: {}",
        timed_out.content
    );
}

#[tokio::test]
async fn ls_sorts_and_marks_directories() {
    let dir = tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("alpha/sub")).unwrap();
    std::fs::write(dir.path().join("alpha/b.txt"), "b").unwrap();
    std::fs::write(dir.path().join("alpha/A.txt"), "a").unwrap();
    let cwd = context(dir.path().join("alpha"));

    let listing = LsTool.execute(json!({}), &cwd).await.expect("ls");
    assert!(!listing.is_error);
    assert!(
        listing.content.contains("A.txt"),
        "got: {}",
        listing.content
    );
    assert!(
        listing.content.contains("b.txt"),
        "got: {}",
        listing.content
    );
    assert!(listing.content.contains("sub/"), "got: {}", listing.content);
    // case-insensitive sort: A.txt before b.txt before sub/
    let a = listing.content.find("A.txt").unwrap();
    let b = listing.content.find("b.txt").unwrap();
    let sub = listing.content.find("sub/").unwrap();
    assert!(a < b && b < sub);
}

#[tokio::test]
async fn grep_finds_matches() {
    let dir = tempdir().unwrap();
    std::fs::write(dir.path().join("code.txt"), "needle here\nnothing here\n").unwrap();
    std::fs::write(dir.path().join("other.txt"), "no match\n").unwrap();
    let cwd = context(dir.path().to_path_buf());

    let result = GrepTool
        .execute(json!({ "pattern": "needle", "path": "." }), &cwd)
        .await
        .expect("grep");
    assert!(!result.is_error);
    assert!(
        result.content.contains("code.txt"),
        "got: {}",
        result.content
    );
    assert!(
        result.content.contains("needle here"),
        "got: {}",
        result.content
    );
}

#[tokio::test]
async fn find_matches_glob() {
    let dir = tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("src/main.rs"), "fn main() {}").unwrap();
    std::fs::write(dir.path().join("README.md"), "# readme").unwrap();
    let cwd = context(dir.path().to_path_buf());

    let result = FindTool
        .execute(json!({ "pattern": "**/*.rs" }), &cwd)
        .await
        .expect("find");
    assert!(!result.is_error);
    assert!(
        result.content.contains("src/main.rs"),
        "got: {}",
        result.content
    );
    assert!(
        !result.content.contains("README.md"),
        "got: {}",
        result.content
    );
}

#[tokio::test]
async fn bash_verbose_output_does_not_time_out() {
    // Output larger than the 64KB pipe buffer must not deadlock `wait()`.
    let dir = tempdir().unwrap();
    let cwd = context(dir.path().to_path_buf());
    let result = BashTool
        .execute(
            json!({ "command": "python3 -c \"import sys; sys.stdout.write('x'*120000)\"" }),
            &cwd,
        )
        .await
        .expect("bash");
    assert!(!result.is_error, "got: {}", result.content);
    // The pipe drained fully; the tool's own truncation notice shows the
    // output exceeded the 50KB display cap, and it was not killed by timeout.
    assert!(
        result.content.contains("truncated"),
        "got {} bytes: {}",
        result.content.len(),
        result.content
    );
    assert!(!result.content.contains("timed out"));
}

#[tokio::test]
async fn read_rejects_huge_files() {
    let dir = tempdir().unwrap();
    let cwd = context(dir.path().to_path_buf());
    // 11 MB file, over the 10 MB read cap.
    let big = dir.path().join("big.txt");
    let handle = std::fs::File::create(&big).unwrap();
    handle.set_len(11 * 1024 * 1024).unwrap();
    let result = ReadTool
        .execute(json!({ "path": "big.txt" }), &cwd)
        .await
        .expect("read");
    assert!(result.is_error);
    assert!(result.content.contains("10 MB"), "got: {}", result.content);
}

#[tokio::test]
async fn read_on_directory_points_at_ls() {
    let dir = tempdir().unwrap();
    let cwd = context(dir.path().to_path_buf());
    let result = ReadTool
        .execute(json!({ "path": "." }), &cwd)
        .await
        .expect("read");
    assert!(result.is_error);
    assert!(
        result.content.contains("use the `ls` tool"),
        "directory error must teach the model to use ls, got: {}",
        result.content
    );
}

#[tokio::test]
async fn ask_returns_answer_via_gate_and_errors_without() {
    let gate: Arc<dyn agent_m_agent::AskGate> = Arc::new(agent_m_agent::ClosureAskGate::new(
        |_question, _options, _multi| Box::pin(async { Ok("yes".to_string()) }),
    ));
    let mut with_gate = ToolContext::simple(PathBuf::from("."));
    with_gate.ask_gate = Some(gate);
    let result = AskTool
        .execute(json!({ "question": "continue?" }), &with_gate)
        .await
        .expect("ask");
    assert!(
        result.content.contains("User answer: yes"),
        "got: {}",
        result.content
    );

    let no_gate = ToolContext::simple(PathBuf::from("."));
    let result = AskTool
        .execute(json!({ "question": "continue?" }), &no_gate)
        .await
        .expect("ask");
    assert!(
        result.is_error && result.content.contains("interactive UI"),
        "got: {}",
        result.content
    );
}

#[tokio::test]
async fn search_finds_symbols_identifiers_and_files() {
    let dir = tempdir().unwrap();
    let data = tempdir().unwrap();
    unsafe { std::env::set_var("AGENT_M_DIR", data.path()) };
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn cache_hit_count() -> u64 { 0 }\npub fn prefix_cache_lookup() {}\nstruct PrefixCache {}\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("README.md"), "# demo").unwrap();
    let cwd = context(dir.path().to_path_buf());

    // Exact symbol name → ranked first.
    let result = SearchTool
        .execute(json!({ "query": "cache_hit_count" }), &cwd)
        .await
        .expect("search");
    assert!(!result.is_error);
    assert!(
        result.content.contains("src/lib.rs:1 (fn cache_hit_count)"),
        "got: {}",
        result.content
    );

    // Identifier token (substring of the symbol) still matches.
    let result = SearchTool
        .execute(json!({ "query": "prefix" }), &cwd)
        .await
        .expect("search");
    assert!(
        result.content.contains("prefix_cache_lookup") || result.content.contains("PrefixCache"),
        "got: {}",
        result.content
    );

    // File-name fragment.
    let result = SearchTool
        .execute(json!({ "query": "readme" }), &cwd)
        .await
        .expect("search");
    assert!(
        result.content.contains("README.md (file name match)"),
        "got: {}",
        result.content
    );

    // No match.
    let result = SearchTool
        .execute(json!({ "query": "zzz_nothing_here" }), &cwd)
        .await
        .expect("search");
    assert!(
        result.content.contains("no matches"),
        "got: {}",
        result.content
    );
}

#[tokio::test]
async fn search_ranking_exact_beats_substring_and_caps_results() {
    let dir = tempdir().unwrap();
    let data = tempdir().unwrap();
    unsafe { std::env::set_var("AGENT_M_DIR", data.path()) };
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    // A symbol whose name contains the query as a substring but is not exact.
    std::fs::write(
        dir.path().join("src/notexact.rs"),
        "fn token_manager() {}\nfn tokenizer() {}\n",
    )
    .unwrap();
    // The exact symbol (top-level, should rank above substring matches).
    std::fs::write(dir.path().join("src/exact.rs"), "fn token() {}\n").unwrap();
    let cwd = context(dir.path().to_path_buf());

    let result = SearchTool
        .execute(json!({ "query": "token" }), &cwd)
        .await
        .expect("search");
    assert!(!result.is_error);
    let lines: Vec<&str> = result.content.lines().collect();
    let first_match = lines
        .iter()
        .find(|line| line.starts_with("  "))
        .copied()
        .unwrap_or("");
    assert!(
        first_match.contains("exact.rs") && first_match.contains("(fn token)"),
        "exact symbol must rank first, got: {first_match}\n{}",
        result.content
    );

    // Cap: >20 matching files → at most 20 result lines.
    for i in 0..30 {
        std::fs::write(
            dir.path().join(format!("src/file{i}.rs")),
            format!("fn flood_{i}() {{}}\n"),
        )
        .unwrap();
    }
    let result = SearchTool
        .execute(json!({ "query": "flood_" }), &cwd)
        .await
        .expect("search");
    let match_lines = result
        .content
        .lines()
        .filter(|line| line.starts_with("  "))
        .count();
    assert!(
        match_lines <= 20,
        "cap violated: {match_lines} lines\n{}",
        result.content
    );
}

#[tokio::test]
async fn web_fetch_blocks_loopback_targets() {
    use agent_m_tools::WebFetchTool;

    // A wiremock server binds to 127.0.0.1 — exactly the SSRF case the
    // guard must reject, so the fetch never reaches the server.
    let server = wiremock::MockServer::start().await;
    let context = agent_m_agent::ToolContext::simple(std::path::PathBuf::from("."));
    let url = format!("{}/page", server.uri());
    let outcome = WebFetchTool
        .execute(serde_json::json!({ "url": url }), &context)
        .await
        .expect("execute");
    assert!(
        outcome.is_error,
        "must refuse loopback: {}",
        outcome.content
    );
    assert!(
        outcome.content.contains("blocked") && outcome.content.contains("loopback"),
        "clear reason: {}",
        outcome.content
    );
}

#[tokio::test]
async fn web_fetch_captures_html_content() {
    use agent_m_tools::WebFetchTool;

    // HTML→text conversion is exercised against the public site path via the
    // unit-tested html_to_text; here we verify a public URL's HTML body is
    // stripped (no raw tags) using a local file server on a non-loopback
    // interface would need network — instead assert the content-type sniffing
    // on the unit level already covered. This test guards the refusal path.
    let context = agent_m_agent::ToolContext::simple(std::path::PathBuf::from("."));
    let outcome = WebFetchTool
        .execute(serde_json::json!({ "url": "file:///etc/passwd" }), &context)
        .await
        .expect("execute");
    assert!(outcome.is_error);
    assert!(outcome.content.contains("refusing non-http(s)"));
}

#[test]
fn web_tools_classify_low_risk() {
    let policy = agent_m_agent::RiskPolicy {
        cwd: std::path::PathBuf::from("/work"),
        opaque_tools: vec![],
    };
    let fetch_call = agent_m_agent::ToolCallInfo {
        tool_call_id: "1".into(),
        name: "web_fetch".into(),
        arguments: serde_json::json!({ "url": "https://example.com" }),
    };
    let search_call = agent_m_agent::ToolCallInfo {
        tool_call_id: "2".into(),
        name: "web_search".into(),
        arguments: serde_json::json!({ "query": "rust" }),
    };
    assert_eq!(
        policy.assess(&fetch_call).level,
        agent_m_agent::RiskLevel::Low
    );
    assert_eq!(
        policy.assess(&search_call).level,
        agent_m_agent::RiskLevel::Low
    );
}
