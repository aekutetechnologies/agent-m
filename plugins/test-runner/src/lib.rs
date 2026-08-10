//! Test-runner plugin: detect the project's test command and run it.
//! Pure std (no HTTP) — the `verify` step's workhorse.

use agent_m_plugin_sdk::PluginEntry;
use agent_m_plugin_sdk::tools::{ToolDef, entry};
use std::process::Command;
use std::sync::OnceLock;

/// Detect a test command from the project root's files.
fn detect_test_command(cwd: &str) -> String {
    for (marker, command) in [
        ("Cargo.toml", "cargo test"),
        ("go.mod", "go test ./..."),
        ("package.json", "npm test"),
        ("pyproject.toml", "python -m pytest"),
        ("Makefile", "make test"),
    ] {
        if std::path::Path::new(cwd).join(marker).is_file() {
            return command.to_string();
        }
    }
    "echo no test command detected".to_string()
}

fn run_tests(arguments: &str, cwd: &str) -> Result<String, String> {
    let value: serde_json::Value = serde_json::from_str(arguments).unwrap_or_default();
    let command = value
        .get("command")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| detect_test_command(cwd));
    let output = Command::new("sh")
        .arg("-c")
        .arg(&command)
        .current_dir(cwd)
        .output()
        .map_err(|error| format!("failed to run `{command}`: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let combined = format!("{stdout}{stderr}").trim().to_string();
    if output.status.success() {
        Ok(if combined.is_empty() {
            "tests passed".to_string()
        } else {
            combined
        })
    } else {
        Err(combined)
    }
}

static DEFS: &[ToolDef] = &[ToolDef {
    name: "run-tests",
    description: "Detect and run the project's test command (Cargo/Go/npm/pytest/Makefile); returns output, errors on failure",
    parameters: r#"{"type":"object","properties":{"command":{"type":"string","description":"Override the detected test command"}}}"#,
    execute: run_tests,
}];

struct EntryHolder(*const PluginEntry);
// SAFETY: the entry points into leaked, immutable plugin state.
unsafe impl Send for EntryHolder {}
unsafe impl Sync for EntryHolder {}

static ENTRY: OnceLock<EntryHolder> = OnceLock::new();

#[unsafe(no_mangle)]
pub extern "C" fn agent_m_plugin_entry() -> *const PluginEntry {
    ENTRY
        .get_or_init(|| EntryHolder(Box::leak(Box::new(entry("test-runner", "0.1.0", DEFS)))))
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_cargo_and_runs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        // `cargo test` needs a valid crate; instead verify detection via the
        // command echo path on an empty dir.
        assert_eq!(
            detect_test_command(dir.path().to_str().unwrap()),
            "cargo test"
        );
    }
}
