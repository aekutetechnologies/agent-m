//! The reference plugin: two trivial tools proving the C-ABI contract.
//! A real plugin lives in its own repo; this one ships in-tree so the loader
//! is testable offline.

use agent_m_plugin_sdk::PluginEntry;
use agent_m_plugin_sdk::tools::{ToolDef, entry};
use std::sync::OnceLock;

fn hello(arguments: &str, _cwd: &str) -> Result<String, String> {
    let name = serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|value| {
            value
                .get("name")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or_else(|| "world".to_string());
    Ok(format!("hello, {name}! (from the fixture plugin)"))
}

fn sum(arguments: &str, _cwd: &str) -> Result<String, String> {
    let value: serde_json::Value =
        serde_json::from_str(arguments).map_err(|error| format!("invalid arguments: {error}"))?;
    let a = value
        .get("a")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    let b = value
        .get("b")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0);
    Ok(format!("{}", a + b))
}

static DEFS: &[ToolDef] = &[
    ToolDef {
        name: "fixture-hello",
        description: "Say hello from the fixture plugin",
        parameters: r#"{"type":"object","properties":{"name":{"type":"string"}}}"#,
        execute: hello,
    },
    ToolDef {
        name: "fixture-sum",
        description: "Sum two integers",
        parameters: r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}"#,
        execute: sum,
    },
];

struct EntryHolder(*const PluginEntry);
// SAFETY: the entry points into leaked, immutable plugin state.
unsafe impl Send for EntryHolder {}
unsafe impl Sync for EntryHolder {}

static ENTRY: OnceLock<EntryHolder> = OnceLock::new();

#[unsafe(no_mangle)]
pub extern "C" fn agent_m_plugin_entry() -> *const PluginEntry {
    ENTRY
        .get_or_init(|| EntryHolder(Box::leak(Box::new(entry("fixture", "0.1.0", DEFS)))))
        .0
}
