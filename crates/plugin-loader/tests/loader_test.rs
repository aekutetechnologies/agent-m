//! End-to-end plugin loading: build the fixture cdylib, load it with
//! libloading, and execute its tools through the agent Tool wrapper.

use agent_m_agent::{Tool, ToolContext};
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;

fn dylib_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libagent_m_plugin_fixture.dylib"
    } else if cfg!(target_os = "windows") {
        "agent_m_plugin_fixture.dll"
    } else {
        "libagent_m_plugin_fixture.so"
    }
}

#[test]
fn builds_and_loads_the_fixture_plugin() {
    // Build the fixture plugin (debug profile) with the workspace toolchain.
    let status = std::process::Command::new("cargo")
        .args([
            "build",
            "--manifest-path",
            "Cargo.toml",
            "-p",
            "agent-m-plugin-fixture",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR").replace("crates/plugin-loader", ""))
        .status()
        .expect("cargo build");
    assert!(status.success(), "fixture plugin build failed");

    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug");
    let dylib = workspace.join(dylib_name());
    assert!(dylib.is_file(), "missing {}", dylib.display());

    // SAFETY: the fixture is our own test plugin.
    let plugin =
        unsafe { agent_m_plugin_loader::LoadedPlugin::load(&dylib) }.expect("load fixture plugin");
    assert_eq!(plugin.name, "fixture");
    assert_eq!(plugin.tools.len(), 2);

    let hello = plugin
        .tools
        .iter()
        .find(|tool| tool.name() == "fixture-hello")
        .expect("fixture-hello tool");
    assert!(hello.description().contains("fixture plugin"));

    let context = ToolContext::simple(PathBuf::from("."));
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let outcome = runtime
        .block_on(hello.execute(json!({ "name": "plugin" }), &context))
        .expect("execute");
    assert!(!outcome.is_error, "got: {}", outcome.content);
    assert!(
        outcome.content.contains("hello, plugin!"),
        "got: {}",
        outcome.content
    );

    let sum = plugin
        .tools
        .iter()
        .find(|tool| tool.name() == "fixture-sum")
        .expect("fixture-sum tool");
    let outcome = runtime
        .block_on(sum.execute(json!({ "a": 2, "b": 3 }), &context))
        .expect("execute");
    assert_eq!(outcome.content, "5");
}

#[test]
fn load_plugins_dir_skips_broken_plugins() {
    let dir = tempfile::tempdir().unwrap();
    // A dir without a manifest (not a plugin) and one with a bad manifest.
    std::fs::create_dir_all(dir.path().join("not-a-plugin")).unwrap();
    let bad = dir.path().join("broken");
    std::fs::create_dir_all(&bad).unwrap();
    std::fs::write(bad.join("plugin.json"), "{ not json").unwrap();
    let loaded = agent_m_plugin_loader::load_plugins_dir(dir.path());
    assert!(loaded.is_empty(), "broken plugins must be skipped");
}

#[tokio::test]
async fn plugin_tool_wraps_into_agent_tool() {
    // The wrapper implements the agent Tool trait, so it can be registered in
    // a ToolRegistry-like vec and driven by the agent loop; here we just
    // verify the trait surface (name/parameters/execute) through the trait
    // object.
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug");
    let dylib = workspace.join(dylib_name());
    // SAFETY: our own fixture.
    let plugin = unsafe { agent_m_plugin_loader::LoadedPlugin::load(&dylib) }.unwrap();
    let tools: Vec<Arc<dyn Tool>> = plugin.tools.clone();
    let tool = &tools[0];
    let schema = tool.parameters();
    assert!(schema.get("type").is_some(), "schema parses");
}
