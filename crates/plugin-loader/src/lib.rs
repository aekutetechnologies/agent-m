//! Loads out-of-tree cdylib plugins and exposes their tools as ordinary agent
//! `Tool`s. Each plugin exports `agent_m_plugin_entry()` returning a
//! `PluginEntry` (see agent-m-plugin-sdk).

use agent_m_agent::{Tool, ToolContext, ToolError, ToolOutcome};
use agent_m_plugin_sdk::{ENTRY_SYMBOL, PluginEntry, PluginTool, cstr_to_string};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One loaded plugin: the dlopen handle (kept alive) + its tools.
pub struct LoadedPlugin {
    pub name: String,
    pub version: String,
    /// Keep the library alive for the tools' lifetimes.
    #[allow(dead_code)]
    library: libloading::Library,
    pub tools: Vec<Arc<dyn Tool>>,
}

impl LoadedPlugin {
    /// Load a plugin dylib from `path`.
    ///
    /// # Safety
    /// Plugins are arbitrary native code executed in-process. Only load
    /// plugins you trust (same trust model as pi's TS extensions).
    pub unsafe fn load(path: &Path) -> Result<LoadedPlugin> {
        // SAFETY: the caller guarantees the plugin is trusted.
        let library = unsafe { libloading::Library::new(path) }
            .with_context(|| format!("cannot load plugin {}", path.display()))?;
        // SAFETY: ENTRY_SYMBOL is a static string.
        let entry: libloading::Symbol<unsafe extern "C" fn() -> *const PluginEntry> =
            unsafe { library.get(ENTRY_SYMBOL.as_bytes()) }
                .with_context(|| format!("plugin {} lacks `{ENTRY_SYMBOL}`", path.display()))?;
        // SAFETY: the entry symbol is the plugin's contract.
        let entry = unsafe { entry() };
        if entry.is_null() {
            return Err(anyhow!("plugin {} returned a null entry", path.display()));
        }
        let entry = unsafe { &*entry };
        let name = unsafe { cstr_to_string(entry.name) };
        let version = unsafe { cstr_to_string(entry.version) };
        let mut tools: Vec<Arc<dyn Tool>> = Vec::new();
        for index in 0..entry.tool_count {
            // SAFETY: the plugin owns a table of `tool_count` tools.
            let tool = unsafe { &*entry.tools.add(index) };
            tools.push(Arc::new(PluginToolWrapper::new(tool)));
        }
        Ok(LoadedPlugin {
            name,
            version,
            library,
            tools,
        })
    }
}

/// The host-side wrapper making a plugin tool an agent `Tool`.
///
/// The raw pointers inside reference plugin memory that stays valid while the
/// `LoadedPlugin` (and thus the dlopen handle) is alive. The wrapper is Send +
/// Sync because plugin execution is synchronous and single-threaded per call.
struct PluginToolWrapper {
    tool: *const PluginTool,
}

// SAFETY: plugin tools are stateless across calls (context is plugin-owned,
// functions are 'static); the wrapper is only invoked while the owning
// LoadedPlugin is alive, so the pointer stays valid.
unsafe impl Send for PluginToolWrapper {}
unsafe impl Sync for PluginToolWrapper {}

impl PluginToolWrapper {
    fn new(tool: *const PluginTool) -> Self {
        Self { tool }
    }

    fn name(&self) -> String {
        // SAFETY: pointer valid for the plugin's lifetime.
        unsafe { cstr_to_string(((*self.tool).name_fn)((*self.tool).context)) }
    }

    fn description(&self) -> String {
        unsafe { cstr_to_string(((*self.tool).description_fn)((*self.tool).context)) }
    }

    fn parameters(&self) -> Value {
        let text = unsafe { cstr_to_string(((*self.tool).parameters_fn)((*self.tool).context)) };
        serde_json::from_str(&text).unwrap_or_else(|_| json!({ "type": "object" }))
    }

    fn execute_sync(&self, arguments: &Value, cwd: &str) -> Result<ToolOutcome, ToolError> {
        let arguments = serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string());
        let cwd = CString::new(cwd).unwrap_or_default();
        let arguments = CString::new(arguments).unwrap_or_default();
        // SAFETY: pointers are valid for the plugin's lifetime; arguments/cwd
        // are NUL-terminated.
        let result = unsafe {
            ((*self.tool).execute_fn)((*self.tool).context, arguments.as_ptr(), cwd.as_ptr())
        };
        let result_text = unsafe { cstr_to_string(result) };
        // SAFETY: the result string was allocated by the plugin's result path
        // and is freed with its result_free.
        unsafe {
            ((*self.tool).result_free)(result);
        }
        let parsed: Value = serde_json::from_str(&result_text)
            .unwrap_or_else(|_| json!({ "content": result_text, "isError": true }));
        Ok(ToolOutcome {
            content: parsed
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            is_error: parsed
                .get("isError")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }
}

#[async_trait]
impl Tool for PluginToolWrapper {
    fn name(&self) -> &str {
        // Names are static plugin strings; leak-free by caching on first use
        // is overkill — plugin names are short-lived constants, so we leak a
        // copy per wrapper (bounded by the number of plugin tools).
        Box::leak(self.name().into_boxed_str())
    }

    fn description(&self) -> String {
        self.description()
    }

    fn parameters(&self) -> Value {
        self.parameters()
    }

    async fn execute(
        &self,
        arguments: Value,
        context: &ToolContext,
    ) -> Result<ToolOutcome, ToolError> {
        let cwd = context.cwd.to_string_lossy().to_string();
        // Blocking plugin execution on the async runtime; plugin calls are
        // short (fixture/test-runner) or network (jira/github, MVP).
        let wrapper = PluginToolWrapper::new(self.tool);
        tokio::task::spawn_blocking(move || wrapper.execute_sync(&arguments, &cwd))
            .await
            .map_err(|error| ToolError::failed("plugin", error.to_string()))?
    }
}

/// Scan a plugins directory: every subdirectory with a `plugin.json` manifest
/// is loaded; broken plugins are skipped with a warning.
pub fn load_plugins_dir(dir: &Path) -> Vec<LoadedPlugin> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut loaded = Vec::new();
    for entry in entries.flatten() {
        let plugin_dir = entry.path();
        if !plugin_dir.is_dir() {
            continue;
        }
        let manifest_path = plugin_dir.join("plugin.json");
        if !manifest_path.is_file() {
            continue;
        }
        // SAFETY: loading plugins is the user's explicit choice; the manifest
        // names the entry artifact, which is a compiled library.
        match unsafe { load_manifest(&plugin_dir) } {
            Ok(plugin) => {
                eprintln!(
                    "plugin loaded: {} v{} ({} tool(s))",
                    plugin.name,
                    plugin.version,
                    plugin.tools.len()
                );
                loaded.push(plugin);
            }
            Err(error) => {
                eprintln!(
                    "warning: skipping broken plugin `{}`: {error}",
                    plugin_dir.display()
                );
            }
        }
    }
    loaded
}

/// Load one plugin directory using its `plugin.json` manifest.
///
/// # Safety
/// See [`LoadedPlugin::load`].
unsafe fn load_manifest(plugin_dir: &Path) -> Result<LoadedPlugin> {
    let manifest_path = plugin_dir.join("plugin.json");
    let text = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("cannot read {}", manifest_path.display()))?;
    let manifest: Value = serde_json::from_str(&text)
        .with_context(|| format!("invalid manifest {}", manifest_path.display()))?;
    let entry_name = manifest
        .get("entry")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("manifest lacks `entry`"))?;
    let entry_path: PathBuf = plugin_dir.join(entry_name);
    if !entry_path.is_file() {
        return Err(anyhow!(
            "entry artifact `{}` missing (run `agent-m plugins install` again)",
            entry_path.display()
        ));
    }
    // SAFETY: caller trusts the plugin.
    unsafe { LoadedPlugin::load(&entry_path) }
}
