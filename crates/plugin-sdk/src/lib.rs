//! agent-m-plugin-sdk: the C-ABI contract out-of-tree plugins implement.
//!
//! A plugin is a cdylib exporting `agent_m_plugin_entry()`. The host loads it
//! with `libloading` and wraps each contributed tool as an ordinary agent
//! `Tool` (permission gate, plan-mode filtering, byte-stable schemas all
//! apply). Plugins live in separate repos, built at install time
//! (`agent-m plugins install <git-url|path>`).

use std::ffi::{CStr, CString, c_char, c_void};

/// A tool contributed by a plugin. Function pointers operate on `context`
/// (plugin-owned state).
#[repr(C)]
pub struct PluginTool {
    /// Plugin-owned state passed to every call.
    pub context: *const c_void,
    /// Returns the tool name (C string, static or plugin-owned).
    pub name_fn: unsafe extern "C" fn(*const c_void) -> *const c_char,
    /// Returns the tool description (C string).
    pub description_fn: unsafe extern "C" fn(*const c_void) -> *const c_char,
    /// Returns the JSON schema for the tool's arguments (C string).
    pub parameters_fn: unsafe extern "C" fn(*const c_void) -> *const c_char,
    /// Executes the tool. Arguments and cwd are JSON/UTF-8 C strings; returns
    /// a JSON result `{"content": "...", "isError": bool}` as a C string the
    /// host frees with `result_free`.
    pub execute_fn: unsafe extern "C" fn(
        *const c_void,
        arguments: *const c_char,
        cwd: *const c_char,
    ) -> *mut c_char,
    /// Frees a result string returned by `execute_fn`.
    pub result_free: unsafe extern "C" fn(*mut c_char),
}

/// The plugin entry: name/version + a contiguous tool table. Returned by
/// `agent_m_plugin_entry()`.
#[repr(C)]
pub struct PluginEntry {
    pub name: *const c_char,
    pub version: *const c_char,
    pub tool_count: usize,
    /// Pointer to `tool_count` consecutive `PluginTool`s (owned by the plugin,
    /// valid for the plugin's lifetime).
    pub tools: *const PluginTool,
}

/// The symbol every plugin must export.
pub const ENTRY_SYMBOL: &str = "agent_m_plugin_entry";

/// Read a C string into a Rust String (empty on null).
///
/// # Safety
/// `ptr` must be a valid NUL-terminated string or null.
pub unsafe fn cstr_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: caller guarantees a valid NUL-terminated string.
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// Convert a Rust string to a heap C string (caller frees with
/// `CString::from_raw`).
pub fn string_to_cstring(text: &str) -> CString {
    CString::new(text).unwrap_or_default()
}

/// A tiny helper for plugin authors: build a static entry from plain
/// functions, so the SDK stays dependency-free (no serde, no async).
///
/// All strings handed to the host are NUL-terminated CStrings (never the
/// string-literal rodata, which the linker may merge without NUL separators).
pub mod tools {
    use super::*;

    /// A plugin tool implemented as plain functions.
    pub struct ToolDef {
        pub name: &'static str,
        pub description: &'static str,
        pub parameters: &'static str,
        /// `(arguments_json, cwd) -> Ok(content)` or `Err(error)`.
        pub execute: fn(&str, &str) -> Result<String, String>,
    }

    /// Per-tool NUL-terminated strings + a back-reference to the def.
    struct ToolCtx {
        def: &'static ToolDef,
        name: CString,
        description: CString,
        parameters: CString,
    }

    extern "C" fn name_fn(ctx: *const c_void) -> *const c_char {
        // SAFETY: ctx points to a leaked ToolCtx.
        let ctx = unsafe { &*(ctx as *const ToolCtx) };
        ctx.name.as_ptr()
    }
    extern "C" fn description_fn(ctx: *const c_void) -> *const c_char {
        let ctx = unsafe { &*(ctx as *const ToolCtx) };
        ctx.description.as_ptr()
    }
    extern "C" fn parameters_fn(ctx: *const c_void) -> *const c_char {
        let ctx = unsafe { &*(ctx as *const ToolCtx) };
        ctx.parameters.as_ptr()
    }
    extern "C" fn execute_fn(
        ctx: *const c_void,
        arguments: *const c_char,
        cwd: *const c_char,
    ) -> *mut c_char {
        let ctx = unsafe { &*(ctx as *const ToolCtx) };
        let args = unsafe { cstr_to_string(arguments) };
        let cwd = unsafe { cstr_to_string(cwd) };
        let result = match (ctx.def.execute)(&args, &cwd) {
            Ok(content) => format!("{{\"content\":{},\"isError\":false}}", quote(&content)),
            Err(error) => format!("{{\"content\":{},\"isError\":true}}", quote(&error)),
        };
        // Leak the result string; the host frees it via result_free.
        let c = string_to_cstring(&result);
        c.into_raw() as *mut c_char
    }
    extern "C" fn result_free(ptr: *mut c_char) {
        if !ptr.is_null() {
            // SAFETY: the pointer was produced by string_to_cstring above.
            unsafe {
                let _ = CString::from_raw(ptr);
            }
        }
    }

    /// Wrap a `ToolDef` table into a `PluginEntry`. Everything the host can
    /// point at (tool table, per-tool contexts) is leaked so the pointers stay
    /// valid for the plugin's lifetime.
    pub fn entry(
        name: &'static str,
        version: &'static str,
        defs: &'static [ToolDef],
    ) -> PluginEntry {
        let name_c = Box::leak(Box::new(CString::new(name).unwrap_or_default()));
        let version_c = Box::leak(Box::new(CString::new(version).unwrap_or_default()));
        let tools: Vec<PluginTool> = defs
            .iter()
            .map(|def| {
                let ctx = Box::leak(Box::new(ToolCtx {
                    def,
                    name: CString::new(def.name).unwrap_or_default(),
                    description: CString::new(def.description).unwrap_or_default(),
                    parameters: CString::new(def.parameters).unwrap_or_default(),
                }));
                PluginTool {
                    context: ctx as *const ToolCtx as *const c_void,
                    name_fn,
                    description_fn,
                    parameters_fn,
                    execute_fn,
                    result_free,
                }
            })
            .collect();
        let tools: &'static [PluginTool] = Box::leak(tools.into_boxed_slice());
        PluginEntry {
            name: name_c.as_ptr(),
            version: version_c.as_ptr(),
            tool_count: tools.len(),
            tools: tools.as_ptr(),
        }
    }

    fn quote(text: &str) -> String {
        let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
        format!("\"{escaped}\"")
    }
}
