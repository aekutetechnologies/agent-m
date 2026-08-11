//! agent-m: a pi-style coding agent in Rust.
//!
//! Modes: interactive TUI (default on a TTY), `--print` (stream to stdout),
//! `--list-models`. Startup order mirrors pi's `main.ts`: parse args →
//! resolve provider/key/model → build tools → resolve mode → dispatch.

use agent_m_agent::{
    Agent, AgentOptions, AlwaysAllowGate, DangerousCommandGate, DenyAllGate, PermissionGate,
    RiskPolicy, SessionMessage, Tool,
};
use agent_m_ai::{OpenAiCompatibleProvider, Provider, resolve_api_key};
use agent_m_tools::{all_tools, default_tools};

/// ECC recommendation: keep the active tool set under this budget so tool
/// schemas don't eat the context window.
const TOOL_BUDGET: usize = 80;
/// Startup context cap (chars) for the injected AGENTS.md/system prompt.
const STARTUP_CONTEXT_MAX_CHARS: usize = 50_000;
use agent_m_tui::{App, AppInputs, Theme, UiMode};

mod plugins;
use anyhow::{Context, Result};
use clap::Parser;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const DEFAULT_SYSTEM_PROMPT: &str = "You are agent-m, a coding agent running in a terminal. \
You have access to tools (bash, read, write, edit, grep, find, ls) for working with the \
user's codebase. Be concise and precise. Use tools when they help; do not invent file \
contents.";


#[derive(Debug, Parser)]
#[command(name = "agent-m", version, about)]
struct Cli {
    /// Print mode: stream the reply to stdout and exit (default off a TTY).
    #[arg(short = 'p', long = "print")]
    print: bool,

    /// Model id, e.g. `deepseek-chat` or `deepseek-reasoner`.
    #[arg(long)]
    model: Option<String>,

    /// Provider id (default: deepseek).
    #[arg(long)]
    provider: Option<String>,

    /// API key override (otherwise DEEPSEEK_API_KEY env → auth.json → settings.json).
    #[arg(long = "api-key")]
    api_key: Option<String>,

    /// Auto-approve tool calls (no interactive confirmation).
    #[arg(long)]
    yes: bool,

    /// Disable all tools.
    #[arg(long = "no-tools")]
    no_tools: bool,

    /// Restrict tools to this comma-separated allowlist.
    #[arg(long, value_delimiter = ',')]
    tools: Vec<String>,

    /// Exclude these comma-separated tools.
    #[arg(long = "exclude-tools", value_delimiter = ',')]
    exclude_tools: Vec<String>,

    /// Theme: `dark`, `light`, or a path to a theme JSON file.
    #[arg(long)]
    theme: Option<String>,

    /// UI mode: `regular` (scrollback) or `fullscreen` (alternate screen).
    #[arg(long = "ui-mode")]
    ui_mode: Option<String>,

    /// Resume the most recent session for this directory.
    #[arg(long, short = 'r')]
    resume: bool,

    /// Override the agent data directory (default ~/.agent-m/agent).
    #[arg(long = "session-dir")]
    session_dir: Option<PathBuf>,

    /// List available models and exit.
    #[arg(long = "list-models")]
    list_models: bool,

    /// Custom system prompt.
    #[arg(long = "system-prompt")]
    system_prompt: Option<String>,

    /// Start in plan mode (read-only; the model produces a numbered plan).
    #[arg(long = "mode-plan")]
    mode_plan: bool,

    /// Auto-compact threshold as a fraction of the context window (ECC's
    /// strategic-compaction default: 0.5). Compact at turn boundaries once
    /// usage passes this, not mid-implementation.
    #[arg(long = "compact-threshold", default_value_t = 0.5)]
    compact_threshold: f64,

    /// Run a YAML flow (print mode): `agent-m --flow flows/agentic-dev.yml`.
    #[arg(long = "flow")]
    flow: Option<PathBuf>,

    /// Stream machine-readable events as JSON lines (print mode).
    #[arg(long = "stream-json")]
    stream_json: bool,

    /// Serve a stdio JSON-RPC prompt/respond loop (headless embedding).
    #[arg(long = "serve")]
    serve: bool,

    /// Max agent turns per message (default 20). Raise for long tasks.
    #[arg(long = "max-turns", default_value_t = 20)]
    max_turns: usize,

    /// Autonomy level 0-4 (check.md principle 12): 0 observe · 1 suggest ·
    /// 2 assisted (everything asks) · 3 trusted (default; auto low/medium,
    /// ask high/critical) · 4 autonomous (auto everything except critical).
    #[arg(long = "level")]
    level: Option<u8>,

    /// Extra directory the file tools may access (repeatable). bash is NOT contained.
    #[arg(long = "allow-path")]
    allow_path: Vec<PathBuf>,

    /// Manage plugins (install/list/remove/update).
    #[command(subcommand)]
    plugins: Option<PluginsGroup>,

    /// Prompt(s). In print mode, multiple prompts run in sequence.
    #[arg()]
    messages: Vec<String>,
}

/// The `agent-m plugins` group.
#[derive(Debug, clap::Subcommand)]
enum PluginsGroup {
    Plugins {
        #[command(subcommand)]
        command: PluginsCommand,
    },
}

/// `agent-m plugins ...`: manage out-of-tree cdylib plugins.
#[derive(Debug, clap::Subcommand)]
enum PluginsCommand {
    /// Install a plugin from a git URL or local path (builds it with cargo).
    Install {
        source: String,
        #[arg(long)]
        rev: Option<String>,
    },
    /// List installed plugins.
    List,
    /// Remove an installed plugin.
    Remove { name: String },
    /// Reinstall an installed plugin from its manifest `source`.
    Update { name: Option<String> },
}

fn default_agent_dir() -> PathBuf {
    // AGENT_M_DIR overrides the data root (mirrors pi's PI_CODING_AGENT_DIR).
    if let Ok(dir) = std::env::var("AGENT_M_DIR") {
        return PathBuf::from(dir).join("agent");
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".agent-m")
        .join("agent")
}

fn load_settings(agent_dir: &Path) -> serde_json::Value {
    std::fs::read_to_string(agent_dir.join("settings.json"))
        .ok()
        .and_then(|contents| serde_json::from_str(&contents).ok())
        .unwrap_or(serde_json::json!({}))
}

fn resolve_level(cli: &Cli, settings: &serde_json::Value) -> agent_m_agent::AutonomyLevel {
    let number = cli
        .level
        .or_else(|| {
            settings
                .get("level")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as u8)
        })
        .unwrap_or(3);
    agent_m_agent::AutonomyLevel::from_number(number).unwrap_or_default()
}

fn resolve_model(cli: &Cli, settings: &serde_json::Value, provider: &dyn Provider) -> String {
    if let Some(model) = &cli.model {
        return model.clone();
    }
    if let Some(model) = settings
        .get("defaultModel")
        .and_then(serde_json::Value::as_str)
    {
        return model.to_string();
    }
    // Default to the selected provider's primary model (its configured
    // `model`), not a hardcoded fallback — otherwise `--provider openai`
    // would silently talk to deepseek-chat.
    provider
        .models()
        .first()
        .map(|spec| spec.id.clone())
        .unwrap_or_else(|| "deepseek-chat".to_string())
}

fn resolve_theme(cli: &Cli, settings: &serde_json::Value) -> Result<Theme> {
    let choice = cli.theme.clone().or_else(|| {
        settings
            .get("theme")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    });
    let theme = match choice.as_deref() {
        None => Theme::default_for_terminal(),
        Some("dark") => Theme::dark(),
        Some("light") => Theme::light(),
        Some(path) => {
            let contents = std::fs::read_to_string(path)
                .with_context(|| format!("cannot read theme file {path}"))?;
            agent_m_tui::theme::parse_theme(&contents)
                .map_err(|error| anyhow::anyhow!("invalid theme {path}: {error}"))?
        }
    };
    Ok(theme.downgrade_for_terminal())
}

fn resolve_ui_mode(cli: &Cli, settings: &serde_json::Value) -> UiMode {
    let choice = cli.ui_mode.clone().or_else(|| {
        settings
            .get("uiMode")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    });
    match choice.as_deref().and_then(UiMode::parse) {
        Some(mode) => mode,
        None => UiMode::Fullscreen,
    }
}

fn build_tools(cli: &Cli) -> Vec<Arc<dyn Tool>> {
    if cli.no_tools {
        return Vec::new();
    }
    let registry = all_tools();
    let allowed: Vec<String> = if cli.tools.is_empty() {
        // Default active set (pi): read, bash, edit, write.
        default_tools()
            .into_iter()
            .map(|tool| tool.name().to_string())
            .collect()
    } else {
        cli.tools.clone()
    };
    registry
        .into_iter()
        .filter(|tool| allowed.iter().any(|name| name == tool.name()))
        .filter(|tool| !cli.exclude_tools.iter().any(|name| name == tool.name()))
        .collect()
}

/// The gate for modes with no human to ask: print and flow. `--yes` is full
/// trust minus the risk hints; without it, nothing runs.
fn non_interactive_gate(yes: bool, risk: Arc<RiskPolicy>) -> Arc<dyn PermissionGate> {
    if yes {
        Arc::new(DangerousCommandGate::new(risk, AlwaysAllowGate))
    } else {
        Arc::new(DenyAllGate)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let _otel = init_tracing(); // dropped at end of main → flushes batch exporter

    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let agent_dir = cli.session_dir.clone().unwrap_or_else(default_agent_dir);
    if let Some(PluginsGroup::Plugins { command }) = &cli.plugins {
        let result = match command {
            PluginsCommand::Install { source, rev } => {
                plugins::run_install(&agent_dir, source, rev.as_deref())
            }
            PluginsCommand::List => plugins::run_list(&agent_dir),
            PluginsCommand::Remove { name } => plugins::run_remove(&agent_dir, name),
            PluginsCommand::Update { name } => plugins::run_update(&agent_dir, name.as_deref()),
        };
        if let Err(error) = result {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
        std::process::exit(0);
    }
    let settings = load_settings(&agent_dir);

    // Filesystem containment for the file tools: cwd plus user-approved roots.
    let mut allowed = cli.allow_path.clone();
    if let Some(paths) = settings.get("allowedPaths").and_then(|v| v.as_array()) {
        allowed.extend(paths.iter().filter_map(|v| v.as_str()).map(PathBuf::from));
    }
    agent_m_tools::set_allowed_paths(allowed);

    let configured = agent_m_ai::load_provider_configs(&agent_dir);
    let provider_id = cli
        .provider
        .clone()
        .unwrap_or_else(|| "deepseek".to_string());
    let provider: Arc<dyn Provider> =
        if let Some(config) = configured.iter().find(|config| config.id == provider_id) {
            Arc::from(agent_m_ai::provider_from_config(
                config,
                cli.api_key.clone(),
                &agent_dir,
            ))
        } else if provider_id == "deepseek" {
            // Built-in fallback (zero-config behavior unchanged).
            let api_key = cli.api_key.clone().or_else(|| {
                resolve_api_key(
                    &format!("{}_API_KEY", provider_id.to_uppercase()),
                    &provider_id,
                    &agent_dir,
                )
            });
            Arc::new(OpenAiCompatibleProvider::deepseek(api_key))
        } else {
            let available = configured
                .iter()
                .map(|config| config.id.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let available_list = if available.is_empty() {
                "deepseek".to_string()
            } else {
                format!("{available}, deepseek")
            };
            anyhow::bail!(
                "provider `{provider_id}` is not configured. Configure it in \
             settings.json `providers`, or use /provider in the TUI. \
             Available: {available_list}"
            );
        };

    let model = resolve_model(&cli, &settings, provider.as_ref());
    let print_mode = cli.print || !std::io::stdin().is_terminal();
    // Print mode cannot ask for confirmation, so tools are disabled unless the
    // user explicitly opts in with --yes. This prevents a piped prompt from
    // granting the model unapproved bash/file access (security review HIGH-1).
    // ECC-style guardrails: warn when the active tool set exceeds the budget
    // and when the startup context (AGENTS.md + prompt) is oversized.
    if default_tools().len() > TOOL_BUDGET {
        eprintln!(
            "warning: {} tools active (budget {TOOL_BUDGET}); consider pruning tool schemas",
            default_tools().len()
        );
    }
    // One decision: no tools when --no-tools or print mode without --yes.
    let tools_enabled = !cli.no_tools && (!print_mode || cli.yes);
    if print_mode && !cli.yes && !cli.no_tools {
        eprintln!("warning: print mode disables tools; pass --yes to enable them");
    }
    let mut tools = Vec::new();
    let mut opaque_tools = Vec::new();
    if tools_enabled {
        tools = build_tools(&cli);
        // Load plugins and apply the same filters.
        for plugin in agent_m_plugin_loader::load_plugins_dir(&agent_dir.join("plugins")) {
            for tool in plugin.tools {
                if !cli.exclude_tools.iter().any(|n| n == tool.name())
                    && (cli.tools.is_empty() || cli.tools.iter().any(|n| n == tool.name()))
                {
                    // All plugin tools are opaque (trust policy is P1, not P0).
                    opaque_tools.push(tool.name().to_string());
                    tools.push(tool);
                }
            }
        }
        // MCP servers (mcp.json / settings.json, `mcpServers`). Tools are
        // opaque by default → RiskPolicy classifies them Critical, so they
        // always require approval.
        for (name, config) in agent_m_mcp::load_servers(&agent_dir) {
            let connected = if let Some(command) = &config.command {
                let env: Vec<(String, String)> = config
                    .env
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                agent_m_mcp::McpClient::connect_stdio(command, &config.args, &env).await
            } else if let Some(url) = &config.url {
                agent_m_mcp::McpClient::connect_http(url).await
            } else {
                eprintln!("mcp: server `{name}` has neither command nor url");
                continue;
            };
            match connected {
                Ok(client) => match agent_m_mcp::connect_tools(&name, client).await {
                    Ok((mcp_tools, _shared)) => {
                        let count = mcp_tools.len();
                        for tool in mcp_tools {
                            if !cli.exclude_tools.iter().any(|n| n == tool.name())
                                && (cli.tools.is_empty()
                                    || cli.tools.iter().any(|n| n == tool.name()))
                            {
                                opaque_tools.push(tool.name().to_string());
                                tools.push(tool);
                            }
                        }
                        eprintln!("mcp: connected `{name}` ({count} tool(s))");
                    }
                    Err(error) => eprintln!("mcp: `{name}` handshake failed: {error}"),
                },
                Err(error) => eprintln!("mcp: `{name}` connect failed: {error}"),
            }
        }
    }
    let risk = Arc::new(agent_m_agent::RiskPolicy {
        cwd: cwd.clone(),
        opaque_tools,
    });

    if cli.list_models {
        // Configured providers first, then the built-in deepseek fallback.
        let mut seen = std::collections::BTreeSet::new();
        for config in &configured {
            seen.insert(config.id.clone());
            let provider = agent_m_ai::provider_from_config(config, None, &agent_dir);
            for spec in provider.models() {
                println!(
                    "{}\t{}\t{}{}",
                    config.id,
                    spec.id,
                    spec.name.clone().unwrap_or_default(),
                    if spec.reasoning { " (reasoning)" } else { "" }
                );
            }
        }
        if !seen.contains("deepseek") {
            let fallback = OpenAiCompatibleProvider::deepseek(None);
            let fallback_id = "deepseek".to_string();
            for spec in fallback.models() {
                println!(
                    "{}\t{}\t{}{}",
                    fallback_id,
                    spec.id,
                    spec.name.clone().unwrap_or_default(),
                    if spec.reasoning { " (reasoning)" } else { "" }
                );
            }
        }
        return Ok(());
    }

    let base_prompt = cli
        .system_prompt
        .clone()
        .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());
    // Context creation: AGENTS.md instructions (ancestors + global) are
    // appended once per session, keeping the request prefix byte-stable.
    let context_files = agent_m_agent::discover_instructions(&cwd);
    // check.md principle 11: reflect learned preferences back to the model as
    // a static block (rebuilt only when preferences change — byte-stable).
    let preference_block = agent_m_tui::prefs::prompt_block(&agent_m_tui::prefs::load(&agent_dir));
    let system_prompt = format!(
        "{base_prompt}{}{preference_block}",
        agent_m_agent::render_instructions(&context_files)
    );
    if system_prompt.chars().count() > STARTUP_CONTEXT_MAX_CHARS {
        eprintln!(
            "warning: startup context is {} chars (cap {STARTUP_CONTEXT_MAX_CHARS}); consider trimming AGENTS.md files",
            system_prompt.chars().count()
        );
    }
    let agent_options = AgentOptions {
        model: model.clone(),
        system_prompt,
        tools,
        max_turns: cli.max_turns,
        cwd: cwd.clone(),
        mode: if cli.mode_plan {
            agent_m_agent::Mode::Plan
        } else {
            agent_m_agent::Mode::Build
        },
        ask_gate: None,
        context_window: provider
            .models()
            .iter()
            .find(|m| m.id == model)
            .and_then(|m| m.context_window),
        variant: None,
        output_dir: Some(agent_dir.join("tool_outputs")),
    };

    if let Some(flow_path) = &cli.flow {
        let flow_state_dir = agent_dir.join("flows");
        return run_flow_mode(
            provider,
            agent_options,
            flow_path,
            cwd,
            cli.yes,
            risk.clone(),
            flow_state_dir,
        )
        .await;
    }

    if cli.serve {
        let gate = non_interactive_gate(cli.yes, risk.clone());
        return serve_loop(provider, agent_options, gate).await;
    }
    if print_mode {
        let messages = inline_file_args(&cwd, cli.messages.clone());
        let gate = non_interactive_gate(cli.yes, risk.clone());
        return run_print(provider, agent_options, gate, messages, cli.stream_json).await;
    }

    let theme = resolve_theme(&cli, &settings)?;
    let ui_mode = resolve_ui_mode(&cli, &settings);
    let show_cache_notices = settings
        .get("showCacheMissNotices")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    if cli.resume {
        eprintln!("resuming most recent session (agent-m sessions are auto-resumed)");
    }

    App::run(AppInputs {
        provider: provider.clone(),
        agent_options,
        theme,
        ui_mode,
        show_cache_notices,
        models: provider.models().to_vec(),
        context_files,
        approve_tools: cli.yes,
        compact_threshold: cli.compact_threshold,
        level: resolve_level(&cli, &settings),
        agent_dir,
        cwd,
    })
    .await
}

/// Print mode: stream text deltas to stdout and tool activity to stderr.
/// Inline file arguments (`@path` or an existing path) into `<file>` context
/// blocks, capped at 50 KB per file (pi's file-processor pattern). Non-file
/// messages pass through unchanged.
/// Inline `@file` arguments as `<file>` context; `@image.png` tokens are
/// pulled out into base64 image attachments for vision models. Returns
/// (cleaned message, image data URIs).
fn inline_file_args(cwd: &std::path::Path, messages: Vec<String>) -> Vec<(String, Vec<String>)> {
    const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp"];
    const MAX_IMAGE_BYTES: usize = 10 * 1024 * 1024;
    messages
        .into_iter()
        .map(|message| {
            let mut images = Vec::new();
            let mut cleaned: Vec<String> = Vec::new();
            for token in message.split_whitespace() {
                let Some(path) = token.strip_prefix('@') else {
                    cleaned.push(token.to_string());
                    continue;
                };
                let full = cwd.join(path);
                // Containment: canonicalized path must stay under the
                // canonicalized cwd (macOS /tmp → /private/tmp).
                let (Ok(canonical), Ok(cwd_canonical)) = (full.canonicalize(), cwd.canonicalize())
                else {
                    cleaned.push(token.to_string());
                    continue;
                };
                if !canonical.starts_with(&cwd_canonical) {
                    cleaned.push(token.to_string());
                    continue;
                }
                let ext = full
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_lowercase());
                let is_image = ext.as_deref().is_some_and(|e| IMAGE_EXTS.contains(&e));
                if canonical.is_file() && is_image {
                    let Ok(bytes) = std::fs::read(&canonical) else {
                        cleaned.push(token.to_string());
                        continue;
                    };
                    if bytes.len() > MAX_IMAGE_BYTES {
                        // Too big to attach — leave the token so the user sees
                        // it wasn't consumed.
                        cleaned.push(token.to_string());
                        continue;
                    }
                    let mime = match ext.as_deref() {
                        Some("jpg") | Some("jpeg") => "image/jpeg",
                        Some("gif") => "image/gif",
                        Some("webp") => "image/webp",
                        Some("bmp") => "image/bmp",
                        _ => "image/png",
                    };
                    images.push(format!("data:{mime};base64,{}", base64_encode(&bytes)));
                } else if canonical.is_file() {
                    let content = std::fs::read_to_string(&canonical).unwrap_or_default();
                    let capped: String = content.chars().take(50_000).collect();
                    let truncated = if capped.chars().count() < content.chars().count() {
                        "\n… (truncated)"
                    } else {
                        ""
                    };
                    cleaned.push(format!(
                        "<file name=\"{path}\">\n{capped}{truncated}\n</file>"
                    ));
                } else {
                    cleaned.push(token.to_string());
                }
            }
            (cleaned.join(" "), images)
        })
        .collect()
}

/// Minimal base64 encoder for image attachments (3 bytes → 4 chars).
fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(n >> 6) as usize & 63] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[n as usize & 63] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// Run a YAML flow in print mode: sequential steps, status lines to stdout,
/// non-zero exit on failure.
async fn run_flow_mode(
    provider: Arc<dyn Provider>,
    agent_options: AgentOptions,
    flow_path: &std::path::Path,
    cwd: PathBuf,
    yes: bool,
    risk: Arc<RiskPolicy>,
    state_dir: PathBuf,
) -> anyhow::Result<()> {
    let flow = agent_m_flow::load_flow(&flow_path.to_path_buf())?;
    let tools = agent_options.tools.clone();
    let permission = non_interactive_gate(yes, risk);
    let deps = agent_m_flow::FlowDeps {
        provider,
        agent_options,
        tools,
        permission_gate: permission,
        ask_gate: None, // no interactive UI in print mode
        state_dir: Some(state_dir),
        on_progress: None,
    };
    let mut context = agent_m_flow::FlowContext::new();
    context.set("cwd", serde_json::json!(cwd.to_string_lossy()));
    println!("flow: {}", flow.name);
    let run = match agent_m_flow::run_flow(&flow, &mut context, &deps).await {
        Ok(run) => run,
        Err(error) => {
            eprintln!("flow failed: {error}");
            std::process::exit(1);
        }
    };
    let mut failed = false;
    for step in &run.steps {
        let status = step.status.as_str();
        println!("[{status}] {}", step.name);
        if let Some(output) = &step.output {
            if let Some(content) = output.get("content").and_then(serde_json::Value::as_str) {
                println!("    {content}");
            } else if let Some(answer) = output.get("answer").and_then(serde_json::Value::as_str) {
                println!("    answer: {answer}");
            }
        }
        if let Some(error) = &step.error {
            println!("    error: {error}");
        }
        if step.status == agent_m_flow::StepStatus::Failed {
            failed = true;
        }
    }
    println!(
        "flow {}: {}",
        flow.name,
        if failed { "FAILED" } else { "OK" }
    );
    if failed {
        std::process::exit(1);
    }
    Ok(())
}

async fn run_print(
    provider: Arc<dyn Provider>,
    agent_options: AgentOptions,
    gate: Arc<dyn PermissionGate>,
    messages: Vec<(String, Vec<String>)>,
    stream_json: bool,
) -> Result<()> {
    let mut agent = Agent::new(provider, agent_options, gate);
    if stream_json {
        let json_sink = Arc::new(|event: &agent_m_agent::AgentEvent| {
            let json = serde_json::to_string(&event_to_json(event)).unwrap_or_default();
            println!("{json}");
        });
        let sink = json_sink.clone();
        agent.subscribe(move |event| sink(event));
        for (message, images) in messages {
            let result = if images.is_empty() {
                agent.prompt(message).await
            } else {
                agent.prompt_with_images(message, images).await
            };
            result?;
        }
        return Ok(());
    }
    agent.subscribe(|event| match event {
        agent_m_agent::AgentEvent::MessageUpdate {
            delta: agent_m_ai::StreamEvent::TextDelta { delta },
        } => {
            print!("{delta}");
            let _ = std::io::stdout().flush();
        }
        agent_m_agent::AgentEvent::ToolExecutionStart {
            name, arguments, ..
        } => {
            eprintln!(
                "[tool] {name} {}",
                serde_json::to_string(&arguments).unwrap_or_default()
            );
        }
        agent_m_agent::AgentEvent::Notice { message } => {
            eprintln!("[notice] {message}");
        }
        _ => {}
    });

    if messages.is_empty() {
        // Read the prompt from stdin (piped usage).
        use std::io::Read;
        let mut input = String::new();
        std::io::stdin().read_to_string(&mut input)?;
        if !input.trim().is_empty() {
            agent.prompt(input.trim().to_string()).await?;
        }
    } else {
        for (message, images) in messages {
            let result = if images.is_empty() {
                agent.prompt(message).await
            } else {
                agent.prompt_with_images(message, images).await
            };
            result?;
        }
    }

    // Echo the final assistant text on its own line.
    for message in agent.messages() {
        if let SessionMessage::Assistant { content, .. } = message {
            let text: String = content
                .iter()
                .filter_map(|part| match part {
                    agent_m_ai::ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            if !text.is_empty() {
                println!();
            }
        }
    }
    Ok(())
}

/// Initialise the tracing subscriber. When `OTEL_EXPORTER_OTLP_ENDPOINT` is
/// set, a `tracing-opentelemetry` layer is added that exports spans via OTLP/HTTP
/// (proto) using the already-present reqwest client. The returned guard flushes
/// the batch exporter on drop, covering every exit path from `main`.
fn init_tracing() -> OtelGuard {
    use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("agent_m=info,warn"));
    let fmt = tracing_subscriber::fmt::layer().with_target(false);

    if std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").is_ok()
        && let Ok(provider) = build_otlp_provider()
    {
        let tracer = opentelemetry::trace::TracerProvider::tracer(&provider, "agent-m");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        let _ = tracing_subscriber::registry()
            .with(filter)
            .with(fmt)
            .with(otel_layer)
            .try_init();
        return OtelGuard(Some(provider));
    }
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt)
        .try_init();
    OtelGuard(None)
}

/// Flushes the OTLP batch exporter on drop, covering every exit path.
struct OtelGuard(Option<opentelemetry_sdk::trace::TracerProvider>);
impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.0.take() {
            let _ = opentelemetry_sdk::trace::TracerProvider::shutdown(&provider);
        }
    }
}

fn build_otlp_provider()
-> Result<opentelemetry_sdk::trace::TracerProvider, Box<dyn std::error::Error + Send + Sync>> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .build()?;
    let resource = opentelemetry_sdk::Resource::new(vec![opentelemetry::KeyValue::new(
        "service.name",
        "agent-m",
    )]);
    let provider = opentelemetry_sdk::trace::TracerProvider::builder()
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .with_resource(resource)
        .build();
    Ok(provider)
}

/// Map an agent event to a stable JSON shape for `--stream-json` and
/// `--serve` (machine-readable, pi's event ordering preserved).
fn event_to_json(event: &agent_m_agent::AgentEvent) -> serde_json::Value {
    use agent_m_agent::AgentEvent as E;
    let value = match event {
        E::AgentStart => serde_json::json!({}),
        E::TurnStart { turn } => serde_json::json!({ "turn": turn }),
        E::MessageStart { kind } => serde_json::json!({ "kind": format!("{kind:?}") }),
        E::MessageUpdate {
            delta: agent_m_ai::StreamEvent::TextDelta { delta },
        } => serde_json::json!({ "delta": delta }),
        E::MessageUpdate { .. } => serde_json::json!({ "delta": "" }),
        E::MessageEnd { message } => serde_json::json!({ "message": format!("{message:?}") }),
        E::ToolExecutionStart {
            name, arguments, ..
        } => serde_json::json!({ "name": name, "arguments": arguments }),
        E::ToolExecutionEnd { outcome, .. } => serde_json::json!({
            "is_error": outcome.is_error,
            "content": outcome.content,
        }),
        E::Notice { message } => serde_json::json!({ "message": message }),
        E::Compacted { summary, .. } => serde_json::json!({ "summary": summary }),
        E::FlowStep {
            index,
            name,
            status,
        } => {
            serde_json::json!({ "index": index, "name": name, "status": format!("{status:?}") })
        }
        E::TurnEnd { .. } => serde_json::json!({}),
        E::AgentEnd { .. } => serde_json::json!({}),
    };
    serde_json::json!({ "event": event_name(event), "data": value })
}

fn event_name(event: &agent_m_agent::AgentEvent) -> &'static str {
    use agent_m_agent::AgentEvent as E;
    match event {
        E::AgentStart => "agent_start",
        E::TurnStart { .. } => "turn_start",
        E::MessageStart { .. } => "message_start",
        E::MessageUpdate { .. } => "message_update",
        E::MessageEnd { .. } => "message_end",
        E::ToolExecutionStart { .. } => "tool_execution_start",
        E::ToolExecutionEnd { .. } => "tool_execution_end",
        E::Notice { .. } => "notice",
        E::Compacted { .. } => "compacted",
        E::FlowStep { .. } => "flow_step",
        E::TurnEnd { .. } => "turn_end",
        E::AgentEnd { .. } => "agent_end",
    }
}

/// Headless stdio JSON-RPC (pi-style): read framed requests, run prompts,
/// stream `event` notifications, respond with the final result.
async fn serve_loop(
    provider: Arc<dyn Provider>,
    agent_options: AgentOptions,
    gate: Arc<dyn PermissionGate>,
) -> Result<()> {
    let mut agent = Agent::new(provider, agent_options, gate);
    let events = Arc::new(|event: &agent_m_agent::AgentEvent| {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "event",
            "params": event_to_json(event),
        });
        let framed = agent_m_mcp::jsonrpc::frame(&notification);
        let mut stdout = std::io::stdout();
        let _ = std::io::Write::write_all(&mut stdout, &framed);
        let _ = std::io::Write::flush(&mut stdout);
    });
    let sink = events.clone();
    agent.subscribe(move |event| sink(event));

    use std::io::{Read, Write};
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let read = reader.read(&mut chunk).map_err(anyhow::Error::msg)?;
        if read == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..read]);
        while let Some((message, consumed)) = agent_m_mcp::jsonrpc::unframe(&buf) {
            buf.drain(..consumed);
            let method = message
                .get("method")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let id = message
                .get("id")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            // JSON-RPC notifications (no id) get no response — process and
            // continue.
            if id.is_null() {
                if method == "exit" {
                    return Ok(());
                }
                continue;
            }
            let response = match method {
                "prompt" => {
                    let text = message
                        .get("params")
                        .and_then(|params| params.get("text"))
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let result = agent.prompt(text).await;
                    serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": {
                        "ok": result.is_ok(),
                        "error": result.err().map(|e| e.to_string()),
                    }})
                }
                "exit" => {
                    serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": { "ok": true } })
                }
                _ => serde_json::json!({ "jsonrpc": "2.0", "id": id, "error": {
                    "message": format!("unknown method `{method}`"),
                }}),
            };
            let mut stdout = std::io::stdout();
            let framed = agent_m_mcp::jsonrpc::frame(&response);
            stdout.write_all(&framed).map_err(anyhow::Error::msg)?;
            stdout.flush().map_err(anyhow::Error::msg)?;
            if method == "exit" {
                return Ok(());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_m_agent::AgentEvent;

    #[test]
    fn event_to_json_shapes_are_stable() {
        let json = event_to_json(&AgentEvent::AgentStart);
        assert_eq!(json["event"], "agent_start");
        let json = event_to_json(&AgentEvent::ToolExecutionStart {
            tool_call_id: "1".into(),
            name: "read".into(),
            arguments: serde_json::json!({ "path": "a.rs" }),
        });
        assert_eq!(json["event"], "tool_execution_start");
        assert_eq!(json["data"]["name"], "read");
        assert_eq!(json["data"]["arguments"]["path"], "a.rs");
        let json = event_to_json(&AgentEvent::MessageUpdate {
            delta: agent_m_ai::StreamEvent::TextDelta { delta: "hi".into() },
        });
        assert_eq!(json["event"], "message_update");
        assert_eq!(json["data"]["delta"], "hi");
    }
}

#[cfg(test)]
mod inline_file_args_tests {
    use super::inline_file_args;
    use tempfile::tempdir;

    #[test]
    fn image_attachment_keeps_the_rest_of_the_prompt() {
        let dir = tempdir().unwrap();
        // A 1x1 PNG (tiny, valid base64-able bytes).
        std::fs::write(dir.path().join("shot.png"), vec![0x89u8, 0x50, 0x4e, 0x47]).unwrap();
        let (message, images) =
            inline_file_args(dir.path(), vec!["@shot.png explain this".to_string()])
                .pop()
                .unwrap();
        assert!(images.len() == 1, "image attached: {images:?}");
        assert!(images[0].starts_with("data:image/png;base64,"));
        assert_eq!(message, "explain this", "prompt text preserved: {message}");
    }

    #[test]
    fn outside_cwd_paths_are_not_inlined() {
        let dir = tempdir().unwrap();
        let outside = tempdir().unwrap();
        std::fs::write(outside.path().join("secret.png"), vec![1, 2, 3]).unwrap();
        let (message, images) = inline_file_args(
            dir.path(),
            vec![format!(
                "@{} explain",
                outside.path().join("secret.png").display()
            )],
        )
        .pop()
        .unwrap();
        assert!(images.is_empty(), "nothing attached");
        assert!(message.contains("explain"));
        assert!(message.contains("secret.png"), "token left visible");
    }
}
