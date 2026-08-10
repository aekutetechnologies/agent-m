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

const MAX_TURNS: usize = 20;

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

fn resolve_model(cli: &Cli, settings: &serde_json::Value) -> String {
    if let Some(model) = &cli.model {
        return model.clone();
    }
    if let Some(model) = settings
        .get("defaultModel")
        .and_then(serde_json::Value::as_str)
    {
        return model.to_string();
    }
    "deepseek-chat".to_string()
}

fn resolve_theme(cli: &Cli, settings: &serde_json::Value) -> Result<Theme> {
    let choice = cli.theme.clone().or_else(|| {
        settings
            .get("theme")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    });
    match choice.as_deref() {
        None => Ok(Theme::default_for_terminal()),
        Some("dark") => Ok(Theme::dark()),
        Some("light") => Ok(Theme::light()),
        Some(path) => {
            let contents = std::fs::read_to_string(path)
                .with_context(|| format!("cannot read theme file {path}"))?;
            agent_m_tui::theme::parse_theme(&contents)
                .map_err(|error| anyhow::anyhow!("invalid theme {path}: {error}"))
        }
    }
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
    init_tracing();

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

    let provider_id = cli
        .provider
        .clone()
        .unwrap_or_else(|| "deepseek".to_string());
    let api_key = cli.api_key.clone().or_else(|| {
        resolve_api_key(
            &format!("{}_API_KEY", provider_id.to_uppercase()),
            &provider_id,
            &agent_dir,
        )
    });

    if provider_id != "deepseek" {
        anyhow::bail!("provider `{provider_id}` is not implemented; supported providers: deepseek");
    }
    let provider: Arc<dyn Provider> = Arc::new(OpenAiCompatibleProvider::deepseek(api_key));

    let model = resolve_model(&cli, &settings);
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
    }
    let risk = Arc::new(agent_m_agent::RiskPolicy {
        cwd: cwd.clone(),
        opaque_tools,
    });

    if cli.list_models {
        for spec in provider.models() {
            println!(
                "{}\t{}{}",
                spec.id,
                spec.name.clone().unwrap_or_default(),
                if spec.reasoning { " (reasoning)" } else { "" }
            );
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
    let system_prompt = format!(
        "{base_prompt}{}",
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
        max_turns: MAX_TURNS,
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

    if print_mode {
        let messages = inline_file_args(&cwd, cli.messages.clone());
        let gate = non_interactive_gate(cli.yes, risk.clone());
        return run_print(provider, agent_options, gate, messages).await;
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
        agent_dir,
        cwd,
    })
    .await
}

/// Print mode: stream text deltas to stdout and tool activity to stderr.
/// Inline file arguments (`@path` or an existing path) into `<file>` context
/// blocks, capped at 50 KB per file (pi's file-processor pattern). Non-file
/// messages pass through unchanged.
fn inline_file_args(cwd: &std::path::Path, messages: Vec<String>) -> Vec<String> {
    messages
        .into_iter()
        .map(|message| {
            let trimmed = message.trim();
            let path = trimmed.strip_prefix('@').unwrap_or(trimmed);
            let full = cwd.join(path);
            if full.is_file() {
                let content = std::fs::read_to_string(&full).unwrap_or_default();
                let capped: String = content.chars().take(50_000).collect();
                let truncated = if capped.chars().count() < content.chars().count() {
                    "\n… (truncated)"
                } else {
                    ""
                };
                format!("<file name=\"{path}\">\n{capped}{truncated}\n</file>")
            } else {
                message
            }
        })
        .collect()
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
    messages: Vec<String>,
) -> Result<()> {
    let mut agent = Agent::new(provider, agent_options, gate);
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
        for message in messages {
            agent.prompt(message).await?;
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

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}
