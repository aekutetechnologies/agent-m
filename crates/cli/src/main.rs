//! agent-m: a pi-style coding agent in Rust.
//!
//! Modes: interactive TUI (default on a TTY), `--print` (stream to stdout),
//! `--list-models`. Startup order mirrors pi's `main.ts`: parse args →
//! resolve provider/key/model → build tools → resolve mode → dispatch.

use agent_m_agent::{
    Agent, AgentOptions, AlwaysAllowGate, DangerousCommandGate, DenyAllGate, PermissionGate,
    RiskPolicy, SessionMessage, Tool,
};
use agent_m_ai::Provider;
use agent_m_tools::{all_tools, default_tools};

/// ECC recommendation: keep the active tool set under this budget so tool
/// schemas don't eat the context window.
const TOOL_BUDGET: usize = 80;
/// Startup context cap (chars) for the injected AGENTS.md/system prompt.
const STARTUP_CONTEXT_MAX_CHARS: usize = 50_000;
#[allow(dead_code)]
mod harness;
#[allow(dead_code)]
mod plan;
#[allow(dead_code)]
mod prefs;
#[allow(dead_code)]
mod refine;
#[allow(dead_code)]
mod sessions;
mod plugins;
mod repl;
mod commands;
mod gate;
mod toolout;
mod ansi;
mod section;
mod daemon;
mod attach;
mod ask;
mod human;
mod slack;
mod pickup;
mod ticket_log;
use anyhow::{Context, Result};
use clap::Parser;
use std::future::Future;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
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

    /// Reasoning effort variant (e.g. `default`, `low`, `high`, `max`).
    #[arg(long)]
    variant: Option<String>,

    /// Run as a background daemon session with the given session ID.
    #[arg(long)]
    daemon: Option<String>,

    /// Attach to an active daemon session.
    #[arg(long)]
    attach: Option<String>,

    /// List active background daemon sessions.
    #[arg(long = "list-daemons")]
    list_daemons: bool,

    /// Provider id (default: deepseek).
    #[arg(long)]
    provider: Option<String>,

    /// Auto-approve Low/Medium tool calls; High/Critical always ask in the
    /// REPL. In print mode this also enables tools (no human to ask).
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

    /// Seed flow-context values as KEY=VALUE (repeatable), e.g.
    /// `--flow-context ticket=PROJ-42 --flow-context repo=acme/app`.
    /// Referenced from flow steps as `${ticket}`, `${repo}`, …
    #[arg(long = "flow-context", value_name = "KEY=VALUE")]
    flow_context: Vec<String>,

    /// Stream machine-readable events as JSON lines (print mode).
    #[arg(long = "stream-json")]
    stream_json: bool,

    /// Serve a stdio JSON-RPC prompt/respond loop (headless embedding).
    #[arg(long = "serve")]
    serve: bool,

    /// Run this session in a fresh git worktree (`--worktree NAME`), so
    /// parallel sessions can work the same repo without stepping on each
    /// other (Xirp-style). Creates `agent-m/<name>-<ts>` + a checkout under
    /// `~/.agent-m/worktrees/` and starts the agent there.
    #[arg(long = "worktree", num_args = 0..=1, default_missing_value = "")]
    worktree: Option<String>,

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

    /// Route asks and High/Critical approvals over Slack (remote human
    /// channel). Requires SLACK_APP_TOKEN + SLACK_BOT_TOKEN. Also posts flow
    /// step progress.
    #[arg(long = "slack-channel")]
    slack_channel: Option<String>,

    /// Enforce the trust protocol (check.md P2/P4/P9): turns that use tools
    /// must carry a `<trust>` block with a confidence score and real
    /// evidence. warn = notice gaps only (default), ask = ask before running
    /// gapped turns, block = deny without asking, off = display-only.
    #[arg(long = "trust", default_value = "warn", value_parser = ["off", "warn", "ask", "block"])]
    trust: String,

    /// Top-level subcommands (plugins, slack).
    #[command(subcommand)]
    command: Option<Commands>,

    /// Prompt(s). In print mode, multiple prompts run in sequence.
    #[arg()]
    messages: Vec<String>,
}

/// Top-level subcommands.
#[derive(Debug, clap::Subcommand)]
enum Commands {
    /// Manage plugins (install/list/remove/update).
    Plugins {
        #[command(subcommand)]
        command: PluginsCommand,
    },
    /// Connect to Slack (Socket Mode) as the remote human channel.
    Slack {
        /// Channel used for questions/notifications (default: the DM channel).
        #[arg(long)]
        channel: Option<String>,
    },
    /// Pick up the next assigned open Jira ticket and run the flow against it
    /// (auto-pickup for the autonomous SDLC loop).
    Pickup {
        /// Ticket key override (skips the Jira query), e.g. PROJ-42.
        #[arg(long)]
        ticket: Option<String>,
        /// Repo override (git URL or local path); defaults to the repos.json
        /// mapping for the ticket's project key.
        #[arg(long)]
        repo: Option<String>,
        /// JQL override for the ticket query.
        #[arg(long)]
        jql: Option<String>,
        /// Jira transition id for "In Progress" (default 11).
        #[arg(long = "transition-id", default_value = "11")]
        transition_id: String,
        /// Print what would be picked and run, without doing it.
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// Flow to run against the picked ticket.
        #[arg(long = "flow", default_value = "flows/agentic-dev.yml")]
        flow: PathBuf,
        /// Poll for new tickets every N seconds (default 300; run forever
        /// until Ctrl-C). `--poll` without a value uses the default.
        #[arg(long, value_name = "SECONDS", num_args = 0..=1, default_missing_value = "300")]
        poll: Option<u64>,
        /// Run up to N ticket flows in parallel (only meaningful with --poll).
        #[arg(long, default_value = "1")]
        workers: usize,
    },
    /// Run one ticket's full pipeline in its own process (the per-ticket
    /// daemon). Spawned by `pickup --poll --workers N` for each ticket.
    TicketRun {
        /// Ticket key, e.g. PROJ-42 (skips the Jira query).
        #[arg(long)]
        ticket: String,
        /// Repo override (git URL or local path); defaults to the repos.json
        /// mapping for the ticket's project key.
        #[arg(long)]
        repo: Option<String>,
        /// Jira transition id for "In Progress" (default 11).
        #[arg(long = "transition-id", default_value = "11")]
        transition_id: String,
        /// Flow to run against the ticket.
        #[arg(long = "flow", default_value = "flows/agentic-dev.yml")]
        flow: PathBuf,
        /// Agent directory (repos.json, worktrees, reports, sessions).
        #[arg(long = "agent-dir")]
        agent_dir: Option<PathBuf>,
        /// Extra flow context as KEY=VALUE (repeatable).
        #[arg(long = "flow-context")]
        flow_context: Vec<String>,
    },
    /// Tail a per-ticket daemon's report (`<agent_dir>/tickets/<KEY>.jsonl`).
    TicketLog {
        /// Ticket key, e.g. PROJ-42.
        ticket: String,
        /// Keep following the report as new lines are appended.
        #[arg(long)]
        follow: bool,
        /// Agent directory (default: the standard agent dir).
        #[arg(long = "agent-dir")]
        agent_dir: Option<PathBuf>,
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

fn resolve_model(
    cli: &Cli,
    settings_config: &agent_m_ai::SettingsConfig,
    provider: &dyn Provider,
    task_model: Option<String>,
) -> String {
    if let Some(model) = &cli.model {
        return model.clone();
    }
    if let Some(model) = task_model {
        return model;
    }
    if let Some(model) = &settings_config.default_model {
        return model.clone();
    }
    provider
        .models()
        .first()
        .map(|spec| spec.id.clone())
        .unwrap_or_else(|| "default".to_string())
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

/// Which human answers High/Critical approval questions.
#[derive(Clone)]
enum HumanAsk {
    /// The remote Slack channel (--slack-channel).
    Slack {
        remote: Arc<crate::slack::RemoteHuman>,
        channel: String,
    },
    /// The terminal gate (ask_human): readline; denies without a TTY.
    Terminal,
}

/// Route one permission question to the configured human.
fn ask_human_permission(
    human: &HumanAsk,
    policy: Arc<RiskPolicy>,
    call: agent_m_agent::ToolCallInfo,
) -> Pin<Box<dyn Future<Output = agent_m_agent::Permission> + Send>> {
    match human {
        HumanAsk::Slack { remote, channel } => crate::slack::ask_slack_permission(
            remote.clone(),
            channel.clone(),
            (*policy).clone(),
            call,
        ),
        HumanAsk::Terminal => crate::gate::ask_human(policy, call),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    crate::ansi::init_color();
    let _otel = init_tracing(); // dropped at end of main → flushes batch exporter

    let cwd = std::env::current_dir().context("cannot determine current directory")?;
    let agent_dir = cli.session_dir.clone().unwrap_or_else(default_agent_dir);
    // `--worktree`: isolate this session in a fresh git worktree and start
    // the agent there (parallel sessions on one repo, Xirp-style).
    let cwd = if let Some(name) = &cli.worktree {
        let worktree = agent_m_agent::create_worktree(
            &cwd,
            &agent_dir,
            Some(name.as_str()).filter(|n| !n.is_empty()),
        )
        .map_err(|error| anyhow::anyhow!("cannot create worktree: {error}"))?;
        std::env::set_current_dir(&worktree)
            .with_context(|| format!("cannot cd into {}", worktree.display()))?;
        eprintln!("worktree: {}", worktree.display());
        worktree
    } else {
        cwd
    };
    if let Some(Commands::Plugins { command }) = &cli.command {
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

    let _ = agent_m_ai::ensure_default_settings(&agent_dir);
    let _ = agent_m_mcp::ensure_default_mcp(&agent_dir);
    let settings_config = agent_m_ai::load_settings_config(&agent_dir);
    if settings_config.providers.is_empty() && cli.provider.is_none() {
        anyhow::bail!(
            "No LLM providers are configured in ~/.agent-m/agent/settings.json.\n\
             Please configure your providers in settings.json, for example:\n\
             {{\n  \
               \"defaultProvider\": \"openai\",\n  \
               \"defaultModel\": \"gpt-4o-mini\",\n  \
               \"providers\": [\n    \
                 {{\n      \
                   \"id\": \"openai\",\n      \
                   \"baseUrl\": \"https://api.openai.com/v1\",\n      \
                   \"model\": \"gpt-4o-mini\"\n    \
                 }}\n  \
               ]\n\
             }}"
        );
    }

    let (task_provider, task_model) = if cli.provider.is_none() && cli.model.is_none() {
        if let Some((p, m)) = agent_m_ai::resolve_task_route(&settings_config, "build") {
            (Some(p), m)
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    let target_provider_id = cli
        .provider
        .clone()
        .or(task_provider)
        .or_else(|| settings_config.default_provider.clone())
        .or_else(|| settings_config.providers.first().map(|c| c.id.clone()))
        .unwrap_or_else(|| "default".to_string());

    let provider: Arc<dyn Provider> = if let Some(config) = settings_config
        .providers
        .iter()
        .find(|c| c.id == target_provider_id)
    {
        Arc::from(agent_m_ai::provider_from_config(config, None, &agent_dir))
    } else {
        let available = settings_config
            .providers
            .iter()
            .map(|config| config.id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        anyhow::bail!(
            "provider `{target_provider_id}` is not configured in settings.json `providers`. \
             Available: [{available}]"
        );
    };

    let model = resolve_model(&cli, &settings_config, provider.as_ref(), task_model);
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
                    Ok((mcp_tools, hinted_read_only, _shared)) => {
                        let count = mcp_tools.len();
                        for tool in mcp_tools {
                            if !cli.exclude_tools.iter().any(|n| n == tool.name())
                                && (cli.tools.is_empty()
                                    || cli.tools.iter().any(|n| n == tool.name()))
                            {
                                // Read-only MCP tools (server readOnlyHint, or
                                // `readOnlyTools` config override on the bare
                                // name) skip the opaque/always-ask tier so
                                // read-like calls auto-approve.
                                let read_only = hinted_read_only.contains(&tool.name().to_string())
                                    || agent_m_mcp::matches_patterns(
                                        tool.name()
                                            .split("__")
                                            .last()
                                            .unwrap_or(tool.name()),
                                        &config.read_only_tools,
                                    );
                                if !read_only {
                                    opaque_tools.push(tool.name().to_string());
                                }
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
        if settings_config.providers.is_empty() {
            println!("No providers configured in ~/.agent-m/agent/settings.json.");
            println!("Add a provider config under 'providers' in settings.json to get started.");
            return Ok(());
        }
        for config in &settings_config.providers {
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
    let preference_block = crate::prefs::prompt_block(&crate::prefs::load(&agent_dir));

    // Remote human channel (Phase 2): with `--slack-channel` the ask tool,
    // High/Critical approvals, and flow progress go over Slack instead of
    // the terminal. The event loop runs on a spawned task for the whole
    // process.
    let remote = if let Some(channel) = &cli.slack_channel {
        let transport: Arc<dyn crate::slack::SlackTransport> = Arc::new(
            crate::slack::SlackClient::from_env().map_err(anyhow::Error::msg)?,
        );
        Some((crate::slack::RemoteHuman::start(transport), channel.clone()))
    } else {
        None
    };
    // The LevelGate ask closure: remote Slack approval, or the terminal gate.
    // Shared by the daemon and REPL gates so both honor `--slack-channel`.
    let human_ask = match &remote {
        Some((remote, channel)) => HumanAsk::Slack {
            remote: remote.clone(),
            channel: channel.clone(),
        },
        None => HumanAsk::Terminal,
    };

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
        harness_block: None,
        tools,
        max_turns: cli.max_turns,
        cwd: cwd.clone(),
        mode: if cli.mode_plan {
            agent_m_agent::Mode::Plan
        } else {
            agent_m_agent::Mode::Build
        },
        ask_gate: if let Some((remote, channel)) = &remote {
            Some(Arc::new(remote.ask_gate(channel)))
        } else {
            Some(Arc::new(crate::ask::make_repl_ask_gate()))
        },
        context_window: provider
            .models()
            .iter()
            .find(|m| m.id == model)
            .and_then(|m| m.context_window),
        variant: cli.variant.clone(),
        output_dir: Some(agent_dir.join("tool_outputs")),
        trust: agent_m_agent::TrustPolicy {
            mode: match cli.trust.as_str() {
                "off" => agent_m_agent::TrustMode::Off,
                "ask" => agent_m_agent::TrustMode::Ask,
                "block" => agent_m_agent::TrustMode::Block,
                _ => agent_m_agent::TrustMode::Warn,
            },
            confidence_threshold: 50,
        },
        risk_policy: Some((*risk).clone()),
        delegate_depth: 0,
    };

    if let Some(Commands::Slack { channel }) = &cli.command {
        let level = resolve_level(&cli, &settings);
        return run_slack_mode(
            provider,
            agent_options,
            cwd,
            cli.yes,
            risk.clone(),
            agent_dir.clone(),
            level,
            channel.clone(),
            cli.flow_context.clone(),
            cli.flow
                .clone()
                .unwrap_or_else(|| PathBuf::from("flows/agentic-dev.yml")),
        )
        .await;
    }

    if let Some(Commands::Pickup {
        ticket,
        repo,
        jql,
        transition_id,
        dry_run,
        flow,
        poll,
        workers,
    }) = &cli.command
    {
        let level = resolve_level(&cli, &settings);
        let (remote, channel) = match &remote {
            Some((r, ch)) => (Some(r.clone()), Some(ch.clone())),
            None => (None, None),
        };
        let context_pairs = cli.flow_context.clone();
        return run_pickup(
            provider,
            agent_options,
            cwd,
            cli.yes,
            risk.clone(),
            agent_dir.clone(),
            level,
            remote,
            channel,
            &context_pairs,
            ticket.as_deref(),
            repo.as_deref(),
            jql.as_deref(),
            transition_id,
            *dry_run,
            flow,
            *poll,
            *workers,
        )
        .await;
    }

    // Per-ticket daemon: one ticket's whole pipeline (transition → worktree →
    // flow → report) in an isolated process. The pickup supervisor spawns this
    // with `--agent-dir`, so a ticket's crash never takes down the queue.
    if let Some(Commands::TicketRun {
        ticket,
        repo,
        transition_id,
        flow,
        agent_dir,
        flow_context,
    }) = &cli.command
    {
        let agent_dir = agent_dir
            .clone()
            .unwrap_or_else(|| cli.session_dir.clone().unwrap_or_else(default_agent_dir));
        return run_ticket_run(
            provider,
            agent_options,
            cwd,
            cli.yes,
            risk.clone(),
            agent_dir,
            ticket,
            repo.as_deref(),
            transition_id,
            flow,
            flow_context,
        )
        .await;
    }

    // Tail a per-ticket daemon's report. Only needs the agent dir, so it can
    // run even when a provider/tools setup would be impossible (e.g. a plain
    // status check).
    if let Some(Commands::TicketLog {
        ticket,
        follow,
        agent_dir,
    }) = &cli.command
    {
        let agent_dir = agent_dir
            .clone()
            .unwrap_or_else(|| cli.session_dir.clone().unwrap_or_else(default_agent_dir));
        return run_ticket_log(&agent_dir, ticket, *follow).await;
    }

    if let Some(flow_path) = &cli.flow {
        let flow_state_dir = agent_dir.join("flows");
        let level = resolve_level(&cli, &settings);
        let (remote, channel) = match &remote {
            Some((r, ch)) => (Some(r.clone()), Some(ch.clone())),
            None => (None, None),
        };
        let run = run_flow_mode(
            provider,
            agent_options,
            flow_path,
            cwd,
            cli.yes,
            risk.clone(),
            flow_state_dir,
            level,
            remote,
            channel,
            &cli.flow_context,
            None,
        )
        .await?;
        if run
            .steps
            .iter()
            .any(|s| s.status == agent_m_flow::StepStatus::Failed)
        {
            std::process::exit(1);
        }
        return Ok(());
    }

    if cli.serve {
        let gate = non_interactive_gate(cli.yes, risk.clone());
        return serve_loop(provider, agent_options, gate).await;
    }

    // Background sessions win over print mode: the daemon runs with stdin
    // not a TTY, which would otherwise trip the `print_mode` early return
    // below. `--list-daemons`, `--attach`, and `--daemon` are all
    // non-interactive and must dispatch before the print-mode check.
    if cli.list_daemons {
        let dir = crate::daemon::get_sockets_dir(&agent_dir)?;
        println!("Active daemon sockets in {}:", dir.display());
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.ends_with(".sock") {
                    println!("  - {}", name_str.trim_end_matches(".sock"));
                }
            }
        }
        return Ok(());
    }

    if let Some(session_id) = &cli.attach {
        return crate::attach::run_attach(session_id, &agent_dir).await;
    }

    if let Some(session_id) = &cli.daemon {
        let human = human_ask.clone();
        let gate = Arc::new(agent_m_agent::LevelGate::new(
            resolve_level(&cli, &settings),
            (*risk).clone(),
            move |call| ask_human_permission(&human, risk.clone(), call),
        ));
        return crate::daemon::run_daemon(
            session_id.clone(),
            provider,
            agent_options,
            gate,
            agent_dir,
            cwd,
        )
        .await;
    }

    if print_mode {
        let messages = inline_file_args(&cwd, cli.messages.clone());
        let gate = non_interactive_gate(cli.yes, risk.clone());
        return run_print(provider, agent_options, gate, messages, cli.stream_json).await;
    }

    let human = human_ask.clone();
    let level_gate = agent_m_agent::LevelGate::new(
        resolve_level(&cli, &settings),
        (*risk).clone(),
        move |call| ask_human_permission(&human, risk.clone(), call),
    );
    let level_handle = level_gate.level_handle();
    let gate: Arc<dyn agent_m_agent::PermissionGate> = Arc::new(level_gate);

    crate::repl::run_repl(
        provider.clone(),
        agent_options,
        gate,
        agent_dir,
        cwd,
        Some(level_handle),
    )
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

/// Run a YAML flow in print mode: sequential steps, status lines to stdout.
/// Returns the full `FlowRun` (callers decide exit status — steps may be
/// Failed inside a completed run; `Err` only when the flow itself aborts).
/// With a remote channel, ask steps and approvals go over Slack and step
/// progress is posted live.
async fn run_flow_mode(
    provider: Arc<dyn Provider>,
    agent_options: AgentOptions,
    flow_path: &std::path::Path,
    cwd: PathBuf,
    yes: bool,
    risk: Arc<RiskPolicy>,
    state_dir: PathBuf,
    level: agent_m_agent::AutonomyLevel,
    remote: Option<Arc<crate::slack::RemoteHuman>>,
    channel: Option<String>,
    flow_context: &[String],
    ticket_log: Option<&std::path::Path>,
) -> anyhow::Result<agent_m_flow::FlowRun> {
    let flow = agent_m_flow::load_flow(&flow_path.to_path_buf())?;
    let tools = agent_options.tools.clone();
    let permission: Arc<dyn PermissionGate> = match &remote {
        // With a remote human, approvals work: LevelGate auto-approves
        // Low/Medium and asks over Slack for High/Critical.
        Some(remote) => {
            let channel = channel.clone().unwrap_or_default();
            let closure = remote.permission_closure((*risk).clone(), &channel);
            Arc::new(agent_m_agent::LevelGate::new(level, (*risk).clone(), closure))
        }
        // No human to ask: `--yes` is full trust minus risk hints; without
        // it nothing runs.
        None => non_interactive_gate(yes, risk),
    };
    let ask_gate: Option<Arc<dyn agent_m_agent::AskGate>> = remote.as_ref().map(|remote| {
        let gate = remote.ask_gate(&channel.clone().unwrap_or_default());
        Arc::new(gate) as Arc<dyn agent_m_agent::AskGate>
    });
    // Progress sinks: Slack steps when a remote human is attached, and/or the
    // per-ticket daemon's JSONL report (live `step` lines while the flow runs).
    let mut progress_sinks: Vec<Arc<dyn Fn(agent_m_flow::FlowProgress) + Send + Sync>> = Vec::new();
    if let Some(remote) = &remote {
        progress_sinks.push(crate::slack::slack_progress(
            remote.transport.clone(),
            channel.clone().unwrap_or_default(),
        ));
    }
    if let Some(log_path) = ticket_log {
        let log_path = log_path.to_path_buf();
        progress_sinks.push(Arc::new(move |progress| {
            let _ = crate::ticket_log::append(
                log_path.parent().expect("report path has a dir"),
                log_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("ticket"),
                &serde_json::json!({
                    "ts": crate::sessions::now_iso(),
                    "kind": "step",
                    "step": progress.step_name,
                    "status": progress.status.as_str(),
                }),
            );
        }));
    }
    let on_progress = if progress_sinks.is_empty() {
        None
    } else {
        Some(Arc::new(move |progress: agent_m_flow::FlowProgress| {
            for sink in &progress_sinks {
                sink(progress.clone());
            }
        }) as Arc<dyn Fn(agent_m_flow::FlowProgress) + Send + Sync>)
    };
    let deps = agent_m_flow::FlowDeps {
        provider,
        agent_options,
        tools,
        permission_gate: permission,
        ask_gate,
        state_dir: Some(state_dir),
        on_progress,
    };
    let mut context = agent_m_flow::FlowContext::new();
    context.set("cwd", serde_json::json!(cwd.to_string_lossy()));
    for pair in flow_context {
        let Some((key, value)) = pair.split_once('=') else {
            anyhow::bail!("--flow-context expects KEY=VALUE, got `{pair}`");
        };
        if key.is_empty() {
            anyhow::bail!("--flow-context key must not be empty in `{pair}`");
        }
        context.set(key, serde_json::json!(value));
    }
    println!("flow: {}", flow.name);
    let run = match agent_m_flow::run_flow(&flow, &mut context, &deps).await {
        Ok(run) => run,
        Err(error) => return Err(anyhow::anyhow!("flow failed: {error}")),
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
    Ok(run)
}

/// One picked ticket, plus the flow run that executed against it.
struct PickupOutcome {
    ticket: crate::pickup::PickedTicket,
    run: agent_m_flow::FlowRun,
}

/// Pick a ticket and run the flow against it in a fresh worktree. Pure-ish:
/// `dry_run` prints the plan and returns `Ok(None)` before any side effect.
/// Returns `Ok(Some(outcome))` after a real run (the run may contain failed
/// steps — the caller decides how to report).
async fn pick_and_run(
    provider: Arc<dyn Provider>,
    agent_options: AgentOptions,
    cwd: PathBuf,
    yes: bool,
    risk: Arc<RiskPolicy>,
    agent_dir: PathBuf,
    level: agent_m_agent::AutonomyLevel,
    remote: Option<Arc<crate::slack::RemoteHuman>>,
    channel: Option<String>,
    flow_context: &[String],
    ticket_override: Option<&str>,
    repo_override: Option<&str>,
    jql: Option<&str>,
    transition_id: &str,
    dry_run: bool,
    flow_path: &std::path::Path,
    ticket_log: Option<&std::path::Path>,
) -> anyhow::Result<Option<PickupOutcome>> {
    let flow = agent_m_flow::load_flow(&flow_path.to_path_buf())?;
    let inputs = crate::pickup::PickInputs {
        agent_dir: &agent_dir,
        ticket: ticket_override,
        repo: repo_override,
        jql,
    };
    let picked = crate::pickup::pick(inputs).await?;

    if dry_run {
        println!(
            "pickup (dry-run): {} — {}",
            picked.key, picked.summary
        );
        println!("  repo:       {}", picked.repo);
        println!(
            "  worktree:   agent-m/{} in {}/worktrees/",
            picked.key,
            agent_dir.display()
        );
        println!(
            "  transition: {} → In Progress (transition {})",
            picked.key, transition_id
        );
        println!("  flow:       {} ({})", flow_path.display(), flow.name);
        println!("  context:    ticket={} repo={}", picked.key, picked.repo);
        return Ok(None);
    }

    println!(
        "pickup: {} — {}\n  repo:       {}\n  transition: → In Progress ({})",
        picked.key, picked.summary, picked.repo, transition_id
    );

    // Per-ticket daemon: record the pickup in the ticket's report before any
    // side effect, so `ticket-log` shows something immediately.
    if let Some(log_path) = ticket_log {
        let _ = crate::ticket_log::append(
            log_path.parent().expect("report path has a dir"),
            log_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("ticket"),
            &serde_json::json!({
                "ts": crate::sessions::now_iso(),
                "kind": "pickup",
                "ticket": picked.key,
                "summary": picked.summary,
                "repo": picked.repo,
            }),
        );
    }

    // In Progress first, so the loop never re-picks it.
    let base = std::env::var("JIRA_URL")
        .unwrap_or_else(|_| String::from("https://your.atlassian.net"));
    let token = std::env::var("JIRA_TOKEN").unwrap_or_default();
    if let Err(error) =
        crate::pickup::transition_ticket(&base, &token, &picked.key, transition_id).await
    {
        eprintln!("warn: could not transition {}: {}", picked.key, error);
    }

    // Fresh worktree branch, then run the flow inside it. The repo is
    // seeded from the mapping; the worktree gives us branch isolation.
    let worktree = agent_m_agent::create_worktree(&cwd, &agent_dir, Some(&picked.key))
        .map_err(anyhow::Error::msg)?;
    println!("  worktree:   {}", worktree.display());

    let mut context_pairs = flow_context.to_vec();
    context_pairs.push(format!("ticket={}", picked.key));
    context_pairs.push(format!("repo={}", picked.repo));
    context_pairs.push("worktree=true".to_string());
    let state_dir = agent_dir.join("flows");
    let run = run_flow_mode(
        provider,
        agent_options,
        flow_path,
        worktree,
        yes,
        risk,
        state_dir,
        level,
        remote,
        channel,
        &context_pairs,
        ticket_log,
    )
    .await?;
    Ok(Some(PickupOutcome { ticket: picked, run }))
}

/// `agent-m pickup`: pick the next open assigned ticket, move it to In
/// Progress, create a worktree branch, and run the flow inside it with
/// `${ticket}` / `${repo}` seeded. `--dry-run` only prints the plan;
/// `--poll N` keeps picking until Ctrl-C; `--workers M` runs up to M ticket
/// flows in parallel under `--poll`.
async fn run_pickup(
    provider: Arc<dyn Provider>,
    agent_options: AgentOptions,
    cwd: PathBuf,
    yes: bool,
    risk: Arc<RiskPolicy>,
    agent_dir: PathBuf,
    level: agent_m_agent::AutonomyLevel,
    remote: Option<Arc<crate::slack::RemoteHuman>>,
    channel: Option<String>,
    flow_context: &[String],
    ticket_override: Option<&str>,
    repo_override: Option<&str>,
    jql: Option<&str>,
    transition_id: &str,
    dry_run: bool,
    flow_path: &std::path::Path,
    poll: Option<u64>,
    workers: usize,
) -> anyhow::Result<()> {
    let poll_interval = poll.unwrap_or(0);
    let workers = workers.max(1);

    // Parallel path: a poll loop with multiple workers and no explicit
    // ticket (a single explicit ticket cannot be parallelized). Dry-runs
    // stay serial — they print the plan once and exit.
    if poll_interval > 0 && workers > 1 && ticket_override.is_none() && !dry_run {
        return run_pickup_concurrent(
            yes,
            agent_dir,
            flow_context.to_vec(),
            jql.map(str::to_string),
            transition_id.to_string(),
            flow_path.to_path_buf(),
            poll_interval,
            workers,
        )
        .await;
    }

    loop {
        match pick_and_run(
            provider.clone(),
            agent_options.clone(),
            cwd.clone(),
            yes,
            risk.clone(),
            agent_dir.clone(),
            level,
            remote.clone(),
            channel.clone(),
            flow_context,
            ticket_override,
            repo_override,
            jql,
            transition_id,
            dry_run,
            flow_path,
            None,
        )
        .await
        {
            Ok(Some(outcome)) => {
                let failed = outcome
                    .run
                    .steps
                    .iter()
                    .any(|s| s.status == agent_m_flow::StepStatus::Failed);
                if poll_interval == 0 {
                    if failed {
                        // The flow already printed its FAILED summary.
                        std::process::exit(1);
                    }
                    return Ok(());
                }
                if failed {
                    eprintln!("flow failed for {}", outcome.ticket.key);
                }
            }
            Ok(None) => {
                // Dry-run printed the plan; nothing else to do.
                return Ok(());
            }
            Err(error) => {
                eprintln!("pickup failed: {error}");
                if poll_interval == 0 {
                    return Err(error);
                }
            }
        }

        eprintln!("pickup: sleeping {poll_interval}s before next pick (Ctrl-C to stop)");
        tokio::time::sleep(std::time::Duration::from_secs(poll_interval)).await;
    }
}

/// Parallel `--poll` path (Phase 8 + Phase 10): up to `workers` flows run at
/// once, one worktree per ticket. A supervisor acquires a worker slot, picks
/// the next ticket that is not already in flight, and hands it to a spawned
/// **per-ticket daemon** — an independent `agent-m ticket-run` child process
/// (Phase 10), so a ticket's crash/panic never takes down the supervisor or
/// siblings, and its report is tailable with `agent-m ticket-log <KEY>`.
/// Failures are reported per ticket and the loop keeps going until Ctrl-C.
/// Double-picking is prevented twice: the default JQL now excludes
/// `In Progress`, and the in-flight set blocks the window before a worker's
/// transition lands.
async fn run_pickup_concurrent(
    yes: bool,
    agent_dir: PathBuf,
    flow_context: Vec<String>,
    jql: Option<String>,
    transition_id: String,
    flow_path: PathBuf,
    poll_interval: u64,
    workers: usize,
) -> anyhow::Result<()> {
    use std::collections::HashSet;
    use std::sync::Mutex;

    let semaphore = Arc::new(tokio::sync::Semaphore::new(workers));
    let in_flight: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
    loop {
        // Wait for a free worker slot before querying, so a busy backlog
        // doesn't hammer Jira with searches.
        let permit = semaphore.clone().acquire_owned().await;
        let picked = {
            let inputs = crate::pickup::PickInputs {
                agent_dir: &agent_dir,
                ticket: None,
                repo: None,
                jql: jql.as_deref(),
            };
            match crate::pickup::pick(inputs).await {
                Ok(picked) => picked,
                Err(error) => {
                    eprintln!("pickup failed: {error}");
                    drop(permit);
                    eprintln!(
                        "pickup: sleeping {poll_interval}s before next pick (Ctrl-C to stop)"
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(poll_interval)).await;
                    continue;
                }
            }
        };
        {
            let mut in_flight = in_flight.lock().unwrap();
            if in_flight.contains(&picked.key) {
                // Another worker already grabbed it (its In-Progress
                // transition hadn't landed yet).
                drop(permit);
                eprintln!("pickup: {} already in flight, waiting", picked.key);
                tokio::time::sleep(std::time::Duration::from_secs(poll_interval)).await;
                continue;
            }
            in_flight.insert(picked.key.clone());
        }
        // Hand this specific ticket to a worker task. Each worker is its own
        // `agent-m ticket-run` child process (the per-ticket daemon): a
        // ticket's crash/panic never takes down the supervisor or siblings,
        // and its report file is tailable with `agent-m ticket-log <KEY>`.
        let key = picked.key.clone();
        let repo = picked.repo.clone();
        tokio::spawn({
            let in_flight = in_flight.clone();
            let agent_dir = agent_dir.clone();
            let flow_context = flow_context.clone();
            let transition_id = transition_id.clone();
            let flow_path = flow_path.clone();
            async move {
                let mut cmd = tokio::process::Command::new(
                    std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("agent-m")),
                );
                cmd.arg("ticket-run")
                    .arg("--ticket")
                    .arg(&key)
                    .arg("--repo")
                    .arg(&repo)
                    .arg("--flow")
                    .arg(&flow_path)
                    .arg("--transition-id")
                    .arg(&transition_id)
                    .arg("--agent-dir")
                    .arg(&agent_dir);
                if yes {
                    cmd.arg("--yes");
                }
                for pair in &flow_context {
                    cmd.arg("--flow-context").arg(pair);
                }
                let status = cmd.status().await;
                in_flight.lock().unwrap().remove(&key);
                drop(permit);
                match status {
                    Ok(status) if status.success() => {}
                    Ok(status) => eprintln!(
                        "ticket-run for {} exited with {}",
                        key,
                        status
                            .code()
                            .map(|c| c.to_string())
                            .unwrap_or_else(|| "signal".to_string())
                    ),
                    Err(error) => eprintln!("ticket-run for {} failed to start: {error}", key),
                }
            }
        });
    }
}

/// The Slack DM-triggered pickup run (Phase 6): pick → transition → worktree
/// → flow with this channel as the oversight channel (asks, approvals,
/// progress), then post a summary DM back.
async fn run_pickup_slack(
    provider: Arc<dyn Provider>,
    agent_options: AgentOptions,
    cwd: PathBuf,
    yes: bool,
    risk: Arc<RiskPolicy>,
    agent_dir: PathBuf,
    level: agent_m_agent::AutonomyLevel,
    remote: Arc<crate::slack::RemoteHuman>,
    channel: String,
    flow_context: &[String],
    ticket_override: Option<&str>,
    repo_override: Option<&str>,
    jql: Option<&str>,
    transition_id: &str,
    flow_path: &std::path::Path,
) -> anyhow::Result<()> {
    let _ = remote
        .transport
        .post_message(&channel, "🔄 picking it up — I'll report back here.")
        .await;
    let outcome = pick_and_run(
        provider,
        agent_options,
        cwd,
        yes,
        risk,
        agent_dir,
        level,
        Some(remote.clone()),
        Some(channel.clone()),
        flow_context,
        ticket_override,
        repo_override,
        jql,
        transition_id,
        false,
        flow_path,
        None,
    )
    .await?
    .expect("slack pickup never dry-runs");
    let summary = crate::slack::flow_summary(&outcome.ticket.key, &outcome.run);
    let _ = remote.transport.post_message(&channel, &summary).await;
    Ok(())
}

/// `agent-m ticket-run`: the per-ticket daemon body. Runs one ticket's whole
/// pipeline (transition → worktree → flow) in an isolated process and appends
/// a JSONL report to `<agent_dir>/tickets/<KEY>.jsonl` as it goes. Exit code
/// 0 = flow OK, 1 = any step failed (the supervisor reports it).
async fn run_ticket_run(
    provider: Arc<dyn Provider>,
    agent_options: AgentOptions,
    cwd: PathBuf,
    yes: bool,
    risk: Arc<RiskPolicy>,
    agent_dir: PathBuf,
    ticket: &str,
    repo: Option<&str>,
    transition_id: &str,
    flow_path: &std::path::Path,
    flow_context: &[String],
) -> anyhow::Result<()> {
    let report_path = crate::ticket_log::report_path(&agent_dir, ticket);
    let outcome = pick_and_run(
        provider,
        agent_options,
        cwd,
        yes,
        risk,
        agent_dir.clone(),
        agent_m_agent::AutonomyLevel::default(),
        None, // Per-ticket daemons are terminal-driven; Slack oversight stays
        None, // in-process in `agent-m slack` (channel-based, Phase 6).
        flow_context,
        Some(ticket),
        repo,
        None, // explicit ticket → no JQL query
        transition_id,
        false,
        flow_path,
        Some(&report_path),
    )
    .await?
    .expect("ticket-run never dry-runs");

    // Final verdict line, mirroring flow_summary's PR extraction.
    use agent_m_flow::StepStatus;
    let failed = outcome
        .run
        .steps
        .iter()
        .any(|s| s.status == StepStatus::Failed);
    let pr = outcome
        .run
        .steps
        .iter()
        .find(|s| s.name == "pr")
        .and_then(|s| {
            s.output
                .as_ref()
                .and_then(|o| o.get("content"))
                .and_then(serde_json::Value::as_str)
        })
        .map(|content| content.strip_prefix("PR created: ").unwrap_or(content))
        .filter(|content| content.contains("http"))
        .unwrap_or_default()
        .to_string();
    let _ = crate::ticket_log::append(
        &agent_dir,
        ticket,
        &serde_json::json!({
            "ts": crate::sessions::now_iso(),
            "kind": "verdict",
            "status": if failed { "FAILED" } else { "OK" },
            "fix_rounds": outcome.run.fix_rounds,
            "pr": pr,
        }),
    );

    if failed {
        std::process::exit(1);
    }
    Ok(())
}

/// `agent-m ticket-log <KEY>`: print (or `--follow` tail) a per-ticket
/// daemon's report. Friendly error when the ticket has no report yet.
async fn run_ticket_log(
    agent_dir: &std::path::Path,
    ticket: &str,
    follow: bool,
) -> anyhow::Result<()> {
    let path = crate::ticket_log::report_path(agent_dir, ticket);
    if !path.exists() {
        anyhow::bail!(
            "no ticket report for `{ticket}` at {} (run `agent-m pickup --poll --workers N` \
             or `agent-m ticket-run --ticket {ticket}` first)",
            path.display()
        );
    }
    let mut printed = 0usize;
    loop {
        match crate::ticket_log::read_lines(agent_dir, ticket) {
            Ok(lines) => {
                for line in &lines[printed..] {
                    println!("{}", crate::ticket_log::render_line(line));
                }
                printed = lines.len();
            }
            Err(error) => {
                if follow {
                    // The report may be transiently unreadable (rotating);
                    // keep tailing.
                    eprintln!("ticket-log: {error}");
                } else {
                    anyhow::bail!("{error}");
                }
            }
        }
        if !follow {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
}

/// `agent-m slack` (Phase 6 orchestrator): connect Slack and run pickup +
/// flow on every `pick up [TICKET]` DM, with Slack as the oversight channel
/// and a summary DM when the flow finishes.
async fn run_slack_mode(
    provider: Arc<dyn Provider>,
    agent_options: AgentOptions,
    cwd: PathBuf,
    yes: bool,
    risk: Arc<RiskPolicy>,
    agent_dir: PathBuf,
    level: agent_m_agent::AutonomyLevel,
    _default_channel: Option<String>,
    flow_context: Vec<String>,
    flow_path: PathBuf,
) -> anyhow::Result<()> {
    let transport: Arc<dyn crate::slack::SlackTransport> = Arc::new(
        crate::slack::SlackClient::from_env().map_err(anyhow::Error::msg)?,
    );
    let remote = crate::slack::RemoteHuman::new(transport.clone());

    let on_pickup: Arc<dyn Fn(Option<String>, String) + Send + Sync> = Arc::new({
        let provider = provider.clone();
        let agent_options = agent_options.clone();
        let cwd = cwd.clone();
        let risk = risk.clone();
        let agent_dir = agent_dir.clone();
        let flow_context = flow_context.clone();
        let flow_path = flow_path.clone();
        let remote = remote.clone();
        let transport = transport.clone();
        move |ticket, channel| {
            let provider = provider.clone();
            let agent_options = agent_options.clone();
            let cwd = cwd.clone();
            let risk = risk.clone();
            let agent_dir = agent_dir.clone();
            let flow_context = flow_context.clone();
            let flow_path = flow_path.clone();
            let remote = remote.clone();
            let transport = transport.clone();
            tokio::spawn(async move {
                let result = run_pickup_slack(
                    provider,
                    agent_options,
                    cwd,
                    yes,
                    risk,
                    agent_dir,
                    level,
                    remote,
                    channel.clone(),
                    &flow_context,
                    ticket.as_deref(),
                    None,
                    None,
                    "11",
                    &flow_path,
                )
                .await;
                if let Err(error) = result {
                    let _ = transport
                        .post_message(&channel, &format!("❌ pickup failed: {error}"))
                        .await;
                }
            });
        }
    });
    crate::slack::RemoteHuman::start_orchestrator(remote.clone(), on_pickup);

    println!("agent-m slack orchestrator online — DM `pick up [TICKET]` to start a flow (Ctrl-C to stop)");
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(3600)).await;
    }
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
    let stream_filter = Arc::new(std::sync::Mutex::new(crate::repl::StreamFilter::default()));
    let filter_listener = stream_filter.clone();

    agent.subscribe(move |event| match event {
        agent_m_agent::AgentEvent::MessageUpdate {
            delta: agent_m_ai::StreamEvent::TextDelta { delta },
        } => {
            if let Ok(mut sf) = filter_listener.lock() {
                let to_print = sf.push(delta);
                if !to_print.is_empty() {
                    print!("{to_print}");
                    let _ = std::io::stdout().flush();
                }
            }
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
            if let Ok(mut sf) = stream_filter.lock() {
                *sf = crate::repl::StreamFilter::default();
            }
            agent.prompt(input.trim().to_string()).await?;
            if let Ok(mut sf) = stream_filter.lock() {
                let rest = sf.finish();
                if !rest.is_empty() {
                    print!("{rest}");
                    let _ = std::io::stdout().flush();
                }
            }
        }
    } else {
        for (message, images) in messages {
            if let Ok(mut sf) = stream_filter.lock() {
                *sf = crate::repl::StreamFilter::default();
            }
            let result = if images.is_empty() {
                agent.prompt(message).await
            } else {
                agent.prompt_with_images(message, images).await
            };
            if let Ok(mut sf) = stream_filter.lock() {
                let rest = sf.finish();
                if !rest.is_empty() {
                    print!("{rest}");
                    let _ = std::io::stdout().flush();
                }
            }
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
    
    let log_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".agent-m")
        .join("logs");
    let _ = std::fs::create_dir_all(&log_dir);
    let file_appender = tracing_appender::rolling::daily(&log_dir, "agent.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    let fmt = tracing_subscriber::fmt::layer()
        .with_target(false)
        .with_writer(non_blocking);

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
        return OtelGuard(Some(provider), Some(guard));
    }
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(fmt)
        .try_init();
    OtelGuard(None, Some(guard))
}

/// Flushes the OTLP batch exporter and tracing appender on drop.
#[allow(dead_code)]
struct OtelGuard(
    Option<opentelemetry_sdk::trace::TracerProvider>,
    Option<tracing_appender::non_blocking::WorkerGuard>,
);
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
        E::RefineResult { .. } => serde_json::json!({}),
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
        E::RefineResult { .. } => "refine_result",
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
