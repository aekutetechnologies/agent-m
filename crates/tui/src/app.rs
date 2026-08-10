//! The pi-style interactive TUI: transcript + dock (status, editor, footer).
//!
//! Owns the terminal, drives the agent from a background task, and renders
//! agent events into the transcript with follow-end scrolling.

use agent_m_agent::{
    Agent, AgentEvent, AgentOptions, Permission, SessionMessage, Tool, ToolContext,
};
use agent_m_ai::{ContentPart, ModelSpec, Provider, StopReason, TrustData, Usage};
use agent_m_tools::BashTool;
use anyhow::{Context, Result};
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyModifiers,
    MouseEventKind,
};
use ratatui::crossterm::execute;
use ratatui::crossterm::terminal::{
    self as crossterm_terminal, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use std::io::Stdout;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot};
use unicode_width::UnicodeWidthStr;

use crate::editor::Editor;
use crate::keybindings::{self, Action, AppAction, EditorAction};
use crate::sessions::SessionStore;
use crate::theme::Theme;
use crate::transcript::TranscriptItem;

/// `ui-mode`: regular renders into the terminal scrollback, fullscreen uses
/// the alternate screen (pi's `TuiMainScreen` / `TuiAltScreen`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiMode {
    Regular,
    Fullscreen,
}

impl UiMode {
    pub fn parse(value: &str) -> Option<UiMode> {
        match value {
            "regular" => Some(UiMode::Regular),
            "fullscreen" => Some(UiMode::Fullscreen),
            _ => None,
        }
    }
}

/// Everything the App needs except the terminal itself.
pub struct AppInputs {
    pub provider: Arc<dyn Provider>,
    /// Agent options without the permission gate (the App installs it).
    pub agent_options: AgentOptions,
    pub theme: Theme,
    pub ui_mode: UiMode,
    pub show_cache_notices: bool,
    /// The models the provider serves (for the model picker + cost display).
    pub models: Vec<ModelSpec>,
    /// Loaded AGENTS.md instruction files (shown in the /info panel).
    pub context_files: Vec<agent_m_agent::InstructionFile>,
    /// `--yes`: auto-approve tool calls instead of asking.
    pub approve_tools: bool,
    /// Auto-compact threshold (fraction of the window) applied at turn
    /// boundaries (ECC strategic compaction; default 0.5).
    pub compact_threshold: f64,
    /// check.md principle 12: progressive autonomy level (default Trusted).
    pub level: agent_m_agent::AutonomyLevel,
    pub agent_dir: PathBuf,
    pub cwd: PathBuf,
}

enum SubmitCommand {
    Prompt(String),
    Interrupt,
    SetModel(String),
    SetMode(agent_m_agent::Mode),
    Compact,
    RunFlow(std::path::PathBuf),
}

type ApprovalRequest = (agent_m_agent::ToolCallInfo, oneshot::Sender<Permission>);

/// One step's view state for the flow sidebar.
#[derive(Debug, Clone)]
struct StepView {
    name: String,
    status: String,
}

/// Live state of the most recent flow run (the sidebar's primary view).
#[derive(Debug, Clone)]
struct FlowRunView {
    name: String,
    steps: Vec<StepView>,
}

/// Sidebar width in columns (right-side flow/tasks/stats panel).
const SIDEBAR_WIDTH: u16 = 32;
/// Progress-bar cells in the flow view.
const BAR_CELLS: usize = 20;
/// Below this terminal width the sidebar auto-hides to protect the transcript.
const SIDEBAR_MIN_WIDTH: u16 = 110;

type AskRequest = (
    String,
    Option<Vec<String>>,
    oneshot::Sender<Result<String, String>>,
);

/// Session-aggregated token usage and cost.
#[derive(Debug, Clone, Copy, Default)]
struct UsageTotals {
    input: u64,
    output: u64,
    cache_read: u64,
    cache_create: u64,
    cost: f64,
}

impl UsageTotals {
    fn add(&mut self, usage: &Usage) {
        self.input += usage.input_tokens;
        self.output += usage.output_tokens;
        self.cache_read += usage.cache_read_tokens;
        self.cache_create += usage.cache_creation_tokens;
        self.cost += usage.cost;
    }
}

/// Compact token count: `785`, `39.4k`, `1.2M`.
fn fmt_tokens(count: u64) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1000 {
        format!("{:.1}k", count as f64 / 1000.0)
    } else {
        count.to_string()
    }
}

const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const SLASH_COMMANDS: &[&str] = &[
    "/help",
    "/hotkeys",
    "/clear",
    "/exit",
    "/quit",
    "/model",
    "/new",
    "/settings",
    "/cache",
];

/// The interactive application.
pub struct App {
    inputs: AppInputs,
    theme: Theme,
    ui_mode: UiMode,
    show_cache_notices: bool,
    cwd: PathBuf,

    items: Vec<TranscriptItem>,
    /// Items before this index are from a completed, no-longer-current
    /// turn: their tool output/thinking render collapsed unless explicitly
    /// expanded. Sealed by `seal_turn()` at the start of each new action.
    collapsed_before: usize,
    editor: Editor,
    editor_view_top: usize,
    completion: Option<String>,
    completion_index: usize,

    streaming: bool,
    cache_hit: u64,
    cache_miss: u64,
    cache_requests: u64,
    current_model: String,
    mode: agent_m_agent::Mode,
    last_error: Option<String>,
    follow_end: bool,
    scroll_back: usize,

    // Flow extras: session usage/cost, clarifying questions, pickers.
    models: Vec<ModelSpec>,
    context_files: Vec<agent_m_agent::InstructionFile>,
    provider_id: String,
    session_path: PathBuf,
    usage: UsageTotals,
    last_input: u64,
    question_pending: bool,
    todos: Vec<crate::plan::TodoItem>,
    /// Live flow progress (right-side sidebar view).
    flow_run: Option<FlowRunView>,
    /// Whether the right-side sidebar is visible (/sidebar toggles it).
    show_sidebar: bool,
    /// Principle 1 narration: what the model's current tool call is doing.
    active_tool: Option<String>,
    /// Trust data of the most recent assistant reply (the /info panel).
    last_trust: agent_m_ai::TrustData,
    /// Undoable file snapshots (principle 8); the top is the most recent.
    undo_stack: Vec<crate::sessions::UndoEntry>,
    /// Live handle to the autonomy level (shared with the gate for /level).
    level_handle: std::sync::Arc<std::sync::atomic::AtomicU8>,
    /// The /journal audit-timeline overlay is open.
    journal_open: bool,
    plan_choice_pending: bool,
    session_stem: String,
    model_picker_open: bool,
    picker_index: usize,
    info_open: bool,

    risk: Arc<agent_m_agent::RiskPolicy>,
    pending_approval: Option<ApprovalRequest>,
    pending_ask: Option<AskRequest>,
    ask_rx: mpsc::UnboundedReceiver<AskRequest>,

    submit_tx: mpsc::UnboundedSender<SubmitCommand>,
    event_rx: mpsc::UnboundedReceiver<AgentEvent>,
    approval_rx: mpsc::UnboundedReceiver<ApprovalRequest>,
    session_store: Option<SessionStore>,
    should_exit: bool,
    last_spinner: Instant,
}

impl App {
    /// Build the app: wires the agent event channel, the permission gate, the
    /// session store, and spawns the agent runner task.
    fn new(inputs: AppInputs) -> Result<Self> {
        let current_model = inputs.agent_options.model.clone();
        let initial_mode = inputs.agent_options.mode;

        // Approval gate: auto-approve (--yes) or route through the UI.
        // Destructive bash commands (rm -rf, git reset --hard, ...) always go
        // through the UI, even under --yes (ECC GateGuard).
        let (approval_tx, approval_rx) = mpsc::unbounded_channel::<ApprovalRequest>();
        let make_gate_closure = |approval_tx: mpsc::UnboundedSender<ApprovalRequest>| {
            move |tool_call: &agent_m_agent::ToolCallInfo| -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Permission> + Send>,
            > {
                let approval_tx = approval_tx.clone();
                let call = tool_call.clone();
                let (response_tx, response_rx) = oneshot::channel();
                let _ = approval_tx.send((call, response_tx));
                Box::pin(async move {
                    match response_rx.await {
                        Ok(permission) => permission,
                        Err(_) => Permission::Denied("approval channel closed".to_string()),
                    }
                })
            }
        };
        let risk = Arc::new(agent_m_agent::RiskPolicy {
            cwd: inputs.cwd.clone(),
            opaque_tools: vec![], // TUI has no plugins yet; will be needed when they load
        });
        // A human is always present in the interactive TUI, so the gate is
        // risk-based unconditionally, `--yes` or not: read-only tools never
        // prompt, risky calls always do (ECC GateGuard), and everything else
        // — including benign shell commands like `ls`/`cat` run via `bash` —
        // auto-approves. `--yes` still matters for print mode and flows,
        // which have no human to ask and default to denying everything.
        let closure = make_gate_closure(approval_tx.clone());
        // check.md principle 12: the autonomy level maps onto the risk tiers
        // (LevelGate). 0-1 observe/suggest (no execution), 2 asks for
        // everything, 3 (default) auto Low/Medium + asks High/Critical, 4
        // auto everything except Critical.
        let level_gate = agent_m_agent::LevelGate::new(
            inputs.level,
            (*risk).clone(),
            move |call: agent_m_agent::ToolCallInfo| closure(&call),
        );
        let level_handle = level_gate.level_handle();
        let gate: Arc<dyn agent_m_agent::PermissionGate> = Arc::new(level_gate);

        let (event_tx, event_rx) = mpsc::unbounded_channel::<AgentEvent>();
        let (submit_tx, mut submit_rx) = mpsc::unbounded_channel::<SubmitCommand>();
        let session_store = SessionStore::create(&inputs.agent_dir, &inputs.cwd)
            .context("cannot create session store")?;
        let session_path = session_store.path().to_path_buf();
        let session_stem = session_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("session")
            .to_string();
        let resumed_todos = crate::sessions::load_todos(&inputs.agent_dir, &session_stem);
        let undo_stack = crate::sessions::load_undo(&inputs.agent_dir, &session_stem);
        let provider_id = inputs.provider.id().to_string();
        let models = inputs.models.clone();
        let context_files = inputs.context_files.clone();
        let resumed = crate::sessions::resume(&inputs.agent_dir, &inputs.cwd).unwrap_or_default();

        // Ask gate: route the model's `ask` tool to a TUI dialog.
        let (ask_tx, ask_rx) = mpsc::unbounded_channel::<AskRequest>();
        let ask_gate: Arc<dyn agent_m_agent::AskGate> = Arc::new(
            agent_m_agent::ClosureAskGate::new(move |question, options| {
                let ask_tx = ask_tx.clone();
                let (response_tx, response_rx) = oneshot::channel();
                let _ = ask_tx.send((question, options, response_tx));
                Box::pin(async move {
                    response_rx
                        .await
                        .map_err(|_| "ask channel closed".to_string())?
                })
            }),
        );

        // Clones the runner needs for `RunFlow` (a flow gets its own fresh
        // agents, so we hand it the provider + options + gates).
        let flow_provider = inputs.provider.clone();
        let flow_options = inputs.agent_options.clone();
        let flow_gate = gate.clone();
        let flow_ask_gate = ask_gate.clone();
        let flow_tools = inputs.agent_options.tools.clone();
        let flow_cwd = inputs.cwd.clone();
        let flow_state_dir = inputs.agent_dir.join("flows");
        let compact_threshold = inputs.compact_threshold;

        let mut agent = Agent::new(
            inputs.provider.clone(),
            AgentOptions {
                ask_gate: Some(ask_gate),
                ..inputs.agent_options.clone()
            },
            gate.clone(),
        );
        agent.restore_messages(resumed);
        let event_tx_runner = event_tx.clone();
        // Forward live flow step progress to the UI (index, name, status).
        let flow_on_progress = {
            let event_tx = event_tx.clone();
            Arc::new(move |progress: agent_m_flow::FlowProgress| {
                let _ = event_tx.send(AgentEvent::FlowStep {
                    index: progress.step_index,
                    name: progress.step_name,
                    status: progress.status.as_str().to_string(),
                });
            })
        };
        agent.subscribe(move |event| {
            let _ = event_tx.send(event.clone());
        });

        let mut items = Vec::new();
        for message in agent.messages() {
            push_message_item(&mut items, message);
        }
        if !resumed_todos.is_empty() {
            items.push(TranscriptItem::Plan {
                todos: resumed_todos.clone(),
            });
        }
        // Resumed history is all past turns: it starts collapsed.
        let collapsed_before = items.len();

        let mut app = App {
            inputs,
            theme: Theme::dark(),
            ui_mode: UiMode::Regular,
            show_cache_notices: false,
            cwd: PathBuf::new(),
            items,
            collapsed_before,
            editor: Editor::new(),
            editor_view_top: 0,
            completion: None,
            completion_index: 0,
            streaming: false,
            cache_hit: 0,
            cache_miss: 0,
            cache_requests: 0,
            current_model,
            mode: initial_mode,
            last_error: None,
            follow_end: true,
            scroll_back: 0,
            models,
            context_files,
            provider_id,
            session_path,
            usage: UsageTotals::default(),
            last_input: 0,
            question_pending: false,
            todos: resumed_todos,
            plan_choice_pending: false,
            flow_run: None,
            show_sidebar: true,
            active_tool: None,
            last_trust: agent_m_ai::TrustData::default(),
            journal_open: false,
            level_handle,
            session_stem,
            undo_stack,
            model_picker_open: false,
            picker_index: 0,
            info_open: false,
            risk,
            pending_approval: None,
            pending_ask: None,
            ask_rx,
            submit_tx,
            event_rx,
            approval_rx,
            session_store: Some(session_store),
            should_exit: false,
            last_spinner: Instant::now(),
        };
        app.theme = app.inputs.theme.clone();
        app.ui_mode = app.inputs.ui_mode;
        app.show_cache_notices = app.inputs.show_cache_notices;
        app.cwd = app.inputs.cwd.clone();

        // The runner task owns the agent for the app's lifetime; dropping the
        // JoinHandle detaches the task, which is what we want here.
        tokio::spawn(async move {
            while let Some(command) = submit_rx.recv().await {
                match command {
                    SubmitCommand::Prompt(text) => {
                        let _ = agent.prompt(text).await;
                        // Strategic compaction (ECC): only at turn boundaries,
                        // once usage passes the threshold — never mid-run.
                        let (tokens, window) = agent.context_usage();
                        if let Some(window) = window
                            && agent_m_flow::should_compact(tokens, window, compact_threshold)
                        {
                            match agent.summarize_and_compact(10).await {
                                Ok(summary) if !summary.is_empty() => {
                                    let _ = event_tx_runner.send(AgentEvent::Notice {
                                        message: format!(
                                            "auto-compacted at the turn boundary ({}% of window) — {summary}",
                                            tokens * 100 / window
                                        ),
                                    });
                                }
                                _ => {}
                            }
                        }
                    }
                    SubmitCommand::Interrupt => agent.interrupt(),
                    SubmitCommand::SetModel(model) => agent.set_model(model),
                    SubmitCommand::SetMode(mode) => agent.set_mode(mode),
                    SubmitCommand::Compact => {
                        match agent.summarize_and_compact(10).await {
                            Ok(summary) if summary.is_empty() => {
                                let _ = event_tx_runner.send(AgentEvent::Notice {
                                    message: "nothing to compact yet".to_string(),
                                });
                            }
                            Ok(_) => {} // the agent emits Compacted itself
                            Err(error) => {
                                let _ = event_tx_runner.send(AgentEvent::Notice {
                                    message: format!("compaction failed: {error}"),
                                });
                            }
                        }
                    }
                    SubmitCommand::RunFlow(path) => {
                        let flow = match agent_m_flow::load_flow(&path) {
                            Ok(flow) => flow,
                            Err(error) => {
                                let _ = event_tx_runner.send(AgentEvent::Notice {
                                    message: format!("flow error: {error}"),
                                });
                                continue;
                            }
                        };
                        let mut options = flow_options.clone();
                        // Flows run in build mode with the session's tools.
                        options.tools = flow_tools.clone();
                        let deps = agent_m_flow::FlowDeps {
                            provider: flow_provider.clone(),
                            agent_options: options,
                            tools: flow_tools.clone(),
                            permission_gate: flow_gate.clone(),
                            ask_gate: Some(flow_ask_gate.clone()),
                            state_dir: Some(flow_state_dir.clone()),
                            on_progress: Some(flow_on_progress.clone()),
                        };
                        let mut context = agent_m_flow::FlowContext::new();
                        context.set("cwd", serde_json::json!(flow_cwd.to_string_lossy()));
                        let _ = event_tx_runner.send(AgentEvent::Notice {
                            message: format!("running flow `{}`", flow.name),
                        });
                        // Seed the sidebar with all steps as pending.
                        for (index, step) in flow.steps.iter().enumerate() {
                            let _ = event_tx_runner.send(AgentEvent::FlowStep {
                                index,
                                name: step.name().to_string(),
                                status: "pending".to_string(),
                            });
                        }
                        match agent_m_flow::run_flow(&flow, &mut context, &deps).await {
                            Ok(run) => {
                                for step in &run.steps {
                                    let _ = event_tx_runner.send(AgentEvent::Notice {
                                        message: format!(
                                            "[{}] {}",
                                            step.status.as_str(),
                                            step.name
                                        ),
                                    });
                                }
                                let failed = run
                                    .steps
                                    .iter()
                                    .any(|s| s.status == agent_m_flow::StepStatus::Failed);
                                let _ = event_tx_runner.send(AgentEvent::Notice {
                                    message: format!(
                                        "flow `{}`: {}",
                                        flow.name,
                                        if failed { "FAILED" } else { "OK" }
                                    ),
                                });
                            }
                            Err(error) => {
                                let _ = event_tx_runner.send(AgentEvent::Notice {
                                    message: format!("flow failed: {error}"),
                                });
                            }
                        }
                    }
                }
            }
        });
        Ok(app)
    }

    /// Run the TUI loop.
    pub async fn run(inputs: AppInputs) -> Result<()> {
        let mut app = App::new(inputs)?;
        app.run_loop().await
    }

    async fn run_loop(&mut self) -> Result<()> {
        let mut stdout = std::io::stdout();
        let mut terminal = if self.ui_mode == UiMode::Fullscreen {
            execute!(stdout, EnterAlternateScreen)?;
            Terminal::new(CrosstermBackend::new(stdout))?
        } else {
            Terminal::new(CrosstermBackend::new(stdout))?
        };
        crossterm_terminal::enable_raw_mode()?;
        // Without this, wheel scroll goes to the host terminal instead of the
        // app — in regular mode that reveals raw scrollback underneath us.
        execute!(terminal.backend_mut(), EnableMouseCapture)?;

        let result = self.event_loop(&mut terminal).await;

        execute!(terminal.backend_mut(), DisableMouseCapture)?;
        crossterm_terminal::disable_raw_mode()?;
        if self.ui_mode == UiMode::Fullscreen {
            execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        }
        terminal.show_cursor()?;
        result
    }

    async fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ) -> Result<()> {
        let mut first_frame = true;
        loop {
            let mut changed = self.drain_channels();

            let poll_timeout = if self.streaming || self.pending_approval.is_some() {
                Duration::from_millis(80)
            } else {
                Duration::from_millis(200)
            };
            if event::poll(poll_timeout)? {
                match event::read()? {
                    Event::Key(key) => {
                        self.handle_key(key).await;
                        changed = true;
                    }
                    Event::Mouse(mouse) => {
                        match mouse.kind {
                            MouseEventKind::ScrollUp => {
                                self.apply_app_action(AppAction::ScrollUp).await
                            }
                            MouseEventKind::ScrollDown => {
                                self.apply_app_action(AppAction::ScrollDown).await
                            }
                            _ => {}
                        }
                        changed = true;
                    }
                    Event::Resize(_, _) => changed = true,
                    _ => {}
                }
            }

            // Draw on change, while streaming/approval-pending (spinner), and
            // once at startup so the initial frame renders.
            if first_frame || changed || self.streaming || self.pending_approval.is_some() {
                terminal.draw(|frame| self.draw(frame))?;
                first_frame = false;
            }

            if self.should_exit {
                break;
            }
        }
        Ok(())
    }

    /// Pull agent events and approval requests into app state; returns true if
    /// anything changed.
    fn drain_channels(&mut self) -> bool {
        let mut changed = false;
        while let Ok(event) = self.event_rx.try_recv() {
            self.handle_agent_event(event);
            changed = true;
        }
        while let Ok(request) = self.approval_rx.try_recv() {
            if self.pending_approval.is_none() {
                self.pending_approval = Some(request);
            } else {
                let (_, response) = request;
                let _ = response.send(Permission::Denied(
                    "previous decision still pending".to_string(),
                ));
            }
            changed = true;
        }
        while let Ok(request) = self.ask_rx.try_recv() {
            if self.pending_ask.is_none() {
                self.pending_ask = Some(request);
            } else {
                let (_, _, response) = request;
                let _ = response.send(Err("another question is already pending".to_string()));
            }
            changed = true;
        }
        changed
    }

    async fn handle_key(&mut self, key: KeyEvent) {
        // The model is asking a question: enter answers, escape cancels.
        if self.pending_ask.is_some() {
            match key.code {
                KeyCode::Enter => {
                    let answer = self.editor.text();
                    self.editor.set_text("");
                    self.editor_view_top = 0;
                    if let Some((_, _, response)) = self.pending_ask.take() {
                        let _ = response.send(Ok(answer));
                    }
                    return;
                }
                KeyCode::Esc => {
                    if let Some((_, _, response)) = self.pending_ask.take() {
                        let _ = response.send(Err("cancelled by user".to_string()));
                    }
                    return;
                }
                _ => {}
            }
        }
        // The plan is ready: e/s/r decide what happens next. Only when the
        // editor is empty so normal typing/messages are never intercepted,
        // and not for ctrl+r (toggle thinking) which shares the 'r' key.
        if self.plan_choice_pending
            && self.editor.text().is_empty()
            && !key.modifiers.contains(KeyModifiers::CONTROL)
        {
            match key.code {
                KeyCode::Char('e' | 'E') => {
                    self.execute_plan();
                    return;
                }
                KeyCode::Char('s' | 'S') => {
                    self.plan_choice_pending = false;
                    return;
                }
                KeyCode::Char('r' | 'R') => {
                    self.refine_plan();
                    return;
                }
                _ => {}
            }
        }
        // The model picker consumes navigation keys while it is open.
        if self.model_picker_open {
            match key.code {
                KeyCode::Up => {
                    self.picker_index = self.picker_index.saturating_sub(1);
                    return;
                }
                KeyCode::Down => {
                    self.picker_index =
                        (self.picker_index + 1).min(self.models.len().saturating_sub(1));
                    return;
                }
                KeyCode::Enter => {
                    if let Some(spec) = self.models.get(self.picker_index) {
                        self.select_model(spec.id.clone());
                    }
                    self.model_picker_open = false;
                    return;
                }
                KeyCode::Esc => {
                    self.model_picker_open = false;
                    return;
                }
                _ => {}
            }
        }
        let context = keybindings::KeyContext {
            editor_empty: self.editor.is_empty(),
            approval_pending: self.pending_approval.is_some(),
        };
        let Some(action) = keybindings::resolve_key(key, context) else {
            return;
        };
        match action {
            Action::Editor(editor_action) => {
                self.apply_editor_action(editor_action);
            }
            Action::App(app_action) => self.apply_app_action(app_action).await,
        }
    }

    /// Switch the agent to a model and close any picker.
    fn select_model(&mut self, model: String) {
        let _ = self.submit_tx.send(SubmitCommand::SetModel(model.clone()));
        self.current_model = model;
        self.push_notice(format!("model set to {}", self.current_model));
    }

    /// Execute the approved plan: flip to build mode and send the plan as a
    /// follow-up prompt that tracks completion with `[DONE:n]` markers.
    /// Seal the transcript boundary: everything currently in `items` becomes
    /// collapsible; new items added by the action about to run stay
    /// expanded until the *next* sealed action.
    fn seal_turn(&mut self) {
        self.collapsed_before = self.items.len();
    }

    fn execute_plan(&mut self) {
        let plan_text: String = self
            .todos
            .iter()
            .map(|todo| format!("{}. {}", todo.step, todo.text))
            .collect::<Vec<_>>()
            .join("\n");
        self.plan_choice_pending = false;
        self.mode = agent_m_agent::Mode::Build;
        let _ = self
            .submit_tx
            .send(SubmitCommand::SetMode(agent_m_agent::Mode::Build));
        let prompt = format!(
            "Execute the plan. Mark each step with [DONE:n] when finished.\nPlan:\n{plan_text}"
        );
        self.streaming = true;
        self.last_error = None;
        self.seal_turn();
        let _ = self.submit_tx.send(SubmitCommand::Prompt(prompt));
        self.push_notice("executing the plan — mark steps with [DONE:n]");
    }

    /// Refine the plan: clear the current todos and ask the model (still in
    /// plan mode) to rewrite it.
    fn refine_plan(&mut self) {
        self.plan_choice_pending = false;
        self.todos.clear();
        // Also clear the persisted plan so a restart doesn't reload stale
        // todos while the model rewrites the plan.
        let _ =
            crate::sessions::save_todos(&self.inputs.agent_dir, &self.session_stem, &self.todos);
        self.items
            .retain(|item| !matches!(item, TranscriptItem::Plan { .. }));
        self.streaming = true;
        self.last_error = None;
        self.seal_turn();
        let _ = self.submit_tx.send(SubmitCommand::Prompt(
            "Please refine the plan above: improve the steps (keep the `Plan:` numbered format)."
                .to_string(),
        ));
    }

    fn apply_editor_action(&mut self, action: EditorAction) {
        match action {
            EditorAction::PasteText(text) => {
                self.editor.insert_text(&text);
            }
            EditorAction::MoveLeft => self.editor.move_left(),
            EditorAction::MoveRight => self.editor.move_right(),
            EditorAction::MoveUp => self.editor.move_up(),
            EditorAction::MoveDown => self.editor.move_down(),
            EditorAction::WordLeft => self.editor.word_left(),
            EditorAction::WordRight => self.editor.word_right(),
            EditorAction::LineStart => self.editor.line_start(),
            EditorAction::LineEnd => self.editor.line_end(),
            EditorAction::Backspace => self.editor.backspace(),
            EditorAction::Delete => self.editor.delete(),
            EditorAction::KillWordBackward => self.editor.kill_word_backward(),
            EditorAction::KillToStart => self.editor.kill_to_start(),
            EditorAction::KillToEnd => self.editor.kill_to_end(),
            EditorAction::Yank => self.editor.yank(),
            EditorAction::Undo => self.editor.undo(),
            EditorAction::Newline => self.editor.newline(),
        }
        self.completion = None;
    }

    async fn apply_app_action(&mut self, action: AppAction) {
        match action {
            AppAction::Submit => {
                let text = self.editor.text();
                if !text.trim().is_empty() {
                    self.submit_text(text).await;
                }
            }
            AppAction::TabComplete => self.complete(),
            AppAction::Clear => {
                self.editor.set_text("");
                self.editor_view_top = 0;
            }
            AppAction::Exit => self.should_exit = true,
            AppAction::Interrupt => {
                let _ = self.submit_tx.send(SubmitCommand::Interrupt);
            }
            AppAction::ModelSelect => {
                // Open the picker (pi/OpenCode style) instead of cycling.
                self.model_picker_open = true;
                self.picker_index = self
                    .models
                    .iter()
                    .position(|spec| spec.id == self.current_model)
                    .unwrap_or(0);
            }
            AppAction::ModelCycleForward => self.cycle_model(1),
            AppAction::ModelCycleBackward => self.cycle_model(-1),
            AppAction::ToggleInfo => self.info_open = !self.info_open,
            AppAction::ToggleToolOutput => {
                let last_tool = self.items.iter_mut().rev().find_map(|item| match item {
                    TranscriptItem::ToolExecution { .. } => Some(item),
                    _ => None,
                });
                if let Some(TranscriptItem::ToolExecution { expanded, .. }) = last_tool {
                    *expanded = !*expanded;
                }
            }
            AppAction::ToggleThinking => {
                let last_thinking = self.items.iter_mut().rev().find_map(|item| match item {
                    TranscriptItem::Assistant { parts, .. }
                        if parts
                            .iter()
                            .any(|part| matches!(part, ContentPart::Thinking { .. })) =>
                    {
                        Some(item)
                    }
                    _ => None,
                });
                if let Some(TranscriptItem::Assistant {
                    thinking_expanded, ..
                }) = last_thinking
                {
                    *thinking_expanded = !*thinking_expanded;
                }
            }
            AppAction::ToggleCacheNotices => {
                self.show_cache_notices = !self.show_cache_notices;
            }
            AppAction::ApproveTool => self.respond_to_approval(Permission::Allowed),
            AppAction::DenyTool => self
                .respond_to_approval(Permission::Denied("user denied the tool call".to_string())),
            AppAction::ScrollUp => {
                self.follow_end = false;
                self.scroll_back += 5;
            }
            AppAction::ScrollDown => {
                self.scroll_back = self.scroll_back.saturating_sub(5);
                if self.scroll_back == 0 {
                    self.follow_end = true;
                }
            }
            AppAction::ScrollTop => {
                self.follow_end = false;
                self.scroll_back = usize::MAX;
            }
            AppAction::ScrollBottom => {
                self.follow_end = true;
                self.scroll_back = 0;
            }
        }
    }

    fn respond_to_approval(&mut self, permission: Permission) {
        if let Some((_, response)) = self.pending_approval.take() {
            let _ = response.send(permission);
        }
    }

    /// Parse a `Plan:` list from a completed assistant message and apply
    /// `[DONE:n]` completion markers (plan mode / execution tracking).
    fn track_plan(&mut self, message: &SessionMessage) {
        let SessionMessage::Assistant { content, .. } = message else {
            return;
        };
        let text: String = content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if self.todos.is_empty() {
            let parsed = crate::plan::parse_plan(&text);
            if !parsed.is_empty() {
                self.todos = parsed;
                self.plan_choice_pending = true;
                self.items.push(TranscriptItem::Plan {
                    todos: self.todos.clone(),
                });
                self.follow_end = true;
            }
        }
        if !self.todos.is_empty()
            && crate::plan::apply_done_markers(&text, &mut self.todos)
            && let Some(todos) = self.items.iter_mut().rev().find_map(|item| match item {
                TranscriptItem::Plan { todos } => Some(todos),
                _ => None,
            })
        {
            *todos = self.todos.clone();
        }
        if !self.todos.is_empty() {
            let _ = crate::sessions::save_todos(
                &self.inputs.agent_dir,
                &self.session_stem,
                &self.todos,
            );
        }
    }

    /// Accumulate usage/cost for a completed assistant message, using the
    /// model's per-1M pricing when the provider did not report a cost.
    fn record_usage(&mut self, message: &SessionMessage) {
        let SessionMessage::Assistant { usage, model, .. } = message else {
            return;
        };
        let Some(usage) = usage else {
            return;
        };
        let mut usage = usage.clone();
        if usage.cost == 0.0
            && let Some(spec) = self.models.iter().find(|spec| &spec.id == model)
        {
            usage.cost = spec.cost_for(&usage);
        }
        self.last_input = usage.input_tokens;
        self.usage.add(&usage);
    }

    async fn submit_text(&mut self, text: String) {
        if text.starts_with('/') {
            self.run_slash_command(&text);
            return;
        }
        if let Some(command) = text.strip_prefix('!') {
            let command = command.trim().to_string();
            if !command.is_empty() {
                crate::prefs::record_command(&self.inputs.agent_dir, &command);
                self.run_inline_bash(command).await;
            }
            return;
        }
        self.editor.set_text("");
        self.editor_view_top = 0;
        self.streaming = true;
        self.last_error = None;
        self.question_pending = false;
        self.plan_choice_pending = false;
        self.seal_turn();
        let _ = self.submit_tx.send(SubmitCommand::Prompt(text));
    }

    /// Execute bash directly (pi's `!cmd`), showing the result as a tool block.
    async fn run_inline_bash(&mut self, command: String) {
        self.seal_turn();
        self.editor.set_text("");
        let tool_call_id = uuid::Uuid::now_v7().simple().to_string();
        self.items.push(TranscriptItem::ToolExecution {
            tool_call_id: tool_call_id.clone(),
            name: "bash".to_string(),
            arguments: serde_json::json!({ "command": command.clone() }),
            result: None,
            expanded: false,
        });
        self.follow_end = true;

        let context = ToolContext {
            cwd: self.cwd.clone(),
            ask_gate: None,
        };
        let outcome = BashTool
            .execute(serde_json::json!({ "command": command }), &context)
            .await
            .map_err(|error| agent_m_agent::ToolOutcome::error(error.to_string()))
            .unwrap_or_else(|outcome| outcome);

        for item in self.items.iter_mut().rev() {
            if let TranscriptItem::ToolExecution {
                tool_call_id: id,
                result,
                ..
            } = item
                && *id == tool_call_id
            {
                *result = Some(outcome);
                break;
            }
        }
    }

    fn run_slash_command(&mut self, input: &str) {
        let mut parts = input.split_whitespace();
        let command = parts.next().unwrap_or_default();
        let argument = parts.next().unwrap_or_default();
        match command {
            "/help" => self.push_notice(
                "agent-m help\n\nSlash commands: /help /hotkeys /clear /exit /quit /model /new /settings /cache /info /plan /build /todos /context /compact /journal /undo /level <0-4>\n!command runs bash directly. Type a prompt and press Enter to chat.",
            ),
            "/hotkeys" => self.push_notice(
                "enter submit · shift+enter/ctrl+j newline · tab autocomplete\nctrl+c clear (empty: exit) · ctrl+d exit · escape interrupt\nctrl+l model select · ctrl+o tool output · ctrl+r toggle thinking · ctrl+n info · ctrl+p/ctrl+shift+p model cycle\nctrl+a/e line · ctrl+b/f word · ctrl+w/u/k kill · ctrl+y yank · ctrl+- undo\npageUp/pageDown/mouse wheel scroll · ctrl+t toggle cache notices",
            ),
            "/clear" => {
                self.items.clear();
                self.collapsed_before = 0;
            }
            "/exit" | "/quit" => self.should_exit = true,
            "/plan" => {
                let _ = self.submit_tx.send(SubmitCommand::SetMode(agent_m_agent::Mode::Plan));
                self.mode = agent_m_agent::Mode::Plan;
                self.push_notice(
                    "plan mode: read-only — ask the agent to plan a task (Plan: numbered list)",
                );
            }
            "/compact" => {
                self.push_notice("compacting conversation… (older messages → summary)");
                let _ = self.submit_tx.send(SubmitCommand::Compact);
            }
            "/flow" => {
                // "/flow <path>" — the path is the whole rest of the line.
                let path = input.trim_start_matches("/flow").trim();
                if path.is_empty() {
                    self.push_notice("usage: /flow <path-to-flow.yml>");
                } else {
                    let _ = self
                        .submit_tx
                        .send(SubmitCommand::RunFlow(std::path::PathBuf::from(path)));
                }
            }
            "/level" => {
                // "/level <0-4>" (the second whitespace token is the argument)
                let arg = if argument.is_empty() {
                    command.trim_start_matches("/level").trim()
                } else {
                    argument
                };
                if let Ok(number) = arg.parse::<u8>()
                    && let Some(level) = agent_m_agent::AutonomyLevel::from_number(number)
                {
                    self.level_handle
                        .store(number, std::sync::atomic::Ordering::Relaxed);
                    if let Ok(mut settings) = std::fs::read_to_string(
                        self.inputs.agent_dir.join("settings.json"),
                    )
                    .map(|text| serde_json::from_str::<serde_json::Value>(&text))
                    .unwrap_or(Ok(serde_json::Value::Object(Default::default())))
                    {
                        settings["level"] = serde_json::json!(number);
                        let _ = std::fs::write(
                            self.inputs.agent_dir.join("settings.json"),
                            serde_json::to_string(&settings).unwrap_or_default(),
                        );
                    }
                    self.push_notice(format!(
                        "autonomy level {number} ({}) — {}",
                        level.label(),
                        match number {
                            0 => "observe only",
                            1 => "suggest, don't execute",
                            2 => "everything asks",
                            3 => "auto low/medium, ask high/critical",
                            _ => "auto everything except critical",
                        }
                    ));
                } else {
                    self.push_notice("usage: /level <0-4>  (0 observe · 1 suggest · 2 assisted · 3 trusted · 4 autonomous)");
                }
            }
            "/undo" => {
                self.undo_last();
            }
            "/journal" => {
                self.journal_open = !self.journal_open;
                if self.journal_open {
                    self.push_notice(format!(
                        "journal: {} entries",
                        crate::sessions::journal(&self.inputs.agent_dir, &self.cwd).len()
                    ));
                }
            }
            "/sidebar" => {
                self.show_sidebar = !self.show_sidebar;
                self.push_notice(if self.show_sidebar {
                    "sidebar shown (auto-hides under 110 columns)"
                } else {
                    "sidebar hidden"
                });
            }
            "/flows" => {
                let dir = self.inputs.agent_dir.join("flows");
                let names: Vec<String> = std::fs::read_dir(&dir)
                    .ok()
                    .map(|entries| {
                        entries
                            .flatten()
                            .filter_map(|entry| {
                                let path = entry.path();
                                if path.extension().and_then(|e| e.to_str()) == Some("yml")
                                    || path.extension().and_then(|e| e.to_str()) == Some("yaml")
                                {
                                    path.file_name()
                                        .and_then(|n| n.to_str())
                                        .map(str::to_string)
                                } else {
                                    None
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if names.is_empty() {
                    self.push_notice(format!(
                        "no flows in {} (ship flows there or use /flow <path>)",
                        dir.display()
                    ));
                } else {
                    self.push_notice(format!("flows: {}", names.join(", ")));
                }
            }
            "/context" => {
                let window = self
                    .models
                    .iter()
                    .find(|spec| spec.id == self.current_model)
                    .and_then(|spec| spec.context_window)
                    .unwrap_or(128_000);
                let percent = self.last_input.checked_mul(100).and_then(|n| n.checked_div(window)).unwrap_or(0);
                self.push_notice(format!(
                    "context: {} / {} tokens ({}%) · reserve 16k · /compact summarizes older messages",
                    fmt_tokens(self.last_input),
                    fmt_tokens(window),
                    percent
                ));
            }
            "/todos" => {
                if self.todos.is_empty() {
                    self.push_notice("no plan yet — ask for one in plan mode");
                } else {
                    let lines: Vec<String> = self
                        .todos
                        .iter()
                        .map(|todo| {
                            format!(
                                "{}{}. {}",
                                if todo.completed { "✓ " } else { "  " },
                                todo.step,
                                todo.text
                            )
                        })
                        .collect();
                    self.push_notice(format!("📋 Plan ({}/{})\n{}", self.todos.iter().filter(|t| t.completed).count(), self.todos.len(), lines.join("\n")));
                }
            }
            "/build" => {
                let _ = self.submit_tx.send(SubmitCommand::SetMode(agent_m_agent::Mode::Build));
                self.mode = agent_m_agent::Mode::Build;
                self.push_notice("build mode: tools enabled");
            }
            "/model" => {
                if argument.is_empty() {
                    // Open the model picker.
                    self.model_picker_open = true;
                    self.picker_index = self
                        .models
                        .iter()
                        .position(|spec| spec.id == self.current_model)
                        .unwrap_or(0);
                } else {
                    let _ = self.submit_tx.send(SubmitCommand::SetModel(argument.to_string()));
                    self.current_model = argument.to_string();
                    self.push_notice(format!("model set to {argument}"));
                }
            }
            "/info" => self.info_open = !self.info_open,
            "/new" => {
                self.items.clear();
                self.collapsed_before = 0;
                self.todos.clear();
                let _ = crate::sessions::save_todos(
                    &self.inputs.agent_dir,
                    &self.session_stem,
                    &[],
                );
                self.push_notice("new session (transcript cleared)");
            }
            "/settings" => self.push_notice(format!(
                "ui-mode: {:?} · cache notices: {} · theme: {}",
                self.ui_mode,
                if self.show_cache_notices { "on" } else { "off" },
                self.theme.name
            )),
            "/cache" => {
                self.show_cache_notices = !self.show_cache_notices;
                self.push_notice(format!(
                    "cache notices: {}",
                    if self.show_cache_notices { "on" } else { "off" }
                ));
            }
            _ => self.push_notice(format!("unknown command {command}; /help for the list")),
        }
        self.editor.set_text("");
    }

    fn push_notice(&mut self, message: impl Into<String>) {
        self.items.push(TranscriptItem::Notice {
            message: message.into(),
        });
        self.follow_end = true;
    }

    fn cycle_model(&mut self, direction: i32) {
        let models: Vec<String> = self
            .inputs
            .provider
            .models()
            .iter()
            .map(|model| model.id.clone())
            .collect();
        if models.is_empty() {
            return;
        }
        let current = self.current_model.clone();
        let position = models
            .iter()
            .position(|model| *model == current)
            .unwrap_or(0);
        let next = (position as i64 + direction as i64).rem_euclid(models.len() as i64) as usize;
        let model = models[next].clone();
        self.current_model = model.clone();
        let _ = self.submit_tx.send(SubmitCommand::SetModel(model.clone()));
        self.push_notice(format!("model: {model}"));
    }

    fn complete(&mut self) {
        let text = self.editor.text();
        let prefix = text.trim_start();
        let candidates: Vec<String> = if prefix.starts_with('/') {
            SLASH_COMMANDS
                .iter()
                .filter(|command| command.starts_with(prefix))
                .map(|command| command.to_string())
                .collect()
        } else if text.is_empty() {
            Vec::new()
        } else {
            let mut files = Vec::new();
            if let Ok(entries) = std::fs::read_dir(&self.cwd) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with(prefix) {
                        files.push(name);
                    }
                }
            }
            files.sort();
            files
        };

        match candidates.len() {
            0 => {}
            1 => {
                self.editor.set_text(&candidates[0]);
                self.completion = Some(candidates[0].clone());
            }
            _ => {
                self.completion_index = (self.completion_index + 1) % candidates.len();
                let completion = candidates[self.completion_index].clone();
                self.editor.set_text(&completion);
                self.completion = Some(completion);
            }
        }
    }

    fn editor_height_hint(&self) -> usize {
        (self.editor.line_count() + 2).clamp(3, 10)
    }

    /// Update the flow sidebar from one `FlowStep` event (delegates to the
    /// pure [`apply_flow_step_state`]).
    fn apply_flow_step(&mut self, index: usize, name: String, status: String) {
        apply_flow_step_state(&mut self.flow_run, index, name, status);
    }

    /// Principle 8: before a write/edit executes, snapshot the target file so
    /// `/undo` can restore it (skips files > 10 MB).
    fn snapshot_for_undo(&mut self, name: &str, arguments: &serde_json::Value) {
        if !matches!(name, "write" | "edit") {
            return;
        }
        let Some(path) = arguments.get("path").and_then(serde_json::Value::as_str) else {
            return;
        };
        let target = if std::path::Path::new(path).is_absolute() {
            std::path::PathBuf::from(path)
        } else {
            self.cwd.join(path)
        };
        let before = std::fs::read(&target)
            .ok()
            .filter(|bytes| bytes.len() <= 10 * 1024 * 1024)
            .map(|bytes| String::from_utf8_lossy(&bytes).to_string());
        self.undo_stack.push(crate::sessions::UndoEntry {
            path: target.to_string_lossy().to_string(),
            before,
        });
        let _ = crate::sessions::save_undo(
            &self.inputs.agent_dir,
            &self.session_stem,
            &self.undo_stack,
        );
    }

    /// Pop the most recent snapshot and restore the file (or delete it when it
    /// did not exist before).
    fn undo_last(&mut self) {
        crate::prefs::record_undo(&self.inputs.agent_dir);
        let Some(entry) = self.undo_stack.pop() else {
            self.push_notice("nothing to undo");
            return;
        };
        match crate::sessions::apply_undo(&entry) {
            Ok("restored") => self.push_notice(format!(
                "restored {} (undo stack has {} left)",
                entry.path,
                self.undo_stack.len()
            )),
            Ok(_) => self.push_notice(format!(
                "deleted {} (it did not exist before; {} undo entries left)",
                entry.path,
                self.undo_stack.len()
            )),
            Err(error) => self.push_notice(format!("undo failed: {error}")),
        }
        let _ = crate::sessions::save_undo(
            &self.inputs.agent_dir,
            &self.session_stem,
            &self.undo_stack,
        );
    }

    fn handle_agent_event(&mut self, event: AgentEvent) {
        match event {
            AgentEvent::MessageEnd {
                message: message @ SessionMessage::User { .. },
            } => {
                push_message_item(&mut self.items, &message);
                if let Some(store) = &mut self.session_store {
                    let _ = store.append(&message);
                }
                self.follow_end = true;
            }
            AgentEvent::MessageStart { kind } => {
                // Create the assistant item up front so streamed deltas render
                // live; MessageEnd replaces it with the final message.
                if kind == agent_m_agent::SessionMessageKind::Assistant {
                    self.items.push(TranscriptItem::Assistant {
                        parts: Vec::new(),
                        stop_reason: StopReason::Pending,
                        thinking_expanded: false,
                        trust: TrustData::default(),
                    });
                }
            }
            AgentEvent::MessageEnd {
                message: message @ SessionMessage::Assistant { .. },
            } => {
                if let SessionMessage::Assistant { trust, .. } = &message {
                    self.last_trust = trust.clone();
                }
                self.streaming = false;
                // Replace the streaming placeholder (tools start only after the
                // assistant message ends, so it is still the last item).
                if matches!(self.items.last(), Some(TranscriptItem::Assistant { .. })) {
                    self.items.pop();
                }
                // Do not persist aborted replies: their partial tool-call JSON
                // would break the byte-stable prefix on resume.
                let aborted = matches!(
                    &message,
                    SessionMessage::Assistant {
                        stop_reason: StopReason::Aborted,
                        ..
                    }
                );
                push_message_item(&mut self.items, &message);
                if !aborted && let Some(store) = &mut self.session_store {
                    let _ = store.append(&message);
                }
                self.follow_end = true;
                self.record_usage(&message);
                self.question_pending = message_ends_with_question(&message);
                // Don't parse plans from interrupted/aborted partial replies.
                if !aborted {
                    self.track_plan(&message);
                }
            }
            AgentEvent::MessageUpdate { delta } => self.apply_delta(delta),
            AgentEvent::ToolExecutionStart {
                tool_call_id,
                name,
                arguments,
            } => {
                self.active_tool = Some(crate::transcript::narration(&name, &arguments));
                self.snapshot_for_undo(&name, &arguments);
                self.items.push(TranscriptItem::ToolExecution {
                    tool_call_id,
                    name,
                    arguments,
                    result: None,
                    expanded: false,
                });
                self.follow_end = true;
            }
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                outcome,
            } => {
                self.active_tool = None;
                for item in self.items.iter_mut().rev() {
                    if let TranscriptItem::ToolExecution {
                        tool_call_id: id,
                        result,
                        ..
                    } = item
                        && *id == tool_call_id
                    {
                        *result = Some(outcome);
                        break;
                    }
                }
            }
            AgentEvent::FlowStep {
                index,
                name,
                status,
            } => {
                self.apply_flow_step(index, name, status);
            }
            AgentEvent::Notice { message } => {
                // The flow's name rides on the "running flow `x`" notice.
                if let Some(name) = message
                    .strip_prefix("running flow `")
                    .and_then(|rest| rest.strip_suffix('`'))
                    && let Some(view) = self.flow_run.as_mut()
                {
                    view.name = name.to_string();
                }
                self.items.push(TranscriptItem::Notice {
                    message: message.clone(),
                });
                self.last_error = Some(message);
            }
            AgentEvent::AgentEnd { cache_stats, .. } => {
                self.streaming = false;
                self.cache_hit = cache_stats.hit_tokens;
                self.cache_miss = cache_stats.miss_tokens;
                self.cache_requests = cache_stats.requests;
            }
            AgentEvent::AgentStart
            | AgentEvent::TurnStart { .. }
            | AgentEvent::TurnEnd { .. }
            | AgentEvent::MessageEnd {
                message: SessionMessage::ToolResult { .. },
            }
            | AgentEvent::MessageEnd {
                message: SessionMessage::Summary { .. },
            } => {}
            AgentEvent::Compacted {
                summary,
                messages_removed,
            } => {
                // Persist the summary as a session entry so compaction memory
                // survives restarts (the resume path already reads "summary").
                let summary_message = SessionMessage::Summary {
                    text: summary.clone(),
                };
                if let Some(store) = &mut self.session_store {
                    let _ = store.append(&summary_message);
                }
                self.items.push(TranscriptItem::Notice {
                    message: format!(
                        "📄 compacted {messages_removed} messages — session summary:\n{summary}"
                    ),
                });
                self.follow_end = true;
            }
        }
    }

    fn apply_delta(&mut self, delta: agent_m_ai::StreamEvent) {
        match delta {
            agent_m_ai::StreamEvent::TextDelta { delta } => {
                if let Some(TranscriptItem::Assistant { parts, .. }) = self.items.last_mut() {
                    let last_text = parts.iter_mut().rev().find_map(|part| match part {
                        ContentPart::Text { text } => Some(text),
                        _ => None,
                    });
                    match last_text {
                        Some(text) => text.push_str(&delta),
                        None => parts.push(ContentPart::Text {
                            text: delta.clone(),
                        }),
                    }
                }
                self.follow_end = true;
            }
            agent_m_ai::StreamEvent::ThinkingDelta { delta } => {
                if let Some(TranscriptItem::Assistant { parts, .. }) = self.items.last_mut() {
                    let last_thinking = parts.iter_mut().rev().find_map(|part| match part {
                        ContentPart::Thinking { thinking } => Some(thinking),
                        _ => None,
                    });
                    match last_thinking {
                        Some(thinking) => thinking.push_str(&delta),
                        None => parts.push(ContentPart::Thinking {
                            thinking: delta.clone(),
                        }),
                    }
                }
                self.follow_end = true;
            }
            _ => {}
        }
    }

    // ------------------------------------------------------------------
    // Drawing
    // ------------------------------------------------------------------

    fn draw(&mut self, frame: &mut Frame) {
        let sidebar_visible = self.show_sidebar && frame.area().width >= SIDEBAR_MIN_WIDTH;
        let (main_area, sidebar_area) = if sidebar_visible {
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Min(0), Constraint::Length(SIDEBAR_WIDTH)])
                .split(frame.area());
            (chunks[0], Some(chunks[1]))
        } else {
            (frame.area(), None)
        };
        let areas = self.layout(main_area);
        self.draw_transcript(frame, areas[0]);
        self.draw_status(frame, areas[1]);
        self.draw_editor(frame, areas[2]);
        self.draw_footer(frame, areas[3]);
        if let Some(sidebar_area) = sidebar_area {
            self.draw_sidebar(frame, sidebar_area);
        }
        self.draw_overlay(frame);
    }

    fn layout(&self, area: Rect) -> Vec<Rect> {
        let editor_height = self.editor_height_hint();
        // The approval/ask prompts get a multi-line box when pending.
        let status_height = if self.pending_approval.is_some() || self.pending_ask.is_some() {
            5
        } else {
            1
        };
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),
                Constraint::Length(status_height),
                Constraint::Length(editor_height as u16),
                Constraint::Length(2),
            ])
            .split(area)
            .to_vec()
    }

    fn transcript_lines(&self, width: usize) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        for (index, item) in self.items.iter().enumerate() {
            lines.extend(item.render(&self.theme, width, index < self.collapsed_before));
        }
        lines
    }

    fn draw_transcript(&mut self, frame: &mut Frame, area: Rect) {
        if area.height == 0 {
            return;
        }
        let width = area.width.max(10) as usize;
        let lines = self.transcript_lines(width);
        let total: usize = self
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| item.height(&self.theme, width, index < self.collapsed_before))
            .sum();
        let viewport = area.height as usize;
        let top = if self.follow_end {
            total.saturating_sub(viewport)
        } else {
            total
                .saturating_sub(viewport)
                .saturating_sub(self.scroll_back.min(total))
        };

        let paragraph = Paragraph::new(lines)
            .scroll((top as u16, 0))
            .wrap(Wrap { trim: false });
        frame.render_widget(paragraph, area);
    }

    fn draw_status(&mut self, frame: &mut Frame, area: Rect) {
        let spinner_index =
            (self.last_spinner.elapsed().as_millis() / 100) as usize % SPINNER.len();
        let spinner = SPINNER[spinner_index];
        let cache = if self.show_cache_notices {
            format!(
                " · cache {}% hit ({}k/{})",
                self.cache_ratio(),
                self.cache_hit / 1000,
                self.cache_hit + self.cache_miss
            )
        } else {
            String::new()
        };
        if let Some((call, _)) = &self.pending_approval {
            // Multi-line approval prompt: title, the command/arguments, hint.
            let name = call.name.clone();
            let detail = if name == "bash" {
                format!(
                    "$ {}",
                    call.arguments
                        .get("command")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("")
                )
            } else {
                serde_json::to_string(&call.arguments).unwrap_or_default()
            };
            let assessment = self.risk.assess(call);
            let (badge, color) = match assessment.level {
                agent_m_agent::RiskLevel::High => ("⚠️ HIGH", self.theme.warning),
                agent_m_agent::RiskLevel::Critical => ("🔴 CRITICAL", self.theme.error),
                _ => ("⚠️", self.theme.warning),
            };
            let title = match assessment.reason {
                Some(reason) => format!("{badge} Approve tool call: {name} ({reason})"),
                None => format!("{badge} Approve tool call: {name}"),
            };
            let mut lines = vec![Line::from(Span::styled(
                title,
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ))];
            // Consequence framing (principle 6): "This will …".
            if let Some(consequence) = self.risk.consequence(call) {
                lines.push(Line::from(Span::styled(
                    consequence,
                    Style::default().fg(self.theme.warning),
                )));
            }
            let width = area.width.max(20) as usize;
            for wrapped in wrap_lines(&detail, width.saturating_sub(2)) {
                if lines.len() >= 4 {
                    break;
                }
                lines.push(Line::from(Span::styled(
                    wrapped,
                    Style::default().fg(self.theme.user_message_text),
                )));
            }
            lines.push(Line::from(Span::styled(
                "[y] approve   [n] deny   (esc) deny",
                Style::default().fg(self.theme.dim),
            )));
            frame.render_widget(Paragraph::new(lines), area);
            return;
        }
        if let Some((question, options, _)) = &self.pending_ask {
            // The model asked a question: show it and let the user answer in
            // the editor (enter sends, escape cancels).
            let mut text = format!("❓ {question}");
            if let Some(options) = options {
                for (index, option) in options.iter().enumerate() {
                    text.push_str(&format!("  {}. {option}", index + 1));
                }
            }
            let mut lines = vec![Line::from(Span::styled(
                text,
                Style::default()
                    .fg(self.theme.accent)
                    .add_modifier(Modifier::BOLD),
            ))];
            let width = area.width.max(20) as usize;
            for wrapped in wrap_lines(
                "type your answer below and press enter (esc to cancel)",
                width.saturating_sub(2),
            ) {
                if lines.len() >= 4 {
                    break;
                }
                lines.push(Line::from(Span::styled(
                    wrapped,
                    Style::default().fg(self.theme.dim),
                )));
            }
            frame.render_widget(Paragraph::new(lines), area);
            return;
        }
        let (content, style) = if self.streaming {
            let text = match &self.active_tool {
                Some(narration) => format!("{narration} {spinner}{cache}"),
                None => format!("Working… {spinner}{cache}"),
            };
            (text, Style::default().fg(self.theme.accent))
        } else if let Some(error) = &self.last_error {
            (error.clone(), Style::default().fg(self.theme.error))
        } else if self.question_pending {
            (
                "❓ agent is asking — type your answer (enter to send)".to_string(),
                Style::default().fg(self.theme.accent),
            )
        } else if self.plan_choice_pending {
            (
                "📋 Plan ready — [e]xecute  [s]tay in plan mode  [r]efine".to_string(),
                Style::default()
                    .fg(self.theme.warning)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            // Idle status: build a Line of colored spans so the context-%
            // can be color-coded (yellow >70%, red >90%).
            let muted = Style::default().fg(self.theme.muted);
            let mode_badge = if self.mode == agent_m_agent::Mode::Plan {
                "[plan] "
            } else {
                ""
            };
            let mut spans = vec![Span::styled(
                format!(
                    "{mode_badge}{} · {} messages",
                    self.current_model,
                    self.items.len()
                ),
                muted,
            )];
            if !self.undo_stack.is_empty() {
                spans.push(Span::styled(
                    format!(" · undo: /undo ({})", self.undo_stack.len()),
                    Style::default().fg(self.theme.warning),
                ));
            }
            if self.usage.input > 0 {
                spans.push(Span::styled(
                    format!(
                        " · ↑{} ↓{} · ${:.4}",
                        fmt_tokens(self.usage.input),
                        fmt_tokens(self.usage.output),
                        self.usage.cost
                    ),
                    muted,
                ));
            }
            if let Some(window) = self
                .models
                .iter()
                .find(|spec| spec.id == self.current_model)
                .and_then(|spec| spec.context_window)
            {
                let percent = self
                    .last_input
                    .checked_mul(100)
                    .and_then(|n| n.checked_div(window))
                    .unwrap_or(0);
                let color = if percent > 90 {
                    self.theme.error
                } else if percent > 70 {
                    self.theme.warning
                } else {
                    self.theme.muted
                };
                spans.push(Span::styled(
                    format!(" · {percent}% of {}", fmt_tokens(window)),
                    Style::default().fg(color),
                ));
            }
            if !self.todos.is_empty() {
                let done = self.todos.iter().filter(|todo| todo.completed).count();
                spans.push(Span::styled(
                    format!(" · 📋 {done}/{}", self.todos.len()),
                    muted,
                ));
            }
            if !cache.is_empty() {
                spans.push(Span::styled(cache, muted));
            }
            frame.render_widget(Paragraph::new(Line::from(spans)), area);
            return;
        };
        frame.render_widget(Paragraph::new(Span::styled(content, style)), area);
    }

    fn cache_ratio(&self) -> u64 {
        let total = self.cache_hit + self.cache_miss;
        match total {
            0 => 0,
            _ => self.cache_hit * 100 / total,
        }
    }

    fn draw_editor(&mut self, frame: &mut Frame, area: Rect) {
        let inner_height = area.height.saturating_sub(1).max(1) as usize;
        let view_top = self
            .editor
            .viewport_for_cursor(self.editor_view_top, inner_height);
        self.editor_view_top = view_top;
        let (visible, cursor_line, cursor_col) = self.editor.visible_lines(view_top, inner_height);

        let block = Block::default()
            .borders(Borders::TOP)
            .title(" agent-m ")
            .title_style(Style::default().fg(self.theme.accent));
        let paragraph = Paragraph::new(
            visible
                .iter()
                .map(|line| Line::from(line.clone()))
                .collect::<Vec<_>>(),
        )
        .block(block)
        .style(Style::default().fg(self.theme.user_message_text));
        frame.render_widget(paragraph, area);

        if cursor_line < visible.len() {
            let prefix: String = visible[cursor_line].chars().take(cursor_col).collect();
            let x = area.x + UnicodeWidthStr::width(prefix.as_str()) as u16;
            let y = area.y + cursor_line as u16 + 1;
            if y < frame.area().height {
                frame.set_cursor_position((x, y));
            }
        }
    }

    fn draw_footer(&mut self, frame: &mut Frame, area: Rect) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Length(1)])
            .split(area);
        // Line 1: keybinding hints (pi/Claude style footer badges).
        frame.render_widget(
            Paragraph::new(Span::styled(
                "enter send · tab complete · ctrl+l model · ctrl+o output · ctrl+d exit · /help",
                Style::default().fg(self.theme.dim),
            )),
            rows[0],
        );
        // Line 2: working directory (home shortened to ~) and the model.
        let cwd = self.cwd.display().to_string();
        let home = dirs::home_dir()
            .map(|h| h.display().to_string())
            .unwrap_or_default();
        let path = if !home.is_empty() && cwd.starts_with(&home) {
            format!("~{}", &cwd[home.len()..])
        } else {
            cwd
        };
        let right = Span::styled(
            self.current_model.clone(),
            Style::default().fg(self.theme.accent),
        );
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(path, Style::default().fg(self.theme.dim)),
                Span::raw("  "),
                right,
            ])),
            rows[1],
        );
    }

    /// Render the centered overlays: model picker and session info.
    /// Right-side panel: live flow steps while a flow runs, the plan's task
    /// list otherwise (tasks done/pending), and session stats as the fallback.
    fn draw_sidebar(&self, frame: &mut Frame, area: Rect) {
        if area.width < 8 {
            return;
        }
        let content = self.sidebar_lines();
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.theme.muted))
            .title(Span::styled(
                " status ",
                Style::default().fg(self.theme.accent),
            ));
        let paragraph = Paragraph::new(content)
            .block(block)
            .wrap(Wrap { trim: false });
        frame.render_widget(paragraph, area);
    }

    fn sidebar_lines(&self) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        if let Some(view) = &self.flow_run {
            let (done, total) = flow_progress(view);
            let name = if view.name.is_empty() {
                "flow".to_string()
            } else {
                view.name.clone()
            };
            lines.push(Line::from(vec![
                Span::styled("⚙ ", Style::default().fg(self.theme.accent)),
                Span::styled(
                    format!("{name} ({done}/{total})"),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
            ]));
            // Progress bar.
            let filled = done
                .checked_mul(BAR_CELLS)
                .and_then(|n| n.checked_div(total))
                .unwrap_or(0);
            let bar: String = "█".repeat(filled) + &"░".repeat(BAR_CELLS - filled);
            lines.push(Line::from(Span::styled(
                bar,
                Style::default().fg(self.theme.accent),
            )));
            lines.push(Line::from(""));
            for step in &view.steps {
                let (icon, style) = match step.status.as_str() {
                    "succeeded" => ("✓", Style::default().fg(Color::Green)),
                    "running" => (
                        "▶",
                        Style::default()
                            .fg(self.theme.accent)
                            .add_modifier(Modifier::BOLD),
                    ),
                    "failed" => ("✗", Style::default().fg(self.theme.error)),
                    _ => ("○", Style::default().fg(self.theme.muted)),
                };
                lines.push(Line::from(vec![
                    Span::styled(format!("{icon} "), style),
                    Span::styled(step.name.clone(), style),
                ]));
            }
            return lines;
        }
        if !self.todos.is_empty() {
            let done = self.todos.iter().filter(|t| t.completed).count();
            lines.push(Line::from(vec![
                Span::styled("📋 ", Style::default().fg(self.theme.warning)),
                Span::styled(
                    format!("plan ({done}/{})", self.todos.len()),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
            ]));
            lines.push(Line::from(""));
            for todo in &self.todos {
                let (icon, style) = if todo.completed {
                    ("[x]", Style::default().fg(Color::Green))
                } else {
                    ("[ ]", Style::default().fg(self.theme.muted))
                };
                lines.push(Line::from(Span::styled(
                    format!("{icon} {}", todo.text),
                    style,
                )));
            }
            return lines;
        }
        // Session stats fallback so the panel is never empty.
        lines.push(Line::from(Span::styled(
            "ℹ session",
            Style::default().add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(format!("model: {}", self.current_model)));
        lines.push(Line::from(format!(
            "tokens in: {}",
            fmt_tokens(self.usage.input)
        )));
        lines.push(Line::from(format!(
            "cache read: {}",
            fmt_tokens(self.usage.cache_read)
        )));
        if let Some(window) = self
            .models
            .iter()
            .find(|spec| spec.id == self.current_model)
            .and_then(|spec| spec.context_window)
        {
            let percent = self
                .last_input
                .checked_mul(100)
                .and_then(|n| n.checked_div(window))
                .unwrap_or(0);
            lines.push(Line::from(format!(
                "context: {percent}% of {}",
                fmt_tokens(window)
            )));
        }
        lines
    }

    fn draw_overlay(&mut self, frame: &mut Frame) {
        if self.model_picker_open {
            let lines: Vec<Line> = self
                .models
                .iter()
                .enumerate()
                .map(|(index, spec)| {
                    let selected = index == self.picker_index;
                    let active = spec.id == self.current_model;
                    let prefix = if selected { "› " } else { "  " };
                    let marker = if active { "● " } else { "  " };
                    let label = match &spec.name {
                        Some(name) if name != &spec.id => {
                            format!("{prefix}{marker}{} — {name}", spec.id)
                        }
                        _ => format!("{prefix}{marker}{}", spec.id),
                    };
                    let style = if selected {
                        Style::default()
                            .fg(self.theme.accent)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(self.theme.user_message_text)
                    };
                    Line::from(Span::styled(label, style))
                })
                .collect();
            let block = Block::default()
                .borders(Borders::ALL)
                .title(" select model ")
                .title_style(Style::default().fg(self.theme.accent))
                .border_style(Style::default().fg(self.theme.dim));
            let rect = overlay_area(frame.area(), 52, lines.len() as u16 + 2);
            frame.render_widget(Paragraph::new(lines).block(block), rect);
        }

        if self.journal_open {
            let rows = crate::sessions::journal(&self.inputs.agent_dir, &self.cwd);
            let mut lines: Vec<Line> = Vec::new();
            for row in rows {
                let time = if row.time.len() >= 19 {
                    &row.time[11..19]
                } else {
                    ""
                };
                lines.push(Line::from(vec![
                    Span::styled(format!("{time} "), Style::default().fg(self.theme.muted)),
                    Span::styled(
                        format!("[{}] ", row.kind),
                        Style::default().fg(self.theme.accent),
                    ),
                    Span::styled(row.text, Style::default().fg(self.theme.dim)),
                ]));
            }
            if lines.is_empty() {
                lines.push(Line::from(Span::styled(
                    "no journal entries yet",
                    Style::default().fg(self.theme.muted),
                )));
            }
            let block = Block::default()
                .borders(Borders::ALL)
                .title(" journal — narrated audit trail (/journal to close) ");
            let area = overlay_area(frame.area(), 78, lines.len() as u16 + 2);
            frame.render_widget(Paragraph::new(lines).block(block), area);
            return;
        }
        if self.info_open {
            let session_name = self
                .session_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("(none)")
                .to_string();
            let cwd = self.cwd.display().to_string();
            let pairs: Vec<(String, String)> = vec![
                ("provider".to_string(), self.provider_id.clone()),
                ("model".to_string(), self.current_model.clone()),
                ("mode".to_string(), self.mode.as_str().to_string()),
                ("cwd".to_string(), cwd),
                ("session".to_string(), session_name),
                (
                    "context files".to_string(),
                    if self.context_files.is_empty() {
                        "none".to_string()
                    } else {
                        self.context_files
                            .iter()
                            .map(|file| {
                                file.path
                                    .file_name()
                                    .and_then(|name| name.to_str())
                                    .unwrap_or("AGENTS.md")
                            })
                            .collect::<Vec<_>>()
                            .join(", ")
                    },
                ),
                ("tokens in".to_string(), fmt_tokens(self.usage.input)),
                ("tokens out".to_string(), fmt_tokens(self.usage.output)),
                ("cache read".to_string(), fmt_tokens(self.usage.cache_read)),
                (
                    "cache write".to_string(),
                    fmt_tokens(self.usage.cache_create),
                ),
                ("search index".to_string(), {
                    let index = agent_m_tools::load_or_build(&self.cwd);
                    format!(
                        "{} files · {} symbols",
                        index.files.len(),
                        index.total_symbols
                    )
                }),
                ("cost".to_string(), format!("${:.4}", self.usage.cost)),
                (
                    "confidence".to_string(),
                    self.last_trust
                        .confidence
                        .map(|c| format!("{c}%"))
                        .unwrap_or_else(|| "—".to_string()),
                ),
                (
                    "last decision".to_string(),
                    self.last_trust
                        .reason
                        .clone()
                        .unwrap_or_else(|| "—".to_string()),
                ),
                ("preferences".to_string(), {
                    let prefs = crate::prefs::load(&self.inputs.agent_dir);
                    let families: Vec<String> = prefs
                        .command_usage
                        .iter()
                        .map(|(f, c)| format!("{f}×{c}"))
                        .collect();
                    if families.is_empty() && prefs.undos == 0 {
                        "none learned yet".to_string()
                    } else {
                        format!(
                            "{} {}",
                            families.join(" "),
                            if prefs.undos > 0 {
                                format!("undo×{}", prefs.undos)
                            } else {
                                String::new()
                            }
                        )
                        .trim()
                        .to_string()
                    }
                }),
            ];
            let lines: Vec<Line> = pairs
                .into_iter()
                .map(|(key, value)| {
                    Line::from(vec![
                        Span::styled(format!("{key:<12}"), Style::default().fg(self.theme.dim)),
                        Span::styled(value, Style::default().fg(self.theme.user_message_text)),
                    ])
                })
                .collect();
            let block = Block::default()
                .borders(Borders::ALL)
                .title(" session ")
                .title_style(Style::default().fg(self.theme.accent))
                .border_style(Style::default().fg(self.theme.dim));
            let rect = overlay_area(frame.area(), 64, lines.len() as u16 + 2);
            frame.render_widget(Paragraph::new(lines).block(block), rect);
        }
    }
}

/// A centered `width`×`height` rect clamped to the frame.
fn overlay_area(area: Rect, width: u16, height: u16) -> Rect {
    let w = width.min(area.width.saturating_sub(2).max(20));
    let h = height.min(area.height.saturating_sub(2).max(5));
    Rect::new(
        area.x + (area.width - w) / 2,
        area.y + (area.height - h) / 2,
        w,
        h,
    )
}

/// Does this assistant message end with a question (and no pending tool calls)?
/// Used to surface the clarifying-question flow like OpenCode's.
fn message_ends_with_question(message: &SessionMessage) -> bool {
    let SessionMessage::Assistant {
        content,
        stop_reason,
        ..
    } = message
    else {
        return false;
    };
    if !matches!(stop_reason, StopReason::Stop) {
        return false;
    }
    if content
        .iter()
        .any(|part| matches!(part, ContentPart::ToolCall { .. }))
    {
        return false;
    }
    content
        .iter()
        .rev()
        .find_map(|part| match part {
            ContentPart::Text { text } => Some(text),
            _ => None,
        })
        .is_some_and(|text| text.trim_end().ends_with('?'))
}

/// Greedy word-wrap to `width` columns for the approval prompt.
fn wrap_lines(text: &str, width: usize) -> Vec<String> {
    use unicode_width::UnicodeWidthStr;
    let mut wrapped = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;
    for word in text.split_whitespace() {
        let word_width = UnicodeWidthStr::width(word);
        if !current.is_empty() && current_width + 1 + word_width > width {
            wrapped.push(std::mem::take(&mut current));
            current_width = 0;
        }
        if !current.is_empty() {
            current.push(' ');
            current_width += 1;
        }
        current.push_str(word);
        current_width += word_width;
    }
    if !current.is_empty() {
        wrapped.push(current);
    }
    if wrapped.is_empty() {
        wrapped.push(String::new());
    }
    wrapped
}

fn push_message_item(items: &mut Vec<TranscriptItem>, message: &SessionMessage) {
    match message {
        SessionMessage::User { content } => items.push(TranscriptItem::User {
            content: content.clone(),
        }),
        SessionMessage::Assistant {
            content,
            stop_reason,
            trust,
            ..
        } => items.push(TranscriptItem::Assistant {
            parts: content.clone(),
            stop_reason: *stop_reason,
            thinking_expanded: false,
            trust: trust.clone(),
        }),
        SessionMessage::ToolResult { .. } => {}
        SessionMessage::Summary { text } => items.push(TranscriptItem::Notice {
            message: format!("📄 [session summary] {text}"),
        }),
    }
}

/// Update the flow sidebar from one `FlowStep` event. The first `pending`
/// event seeds the view; later events update the matching step by index.
/// Pure so it is unit-testable without an App.
fn apply_flow_step_state(
    view: &mut Option<FlowRunView>,
    index: usize,
    name: String,
    status: String,
) {
    if view.is_none() {
        if status != "pending" {
            return;
        }
        *view = Some(FlowRunView {
            name: String::new(),
            steps: Vec::new(),
        });
    }
    let view = view.as_mut().expect("flow view exists");
    if index >= view.steps.len() {
        view.steps.resize_with(index + 1, || StepView {
            name: String::new(),
            status: "pending".to_string(),
        });
    }
    view.steps[index] = StepView { name, status };
}

/// (done, total) for the flow sidebar counter: done = succeeded steps.
fn flow_progress(view: &FlowRunView) -> (usize, usize) {
    let done = view
        .steps
        .iter()
        .filter(|step| step.status == "succeeded")
        .count();
    (done, view.steps.len())
}

#[cfg(test)]
mod sidebar_tests {
    use super::*;

    #[test]
    fn apply_flow_step_transitions_and_counts() {
        let mut view: Option<FlowRunView> = None;
        // A status event before the seed is ignored.
        apply_flow_step_state(&mut view, 0, "jira".to_string(), "running".to_string());
        assert!(view.is_none(), "no seed yet → ignored");

        // Seed: all steps pending.
        apply_flow_step_state(&mut view, 0, "jira".to_string(), "pending".to_string());
        apply_flow_step_state(&mut view, 1, "plan".to_string(), "pending".to_string());
        apply_flow_step_state(&mut view, 2, "ship".to_string(), "pending".to_string());
        let v = view.as_ref().unwrap();
        assert_eq!(v.steps.len(), 3);
        assert_eq!(flow_progress(v), (0, 3));

        // Live transitions: pending → running → succeeded.
        apply_flow_step_state(&mut view, 0, "jira".to_string(), "running".to_string());
        assert_eq!(flow_progress(view.as_ref().unwrap()), (0, 3));
        apply_flow_step_state(&mut view, 0, "jira".to_string(), "succeeded".to_string());
        apply_flow_step_state(&mut view, 1, "plan".to_string(), "failed".to_string());
        assert_eq!(
            flow_progress(view.as_ref().unwrap()),
            (1, 3),
            "done = succeeded only"
        );

        // Out-of-order index still lands on the right slot.
        apply_flow_step_state(&mut view, 2, "ship".to_string(), "succeeded".to_string());
        assert_eq!(flow_progress(view.as_ref().unwrap()), (2, 3));
        assert_eq!(view.as_ref().unwrap().steps[1].status, "failed");
        assert_eq!(view.as_ref().unwrap().steps[2].name, "ship");
    }
}
