# agent-m

A pi-style coding agent in Rust: an interactive terminal UI (ratatui) with a
DeepSeek-backed streaming agent, tool calling, byte-stable prefix caching, and
JSONL session persistence.

The UI is modeled on [pi](https://pi.dev)'s interactive mode (transcript +
fixed editor/footer dock, same default keybindings), implemented with the same
stack warp's terminal agent CLI uses (ratatui + crossterm).

## Features

- Interactive TUI: streaming markdown replies, tool-execution blocks with
  expand/collapse, status line with a working spinner, footer keybinding hints.
- `ui-mode regular|fullscreen` (terminal scrollback vs alternate screen).
- Tool calling with per-call approval (`y`/`n`) or `--yes` auto-approve:
  `bash`, `read`, `write`, `edit` (exact-text multi-edit + unified diff),
  `grep`, `find`, `ls` (default active set: read, bash, ls, grep, find,
  edit, write). Plan mode keeps the read-only subset (`ls`, `read`, `grep`,
  `find`, `ask`).
- Model-agnostic provider layer (OpenAI-compatible chat completions), with
  DeepSeek configured out of the box (`deepseek-chat`, `deepseek-reasoner`).
- Byte-stable prefix caching: request bodies are serialized deterministically
  (sorted keys, no volatile fields), the system prompt and tool schemas are
  assembled once per session, and only the new message is appended per turn —
  so the provider's context cache is served instead of recomputed. Cache
  hit/miss tokens are parsed into usage and shown in the status line
  (`/cache` or `ctrl+t` to toggle).
- JSONL sessions under `~/.agent-m/agent/sessions/--<cwd>--/`, auto-resumed.

## Install & run

Requires Rust (stable) and a DeepSeek API key.

```bash
./check.sh                          # fmt + clippy -D warnings + tests
cargo build --release               # target/release/agent-m
# or run from source:
./agent-m.sh
```

### Use it from any folder (like `pi`)

Install the release binary once, then `agent-m` works everywhere — it starts the
chat in the folder you're in, and tools/sessions are scoped to that folder:

```bash
cargo build --release
cp target/release/agent-m ~/.cargo/bin/agent-m   # ~/.cargo/bin is on your PATH

cd ~/some/project
agent-m        # chat mode starts, scoped to ~/some/project
```

### Setup

```bash
export DEEPSEEK_API_KEY="sk-..."
```

Key resolution order: `DEEPSEEK_API_KEY` env var → `~/.agent-m/agent/auth.json`
(`{"providers": {"deepseek": {"apiKey": "..."}}}` or `{"deepseek": "..."}`) →
`~/.agent-m/agent/settings.json` (same shapes).

### Usage

```bash
agent-m                                # interactive TUI (default on a TTY)
agent-m --model deepseek-reasoner      # reasoning model
agent-m -p "explain this repo"         # print mode: stream reply to stdout
echo "summarize README" | agent-m      # non-TTY stdin → print mode
agent-m --yes                          # auto-approve tool calls (also enables tools in print mode)
agent-m --no-tools                     # no tools
agent-m --mode-plan                    # start in plan mode (read-only planning)
agent-m --list-models                  # list available models
agent-m --help                         # all flags
```

Note: print mode (piped stdin or `-p`) disables tools unless `--yes` is passed —
there is no interactive approval in print mode, so tools require explicit opt-in.

Security: destructive shell commands (`rm -rf`, `git reset --hard`, `git
checkout --force`, `git clean -f`, `find -exec/-delete`, device writes,
recursive `chmod`) are **never auto-approved** (ECC GateGuard pattern): in the
TUI they always show a ⚠️ DESTRUCTIVE approval prompt even with `--yes`, and
in print mode `--yes` denies them outright. Flow `tool`/`verify` steps run
through the same permission gate. `--compact-threshold` (default 0.5) sets the
strategic-compaction boundary, and startup warns if the active tool set
exceeds the 80-tool budget or the injected context exceeds 50k chars.

## Modes, planning, and the ask tool

- **`/plan`** (or `--mode-plan`) switches to **plan mode**: only read-only tools
  (`read`, `grep`, `find`, `ls`) plus `ask` are available, and the model is
  prompted to emit a numbered `Plan:` list. `/build` returns to normal mode.
- The plan is parsed into a task list (rendered as a `📋` block with a `n/m`
  counter in the status line) and **persisted to
  `~/.agent-m/agent/tasks/<session>.json`** — it survives restarts and
  compaction.
- After the plan is ready: `[e]xecute` (flips to build and sends the plan as
  the follow-up prompt), `[s]tay in plan mode`, `[r]efine` (rewrites the plan).
  During execution the model marks steps with `[DONE:n]`; `/todos` shows the
  current list.
- **`ask` tool**: the model can stop and ask you a clarifying question
  mid-task. The TUI shows the question in the status area — type your answer
  in the editor and press Enter (Escape cancels). In print mode `ask` fails
  with a clear message.

## Right-side status sidebar

While a flow runs, the right side of the TUI shows a live stage diagram: the
flow name with a `done/total` counter, a progress bar, and one row per step —
`✓` done (green), `▶` running (highlighted), `○` pending, `✗` failed — updated
as each step executes. When no flow is active it shows the plan's task list
(`[x]`/`[ ]` + `n/m`, the tasks done/pending view) if a plan exists, otherwise
session stats (model, tokens, cache read, context %). `/sidebar` toggles the
panel; it auto-hides below 110 terminal columns.

## Flows (Devin-style pipelines)

`agent-m --flow flows/agentic-dev.yml --yes` runs a YAML pipeline of steps
(tool / prompt / ask / condition / phase / verify) with a shared
`FlowContext` and `${step.output}` references. In the TUI, `/flow <path>` runs
one interactively and `/flows` lists the flows in `~/.agent-m/flows/`. The
shipped `flows/agentic-dev.yml` is the canonical GSD loop (Discuss → Plan →
Execute → Verify → Ship). Flow `tool` and `verify` steps go through the
permission gate and the destructive-command rules, and each run writes
`STATE.md` + `CONTEXT.json` state artifacts under `~/.agent-m/flows/<name>/`.

## Plugins (out-of-tree extensions)

`agent-m plugins install <git-url|local-path>` clones/builds a cdylib plugin
and installs it into `~/.agent-m/plugins/<name>/`; `plugins list`,
`plugins remove <name>`, and `plugins update` manage them. Every agent-m
startup loads installed plugins and merges their tools into the tool set
(permission gating and plan-mode filtering apply). A plugin is a separate repo
exporting `agent_m_plugin_entry()` from the `agent-m-plugin-sdk` C-ABI
contract; reference plugins ship in `plugins/`: `fixture`, `jira`
(jira-search / jira-comment / jira-transition, `JIRA_URL` + `JIRA_TOKEN`),
`github` (github-repo-info / github-create-pr, `GITHUB_TOKEN`), and
`test-runner` (run-tests). The `agentic-dev.yml` flow calls the jira/github
plugin tools.

## Context, memory, and context size

- **Context**: on startup the agent loads `AGENTS.md` from the working
  directory up to your home, plus `~/.agent-m/agent/AGENTS.md`, and wraps them
  into the system prompt (`<project_instructions path="…">…`). File arguments
  (`agent-m -p @src/main.ts "fix it"` or a bare path) inline the file as
  `<file>` context. `/info` shows the loaded context files, mode, usage, and
  cost.
- **Memory**: the conversation persists as JSONL sessions (auto-resumed), and
  `/compact` summarizes older messages into a `[session summary]` entry that
  stays in context — pi's cross-session memory model. The plan file is a
  second durable memory.
- **Context size**: the status line shows `NN% of 64k` (yellow >70%, red
  >90%); `/context` reports tokens, window, percent, and the 16k reserve.
- **`search` tool (local index)**: agent-m builds a per-project symbol/keyword
  index (file paths, language-aware symbols with line numbers, identifier
  tokens — pure Rust, no embedding API) cached at `~/.agent-m/index/`. The
  `search { query }` tool scores hits (exact symbol > prefix > substring >
  token overlap), returns the top 20 with `file:line (kind name) — snippet`,
  and re-indexes automatically when files change. It is read-only, so plan
  mode gets it too. `/info` shows index stats.

## Keybindings (pi defaults)

| Key | Action |
|-----|--------|
| `Enter` | submit |
| `Shift+Enter` / `ctrl+j` | newline |
| `Tab` | autocomplete (slash commands, file paths) |
| `ctrl+c` | clear editor (exit when empty) |
| `ctrl+d` | exit |
| `Escape` | interrupt the running reply |
| `ctrl+l` | cycle model |
| `ctrl+o` | expand/collapse tool output |
| `ctrl+p` / `ctrl+shift+p` | cycle model forward/backward |
| `ctrl+a`/`ctrl+e` | line start/end |
| `ctrl+b`/`ctrl+f` | word left/right |
| `ctrl+w`/`ctrl+u`/`ctrl+k` | kill word/backward/forward |
| `ctrl+y` / `ctrl+-` | yank / undo |
| `PageUp`/`PageDown` | scroll transcript |

Slash commands: `/help`, `/hotkeys`, `/clear`, `/exit`, `/quit`, `/model`,
`/new`, `/settings`, `/cache`. `!command` runs bash directly.

## Architecture

```
crates/ai      model-agnostic provider layer, byte-stable serializer, cache stats
crates/agent   agent loop (pi event ordering), session messages, tool trait, permission gate
crates/tools   built-in tools (bash, read, write, edit, grep, find, ls)
crates/tui     pi-style terminal UI (transcript, editor, markdown, themes, sessions)
crates/cli     the agent-m binary (args, config, print mode, interactive mode)
```

## Roadmap

- More providers via the OpenAI-compatible client (OpenAI, OpenRouter, Ollama, ...)
- Live end-to-end smoke with a real `DEEPSEEK_API_KEY` (see `scripts/tmux-smoke.sh`)
- OSC-11 background detection, Kitty images, project-trust model, session tree

## License

MIT.
