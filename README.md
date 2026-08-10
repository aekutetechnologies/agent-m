# agent-m

A pi-style coding agent in Rust: an interactive terminal UI (ratatui) with a
DeepSeek-backed streaming agent, tool calling, byte-stable prefix caching, and
JSONL session persistence.

The UI is modeled on [pi](https://pi.dev)'s interactive mode (transcript +
fixed editor/footer dock, same default keybindings), implemented with the same
stack warp's terminal agent CLI uses (ratatui + crossterm).

📚 **Docs**: a Mintlify documentation site lives in [`docs/`](docs/overview.mdx)
— quickstart, CLI reference, trust principles, flows, plugins, and
architecture (`mint.json` + `.mdx` pages, validated by `scripts/docs-check.sh`).

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

## Security boundaries and risk hints

agent-m runs with the full privileges of the invoking user. It has no OS-level sandbox. The security model consists of three layers, in order of strength:

### 1. No tool registered (strongest)
The most secure boundary: if a tool is not registered (`--no-tools`, `--exclude-tools bash`), the model cannot invoke it. Plan mode (`/plan` or `--mode-plan`) registers only read-only tools (`read`, `grep`, `find`, `ls`, `search`) plus `ask`, so destructive operations cannot reach the filesystem.

### 2. Human approval (interactive TUI only)
A human is present in the interactive TUI, so the gate is risk-based whether or not you pass `--yes`: read-only tools (`read`, `grep`, `find`, `ls`, `search`, `ask`) never prompt, and neither does a benign shell command — `ls`, `cat somefile`, `git status` — even when the model runs it via `bash` rather than a dedicated tool. Only calls that look destructive wait for your approval. **Risk hints** — cheap heuristics over command strings and write targets — flag calls that look destructive (recursive deletes, git force operations, writes outside the workspace, writes to `.git/hooks`) with a **⚠️ RISKY** prompt; these always ask, even under `--yes` (ECC GateGuard). These hints catch accidents from a cooperative model, not adversarial prompts. A bash command can hide anything via `eval "$(base64 -d …)"`, so risk detection is advisory, never a containment boundary.

In print mode with `--yes`, and in flow execution with `--yes`, risk-hinted calls are **denied outright** — there is no human to ask. Without `--yes`, print mode and flows deny all tool calls.

### 3. The OS user agent-m runs as (weakest)
The `bash` tool inherits the session's full environment and runs with your privileges. Filesystem containment applies to the seven file tools (`read`, `write`, `edit`, `ls`, `grep`, `find`, `search`) — they resolve symlinks, check `..` escapes, enforce allowed roots (default: cwd only, extend with `--allow-path`), and skip sensitive files (`.env*`, `.ssh`, `*.pem`, API keys) even inside the workspace — but `bash` can access anything you can. Untrusted plugin tools (those not marked `--trust` at install) are always flagged for approval.

**Honest ceiling**: an OS sandbox (`sandbox-exec` / `bubblewrap` / container) around the `bash` tool is the real upgrade path. Until then, agent-m is safe for work on codebases you trust, not for untrusted repos or adversarial prompts. Prompt injection (hidden instructions in file contents the agent reads) reaches the tool loop.

### Why risk hints, not a denylist?
The prior regex-based "destructive command" denylist had 12+ confirmed bypasses: `rm -r --force /`, `chmod -R a+rwx /`, `git -C /repo reset --hard`, `>/dev/sda`, and many more. Shell strings are not parseable without a full interpreter, so the new design treats risk detection as **accident prevention for a cooperative model**, not as a security boundary. Risk hints are tool-agnostic (any tool with a `command` argument is treated as a shell) and match on behavior, not names.

The denylist arms race cannot be won. The real boundaries are: no tool registered, you reading the call, and the OS process sandbox (roadmap).

`--compact-threshold` (default 0.5) sets the strategic-compaction boundary, and startup warns if the active tool set exceeds the 80-tool budget or the injected context exceeds 50k chars.

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
| `ctrl+o` | expand/collapse the most recent tool output |
| `ctrl+r` | expand/collapse the most recent thinking trace |
| `ctrl+p` / `ctrl+shift+p` | cycle model forward/backward |
| `ctrl+a`/`ctrl+e` | line start/end |
| `ctrl+b`/`ctrl+f` | word left/right |
| `ctrl+w`/`ctrl+u`/`ctrl+k` | kill word/backward/forward |
| `ctrl+y` / `ctrl+-` | yank / undo |
| `PageUp`/`PageDown` / mouse wheel | scroll transcript |

Tool output and thinking from a finished turn collapse to a one-line receipt/summary once you
move on to the next prompt (or `!command`) — `ctrl+o`/`ctrl+r` re-expand the most recent one.

Slash commands: `/help`, `/hotkeys`, `/clear`, `/exit`, `/quit`, `/model`,
`/new`, `/settings`, `/cache`. `!command` runs bash directly.

## Trust (check.md principles)

The harness — never the LLM — decides what is safe. All twelve principles from
`check.md` are addressed; the model only *reports* (reason, confidence, evidence)
and the harness *enforces* (risk tiers, autonomy levels).

| # | Principle | Status |
|---|-----------|--------|
| 1 | Transparency | ✅ Tool narration in the status line (`Reading …`, `Running \`…\``) |
| 2 | Explain decisions | ✅ `<trust>` block parsed into a "── decision ──" block under every reply |
| 3 | Plan before execution | ✅ Plan mode + `<plan>` items + time estimate in the decision block |
| 4 | Confidence | ✅ 0-100 gauge, color-coded by tier, in the decision block and `/info` |
| 5 | Risk-based permissions | ✅ 4 tiers (Low/Medium/High/Critical) by `RiskPolicy`; `TierGate`/`LevelGate` |
| 6 | Meaningful interruptions | ✅ Consequence framing + tier badge (⚠️ HIGH / 🔴 CRITICAL); High/Critical always ask, even `--yes` |
| 7 | Audit trail | ✅ Timestamped JSONL entries + `/journal` narrated timeline |
| 8 | Reversible actions | ✅ `/undo` restores write/edit snapshots; ledger persists per session |
| 9 | Evidence | ✅ `file:line — note` citations rendered from `<evidence>` |
| 10 | Uncertainty | ✅ `<uncertainty>` note shown in the decision block |
| 11 | Preference learning | ✅ Learns `!command` families + `/undo` → `preferences.json` → static prompt block |
| 12 | Autonomy levels | ✅ `--level 0-4` / `/level`: observe · suggest · assisted · trusted · autonomous |

**Risk tiers** (harness-assigned): reads/searches → Low; workspace writes and
ordinary commands → Medium; outside-cwd / `.git` / force-git / `find -exec` →
High; recursive deletes / sudo / device writes / opaque plugin tools → Critical.

**Autonomy levels**: 0 observe (no tools), 1 suggest, 2 everything asks,
3 trusted (default: auto Low/Medium, ask High/Critical), 4 autonomous (auto
everything except Critical). Persisted in `settings.json`.

**Slash commands added**: `/journal` (audit timeline), `/undo` (restore the last
file snapshot), `/level <0-4>` (autonomy). `!command` still runs bash directly
and is what the preference learner watches.

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
