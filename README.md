# agent-m

A pi-style coding agent in Rust: an interactive terminal REPL with a
DeepSeek-backed streaming agent, tool calling, byte-stable prefix caching, and
JSONL session persistence.

The REPL (rustyline + crossterm-styled section panels) is modeled on
[pi](https://pi.dev)'s interactive flow: a streaming transcript with tool
execution blocks and slash commands. For embedding and scripting there is
print mode (`-p`), `--stream-json`, and a stdio JSON-RPC server (`--serve`).

📚 **Docs**: a Mintlify documentation site lives in [`docs/`](docs/overview.mdx)
— quickstart, CLI reference, trust principles, flows, plugins, and
architecture (`mint.json` + `.mdx` pages, validated by `scripts/docs-check.sh`).

## Market-standard features

Beyond the pi-style core, agent-m ships the tooling conventions the market
expects from a coding agent:

| Feature | How |
|---------|-----|
| MCP client | `mcpServers` in `~/.agent-m/agent/mcp.json` (stdio + Streamable HTTP); tools appear as `server__tool`, gated Critical-by-default |
| Web tools | `web_fetch` (read-only, 10 MB / 10 s caps) + `web_search` (SearXNG-style endpoint via `AGENT_M_SEARCH_URL`) |
| Subagents | `delegate` tool — fresh-context sub-agent with its own tool/turn budget |
| Git checkpoints | `/checkpoint` + `/restore` + auto-snapshot infra exist (`checkpoint.rs`) but are **not wired** — stubs today |
| Image input | `@image.png` → base64 image parts; provider `supports_images` gate with a clear error on text-only models |
| Headless | `--stream-json` event lines + `--serve` stdio JSON-RPC (prompt/exit + `event` notifications) |
| Cross-tool rules | `AGENTS.md` + `CLAUDE.md` + `.cursorrules` + `GEMINI.md` loaded with fixed precedence |
| Custom slash commands | `~/.agent-m/commands/*.md` prompt templates with `${cwd}` / `${input}` |

## Providers (OpenAI-compatible)

agent-m is model-agnostic over **OpenAI-compatible** endpoints — any service
with a `/chat/completions` API (OpenAI, DeepSeek, Groq, OpenRouter, Ollama,
LM Studio, Together, …). The built-in `deepseek` is the zero-config default.

**Config file** — add a `providers` array to `~/.agent-m/agent/settings.json`:

```json
{ "providers": [
  { "id": "openai", "name": "OpenAI", "baseUrl": "https://api.openai.com/v1",
    "model": "gpt-4o-mini", "apiKeyEnv": "OPENAI_API_KEY" },
  { "id": "local", "name": "Ollama", "baseUrl": "http://localhost:11434/v1",
    "model": "llama3.2", "contextWindow": 131072 }
]}
```

Fields: `id` (`[a-z0-9-_]`, used by `--provider` and the error-hint env var
`<ID>_API_KEY`), `name?`, `baseUrl` (no trailing `/chat/completions`),
`model`, `reasoning?`, `supportsImages?`, `contextWindow?` (default 128000),
`pricing?` (`inMiss`/`inHit`/`out`, USD per 1M tokens), `apiKeyEnv?` (default
`<ID>_API_KEY`). Keys are never stored in the chat logs — they resolve from
`~/.agent-m/agent/auth.json` → `~/.agent-m/agent/settings.json` (provider-key
env vars are not read).

**Providers** — configure via `~/.agent-m/agent/settings.json` or `--provider
<id>`; `/provider` reports the active provider's models, `/model <id>` and
`/variant <id>` switch live, and `/tasks` routes roles (build/plan/compact/
subagent/refine) to different providers.

**CLI** — `--provider <id>` selects a configured provider (or `deepseek`),
`--list-models` shows every configured
provider's model plus the built-in DeepSeek pair. Keys come from the
environment or `~/.agent-m/agent/auth.json` — never from a CLI flag (argv is
visible in `ps`).

## Features

- Interactive REPL: streaming markdown replies, color-coded tool-execution
  blocks, a per-turn "thinking…" indicator, and a `── decision ──` panel
  rendering the model's `<trust>` block (confidence, reason, evidence).
- Print mode (`-p` / piped stdin), `--stream-json` event lines, and `--serve`
  stdio JSON-RPC for embedding and other agents.
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
  hit/miss tokens are parsed into usage and shown in the `/usage` output.
- JSONL sessions under `~/.agent-m/agent/sessions/--<cwd>--/`, auto-resumed.

## Install & run

```bash
curl -LsSf https://raw.githubusercontent.com/aekutetechnologies/agent-m/main/install.sh | sh
```

Installs the latest pre-built binary to `~/.local/bin/agent-m`. No Rust required.

**Build from source:**

```bash
./check.sh                          # fmt + clippy -D warnings + tests
cargo build --release               # target/release/agent-m
```

### Use it from any folder (like `pi`)

```bash
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
agent-m                                # interactive REPL (default on a TTY)
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
The most secure boundary: if a tool is not registered (`--no-tools`, `--exclude-tools bash`), the model cannot invoke it. Plan mode (`/mode plan` or `--mode-plan`) registers only read-only tools (`read`, `grep`, `find`, `ls`, `search`) plus `ask`, so destructive operations cannot reach the filesystem.

### 2. Human approval (interactive mode only)
A human is present in the interactive REPL, so the gate is risk-based: read-only tools (`read`, `grep`, `find`, `ls`, `search`, `ask`) never prompt, and neither does a benign shell command — `ls`, `cat somefile`, `git status` — even when the model runs it via `bash` rather than a dedicated tool. Only calls that look destructive wait for your approval. **Risk hints** — cheap heuristics over command strings and write targets — flag calls that look destructive (recursive deletes, git force operations, writes outside the workspace, writes to `.git/hooks`) with a ⚠️ prompt that shows the risk reason and a consequence framing ("This can destroy data…"). These catch accidents from a cooperative model, not adversarial prompts. A bash command can hide anything via `eval "$(base64 -d …)"`, so risk detection is advisory, never a containment boundary.

**Caveat**: `--yes` auto-approves Low/Medium calls, but **High/Critical calls still prompt in the REPL** — the gate is tiered, not bypassed. There is no way to silence the Critical prompt (see `TRUST_AUDIT.md`).

In print mode with `--yes`, and in flow execution with `--yes`, risk-hinted calls are **denied outright** — there is no human to ask. Without `--yes`, print mode and flows deny all tool calls.

### 3. The OS user agent-m runs as (weakest)
The `bash` tool inherits the session's full environment and runs with your privileges. Filesystem containment applies to the seven file tools (`read`, `write`, `edit`, `ls`, `grep`, `find`, `search`) — they resolve symlinks, check `..` escapes, enforce allowed roots (default: cwd only, extend with `--allow-path`), and skip sensitive files (`.env*`, `.ssh`, `*.pem`, API keys) even inside the workspace — but `bash` can access anything you can. Untrusted plugin tools (those not marked `--trust` at install) are always flagged for approval.

**Honest ceiling**: an OS sandbox (`sandbox-exec` / `bubblewrap` / container) around the `bash` tool is the real upgrade path. Until then, agent-m is safe for work on codebases you trust, not for untrusted repos or adversarial prompts. Prompt injection (hidden instructions in file contents the agent reads) reaches the tool loop.

### Why risk hints, not a denylist?
The prior regex-based "destructive command" denylist had 12+ confirmed bypasses: `rm -r --force /`, `chmod -R a+rwx /`, `git -C /repo reset --hard`, `>/dev/sda`, and many more. Shell strings are not parseable without a full interpreter, so the new design treats risk detection as **accident prevention for a cooperative model**, not as a security boundary. Risk hints are tool-agnostic (any tool with a `command` argument is treated as a shell) and match on behavior, not names.

The denylist arms race cannot be won. The real boundaries are: no tool registered, you reading the call, and the OS process sandbox (roadmap).

`--compact-threshold` (default 0.5) sets the strategic-compaction boundary, and startup warns if the active tool set exceeds the 80-tool budget or the injected context exceeds 50k chars.

## Modes, planning, and the ask tool

- **Plan mode** (`/mode plan` or `--mode-plan`): only read-only tools
  (`read`, `grep`, `find`, `ls`) plus `ask` are available, and the model is
  prompted to emit a numbered `Plan:` list. `/mode build` returns to normal
  mode.
- The plan is parsed into a task list (rendered as a `plan (n/m)` panel with
  ✓/○ markers) and **persisted to
  `~/.agent-m/agent/tasks/<session>.json`** — it survives restarts and
  compaction.
- After the plan is ready, the model executes it (plan mode is read-only);
  `/mode build` flips to normal mode with the plan as the follow-up prompt.
  During execution the model marks steps with `[DONE:n]`; `/todos` shows the
  current list.
- **`ask` tool**: the model can stop and ask you a clarifying question
  mid-task. The REPL prints the question (numbered options when provided) and
  reads your answer from stdin — type the option number or your answer and
  press Enter (blank cancels). In print mode `ask` fails with a clear message.

## Flows (Devin-style pipelines)

`agent-m --flow flows/agentic-dev.yml --yes` runs a YAML pipeline of steps
(tool / prompt / ask / condition / phase / verify) with a shared
`FlowContext` and `${step.output}` references. `/flow <path>` runs one
interactively and `/flows` lists the flows in `~/.agent-m/flows/`. The
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
  `<file>` context.
- **Memory**: the conversation persists as JSONL sessions (auto-resumed), and
  `/compact` summarizes older messages into a `[session summary]` entry that
  stays in context — pi's cross-session memory model. The plan file is a
  second durable memory.
- **Context size**: `/usage` reports tokens in/out, context window usage
  percent, and cache hit/miss stats.
- **`search` tool (local index)**: agent-m builds a per-project symbol/keyword
  index (file paths, language-aware symbols with line numbers, identifier
  tokens — pure Rust, no embedding API) cached at `~/.agent-m/index/`. The
  `search { query }` tool scores hits (exact symbol > prefix > substring >
  token overlap), returns the top 20 with `file:line (kind name) — snippet`,
  and re-indexes automatically when files change. It is read-only, so plan
  mode gets it too.

## REPL editing keys (rustyline defaults)

| Key | Action |
|-----|--------|
| `Enter` | submit |
| `Tab` | autocomplete (slash commands, file paths) |
| `ctrl+c` | exit |
| `ctrl+d` | exit (EOF) |
| `↑` / `↓` (`ctrl+p` / `ctrl+n`) | history |
| `ctrl+a` / `ctrl+e` | line start / end |
| `ctrl+w` / `ctrl+u` / `ctrl+k` | kill word / backward / forward |
| `ctrl+y` | yank |

While the model streams, a `thinking…` indicator shows with an elapsed timer,
tool executions are printed as one-line summaries (`reading "src/main.rs"`,
`running "cargo test"`), and `/tool-output last|<n>` reprints the full output
of any of the last 20 tool calls.

Slash commands: `/exit`, `/quit`, `/sessions`, `/undo`, `/model`, `/variant`,
`/mode`, `/usage`, `/level`, `/harness`, `/refine`, `/todos`, `/worktree`,
`/journal`, `/checkpoint`, `/restore`, `/flows`, `/compact`, `/tool-output`,
`/tools`, `/color`, `/provider`, `/tasks`, `/help`.

## Trust (check.md principles)

The harness — never the LLM — decides what is safe. The model *reports*
(reason, confidence, evidence) and the harness *enforces* (risk tiers,
approvals). Status is against the current code, not the ideal: **three** of
the twelve principles are fully implemented (risk-based permissions,
meaningful interruptions, autonomy levels), nine are partially implemented
(parsed and displayed, or half-wired), and none are entirely absent. See
`TRUST_AUDIT.md` for the evidence-based audit.

| # | Principle | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Transparency | **Partial** | One-line tool narration (`reading "src/main.rs"`, `running "cargo test"`) via `toolout.rs` — no "why" or "what's next" |
| 2 | Explain decisions | **Partial** | `<trust>` block parsed (`trust.rs`) into a `── decision ──` panel (`section.rs`); model may omit it |
| 3 | Plan before execution | **Partial** | Plan mode (`--mode-plan`/`/mode plan`), `plan (n/m)` panel, `[DONE:n]` markers — opt-in |
| 4 | Confidence | **Partial** | 0-100 parsed, tiered, displayed — has no effect on gating or retries |
| 5 | Risk-based permissions | **Yes** | 4 tiers classified by `RiskPolicy::assess`; `LevelGate` gates both interactive modes (repl + daemon): Low/Medium auto, High/Critical ask |
| 6 | Meaningful interruptions | **Yes** | Consequence-framed prompt (`gate.rs::ask_human`); only High/Critical interrupt; `--yes` cannot silence them |
| 7 | Audit trail | **Partial** | Timestamped JSONL + `/journal` narrated rows (`sessions.rs`) — message log, not a narrated rationale |
| 8 | Reversible actions | **Partial** | `/undo` is live: `write`/`edit` targets snapshotted before execution and restored by `/undo`. `/checkpoint` + `/restore` still return fake success strings |
| 9 | Evidence | **Partial** | `file:line — note` citations parsed + rendered — model may omit them |
| 10 | Uncertainty | **Partial** | `<uncertainty>` parsed + rendered — model may omit it |
| 11 | Preference learning | **Partial** | Prompt block built from real usage (`prefs.rs`); `/undo` recorded. No `!command` signal — no `!` command exists |
| 12 | Autonomy levels | **Yes** | `--level <0-4>` feeds `LevelGate` at startup; `/level N` changes it live via the atomic handle; `/level` reports the current tier by name |

**Self-improvement (`/refine`)**: a Continual-Harness layer
(`~/.agent-m/harness.json`) holds memories, prompt notes, and skills that the
model proposes via `/refine`. Proposals print as a text list — there is **no
apply step, no rollback, and no background auto-trigger yet** (see
`TRUST_AUDIT.md`); `/harness` lists the current state.

**Risk tiers** (harness-assigned): reads/searches → Low; workspace writes and
ordinary commands → Medium; outside-cwd / `.git` / force-git / `find -exec` →
High; recursive deletes / sudo / device writes / opaque plugin tools → Critical.

**Autonomy levels**: 0 observe (no tools), 1 suggest, 2 everything asks,
3 trusted (default: auto Low/Medium, ask High/Critical), 4 autonomous (auto
everything except Critical — High/Critical still ask). Wired end-to-end:
`--level <0-4>` or `settings.json` `"level"` at startup, `/level N` live in
the session, `/level` with no arg reports the current tier.

**Slash commands added for trust**: `/journal` (audit timeline), `/undo`
(restores the last `write`/`edit` — targets are snapshotted before they run),
`/level [0-4]` (show or set the live autonomy level), `/checkpoint` +
`/restore` (git snapshots — infra present, not yet wired).

## Architecture

```
crates/ai      model-agnostic provider layer, byte-stable serializer, cache stats
crates/agent   agent loop (pi event ordering), session messages, tool trait, permission gates
crates/tools   built-in tools (bash, read, write, edit, grep, find, ls, search, web)
crates/flow    YAML flow engine (prompt/ask/tool/condition/phase/verify steps, ${ref}s)
crates/mcp     MCP client (stdio + Streamable HTTP)
crates/plugin-sdk / plugin-loader   C-ABI contract + host for out-of-tree plugin tools
crates/cli     the agent-m binary (args, config, print mode, interactive REPL)
```

## Roadmap

- More providers via the OpenAI-compatible client (OpenAI, OpenRouter, Ollama, ...)
- Live end-to-end smoke with a real `DEEPSEEK_API_KEY` (see `scripts/tmux-smoke.sh`)
- OSC-11 background detection, Kitty images, project-trust model, session tree

## License

MIT.
