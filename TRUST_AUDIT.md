# agent-m vs. check.md's trust principles — gap audit

Re-audited against the current tree (2026-08-13, second pass). Every verdict
below was verified by reading the code, not the docs — and the docs disagreed
with the code in several places (the previous version of this audit cited
deleted TUI files: `crates/tui`, `app.rs:1529`, `transcript.rs`, `editor.rs:273`;
none exist anymore). The current UI is a rustyline REPL
(`crates/cli/src/repl.rs`).

This pass re-verifies after wiring the trust harness into the CLI (LevelGate,
undo snapshots, preferences, live `/level`) and enforcing the trust protocol
(`--trust warn|ask|block`, evidence checks, low-confidence escalation).

| # | Principle | Verdict | Evidence (current tree) |
|---|-----------|---------|--------------------------|
| 1 | Transparency (what/why/next) | **Partial** | One-line tool narration is real: `toolout.rs::humanize` renders `reading "src/main.jsx"`, `running "ls -la"`, shown as a boxed `tools` panel in `repl.rs` (ToolExecutionEnd). But it shows *what*, not *why* or *what's next*. |
| 2 | Explain every decision (reason/outcome) | **Yes** | `<trust>` blocks are parsed (`crates/ai/src/trust.rs`, tolerant, last-block-wins, stripped from transcript) and rendered as a `── decision ──` panel (`section.rs::print_decision`, with reason + expected outcome). **Now enforced**: `crates/agent/src/trust_policy.rs` — a tool-using turn with no `<trust>` block triggers `enforce()` before the tools run (Notice in `warn`, human ask in `ask`, denial in `block`; CLI `--trust` flag, default `warn`). Denied calls become error `ToolResult`s so the model repairs the block. Tests: `trust_policy_blocks_tool_call_without_trust_block`, `trust_policy_ask_mode_consults_human` (loop_test.rs). |
| 3 | Show plan before execution | **Partial** | Plan mode (`--mode plan`, `Mode::Plan`) produces a numbered `Plan:` list; parsed into `plan.rs` todos, rendered as `plan (n/m)` with `[DONE:n]` markers, persisted to `tasks/<session>.json`. But plans **gate nothing**: the agent runs tools without one, and plan mode is opt-in. |
| 4 | Confidence scoring | **Yes** | 0-100 `<confidence>` parsed (`trust.rs`) and color-coded by tier (`section.rs` `SectionKind::Decision`). **Now enforced** (P4): a turn whose `<confidence>` is below the threshold (default 50) that also calls risk>Low tools escalates via `enforce()` — noticed (`warn`), asked (`ask`), or denied (`block`). `TrustPolicy.confidence_threshold` is configurable; `RiskPolicy` (or `None` → all calls risky) classifies the calls. Test: `assess_escalates_low_confidence_only_with_risky_calls`. |
| 5 | Risk-based permissions (4 tiers) | **Yes** | `RiskPolicy::assess` classifies into Low/Medium/High/Critical (`crates/agent/src/risk.rs`) with consequence framing. `LevelGate` is now constructed for **both** interactive modes (`main.rs`, repl + daemon): Low/Medium auto-approve, High/Critical route to `ask_human` (`crates/cli/src/gate.rs`). **Remote approval is live**: `--slack-channel` routes the same High/Critical prompt to Slack via `RemoteHuman::permission_closure` / `ask_slack_permission` (`crates/cli/src/slack.rs`), so headless daemons and flow runs are no longer deny-only when a human is attached. `TierGate` (`tool.rs:322`) remains tests-only — the tiering it encodes (ask on High/Critical) is what `LevelGate` provides at runtime. |
| 6 | Meaningful interruptions only | **Yes** | The interactive prompt is consequence-framed and tier-gated: only High/Critical interrupt (`gate.rs` `ask_human` / `slack.rs` `ask_slack_permission`: "Risk Level:", "Consequence:", approve/deny). `--yes` **no longer bypasses the gate** — it only auto-approves Low/Medium; High/Critical still ask. Remote flows (`--slack-channel`) get the same tiered prompt over Slack. Without `--slack-channel`, headless print/flow modes still deny risk-hinted calls outright (`DangerousCommandGate`) or deny everything without `--yes` (`DenyAllGate`) — no human to ask is treated as denial, which is the safe default. |
| 7 | Complete audit trail | **Partial** | Sessions persist as timestamped JSONL (`sessions.rs`); `/journal` (`commands.rs:327`) renders narrated rows (`JournalEntry { time, kind, text }`). It's a message/tool log with timestamps — not a rationale trail (no reasoning, no decision records). |
| 8 | Reversible actions | **Partial** | **Undo is now live**: the REPL snapshots `write`/`edit` targets before execution (`repl.rs` `ToolExecutionStart` → `record_undo_snapshot`), and `/undo` (`commands.rs:48`) restores them. Checkpoints are still unwired: `/checkpoint`/`/restore` (`commands.rs:339-340`) return hardcoded success strings without calling `create_checkpoint`/`restore_checkpoint` (`checkpoint.rs`). |
| 9 | Evidence-driven conclusions | **Yes** | `<evidence>` items (`file:line — note`) are parsed and rendered (`section.rs`, `trust.rs`). **Now verified**: `trust_policy::check_evidence` checks every citation against the working directory (file must exist; given line must be in range) — reported as `[evidence] …` at the REPL (`repl.rs` TurnEnd) and folded into the trust gap that `--trust` notices/asks/blocks. Test: `check_evidence_reports_missing_files_and_bad_lines`. |
| 10 | Honest uncertainty | **Partial** | `<uncertainty>` parsed + shown in the decision panel. Optional; display-only. |
| 11 | Learn user preferences | **Partial** | `prefs.rs` is now wired: the system-prompt block is built from `preferences.json` (`main.rs` → `prefs::prompt_block(&prefs::load(&agent_dir))`) and `/undo` usage is recorded (`commands.rs:52` `prefs::record_undo`). The second signal — `!command` shell-usage learning — has **no code path** (there is no `!` command in the REPL), so only the undo signal feeds the block. |
| 12 | Progressive autonomy levels | **Yes** | `AutonomyLevel` 0–4 + `LevelGate` (`crates/agent/src/tool.rs:482+`) are constructed in the CLI: `resolve_level` (`main.rs:210`) reads `--level`/`settings.json` and feeds `LevelGate::new`. `/level N` (`commands.rs:247`) writes the live `AtomicU8` handle, so the level changes at runtime; `/level` with no arg reports the current level by name. `--level 0` (observe) denies tool calls at the gate; `--level 4` (autonomous) still asks on High/Critical. |

## Summary

**Seven fully-implemented principles (2, 4, 5, 6, 9, 12 + evidence-backed 11 partial); five partial; zero absent.**
Risk-based permissions gate the CLI at runtime (5), interruptions are
tiered and `--yes` can no longer silence Critical (6), autonomy levels
are live via `--level` and `/level N` (12), and **Phase 3 makes the trust
protocol enforceable**: missing `<trust>` blocks (2), low confidence on
risky turns (4), and broken evidence citations (9) are now noticed, asked
about, or denied via `--trust warn|ask|block` (`trust_policy.rs`). Phase 2
closed the headless gap: `--slack-channel` gives the daemon and flow runner
a remote human for High/Critical approvals, and the daemon is a real
persistent session (`prompt`/`status`/`resume`). Rollback is half-wired (8):
undo is live, checkpoints are not. Preference learning (11) feeds the prompt
block from `/undo` usage but has no `!command` signal. The audit trail (7)
is a timestamped message log, not a rationale trail.

## Disconnected infrastructure (code exists, nothing calls it)

| Infra | Where | Wired? |
|-------|-------|--------|
| `create_checkpoint` / `restore_checkpoint` | `crates/agent/src/checkpoint.rs` | No — `/checkpoint`/`/restore` return fake success strings |
| `prefs::record_command` (`!command` family) | `crates/cli/src/prefs.rs:45` | No — no `!` command exists in the REPL; only `record_undo` is called |
| `TierGate` | `crates/agent/src/tool.rs:322` | No — tests only (`LevelGate` now provides the same tiering at runtime) |
| `/sessions` resume-from-list | `commands.rs` | No — list-only; auto-resume is by folder at startup |
| `/refine` apply/rollback/auto-trigger | `crates/cli/src/refine.rs`, `commands.rs:266` | No — propose-only, prints a text list |
| `/worktree detach`, `/flows`, `/compact` | `commands.rs` | Verify before relying — several are surface commands without documented backing |
| `/provider` | `commands.rs` | Prints the configured provider/model count only — not a wizard |
| `start_progress_notifier` (slack) | `crates/cli/src/slack.rs` | Partial — flow runs post step transitions via `slack_progress`; the daemon streams raw `EVENT` lines instead (attach client renders) |
| Daemon events → Slack progress lines | `daemon.rs` | No — daemon events go to socket clients; Slack gets only flow progress + gates |

## Shipped with this audit pass (wiring, not fiction)

- `LevelGate` constructed for repl + daemon (`main.rs`); `resolve_level` is
  live; `--yes` no longer bypasses High/Critical (new `gate.rs::ask_human`).
- `/level N` writes the live autonomy handle; `/level` reports it.
- `write`/`edit` targets are snapshotted before execution; `/undo` restores.
- Preferences prompt block built from real usage; `/undo` increments it.
- **Phase 2 — remote human over Slack**: `--slack-channel` routes the ask
  tool, High/Critical approvals, and flow step progress to Slack
  (`slack.rs` `RemoteHuman`/`ask_slack_permission`/`slack_progress`);
  `HumanChannel` is the transport-agnostic pending-question registry.
- **Phase 2 — daemon resurrected**: `--daemon <id>` runs a real persistent
  session (unix socket, `prompt`/`status`/`resume`, `EVENT` streaming);
  `--attach`/`--list-daemons` manage it. Daemon/attach now dispatch before
  the print-mode early return, so background sessions actually start.
- **Phase 3 — enforced trust protocol**: `crates/agent/src/trust_policy.rs`
  (`TrustMode` Off/Warn/Ask/Block, `TrustPolicy`, `check_evidence`, `assess`,
  `enforce`); `AgentOptions.trust` + `.risk_policy`; hook in `run_turns`
  before tool execution — denied turns get error ToolResults and the model
  adapts. CLI `--trust <off|warn|ask|block>` (default `warn`); REPL reports
  `[evidence]` problems at TurnEnd. 7 trust-policy tests + 2 loop tests.

## Known doc-vs-code fixes shipped alongside this audit

- README/docs previously described a ratatui TUI (sidebar, keybindings,
  `--ui-mode`, `/cache`, `ctrl+t`, "always ask even under `--yes`"). The TUI
  was deleted; the UI is the REPL. Claims have been rewritten to match code.
- The `--yes` "always asks" guarantee was **false**: the interactive gate
  short-circuited to `Allowed` under `--yes`. The guarantee is now restored
  for High/Critical (the gate is tiered, not bypassed).
- Smoke scripts asserted TUI strings (`"enter send"`, `"Working"`,
  `--ui-mode`); updated to REPL reality (`"agent-m REPL mode"`,
  `"thinking..."`, Ctrl-C interrupt).

Audited against the current tree. Re-verify claims (grep the cited files/lines)
before relying on this if the code has moved on.
