# agent-m vs. check.md's trust principles — gap audit

Answered by reading the actual code (`crates/agent`, `crates/tui`, `crates/tools`) and a real
session transcript, not by inference.

| # | Principle | Verdict | Evidence |
|---|---|---|---|
| 1 | Transparency (what/why/next) | **No** | `app.rs:1529` — the only status string during model work is `"Working… {spinner}"`. No step narration ("Reading X… Searching Y…"). Tool calls do get a one-line title (`transcript.rs` `tool_title()`) — command/path shown — but no "why" and no "what's next." |
| 2 | Explain every decision (reason/evidence/outcome) | **No** | Tool execution blocks show name + arguments + raw output only (`transcript.rs`). No structured reason/evidence/expected-outcome anywhere in the codebase. |
| 3 | Show plan before execution | **Partial** | Plan mode (`/plan`, `Mode::Plan`) exists and does show a numbered plan + a `📋` checklist with done/total (`plan.rs`, `transcript.rs` Plan rendering) — this is real and matches the spirit. But it's opt-in (default mode has no upfront plan) and has no time estimate. |
| 4 | Confidence scoring | **No** | `grep -rn "confidence"` across `crates/` — zero hits. No confidence value exists anywhere in the type system or output. |
| 5 | Risk-based permissions (4 tiers) | **Partial** | `RiskPolicy::risk()` (`crates/agent/src/risk.rs`) is real and evidence-based (flags recursive deletes, git force-push, writes outside cwd, opaque plugin tools), and read-only auto-approval (`ReadOnlyAutoApproveGate`) was added recently. But it's **binary** (flagged / not flagged), not the Low/Medium/High/Critical tiers check.md describes, and there's no per-action approval-required-vs-optional distinction beyond that. |
| 6 | Meaningful interruptions only | **Partial, recently improved** | Previously: every tool call prompted, including reads and benign shell commands (`ls`, `cat` via `bash`) — exactly check.md's "Bad" example (`Approve reading package.json?`). Fixed: the interactive gate is now risk-based unconditionally (`--yes` or not) — read-only tools and non-risky commands of any kind auto-approve; only risk-flagged calls prompt. Prompts still show name+args only, not consequence framing ("This will delete 2,430 files"). |
| 7 | Complete audit trail | **Partial** | Sessions are persisted as JSONL (`sessions.rs`) and every message/tool-call/result is recorded — replayable in that sense. But verified directly: only the session header carries a timestamp; individual turns do not. No reasoning/evidence is captured (the model's own chain-of-thought, `ContentPart::Thinking`, is shown live but not distinct from an audit rationale). It's a message log, not a narrated timeline. |
| 8 | Reversible actions | **No** (for agent actions) | `ctrl+-` undo exists (`editor.rs:273`) but only for the *text input box*, confirmed by reading the code — it has nothing to do with file edits. No rollback command is surfaced after a `write`/`edit`/`bash` call; the user must know to `git restore` themselves. Session resume exists but that's continuity, not rollback. |
| 9 | Evidence-driven conclusions | **No** | No hits for "evidence" as a concept anywhere in the codebase. Assistant prose is free text; there's no structured citation of which file/line/test backs a claim. |
| 10 | Honest uncertainty | **No** (harness-level) | Whatever hedging appears is only what the model chooses to write in prose — the harness has no uncertainty field, no display for it, and never appends one to a message. |
| 11 | Learn user preferences | **No** | Only hit for "preference" is the compaction summary prompt asking the *model* to summarize "user preferences" into the next context window (`agent.rs:43`) — that's short-term conversation memory, not a persistent, structured preference store. |
| 12 | Progressive autonomy levels | **No** | There's `--yes` (all-or-nothing auto-approve) and the risk gates from #5/#6, but no concept of levels, no history-based trust that raises autonomy over time. |

## Summary

agent-m has real infrastructure for exactly two of the twelve — plan mode (#3) and risk-based
gating (#5/#6). The other ten are either raw building blocks with no harness-level structure
(the audit trail is a message log, not a narrated journal) or entirely absent (confidence,
evidence citations, rollback, preference learning, autonomy levels).

Audited against the commit current at the time of writing. Re-verify claims (grep the cited
files/lines) before relying on this if the code has moved on.
