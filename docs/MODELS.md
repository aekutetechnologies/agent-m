# Model Capability Reference

Canonical source of truth for hardcoding `ModelSpec` entries in
`crates/ai/src/openai.rs`, `crates/ai/src/anthropic.rs`, and
`crates/ai/src/providers.rs`.

**Fields explained**

| Field | Meaning in code |
|---|---|
| `reasoning` | Model emits thinking/reasoning deltas (`ContentPart::Thinking`) |
| `variants` | Effort levels sent as `reasoning_effort` on the wire (empty = no picker) |
| `thinking_toggle` | Model supports disabling thinking per-request (separate from `reasoning_effort`) |
| `vision` | `supports_images: true` in ModelSpec |
| Pricing | USD per 1M tokens: `(in_miss, in_hit, out)` → `.pricing(in_miss, in_hit, out)` |

Prices sourced from official pricing pages. Mark stale entries with `*` and update.

---

## Anthropic

API base: `https://api.anthropic.com/v1`  
Wire format: native Anthropic (not OpenAI-compatible) — use `AnthropicProvider`.  
Caching: prefix caching on system prompt + last N user turns; `cache_creation` billed at 1.25× input.  
Thinking: enabled via `thinking: {type: "enabled", budget_tokens: N}` in the request body.

| Model ID | Display Name | Context | in_miss | in_hit | out | `reasoning` | Variants | `thinking_toggle` | Vision | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| `claude-opus-5` | Claude Opus 5 | 200K | 15.00 | 1.50 | 75.00 | ✓ | — | ✓ | ✓ | Highest capability; thinking always available |
| `claude-sonnet-5` | Claude Sonnet 5 | 200K | 3.00 | 0.30 | 15.00 | ✓ | — | ✓ | ✓ | Best balance; default thinking budget 16K |
| `claude-sonnet-4-6` | Claude Sonnet 4.6 | 200K | 3.00 | 0.30 | 15.00 | ✓ | — | ✓ | ✓ | Current session model |
| `claude-haiku-4-5` | Claude Haiku 4.5 | 200K | 0.80 | 0.08 | 4.00 | — | — | — | ✓ | Fast, cheap; no extended thinking |
| `claude-fable-5` | Claude Fable 5 | 200K | TBD | TBD | TBD | ✓ | — | ✓ | ✓ | Pricing TBD at release |

**Thinking budget tiers** (map to variant selector for Anthropic models with `thinking_toggle`):

> ⚠️ *Planned, not yet wired* — `anthropic.rs` currently sends a fixed
> `max_tokens: 8192` and ignores `variant`. The `variant` selector maps to
> `reasoning_effort` only on OpenAI-compatible providers (`openai.rs`).
> These tiers describe the intended mapping when thinking-budget selection
> lands.

| Variant label | `budget_tokens` |
|---|---|
| `low` | 1 024 |
| `medium` | 8 192 |
| `high` | 16 000 |
| `max` | 32 000 |

Wire: add `"thinking": {"type": "enabled", "budget_tokens": N}` to the request body.  
Disable: omit the `thinking` key entirely (or send `{"type": "disabled"}`).

---

## OpenAI

API base: `https://api.openai.com/v1`  
OpenAI-compatible wire format — use `OpenAiCompatibleProvider`.  
Caching: automatic prompt caching; cached tokens billed at 50% of input price.  
Effort: `reasoning_effort` field on the wire (`"low"` / `"medium"` / `"high"`).

### GPT family (no reasoning)

| Model ID | Display Name | Context | in_miss | in_hit | out | `reasoning` | Variants | Vision | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `gpt-4o` | GPT-4o | 128K | 2.50 | 1.25 | 10.00 | — | — | ✓ | Multimodal flagship |
| `gpt-4o-mini` | GPT-4o Mini | 128K | 0.15 | 0.075 | 0.60 | — | — | ✓ | Cheap, fast |
| `gpt-4.1` | GPT-4.1 | 1M | 2.00 | 0.50 | 8.00 | — | — | ✓ | Long-context, instruction-following |
| `gpt-4.1-mini` | GPT-4.1 Mini | 1M | 0.40 | 0.10 | 1.60 | — | — | ✓ | |
| `gpt-4.1-nano` | GPT-4.1 Nano | 1M | 0.10 | 0.025 | 0.40 | — | — | ✓ | Cheapest GPT |

### o-series (reasoning)

Effort is controlled via `reasoning_effort`; thinking tokens are billed as output.

| Model ID | Display Name | Context | in_miss | in_hit | out | `reasoning` | Variants | Vision | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `o3` | o3 | 200K | 10.00 | 2.50 | 40.00 | ✓ | low/medium/high | ✓ | Highest-capability reasoning |
| `o3-mini` | o3 Mini | 200K | 1.10 | 0.55 | 4.40 | ✓ | low/medium/high | — | Budget reasoning |
| `o4-mini` | o4 Mini | 200K | 1.10 | 0.275 | 4.40 | ✓ | low/medium/high | ✓ | Replaces o3-mini; vision added |
| `o1` | o1 | 200K | 15.00 | 7.50 | 60.00 | ✓ | — | ✓ | No effort param; always full reasoning |

**Wire for variants:** `"reasoning_effort": "low" | "medium" | "high"` in the request body.  
Omit the field to use provider default (equivalent to `"medium"` for o-series).

---

## DeepSeek

API base: `https://api.deepseek.com`  
OpenAI-compatible wire format — use `OpenAiCompatibleProvider`.  
Caching: prefix caching; cache-hit tokens billed at ~25% of miss price.

### V3 / non-thinking

| Model ID | Display Name | Context | in_miss | in_hit | out | `reasoning` | Variants | Vision | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `deepseek-chat` | DeepSeek Chat | 1M | 0.27 | 0.07 | 1.10 | — | — | — | V3 non-thinking |
| `deepseek-reasoner` | DeepSeek Reasoner | 1M | 0.55 | 0.14 | 2.19 | ✓ | — | — | R1; thinking always on, no toggle |

### V4 (thinking on by default)

DeepSeek V4 supports disabling thinking via `"thinking": false` (or omitting the thinking budget).  
There is **no `reasoning_effort` parameter** — effort is not a supported field.  
Thinking on/off is the only control; use `thinking_toggle` rather than `variants`.

| Model ID | Display Name | Context | in_miss | in_hit | out | `reasoning` | `thinking_toggle` | Vision | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `deepseek-v4-flash` | DeepSeek V4 Flash | 1M | 0.14 | 0.003 | 0.28 | ✓ | ✓ | — | Thinking on by default; can disable |
| `deepseek-v4-pro` | DeepSeek V4 Pro | 1M | 0.42 | 0.004 | 0.84 | ✓ | ✓ | — | Thinking on by default; can disable |

**Wire for thinking toggle:**
- Enable (default): omit `thinking` field, or `"thinking": true`
- Disable: `"thinking": false`

> **TODO:** Confirm exact wire field name from DeepSeek V4 API docs (`thinking`, `enable_thinking`, or similar). Update this table and `openai.rs` once confirmed.

---

## Google Gemini

API base: `https://generativelanguage.googleapis.com/v1beta/openai/` (OpenAI-compatible shim)  
Use `OpenAiCompatibleProvider` with the Gemini OpenAI-compat base URL.  
Thinking: controlled via `thinking_config` (native) or a budget parameter on the OpenAI-compat endpoint.

| Model ID | Display Name | Context | in_miss | in_hit | out | `reasoning` | Variants | Vision | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `gemini-2.5-pro` | Gemini 2.5 Pro | 1M | 1.25 | 0.31 | 10.00 | ✓ | — | ✓ | Thinking enabled; no effort param on compat endpoint |
| `gemini-2.5-flash` | Gemini 2.5 Flash | 1M | 0.075 | 0.018 | 0.30 | ✓ | — | ✓ | Thinking optional; cheapest Gemini |
| `gemini-2.5-flash-lite` | Gemini 2.5 Flash Lite | 1M | 0.015 | — | 0.04 | — | — | ✓ | No thinking; cheapest overall |

> **TODO:** Verify whether `reasoning_effort` or a budget token field is accepted on the OpenAI-compat Gemini endpoint.

---

## Mistral

API base: `https://api.mistral.ai/v1`  
OpenAI-compatible. No reasoning models as of mid-2025.

| Model ID | Display Name | Context | in_miss | in_hit | out | `reasoning` | Vision | Notes |
|---|---|---|---|---|---|---|---|---|
| `mistral-large-2` | Mistral Large 2 | 128K | 2.00 | — | 6.00 | — | ✓ | General flagship |
| `mistral-small-3` | Mistral Small 3 | 128K | 0.10 | — | 0.30 | — | — | Cheap, fast |
| `codestral-2501` | Codestral | 256K | 0.30 | — | 0.90 | — | — | Code-optimised |

---

## Implementation checklist

When adding a new model to agent-m:

1. Add a row to this table with all fields filled in.
2. In `openai.rs` (or `anthropic.rs`): add `ModelSpec::new(id).name(…).context_window(…).pricing(…)`, then chain `.reasoning(true)` and `.effort(&[…])` as applicable.
3. For `thinking_toggle` models (DeepSeek V4, Anthropic): wire the toggle into the request builder — send the appropriate field when `AgentOptions::thinking_enabled` is `false`.
4. Update the `/list-models` output to surface the new model.
5. Run `cargo test -p agent-m-ai` to catch any broken spec assertions.
