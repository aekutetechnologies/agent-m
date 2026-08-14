//! Enforced trust protocol (check.md principles 2, 4, 9).
//!
//! The display-only pipeline (parse → render) already exists; this module
//! makes the protocol actionable:
//!
//! - **P2** — a turn that uses tools without a `<trust>` block is a trust
//!   gap (the model did not explain its decision).
//! - **P4** — a turn whose `<confidence>` is below the threshold (default 50)
//!   that also calls risk>Low tools escalates to a human.
//! - **P9** — `<evidence>` citations are checked against the working
//!   directory: every `file` must exist and any given `line` must be within
//!   the file.
//!
//! Modes (`TrustMode`):
//! - `Off` — today's behavior: display-only.
//! - `Warn` — emit a Notice for each trust gap; never blocks.
//! - `Ask` — before running a turn's tool calls, ask the human (AskGate)
//!   whether to continue. No ask gate → deny (safe default).
//! - `Block` — deny the tool calls outright, no question.
//!
//! The library default is `Off` so embedding tests and prior behavior are
//! unchanged; the CLI sets its own default (`warn`) via `--trust`.

use crate::risk::{RiskLevel, RiskPolicy};
use crate::tool::{AskGate, ToolCallInfo};
use agent_m_ai::TrustData;
use std::path::{Path, PathBuf};

#[cfg(test)]
use std::future::Future;
#[cfg(test)]
use std::pin::Pin;

/// How strictly the harness enforces the trust protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustMode {
    /// Trust blocks are display-only (today's behavior).
    Off,
    /// Notice each trust gap; never blocks.
    Warn,
    /// Ask the human before running tool calls from a gapped turn.
    Ask,
    /// Deny tool calls from a gapped turn without asking.
    Block,
}

impl Default for TrustMode {
    fn default() -> Self {
        TrustMode::Off
    }
}

/// Per-session trust enforcement configuration.
#[derive(Debug, Clone)]
pub struct TrustPolicy {
    /// Enforcement mode.
    pub mode: TrustMode,
    /// Confidence below this (0-100) counts as low (P4). Default 50.
    pub confidence_threshold: u8,
}

impl Default for TrustPolicy {
    fn default() -> Self {
        TrustPolicy {
            mode: TrustMode::Off,
            confidence_threshold: 50,
        }
    }
}

/// The trust gaps found in one turn, as structured flags + human lines.
#[derive(Debug, Default)]
pub struct TrustIssues {
    /// P2: the turn used tools but carried no `<trust>` block.
    pub missing_block: bool,
    /// P4: `Some(confidence)` below the policy threshold.
    pub low_confidence: Option<u8>,
    /// P9: human-readable evidence problems (`file:line …`).
    pub bad_evidence: Vec<String>,
}

impl TrustIssues {
    /// No gaps at all: nothing to escalate or warn about.
    pub fn is_clean(&self) -> bool {
        !self.missing_block && self.low_confidence.is_none() && self.bad_evidence.is_empty()
    }

    /// One human-readable line describing every gap ("" when clean).
    pub fn summary(&self, threshold: u8) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.missing_block {
            parts.push("no <trust> block explaining the turn".to_string());
        }
        if let Some(confidence) = self.low_confidence {
            parts.push(format!(
                "confidence {confidence}/100 is below the {threshold} threshold"
            ));
        }
        parts.extend(self.bad_evidence.iter().cloned());
        parts.join("; ")
    }
}

/// Verify every `<evidence>` citation against the working directory (P9).
/// Returns human-readable problems; an empty Vec means all citations check
/// out. File must exist (and be a file); a given `line` must be within the
/// file's line count. Unreadable files skip the line check but not the
/// existence check.
pub fn check_evidence(trust: &TrustData, cwd: &Path) -> Vec<String> {
    let mut problems = Vec::new();
    for item in &trust.evidence {
        let path = if Path::new(&item.file).is_absolute() {
            PathBuf::from(&item.file)
        } else {
            cwd.join(&item.file)
        };
        if !path.is_file() {
            problems.push(format!("evidence cites `{}`: file not found", item.file));
            continue;
        }
        if let Some(line) = item.line {
            if line == 0 {
                problems.push(format!(
                    "evidence cites `{}:{}`: line 0 is invalid",
                    item.file, line
                ));
                continue;
            }
            if let Some(total) = count_lines(&path)
                && line > total
            {
                problems.push(format!(
                    "evidence cites `{}:{}`: file has only {total} line(s)",
                    item.file, line
                ));
            }
        }
    }
    problems
}

fn count_lines(path: &Path) -> Option<u64> {
    let content = std::fs::read_to_string(path).ok()?;
    Some(content.lines().count() as u64)
}

/// Evaluate one turn's trust compliance without deciding. `calls` are the
/// tool calls the turn is about to run; `risk` classifies them (None → every
/// call counts as risky, the conservative default for a host that cannot
/// inspect tools).
pub fn assess(
    policy: &TrustPolicy,
    trust: &TrustData,
    calls: &[ToolCallInfo],
    risk: Option<&RiskPolicy>,
    cwd: &Path,
) -> TrustIssues {
    let mut issues = TrustIssues::default();
    if calls.is_empty() {
        return issues;
    }
    issues.missing_block = trust.is_empty();
    if let Some(confidence) = trust.confidence
        && confidence < policy.confidence_threshold
        && calls_are_risky(calls, risk)
    {
        issues.low_confidence = Some(confidence);
    }
    issues.bad_evidence = check_evidence(trust, cwd);
    issues
}

/// Any call classified High/Critical? (None policy → unknown = risky.)
fn calls_are_risky(calls: &[ToolCallInfo], risk: Option<&RiskPolicy>) -> bool {
    match risk {
        Some(policy) => calls
            .iter()
            .any(|call| policy.assess(call).level >= RiskLevel::High),
        None => !calls.is_empty(),
    }
}

/// What to do with a tool-using turn after enforcement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustDecision {
    /// No gaps; run the tool calls.
    Proceed,
    /// Gaps were noticed (Warn mode); run anyway.
    ProceedWarned,
    /// The human approved the calls despite the gaps.
    ApprovedByHuman,
    /// Denied: the calls must not run.
    Denied(String),
}

/// Enforce `policy` for a turn that produced tool calls. Returns the decision
/// plus any notices the harness should emit (so each gap is reported exactly
/// once, whether or not it blocks).
pub async fn enforce(
    policy: &TrustPolicy,
    trust: &TrustData,
    calls: &[ToolCallInfo],
    risk: Option<&RiskPolicy>,
    ask: Option<&dyn AskGate>,
    cwd: &Path,
) -> (TrustDecision, Vec<String>) {
    let issues = assess(policy, trust, calls, risk, cwd);
    if policy.mode == TrustMode::Off || calls.is_empty() || issues.is_clean() {
        return (TrustDecision::Proceed, Vec::new());
    }
    let summary = issues.summary(policy.confidence_threshold);
    match policy.mode {
        TrustMode::Off => (TrustDecision::Proceed, Vec::new()),
        TrustMode::Warn => {
            let notice = format!("trust gap: {summary}");
            (TrustDecision::ProceedWarned, vec![notice])
        }
        TrustMode::Block => (
            TrustDecision::Denied(format!("blocked by trust policy: {summary}")),
            vec![format!("trust gap: {summary}")],
        ),
        TrustMode::Ask => {
            let question = format!(
                "The agent's turn has a trust gap:\n{summary}\nRun its tool calls anyway?"
            );
            match ask {
                Some(gate) => match gate
                    .ask(question, Some(vec!["yes".to_string(), "no".to_string()]), false)
                    .await
                {
                    Ok(answer) if answer.trim().eq_ignore_ascii_case("yes") => (
                        TrustDecision::ApprovedByHuman,
                        vec![format!("trust gap: {summary}; approved by human")],
                    ),
                    _ => (
                        TrustDecision::Denied(format!(
                            "denied over trust gap (human did not approve): {summary}"
                        )),
                        vec![format!("trust gap: {summary}; not approved")],
                    ),
                },
                None => (
                    TrustDecision::Denied(format!(
                        "blocked by trust policy (no human to ask): {summary}"
                    )),
                    vec![format!("trust gap: {summary}; no human to ask")],
                ),
            }
        }
    }
}

/// A test ask gate that returns a canned answer.
#[cfg(test)]
struct StubGate {
    answer: &'static str,
}

#[cfg(test)]
impl AskGate for StubGate {
    fn ask(
        &self,
        _question: String,
        _options: Option<Vec<String>>,
        _multi_select: bool,
    ) -> Pin<Box<dyn Future<Output = Result<String, String>> + Send>> {
        let answer = self.answer.to_string();
        Box::pin(async move { Ok(answer) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn call(name: &str, args: serde_json::Value) -> ToolCallInfo {
        ToolCallInfo {
            tool_call_id: "t1".into(),
            name: name.into(),
            arguments: args,
        }
    }

    fn bash_call() -> ToolCallInfo {
        call("bash", serde_json::json!({ "command": "rm -rf /tmp/x" }))
    }

    fn read_call() -> ToolCallInfo {
        call("read", serde_json::json!({ "path": "src/main.rs" }))
    }

    #[test]
    fn assess_flags_missing_block_for_tool_turns() {
        let policy = TrustPolicy::default();
        let tmp = std::env::temp_dir().join(format!("trust-policy-{}", std::process::id()));
        let issues = assess(&policy, &TrustData::default(), &[bash_call()], None, &tmp);
        assert!(issues.missing_block);
        assert!(issues.low_confidence.is_none());
        assert!(!issues.is_clean());
        // No calls → clean even without a block.
        assert!(assess(&policy, &TrustData::default(), &[], None, &tmp).is_clean());
    }

    #[test]
    fn assess_escalates_low_confidence_only_with_risky_calls() {
        let policy = TrustPolicy {
            mode: TrustMode::Ask,
            confidence_threshold: 50,
        };
        let tmp = std::env::temp_dir().join(format!("trust-policy-{}", std::process::id()));
        let risk = RiskPolicy {
            cwd: tmp.clone(),
            opaque_tools: vec![],
        };
        let low = TrustData {
            confidence: Some(30),
            ..TrustData::default()
        };
        // High-risk bash + low confidence → escalated.
        let issues = assess(&policy, &low, &[bash_call()], Some(&risk), &tmp);
        assert_eq!(issues.low_confidence, Some(30));
        // Low-risk read + low confidence → not escalated.
        let issues = assess(&policy, &low, &[read_call()], Some(&risk), &tmp);
        assert!(issues.low_confidence.is_none());
        // High confidence + risky call → clean.
        let high = TrustData {
            confidence: Some(90),
            ..TrustData::default()
        };
        assert!(assess(&policy, &high, &[bash_call()], Some(&risk), &tmp).is_clean());
    }

    #[test]
    fn check_evidence_reports_missing_files_and_bad_lines() {
        let dir = std::env::temp_dir().join(format!("trust-evidence-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("real.rs"), "fn main() {}\n").unwrap();

        let trust = TrustData {
            evidence: vec![
                agent_m_ai::Evidence {
                    file: "real.rs".into(),
                    line: Some(1),
                    note: None,
                },
                agent_m_ai::Evidence {
                    file: "missing.rs".into(),
                    line: None,
                    note: None,
                },
                agent_m_ai::Evidence {
                    file: "real.rs".into(),
                    line: Some(99),
                    note: None,
                },
            ],
            ..TrustData::default()
        };
        let problems = check_evidence(&trust, &dir);
        assert_eq!(problems.len(), 2, "one per broken citation: {problems:?}");
        assert!(problems[0].contains("missing.rs"));
        assert!(problems[1].contains("99"));
        // Clean citation → no problems.
        let ok = TrustData {
            evidence: vec![agent_m_ai::Evidence {
                file: "real.rs".into(),
                line: Some(1),
                note: None,
            }],
            ..TrustData::default()
        };
        assert!(check_evidence(&ok, &dir).is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn enforce_modes() {
        let policy = TrustPolicy {
            mode: TrustMode::Warn,
            confidence_threshold: 50,
        };
        let tmp = std::env::temp_dir().join(format!("trust-policy-{}", std::process::id()));
        // Warn never blocks.
        let (decision, notices) = enforce(&policy, &TrustData::default(), &[bash_call()], None, None, &tmp).await;
        assert_eq!(decision, TrustDecision::ProceedWarned);
        assert_eq!(notices.len(), 1);
        // Off proceeds silently.
        let off = TrustPolicy::default();
        let (decision, notices) = enforce(&off, &TrustData::default(), &[bash_call()], None, None, &tmp).await;
        assert_eq!(decision, TrustDecision::Proceed);
        assert!(notices.is_empty());
        // Block denies without asking.
        let block = TrustPolicy {
            mode: TrustMode::Block,
            confidence_threshold: 50,
        };
        let (decision, _) = enforce(&block, &TrustData::default(), &[bash_call()], None, None, &tmp).await;
        assert!(matches!(decision, TrustDecision::Denied(reason) if reason.contains("<trust>")));
        // Ask: yes → approved; no → denied.
        let ask = TrustPolicy {
            mode: TrustMode::Ask,
            confidence_threshold: 50,
        };
        let (decision, _) = enforce(
            &ask,
            &TrustData::default(),
            &[bash_call()],
            None,
            Some(&StubGate { answer: "yes" }),
            &tmp,
        )
        .await;
        assert_eq!(decision, TrustDecision::ApprovedByHuman);
        let (decision, _) = enforce(
            &ask,
            &TrustData::default(),
            &[bash_call()],
            None,
            Some(&StubGate { answer: "no" }),
            &tmp,
        )
        .await;
        assert!(matches!(decision, TrustDecision::Denied(_)));
        // Ask with no gate → deny.
        let (decision, _) = enforce(&ask, &TrustData::default(), &[bash_call()], None, None, &tmp).await;
        assert!(matches!(decision, TrustDecision::Denied(_)));
    }

    #[tokio::test]
    async fn enforce_clean_turn_never_asks() {
        let ask = TrustPolicy {
            mode: TrustMode::Ask,
            confidence_threshold: 50,
        };
        let tmp = std::env::temp_dir().join(format!("trust-policy-{}", std::process::id()));
        let trust = TrustData {
            confidence: Some(90),
            reason: Some("it is correct".into()),
            ..TrustData::default()
        };
        let (decision, notices) = enforce(&ask, &trust, &[read_call()], None, None, &tmp).await;
        assert_eq!(decision, TrustDecision::Proceed);
        assert!(notices.is_empty());
    }
}
