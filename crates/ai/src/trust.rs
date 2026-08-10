//! Trust metadata for assistant replies (check.md principles 2, 4, 9, 10).
//!
//! The model ends each reply with an XML `<trust>` block; the harness parses
//! it into structured `TrustData`, persists it in the session, and renders it
//! in the UI (confidence gauge, reason, evidence citations, uncertainty,
//! plan + estimate). The block is machine-read — it is stripped from the text
//! shown to the user and from messages re-sent to the model, so it never
//! pollutes the transcript or the byte-stable prefix.

use serde::{Deserialize, Serialize};

/// One piece of evidence supporting a claim (`file:line — note`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Structured trust data extracted from an assistant reply's `<trust>` block.
/// All fields are optional: a model that omits the block degrades gracefully
/// to today's behavior.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrustData {
    /// 0-100; higher is more confident.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<u8>,
    /// Why the model chose this action (principle 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// What the model expects to change (principle 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_outcome: Option<String>,
    /// Evidence backing the claim (principle 9).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<Evidence>,
    /// Honest uncertainty note (principle 10).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<String>,
    /// The plan before execution, one item per step (principle 3).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plan: Vec<String>,
    /// Human-readable time estimate for the plan (principle 3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_time: Option<String>,
}

impl TrustData {
    /// True when no field carries a value (no block or an empty one).
    pub fn is_empty(&self) -> bool {
        self.confidence.is_none()
            && self.reason.is_none()
            && self.expected_outcome.is_none()
            && self.evidence.is_empty()
            && self.uncertainty.is_none()
            && self.plan.is_empty()
            && self.estimated_time.is_none()
    }

    /// Confidence tier for display/gating: low < 50, medium < 75, else high.
    pub fn confidence_tier(&self) -> Option<(&'static str, u8)> {
        self.confidence.map(|value| match value {
            0..=49 => ("low", value),
            50..=74 => ("medium", value),
            _ => ("high", value),
        })
    }
}

/// A `<trust>` block as the model is asked to emit it (kept as a doc example;
/// the live instruction lives in the agent's system prompt).
#[allow(dead_code)]
pub const TRUST_BLOCK_DOC: &str = r#"<trust>
<confidence>85</confidence>
<reason>Expired JWT tokens are not validated.</reason>
<expected_outcome>Expired sessions will now be rejected.</expected_outcome>
<evidence><item file="auth.ts" line="83">no expiry check</item><item file="login.test.ts">failing test</item></evidence>
<uncertainty>Verified against the local tests only; not load-tested.</uncertainty>
<plan><item>Inspect logs</item><item>Update middleware</item><item>Run tests</item></plan>
<estimated_time>45 seconds</estimated_time>
</trust>"#;

/// Extract a `<trust>…</trust>` block from assistant text. Returns the parsed
/// `TrustData` and the text with the block removed (the caller decides what
/// to do with each). The LAST block wins (the model may draft one mid-reply);
/// malformed blocks are dropped, not fatal.
pub fn extract_trust_block(text: &str) -> (TrustData, String) {
    let mut cleaned = text.to_string();
    let mut trust = TrustData::default();
    let mut start = 0;
    while let Some(open) = cleaned[start..].find("<trust>") {
        let open = start + open;
        let Some(close) = cleaned[open..].find("</trust>") else {
            break;
        };
        let close = open + close + "</trust>".len();
        let block = &cleaned[open..close];
        // Parse the block; on failure keep the text (model wrote pseudo-XML)
        // and move past it.
        if let Some(parsed) = parse_block(block) {
            trust = parsed;
            cleaned.replace_range(open..close, "");
            start = open;
        } else {
            start = close;
        }
    }
    (trust, cleaned.trim().to_string())
}

/// Hand-rolled, forgiving XML-ish parser for the small, fixed set of tags the
/// model is instructed to emit. Returns None when the block has no content.
fn parse_block(block: &str) -> Option<TrustData> {
    let mut trust = TrustData::default();
    let mut any = false;
    if let Some(value) = tag_value(block, "confidence")
        && let Ok(number) = value.trim().parse::<u8>()
    {
        trust.confidence = Some(number.min(100));
        any = true;
    }
    if let Some(value) = tag_value(block, "reason") {
        trust.reason = Some(value.trim().to_string());
        any = true;
    }
    if let Some(value) = tag_value(block, "expected_outcome") {
        trust.expected_outcome = Some(value.trim().to_string());
        any = true;
    }
    if let Some(value) = tag_value(block, "uncertainty") {
        trust.uncertainty = Some(value.trim().to_string());
        any = true;
    }
    if let Some(value) = tag_value(block, "estimated_time") {
        trust.estimated_time = Some(value.trim().to_string());
        any = true;
    }
    // Evidence items: <item file="…" line="…">note</item>
    let mut evidence = Vec::new();
    let mut rest = block;
    while let Some(open) = rest.find("<item") {
        let Some(close) = rest[open..].find("</item>") else {
            break;
        };
        let item = &rest[open..open + close + "</item>".len()];
        rest = &rest[open + close + "</item>".len()..];
        let file = attr_value(item, "file").unwrap_or_default();
        if file.is_empty() {
            continue;
        }
        let line = attr_value(item, "line").and_then(|l| l.parse::<u64>().ok());
        let note = inner_text(item).map(|n| n.trim().to_string());
        evidence.push(Evidence { file, line, note });
        any = true;
    }
    trust.evidence = evidence;
    // Plan items: <plan><item>…</item></plan>
    if let Some(plan_block) = tag_block(block, "plan") {
        let mut plan = Vec::new();
        let mut rest = plan_block;
        while let Some(open) = rest.find("<item>") {
            let Some(close) = rest[open..].find("</item>") else {
                break;
            };
            let item = &rest[open + "<item>".len()..open + close];
            plan.push(item.trim().to_string());
            rest = &rest[open + close + "</item>".len()..];
        }
        if !plan.is_empty() {
            trust.plan = plan;
            any = true;
        }
    }
    any.then_some(trust)
}

/// Value of the first occurrence of `<tag>…</tag>`, if well-formed.
fn tag_value<'a>(block: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = block.find(&open)? + open.len();
    let end = block[start..].find(&close)? + start;
    Some(&block[start..end])
}

/// Inner text of the first `<tag>…</tag>` (used for the plan list).
fn tag_block<'a>(block: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = block.find(&open)? + open.len();
    let end = block[start..].find(&close)? + start;
    Some(&block[start..end])
}

/// Value of an attribute in `<item file="…" line="…">`.
fn attr_value(item: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = item.find(&needle)? + needle.len();
    let rest = &item[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Inner text of an `<item>…</item>`.
fn inner_text(item: &str) -> Option<&str> {
    let start = item.find('>')? + 1;
    let end = item.rfind("</item>")?;
    Some(&item[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_complete_trust_block() {
        let (trust, cleaned) =
            extract_trust_block(&format!("Let me fix that.\n\n{}\n", TRUST_BLOCK_DOC));
        assert_eq!(trust.confidence, Some(85));
        assert_eq!(
            trust.reason.as_deref(),
            Some("Expired JWT tokens are not validated.")
        );
        assert_eq!(trust.evidence.len(), 2);
        assert_eq!(trust.evidence[0].file, "auth.ts");
        assert_eq!(trust.evidence[0].line, Some(83));
        assert_eq!(trust.evidence[1].file, "login.test.ts");
        assert!(trust.plan.contains(&"Run tests".to_string()));
        assert_eq!(trust.estimated_time.as_deref(), Some("45 seconds"));
        assert!(!cleaned.contains("<trust>"), "block stripped");
    }

    #[test]
    fn missing_block_yields_empty_trust_and_unchanged_text() {
        let (trust, cleaned) = extract_trust_block("Plain reply, no block.");
        assert!(trust.is_empty());
        assert_eq!(cleaned, "Plain reply, no block.");
    }

    #[test]
    fn malformed_block_is_dropped_not_fatal() {
        let (trust, cleaned) =
            extract_trust_block("ok\n\n<trust><confidence>abc</confidence></trust>");
        assert!(trust.confidence.is_none());
        assert!(trust.is_empty());
        assert!(cleaned.contains("<trust>"), "malformed kept verbatim");
    }

    #[test]
    fn duplicate_blocks_last_wins_and_both_are_stripped() {
        let (trust, cleaned) = extract_trust_block(
            "<trust><confidence>40</confidence></trust> then <trust><confidence>90</confidence><reason>revised</reason></trust>",
        );
        assert_eq!(trust.confidence, Some(90));
        assert_eq!(trust.reason.as_deref(), Some("revised"));
        assert!(!cleaned.contains("<trust>"));
    }

    #[test]
    fn confidence_tiers() {
        let trust = TrustData {
            confidence: Some(30),
            ..Default::default()
        };
        assert_eq!(trust.confidence_tier(), Some(("low", 30)));
        let trust = TrustData {
            confidence: Some(80),
            ..Default::default()
        };
        assert_eq!(trust.confidence_tier(), Some(("high", 80)));
    }
}

#[cfg(test)]
mod wire_tests {
    use super::*;
    use crate::types::{ContentPart, LlmMessage};
    use crate::wire::serialize_messages;

    #[test]
    fn trust_block_is_stripped_and_wire_stays_byte_stable() {
        let raw = "Fixed it.\n\n<trust><confidence>88</confidence><reason>r</reason></trust>";
        let (trust, cleaned) = extract_trust_block(raw);
        assert_eq!(trust.confidence, Some(88));
        // The cleaned text is what gets stored and re-sent; the block must
        // never reach the wire.
        let messages = vec![
            LlmMessage::User {
                content: "hi".to_string(),
            },
            LlmMessage::Assistant {
                content: vec![ContentPart::Text {
                    text: cleaned.clone(),
                }],
                usage: None,
                stop_reason: Some(crate::types::StopReason::Stop),
            },
        ];
        let first = serde_json::to_string(&serialize_messages(&messages)).unwrap_or_default();
        let second = serde_json::to_string(&serialize_messages(&messages)).unwrap_or_default();
        assert!(!first.contains("<trust>"), "no raw block on the wire");
        assert_eq!(first, second, "byte-stable across turns");
    }
}
