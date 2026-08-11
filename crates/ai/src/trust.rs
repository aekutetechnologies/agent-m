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
/// to do with each). The LAST block wins (the model may draft one mid-reply).
///
/// Tolerant by design — live models (DeepSeek observed) emit sloppy markup:
/// missing `<trust>` wrappers, `<confidence 100 …` without a closing `>`,
/// fused tags. Two modes:
/// - wrapped: `<trust> … </trust>` anywhere in the reply;
/// - loose: a trailing cluster of trust tags with no wrapper.
///
/// In BOTH modes the detected region is stripped even if nothing parses, so
/// raw XML never leaks into the transcript or onto the wire.
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
        let parsed = parse_block(block).unwrap_or_default();
        if !parsed.is_empty() {
            trust = parsed;
        }
        cleaned.replace_range(open..close, "");
        start = open;
    }
    // Loose mode: no wrapper tags anywhere — look for a trailing cluster of
    // trust tags (the model emitted the fields without <trust>).
    if trust.is_empty()
        && !cleaned.contains("<trust>")
        && let Some(cut) = trailing_trust_start(&cleaned)
    {
        let block = &cleaned[cut..];
        let parsed = parse_block(block).unwrap_or_default();
        if !parsed.is_empty() {
            trust = parsed;
        }
        cleaned.replace_range(cut.., "");
    }
    (trust, cleaned.trim().to_string())
}

/// Start of the trailing trust-tag cluster, or None. A cluster is the last
/// occurrence of any known trust tag, provided it sits in the trailing half
/// of the reply (a mid-reply `<confidence` mention is prose, not a block).
fn trailing_trust_start(text: &str) -> Option<usize> {
    const TAGS: &[&str] = &[
        "<confidence",
        "<reason",
        "<expected_outcome",
        "<evidence",
        "<uncertainty",
        "<plan",
        "<estimated_time",
    ];
    // The cluster starts at the EARLIEST trust tag; require the cluster to be
    // trailing (past the midpoint, or the remainder of the reply is clearly
    // the block — it ends with closing tags).
    let cut = TAGS.iter().filter_map(|tag| text.rfind(tag)).min()?;
    let trailing_limit = text.len() / 2;
    let is_trailing = cut >= trailing_limit || text[cut..].contains("</");
    is_trailing.then_some(cut)
}

/// Hand-rolled, forgiving XML-ish parser for the small, fixed set of tags the
/// model is instructed to emit. Returns None when the block has no content.
/// Every tag is read loosely: `<tag>value</tag>` when well-formed, otherwise
/// text after the `<tag` marker up to the next `<` (or end of block).
fn parse_block(block: &str) -> Option<TrustData> {
    let mut trust = TrustData::default();
    let mut any = false;
    if let Some(value) = loose_value(block, "confidence") {
        // The model often fuses the number with prose ("confidence 100 Simple
        // …") — take the first run of digits.
        let digits: String = value
            .chars()
            .skip_while(|ch| !ch.is_ascii_digit())
            .take_while(|ch| ch.is_ascii_digit())
            .collect();
        if let Ok(number) = digits.parse::<u8>() {
            trust.confidence = Some(number.min(100));
            any = true;
        }
    }
    if let Some(value) = loose_value(block, "reason") {
        trust.reason = Some(value.trim().to_string());
        any = true;
    }
    if let Some(value) = loose_value(block, "expected_outcome") {
        trust.expected_outcome = Some(value.trim().to_string());
        any = true;
    }
    if let Some(value) = loose_value(block, "uncertainty") {
        trust.uncertainty = Some(value.trim().to_string());
        any = true;
    }
    if let Some(value) = loose_value(block, "estimated_time") {
        trust.estimated_time = Some(value.trim().to_string());
        any = true;
    }
    // Evidence items: <item file="…" line="…">note</item> — tolerated even
    // without the closing `>` on the open tag.
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
    // Plan items: <plan><item>…</item></plan> — also tolerated as loose tags.
    let mut plan = Vec::new();
    if let Some(plan_block) = loose_block(block, "plan") {
        let mut rest = plan_block;
        while let Some(open) = rest.find("<item>") {
            let Some(close) = rest[open..].find("</item>") else {
                break;
            };
            let item = &rest[open + "<item>".len()..open + close];
            plan.push(item.trim().to_string());
            rest = &rest[open + close + "</item>".len()..];
        }
    }
    // Loose plan fallback: <plan item>…, <plan>…, or plan: …
    if plan.is_empty()
        && let Some(raw) = loose_value(block, "plan")
    {
        let raw = raw.trim();
        if !raw.is_empty() {
            plan = raw
                .split([';', '\n', ',', '→', '-'])
                .map(str::trim)
                .filter(|item| !item.is_empty() && *item != "-")
                .map(str::to_string)
                .collect();
        }
    }
    if !plan.is_empty() {
        trust.plan = plan;
        any = true;
    }
    any.then_some(trust)
}

/// Loosely read `<tag>value</tag>`: returns the value when well-formed, else
/// the text after the `<tag` marker up to the next `<` or end of block.
fn loose_value<'a>(block: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}");
    let start = block.find(&open)? + open.len();
    let rest = block[start..].strip_prefix('>').unwrap_or(&block[start..]);
    // Well-formed close wins.
    let close = format!("</{tag}>");
    if let Some(end) = rest.find(&close) {
        return Some(&rest[..end]);
    }
    // Otherwise: skip an attribute-ish tail (`>` or whitespace) and read to
    // the next `<` or end.
    let value_start = rest
        .find(['>', ' ', '=', ':'])
        .map(|index| index + 1)
        .unwrap_or(0);
    let end = rest[value_start..]
        .find('<')
        .map(|index| value_start + index)
        .unwrap_or(rest.len());
    let value = &rest[value_start..end];
    (!value.trim().is_empty()).then_some(value)
}

/// The inner region of a `<tag>…</tag>` block (used for the plan list), with
/// tolerance for a missing `>` on the open tag.
fn loose_block<'a>(block: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}");
    let start = block.find(&open)? + open.len();
    let rest = &block[start..];
    let close = format!("</{tag}>");
    let content_start = rest.find('>').map(|index| index + 1).unwrap_or(0);
    let end = rest[content_start..].find(&close)? + content_start;
    Some(&rest[content_start..end])
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
    fn malformed_block_is_stripped_not_fatal() {
        let (trust, cleaned) =
            extract_trust_block("ok\n\n<trust><confidence>abc</confidence></trust>");
        assert!(trust.confidence.is_none());
        assert!(trust.is_empty());
        assert!(!cleaned.contains("<trust>"), "raw markup stripped");
    }

    #[test]
    fn live_deepseek_loose_block_parses_and_strips() {
        // Real output captured from a live DeepSeek reply: no <trust> wrapper,
        // `<confidence 100 …` without a closing `>`, fused tags.
        let raw = "Hello, human! I'm agent-m, a coding agent running in a terminal, \
ready to help you work with code and files. <confidence 100 Simple informational \
response.<expected_outcome>User receives a greeting and \
identification.</expected_outcome><evidence plan>Greet the user and identify \
myself.<estimated_time>2 minutes</estimated_time>";
        let (trust, cleaned) = extract_trust_block(raw);
        assert_eq!(trust.confidence, Some(100), "fused number extracted");
        assert_eq!(
            trust.expected_outcome.as_deref(),
            Some("User receives a greeting and identification.")
        );
        assert_eq!(trust.estimated_time.as_deref(), Some("2 minutes"));
        assert!(
            !cleaned.contains('<'),
            "no raw markup leaks to the transcript: {cleaned:?}"
        );
        assert!(cleaned.contains("Hello, human!"), "prose kept");
    }

    #[test]
    fn mid_reply_confidence_mention_is_not_a_block() {
        // A prose mention of confidence early in the reply must not be cut.
        let raw = "My confidence in that approach is high, but let me check the code first.";
        let (trust, cleaned) = extract_trust_block(raw);
        assert!(trust.is_empty());
        assert_eq!(cleaned, raw);
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
                images: Vec::new(),
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

#[cfg(test)]
mod image_wire_tests {
    use crate::models::ModelSpec;
    use crate::types::LlmMessage;
    use crate::wire::serialize_messages;

    #[test]
    fn user_images_serialize_as_image_url_parts() {
        let messages = vec![LlmMessage::User {
            content: "what is this?".to_string(),
            images: vec!["data:image/png;base64,AAAA".to_string()],
        }];
        let body = serialize_messages(&messages);
        let json = serde_json::to_string(&body).unwrap();
        assert!(json.contains("image_url"), "image part emitted: {json}");
        assert!(json.contains("data:image/png;base64,AAAA"));
        assert!(json.contains("\"type\":\"text\""), "text part kept: {json}");
        // Byte-stable across turns.
        assert_eq!(serde_json::to_string(&body).unwrap(), json);
    }

    #[test]
    fn no_images_keeps_plain_string_content() {
        let messages = vec![LlmMessage::User {
            content: "hi".to_string(),
            images: Vec::new(),
        }];
        let body = serde_json::to_string(&serialize_messages(&messages)).unwrap();
        assert!(!body.contains("image_url"));
        assert!(body.contains("\"content\":\"hi\""), "plain content: {body}");
    }

    #[test]
    fn modelspec_defaults_to_no_vision() {
        let spec = ModelSpec::new("deepseek-chat");
        assert!(!spec.supports_images);
    }
}
