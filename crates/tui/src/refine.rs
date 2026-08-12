//! The refinement planner for the Continual Harness (`/refine`): collects the
//! recent session trajectory, asks the model for the smallest useful harness
//! edit (create/update/delete a memory, note, or skill), and parses the
//! proposal tolerantly. The proposal is always *reviewed by the user* before
//! it is applied — the planner itself never writes.

use std::path::Path;

use agent_m_ai::{ChatRequest, LlmMessage, Provider, StreamEvent};
use futures_util::StreamExt;

/// One proposed harness edit.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProposedOp {
    /// create | update | delete
    pub action: String,
    /// memory | note | skill
    pub kind: String,
    /// Entry id (may be empty for a create without an explicit id).
    pub id: String,
    pub text: String,
    pub reason: String,
}

/// A parsed refinement proposal (already validated + deduped).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RefineProposal {
    pub ops: Vec<ProposedOp>,
}

/// The fixed planner system prompt (byte-stable — it is not part of the chat
/// session prompt, only of the one-shot refine request).
const PLANNER_SYSTEM: &str = "\
You are the refinement planner for a coding-agent harness. The harness keeps a \
small, durable layer of memories, prompt notes, and skills that is injected into \
the agent's system prompt. Your job is to propose the SMALLEST useful edit to \
that layer based on the session trajectory below — never rewrite the whole \
harness, and never invent evidence.

Respond with ONLY a JSON object, no prose:
{\"ops\": [{\"action\": \"create\"|\"update\"|\"delete\", \"kind\": \"memory\"|\"note\"|\"skill\", \"id\": \"entry-id-or-empty-for-create\", \"text\": \"new text (for create/update)\", \"reason\": \"one line why\"}]}

Rules:
- create: new durable fact (memory), instruction (note), or reusable tactic (skill).
- update: only when the existing text is now wrong; id must match an existing id.
- delete: only when an entry is wrong/harmful; id must match an existing id.
- Empty ops array ({\"ops\": []}) is a valid answer when nothing should change.
- Keep text under 200 characters; keep ops under 5.";

/// Collect the recent trajectory as plain-text rows: the last `limit` journal
/// rows of the current session, oldest → newest (chronological), each capped
/// in length so the planner request stays bounded.
pub fn collect_trajectory(agent_dir: &Path, cwd: &Path, limit: usize) -> Vec<String> {
    let mut rows: Vec<String> = crate::sessions::journal(agent_dir, cwd)
        .into_iter()
        .map(|entry| {
            let text: String = entry.text.chars().take(300).collect();
            format!("{} {}: {}", entry.time, entry.kind, text)
        })
        .collect();
    if rows.len() > limit {
        rows = rows.split_off(rows.len() - limit);
    }
    rows
}

/// Render the current harness state for the planner (entries + recent ops).
pub fn render_harness_state(harness: &crate::harness::Harness) -> String {
    if harness.entries.is_empty() {
        return "(empty harness — nothing has been learned yet)".to_string();
    }
    let mut lines: Vec<String> = harness
        .entries
        .iter()
        .map(|entry| format!("{} {}: {}", entry.kind.as_str(), entry.id, entry.text))
        .collect();
    lines.sort();
    lines.join("\n")
}

/// Tolerant parser: extracts the JSON from a model reply (bare JSON or a
/// ```json fence), validates every op, drops invalid ones, and dedupes.
pub fn parse_refine_proposal(text: &str) -> RefineProposal {
    let json_text = extract_json_block(text);
    let Ok(value) = serde_json::from_str::<serde_json::Value>(json_text) else {
        return RefineProposal::default();
    };
    let Some(ops) = value.get("ops").and_then(serde_json::Value::as_array) else {
        return RefineProposal::default();
    };
    let mut proposal = RefineProposal::default();
    let mut seen = std::collections::HashSet::new();
    for op in ops {
        let Some(action) = op.get("action").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let Some(kind) = op.get("kind").and_then(serde_json::Value::as_str) else {
            continue;
        };
        if !matches!(action, "create" | "update" | "delete") {
            continue;
        }
        if !matches!(kind, "memory" | "note" | "skill") {
            continue;
        }
        let id = op
            .get("id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .chars()
            .take(80)
            .collect::<String>();
        let text = op
            .get("text")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .chars()
            .take(300)
            .collect::<String>();
        let reason = op
            .get("reason")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .chars()
            .take(200)
            .collect::<String>();
        // create/update need text; delete does not.
        if action != "delete" && text.trim().is_empty() {
            continue;
        }
        // Creates have no id yet — key on the text so distinct proposals
        // survive; updates/deletes key on the id.
        let key = if action == "create" {
            (action, kind, text.clone())
        } else {
            (action, kind, id.clone())
        };
        if !seen.insert(key) {
            continue;
        }
        proposal.ops.push(ProposedOp {
            action: action.to_string(),
            kind: kind.to_string(),
            id,
            text,
            reason,
        });
        if proposal.ops.len() >= 5 {
            break;
        }
    }
    proposal
}

/// Pull the JSON out of a model reply: a ```json fence, a bare ``` fence, or
/// the first balanced {...} block.
fn extract_json_block(text: &str) -> &str {
    let trimmed = text.trim();
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        &trimmed[start..=end]
    } else {
        trimmed
    }
}

/// Run the planner: stream one completion from `provider` (no tools) with the
/// trajectory + current harness state and parse the proposal. Errors never
/// panic — they surface as `Err(AiError)`.
pub async fn propose_refinement(
    provider: &dyn Provider,
    model: &str,
    trajectory: &[String],
    harness_state: &str,
    focus: Option<&str>,
) -> Result<RefineProposal, agent_m_ai::AiError> {
    let mut body = String::new();
    body.push_str("CURRENT HARNESS STATE:\n");
    body.push_str(harness_state);
    body.push_str("\n\nSESSION TRAJECTORY (oldest first):\n");
    if trajectory.is_empty() {
        body.push_str("(no recent activity)");
    } else {
        body.push_str(&trajectory.join("\n"));
    }
    if let Some(focus) = focus {
        body.push_str(&format!("\n\nFOCUS: {focus}"));
    }
    let request = ChatRequest {
        model: model.to_string(),
        system: PLANNER_SYSTEM.to_string(),
        messages: vec![LlmMessage::User {
            content: body,
            images: Vec::new(),
        }],
        tools: vec![],
        temperature: Some(0.0),
        variant: None,
    };
    let stream = provider.stream_chat(request).await?;
    futures_util::pin_mut!(stream);
    let mut reply = String::new();
    while let Some(event) = stream.next().await {
        match event {
            StreamEvent::TextDelta { delta } => reply.push_str(&delta),
            StreamEvent::Error { .. } => {
                return Err(agent_m_ai::AiError::Api(
                    "stream error during refinement planning".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(parse_refine_proposal(&reply))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_proposal_and_drops_invalid_ops() {
        let reply = r#"{
  "ops": [
    {"action": "create", "kind": "memory", "id": "", "text": "flaky tests need retries", "reason": "saw 3 failures"},
    {"action": "update", "kind": "note", "id": "note-1", "text": "always check the cache", "reason": "stale"},
    {"action": "delete", "kind": "skill", "id": "skill-old", "text": "", "reason": "wrong"},
    {"action": "explode", "kind": "memory", "id": "", "text": "invalid", "reason": "bad action"},
    {"action": "create", "kind": "memory", "id": "", "text": "", "reason": "no text"}
  ]
}"#;
        let proposal = parse_refine_proposal(reply);
        assert_eq!(proposal.ops.len(), 3);
        assert_eq!(proposal.ops[0].action, "create");
        assert_eq!(proposal.ops[1].action, "update");
        assert_eq!(proposal.ops[2].action, "delete");
    }

    #[test]
    fn parses_fenced_json_and_empty_ops() {
        let fenced = "Here you go:\n```json\n{\"ops\": []}\n```\n";
        assert!(parse_refine_proposal(fenced).ops.is_empty());
        assert!(parse_refine_proposal("no json here").ops.is_empty());
        assert!(parse_refine_proposal("{\"unexpected\": 1}").ops.is_empty());
    }

    #[test]
    fn dedupes_repeated_ops_and_caps_at_five() {
        let reply = r#"{"ops": [
            {"action":"create","kind":"memory","id":"","text":"a","reason":"1"},
            {"action":"create","kind":"memory","id":"","text":"a","reason":"dup"},
            {"action":"create","kind":"memory","id":"","text":"b","reason":"2"},
            {"action":"create","kind":"memory","id":"","text":"c","reason":"3"},
            {"action":"create","kind":"memory","id":"","text":"d","reason":"4"},
            {"action":"create","kind":"memory","id":"","text":"e","reason":"5"}
        ]}"#;
        let proposal = parse_refine_proposal(reply);
        assert_eq!(proposal.ops.len(), 5);
        assert_eq!(proposal.ops[0].text, "a");
        assert_eq!(proposal.ops[1].text, "b");
    }

    #[test]
    fn trajectory_is_chronological_and_capped() {
        let dir = tempfile::tempdir().unwrap();
        // Seed a session with 6 rows; journal() reads them in order.
        let session_dir = dir.path().join("sessions").join("--_tmp_proj--");
        std::fs::create_dir_all(&session_dir).unwrap();
        let path = session_dir.join("2026-08-12T10-00-00.000Z-aaa.jsonl");
        let mut body = String::from("{\"type\":\"session\",\"version\":1}\n");
        for i in 0..6 {
            body.push_str(&format!(
                "{{\"type\":\"message\",\"kind\":\"user\",\"content\":\"step {i}\"}}\n"
            ));
        }
        std::fs::write(&path, body).unwrap();
        let cwd = Path::new("/tmp/proj");
        let rows = collect_trajectory(dir.path(), cwd, 3);
        assert_eq!(rows.len(), 3);
        assert!(
            rows.last().unwrap().contains("step 5"),
            "newest last: {rows:?}"
        );
    }

    /// A provider that replays a canned reply as a text stream.
    struct FakePlannerProvider {
        reply: &'static str,
    }

    #[async_trait::async_trait]
    impl Provider for FakePlannerProvider {
        fn id(&self) -> &str {
            "fake"
        }
        fn display_name(&self) -> &str {
            "Fake Planner"
        }
        fn api_key(&self) -> Option<&str> {
            None
        }
        fn set_api_key(&mut self, _key: String) {}
        fn models(&self) -> &[agent_m_ai::ModelSpec] {
            &[]
        }
        async fn stream_chat(
            &self,
            _request: ChatRequest,
        ) -> Result<futures_util::stream::BoxStream<'static, StreamEvent>, agent_m_ai::AiError>
        {
            let events = vec![StreamEvent::TextDelta {
                delta: self.reply.to_string(),
            }];
            Ok(Box::pin(futures_util::stream::iter(events)))
        }
    }

    #[tokio::test]
    async fn propose_refinement_streams_and_parses() {
        let provider = FakePlannerProvider {
            reply: "{\"ops\": [{\"action\": \"create\", \"kind\": \"skill\", \"id\": \"\", \"text\": \"prefer interactive rebase\", \"reason\": \"saw it work\"}]}",
        };
        let trajectory = vec!["t0 tool_result: done".to_string()];
        let proposal = propose_refinement(
            &provider,
            "fake-model",
            &trajectory,
            "(empty harness — nothing has been learned yet)",
            Some("git workflow"),
        )
        .await
        .expect("planner runs");
        assert_eq!(proposal.ops.len(), 1);
        assert_eq!(proposal.ops[0].action, "create");
        assert_eq!(proposal.ops[0].kind, "skill");
        assert_eq!(proposal.ops[0].text, "prefer interactive rebase");
    }
}
