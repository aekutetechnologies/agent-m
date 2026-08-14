//! Remote human channel (check.md principle 4): a shared registry of pending
//! human questions. The Slack connector posts a question, stores a oneshot,
//! and resolves it when a reply arrives. The ask tool and the permission gate
//! use the same registry, so a human who is not at the terminal can answer
//! mid-task — the foundation for unattended long-horizon runs.
//!
//! Transport-agnostic on purpose: the REPL, daemon, flow runner, and Slack
//! all share one `HumanChannel`; only the transport differs.

use crate::slack::SlackTransport;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{broadcast, oneshot};

/// Lifecycle of a question posted to the channel.
/// Wired into the daemon/flow in Phase 2; used by tests now.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum ChannelEvent {
    /// A question was posted and is awaiting an answer.
    Posted { id: String, question: String },
    /// A reply resolved a pending question.
    Answered { id: String, answer: String },
}

/// Shared registry of pending human questions.
/// `ask`/`subscribe` are wired in Phase 2 (daemon + flow); tests use them now.
#[allow(dead_code)]
pub struct HumanChannel {
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<String>>>>,
    next_id: AtomicU64,
    events: broadcast::Sender<ChannelEvent>,
}

#[allow(dead_code)] // wired into daemon/flow ask paths in Phase 2
impl HumanChannel {
    pub fn new() -> Self {
        let (events, _) = broadcast::channel(64);
        Self {
            pending: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(1),
            events,
        }
    }

    /// Subscribe to the question/answer lifecycle (tests, progress panels).
    pub fn subscribe(&self) -> broadcast::Receiver<ChannelEvent> {
        self.events.subscribe()
    }

    /// Post `question` through `transport`, then wait for a reply that
    /// resolves the generated id (a message like `ask-3 <answer>`).
    /// Times out after `timeout` (default 5 minutes) and cleans up the
    /// pending entry.
    pub async fn ask(
        &self,
        transport: &dyn SlackTransport,
        channel: &str,
        question: &str,
        options: Option<Vec<String>>,
        timeout: Option<Duration>,
    ) -> Result<String, String> {
        let id = format!("ask-{}", self.next_id.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .map_err(|_| "human channel lock poisoned".to_string())?
            .insert(id.clone(), tx);

        let mut text = format!("❓ {question}");
        if let Some(opts) = &options {
            for (i, opt) in opts.iter().enumerate() {
                text.push_str(&format!("\n  {}. {}", i + 1, opt));
            }
        }
        text.push_str(&format!("\nreply with: {id} <answer>"));

        transport.post_message(channel, &text).await?;
        let _ = self.events.send(ChannelEvent::Posted {
            id: id.clone(),
            question: question.to_string(),
        });

        let waited =
            tokio::time::timeout(timeout.unwrap_or(Duration::from_secs(300)), rx).await;
        match waited {
            Ok(Ok(answer)) => {
                let _ = self
                    .events
                    .send(ChannelEvent::Answered { id, answer: answer.clone() });
                Ok(answer)
            }
            _ => {
                self.pending
                    .lock()
                    .map_err(|_| "human channel lock poisoned".to_string())?
                    .remove(&id);
                Err("no answer received before timeout".to_string())
            }
        }
    }

    /// Complete a pending question. Returns false if the id is unknown
    /// (already answered or expired).
    pub fn resolve(&self, id: &str, answer: String) -> bool {
        let tx = self.pending.lock().ok().and_then(|mut pending| pending.remove(id));
        match tx {
            Some(tx) => {
                let _ = tx.send(answer);
                true
            }
            None => false,
        }
    }

    /// How many questions are awaiting an answer.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().map(|p| p.len()).unwrap_or(0)
    }
}
