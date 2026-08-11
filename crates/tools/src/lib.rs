//! agent-m-tools: built-in agent tools.
//!
//! Ports pi's core tool set: `bash`, `read`, `write`, `edit`, `grep`, `find`,
//! `ls`. All tools implement the `Tool` trait from `agent-m-agent`; permission
//! decisions are made by the agent's permission gate, not inside the tools.

mod ask;
mod bash;
mod edit;
mod find;
mod grep;
mod index;
mod ls;
mod paths;
mod read;
mod sandbox;
mod search;
mod truncate;
mod web;
mod write;

pub use ask::AskTool;
pub use bash::BashTool;
pub use edit::EditTool;
pub use find::FindTool;
pub use grep::GrepTool;
pub use index::{SymbolIndex, build_index, load_or_build};
pub use ls::LsTool;
pub use paths::{resolve_path, set_allowed_paths};
pub use read::ReadTool;
pub use search::SearchTool;
pub use truncate::{MAX_BYTES, MAX_LINES};
pub use web::{WebFetchTool, WebSearchTool};
pub use write::WriteTool;

use agent_m_agent::Tool;
use std::sync::Arc;

/// All built-in tools.
pub fn all_tools() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(BashTool),
        Arc::new(ReadTool),
        Arc::new(WriteTool),
        Arc::new(EditTool),
        Arc::new(GrepTool),
        Arc::new(FindTool),
        Arc::new(LsTool),
        Arc::new(AskTool),
        Arc::new(SearchTool),
        Arc::new(WebFetchTool),
        Arc::new(WebSearchTool),
    ]
}

/// The default active set, mirroring pi (all tools active): `read`, `bash`,
/// `ls`, `grep`, `find`, `edit`, `write`, plus `ask` and `search`.
/// Registering the read-only tools also gives plan mode its exploration set
/// (the agent filters them by mode).
pub fn default_tools() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(ReadTool),
        Arc::new(BashTool),
        Arc::new(LsTool),
        Arc::new(GrepTool),
        Arc::new(FindTool),
        Arc::new(EditTool),
        Arc::new(WriteTool),
        Arc::new(AskTool),
        Arc::new(SearchTool),
        Arc::new(WebFetchTool),
        Arc::new(WebSearchTool),
    ]
}
