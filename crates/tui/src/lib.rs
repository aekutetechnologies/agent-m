//! agent-m-tui: pi-style terminal UI built on ratatui.

pub mod app;
pub mod editor;
pub mod keybindings;
pub mod markdown;
pub mod plan;
pub mod sessions;
pub mod theme;
pub mod transcript;

pub use app::{App, AppInputs, UiMode};
pub use editor::Editor;
pub use keybindings::{Action, AppAction, EditorAction, KeyContext};
pub use theme::Theme;
pub use transcript::TranscriptItem;
