//! Key handling: maps crossterm keys to actions, following pi's default
//! keybindings (see `packages/coding-agent/src/core/keybindings.ts`).

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

/// Editor-level actions (pi's `tui.editor.*` bindings).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorAction {
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    WordLeft,
    WordRight,
    LineStart,
    LineEnd,
    Backspace,
    Delete,
    KillWordBackward,
    KillToStart,
    KillToEnd,
    Yank,
    Undo,
    Newline,
    /// Insert a typed character.
    PasteText(String),
}

/// App-level actions (pi's `app.*` bindings).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppAction {
    Submit,
    TabComplete,
    Clear,
    Exit,
    Interrupt,
    ModelSelect,
    ModelCycleForward,
    ModelCycleBackward,
    ToggleToolOutput,
    ToggleThinking,
    /// Open/close the session info panel (ctrl+i).
    ToggleInfo,
    ApproveTool,
    DenyTool,
    ScrollUp,
    ScrollDown,
    ScrollTop,
    ScrollBottom,
}

/// A single resolved key press.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    Editor(EditorAction),
    App(AppAction),
}

/// Context that changes key interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyContext {
    /// The editor is empty (pi: ctrl+d exits only when empty; ctrl+c on an
    /// empty editor exits).
    pub editor_empty: bool,
    /// A tool call is awaiting approval (y/n keys).
    pub approval_pending: bool,
}

/// Resolve a key event to an action. Mirrors pi's defaults:
/// enter submit, shift+enter/ctrl+j newline, tab autocomplete, ctrl+c clear
/// (double ctrl+c exits), ctrl+d exit when empty, escape interrupt,
/// ctrl+l model select, ctrl+o toggle tool output, ctrl+p/ctrl+shift+p model
/// cycle, ctrl+t toggle thinking, y/n approve/deny while a call is pending.
pub fn resolve_key(key: KeyEvent, context: KeyContext) -> Option<Action> {
    let modifiers = key.modifiers;
    let ctrl = modifiers.contains(KeyModifiers::CONTROL);
    let alt = modifiers.contains(KeyModifiers::ALT);
    let shift = modifiers.contains(KeyModifiers::SHIFT);

    // Approval mode: y/n take over, but the exit keys (ctrl+d, ctrl+c) still
    // work — a user stuck at the approval prompt must be able to quit.
    if context.approval_pending {
        match key.code {
            KeyCode::Char('y' | 'Y') => return Some(Action::App(AppAction::ApproveTool)),
            KeyCode::Char('n' | 'N') => return Some(Action::App(AppAction::DenyTool)),
            KeyCode::Esc => return Some(Action::App(AppAction::DenyTool)),
            // Fall through to the general bindings so exit works.
            KeyCode::Char('d' | 'c') if ctrl => {}
            _ => return None,
        }
    }

    match key.code {
        KeyCode::Enter if shift || ctrl => Some(Action::Editor(EditorAction::Newline)),
        KeyCode::Enter => Some(Action::App(AppAction::Submit)),
        KeyCode::Tab => Some(Action::App(AppAction::TabComplete)),
        KeyCode::Esc => Some(Action::App(AppAction::Interrupt)),
        KeyCode::Char('c') if ctrl => {
            if context.editor_empty {
                Some(Action::App(AppAction::Exit))
            } else {
                Some(Action::App(AppAction::Clear))
            }
        }
        KeyCode::Char('d') if ctrl => Some(Action::App(AppAction::Exit)),
        KeyCode::Char('j') if ctrl => Some(Action::Editor(EditorAction::Newline)),
        KeyCode::Char('l') if ctrl => Some(Action::App(AppAction::ModelSelect)),
        KeyCode::Char('o') if ctrl => Some(Action::App(AppAction::ToggleToolOutput)),
        // ctrl+i is Tab in terminals, so the info panel uses ctrl+n (and /info).
        KeyCode::Char('n') if ctrl => Some(Action::App(AppAction::ToggleInfo)),
        KeyCode::Char('p') if ctrl && shift => Some(Action::App(AppAction::ModelCycleBackward)),
        KeyCode::Char('p') if ctrl => Some(Action::App(AppAction::ModelCycleForward)),
        KeyCode::Char('t') if ctrl => Some(Action::App(AppAction::ToggleThinking)),
        KeyCode::Char('b') if ctrl => Some(Action::Editor(EditorAction::MoveLeft)),
        KeyCode::Char('f') if ctrl => Some(Action::Editor(EditorAction::MoveRight)),
        KeyCode::Char('a') if ctrl => Some(Action::Editor(EditorAction::LineStart)),
        KeyCode::Char('e') if ctrl => Some(Action::Editor(EditorAction::LineEnd)),
        KeyCode::Char('w') if ctrl => Some(Action::Editor(EditorAction::KillWordBackward)),
        KeyCode::Char('u') if ctrl => Some(Action::Editor(EditorAction::KillToStart)),
        KeyCode::Char('k') if ctrl => Some(Action::Editor(EditorAction::KillToEnd)),
        KeyCode::Char('y') if ctrl => Some(Action::Editor(EditorAction::Yank)),
        KeyCode::Char('-') if ctrl => Some(Action::Editor(EditorAction::Undo)),
        KeyCode::Char(' ') if ctrl => Some(Action::App(AppAction::Interrupt)),
        KeyCode::Up => Some(Action::Editor(EditorAction::MoveUp)),
        KeyCode::Down => Some(Action::Editor(EditorAction::MoveDown)),
        KeyCode::Left if ctrl => Some(Action::Editor(EditorAction::WordLeft)),
        KeyCode::Left if alt => Some(Action::Editor(EditorAction::WordLeft)),
        KeyCode::Left => Some(Action::Editor(EditorAction::MoveLeft)),
        KeyCode::Right if ctrl => Some(Action::Editor(EditorAction::WordRight)),
        KeyCode::Right if alt => Some(Action::Editor(EditorAction::WordRight)),
        KeyCode::Right => Some(Action::Editor(EditorAction::MoveRight)),
        KeyCode::Home => Some(Action::Editor(EditorAction::LineStart)),
        KeyCode::End => Some(Action::Editor(EditorAction::LineEnd)),
        KeyCode::Backspace => Some(Action::Editor(EditorAction::Backspace)),
        KeyCode::Delete => Some(Action::Editor(EditorAction::Delete)),
        KeyCode::PageUp => Some(Action::App(AppAction::ScrollUp)),
        KeyCode::PageDown => Some(Action::App(AppAction::ScrollDown)),
        KeyCode::Char(character) => {
            if ctrl || alt {
                None
            } else {
                Some(Action::Editor(EditorAction::PasteText(
                    character.to_string(),
                )))
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::KeyEvent;

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, modifiers)
    }

    fn idle() -> KeyContext {
        KeyContext {
            editor_empty: true,
            approval_pending: false,
        }
    }

    #[test]
    fn basic_submit_and_newline() {
        assert_eq!(
            resolve_key(key(KeyCode::Enter, KeyModifiers::NONE), idle()),
            Some(Action::App(AppAction::Submit))
        );
        assert_eq!(
            resolve_key(key(KeyCode::Enter, KeyModifiers::SHIFT), idle()),
            Some(Action::Editor(EditorAction::Newline))
        );
        assert_eq!(
            resolve_key(key(KeyCode::Char('j'), KeyModifiers::CONTROL), idle()),
            Some(Action::Editor(EditorAction::Newline))
        );
    }

    #[test]
    fn ctrl_c_clears_or_exits() {
        let ctrl_c = key(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(
            resolve_key(ctrl_c, idle()),
            Some(Action::App(AppAction::Exit))
        );
        assert_eq!(
            resolve_key(
                ctrl_c,
                KeyContext {
                    editor_empty: false,
                    approval_pending: false
                }
            ),
            Some(Action::App(AppAction::Clear))
        );
    }

    #[test]
    fn pi_app_bindings() {
        assert_eq!(
            resolve_key(key(KeyCode::Char('l'), KeyModifiers::CONTROL), idle()),
            Some(Action::App(AppAction::ModelSelect))
        );
        assert_eq!(
            resolve_key(key(KeyCode::Char('o'), KeyModifiers::CONTROL), idle()),
            Some(Action::App(AppAction::ToggleToolOutput))
        );
        assert_eq!(
            resolve_key(key(KeyCode::Char('n'), KeyModifiers::CONTROL), idle()),
            Some(Action::App(AppAction::ToggleInfo))
        );
        assert_eq!(
            resolve_key(key(KeyCode::Esc, KeyModifiers::NONE), idle()),
            Some(Action::App(AppAction::Interrupt))
        );
        assert_eq!(
            resolve_key(key(KeyCode::Char('d'), KeyModifiers::CONTROL), idle()),
            Some(Action::App(AppAction::Exit))
        );
    }

    #[test]
    fn approval_keys_override() {
        let context = KeyContext {
            editor_empty: true,
            approval_pending: true,
        };
        assert_eq!(
            resolve_key(key(KeyCode::Char('y'), KeyModifiers::NONE), context),
            Some(Action::App(AppAction::ApproveTool))
        );
        assert_eq!(
            resolve_key(key(KeyCode::Char('n'), KeyModifiers::NONE), context),
            Some(Action::App(AppAction::DenyTool))
        );
        // Regular keys are swallowed while a decision is pending.
        assert_eq!(
            resolve_key(key(KeyCode::Char('a'), KeyModifiers::NONE), context),
            None
        );
        // …but the exit keys must still work so the user is never trapped.
        assert_eq!(
            resolve_key(key(KeyCode::Char('d'), KeyModifiers::CONTROL), context),
            Some(Action::App(AppAction::Exit))
        );
        assert_eq!(
            resolve_key(key(KeyCode::Char('c'), KeyModifiers::CONTROL), context),
            Some(Action::App(AppAction::Exit))
        );
    }
}
