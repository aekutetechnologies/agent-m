//! A small multi-line text editor with pi's keybinding semantics (kill ring,
//! undo, word navigation).

use unicode_width::UnicodeWidthStr;

/// A multi-line editor buffer with char-based cursor positions.
#[derive(Debug, Clone, Default)]
pub struct Editor {
    lines: Vec<String>,
    cursor_line: usize,
    cursor_col: usize,
    kill_ring: Option<String>,
    undo_stack: Vec<Snapshot>,
    /// Marks whether the last edit pushed an undo snapshot (avoid duplicates).
    last_snapshot_dirty: bool,
}

type Snapshot = (Vec<String>, usize, usize);

const MAX_UNDO: usize = 100;

impl Editor {
    pub fn new() -> Self {
        Self {
            lines: vec![String::new()],
            ..Default::default()
        }
    }

    pub fn set_text(&mut self, text: &str) {
        self.snapshot();
        self.lines = text.lines().map(str::to_string).collect();
        if self.lines.is_empty() {
            self.lines.push(String::new());
        }
        self.cursor_line = self.lines.len() - 1;
        self.cursor_col = self.lines[self.cursor_line].chars().count();
    }

    /// All lines joined with newlines (no trailing newline).
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    pub fn is_empty(&self) -> bool {
        self.lines.iter().all(|line| line.is_empty())
    }

    pub fn cursor_line(&self) -> usize {
        self.cursor_line
    }

    pub fn cursor_col(&self) -> usize {
        self.cursor_col
    }

    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// The character index of the cursor within the current line.
    fn char_index(line: &str, col: usize) -> usize {
        line.chars().take(col).map(char::len_utf8).sum()
    }

    fn clamp_cursor(&mut self) {
        self.cursor_line = self.cursor_line.min(self.lines.len() - 1);
        self.cursor_col = self
            .cursor_col
            .min(self.lines[self.cursor_line].chars().count());
    }

    fn snapshot(&mut self) {
        self.undo_stack
            .push((self.lines.clone(), self.cursor_line, self.cursor_col));
        if self.undo_stack.len() > MAX_UNDO {
            self.undo_stack.remove(0);
        }
        self.last_snapshot_dirty = true;
    }

    fn begin_edit(&mut self) {
        if !self.last_snapshot_dirty {
            self.snapshot();
        }
    }

    pub fn insert_char(&mut self, character: char) {
        self.begin_edit();
        let index = Self::char_index(&self.lines[self.cursor_line], self.cursor_col);
        self.lines[self.cursor_line].insert(index, character);
        self.cursor_col += 1;
    }

    pub fn insert_text(&mut self, text: &str) {
        self.begin_edit();
        let index = Self::char_index(&self.lines[self.cursor_line], self.cursor_col);
        self.lines[self.cursor_line].insert_str(index, text);
        self.cursor_col += text.chars().count();
    }

    pub fn newline(&mut self) {
        self.begin_edit();
        let index = Self::char_index(&self.lines[self.cursor_line], self.cursor_col);
        let rest: String = self.lines[self.cursor_line][index..].to_string();
        self.lines[self.cursor_line].truncate(index);
        self.cursor_line += 1;
        self.lines.insert(self.cursor_line, rest);
        self.cursor_col = 0;
    }

    pub fn backspace(&mut self) {
        if self.cursor_col > 0 {
            self.begin_edit();
            let index = Self::char_index(&self.lines[self.cursor_line], self.cursor_col);
            let previous = self.lines[self.cursor_line][..index]
                .chars()
                .next_back()
                .map(char::len_utf8)
                .unwrap_or(0);
            self.lines[self.cursor_line].remove(index - previous);
            self.cursor_col -= 1;
        } else if self.cursor_line > 0 {
            self.begin_edit();
            let previous_len = self.lines[self.cursor_line - 1].chars().count();
            let rest = self.lines.remove(self.cursor_line);
            self.lines[self.cursor_line - 1].push_str(&rest);
            self.cursor_line -= 1;
            self.cursor_col = previous_len;
        }
    }

    pub fn delete(&mut self) {
        if self.cursor_col < self.lines[self.cursor_line].chars().count() {
            self.begin_edit();
            let index = Self::char_index(&self.lines[self.cursor_line], self.cursor_col);
            let length = self.lines[self.cursor_line][index..]
                .chars()
                .next()
                .map(char::len_utf8)
                .unwrap_or(0);
            self.lines[self.cursor_line].remove(index);
            let _ = length;
        } else if self.cursor_line + 1 < self.lines.len() {
            self.begin_edit();
            let rest = self.lines.remove(self.cursor_line + 1);
            self.lines[self.cursor_line].push_str(&rest);
        }
    }

    pub fn move_left(&mut self) {
        if self.cursor_col > 0 {
            self.cursor_col -= 1;
        } else if self.cursor_line > 0 {
            self.cursor_line -= 1;
            self.cursor_col = self.lines[self.cursor_line].chars().count();
        }
    }

    pub fn move_right(&mut self) {
        if self.cursor_col < self.lines[self.cursor_line].chars().count() {
            self.cursor_col += 1;
        } else if self.cursor_line + 1 < self.lines.len() {
            self.cursor_line += 1;
            self.cursor_col = 0;
        }
    }

    pub fn move_up(&mut self) {
        if self.cursor_line > 0 {
            self.cursor_line -= 1;
            self.clamp_cursor();
        }
    }

    pub fn move_down(&mut self) {
        if self.cursor_line + 1 < self.lines.len() {
            self.cursor_line += 1;
            self.clamp_cursor();
        }
    }

    pub fn word_left(&mut self) {
        let line = &self.lines[self.cursor_line];
        let index = Self::char_index(line, self.cursor_col);
        let mut skipped = false;
        let mut position = index;
        while position > 0 {
            let previous = line[..position].chars().next_back().unwrap();
            if !skipped && previous.is_alphanumeric() {
                skipped = true;
            } else if skipped && !previous.is_alphanumeric() {
                break;
            }
            position -= previous.len_utf8();
        }
        self.cursor_col = line[..position].chars().count();
    }

    pub fn word_right(&mut self) {
        let line = &self.lines[self.cursor_line];
        let mut position = Self::char_index(line, self.cursor_col);
        let mut seen_word = false;
        while position < line.len() {
            let character = line[position..].chars().next().unwrap();
            if character.is_alphanumeric() {
                seen_word = true;
            } else if seen_word {
                break;
            }
            position += character.len_utf8();
        }
        if !seen_word {
            // Skip trailing whitespace.
            while position < line.len()
                && line[position..]
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_whitespace())
            {
                position += line[position..].chars().next().unwrap().len_utf8();
            }
        }
        self.cursor_col = line[..position].chars().count();
    }

    pub fn line_start(&mut self) {
        self.cursor_col = 0;
    }

    pub fn line_end(&mut self) {
        self.cursor_col = self.lines[self.cursor_line].chars().count();
    }

    /// Kill from the cursor to the end of the line (`ctrl+k`).
    pub fn kill_to_end(&mut self) {
        self.begin_edit();
        let index = Self::char_index(&self.lines[self.cursor_line], self.cursor_col);
        self.kill_ring = Some(self.lines[self.cursor_line][index..].to_string());
        self.lines[self.cursor_line].truncate(index);
    }

    /// Kill from the line start to the cursor (`ctrl+u`).
    pub fn kill_to_start(&mut self) {
        self.begin_edit();
        let index = Self::char_index(&self.lines[self.cursor_line], self.cursor_col);
        self.kill_ring = Some(self.lines[self.cursor_line][..index].to_string());
        self.lines[self.cursor_line].drain(..index);
        self.cursor_col = 0;
    }

    /// Delete the word before the cursor (`ctrl+w`).
    pub fn kill_word_backward(&mut self) {
        let before = self.cursor_col;
        self.word_left();
        if self.cursor_col < before {
            self.begin_edit();
            let start = Self::char_index(&self.lines[self.cursor_line], self.cursor_col);
            let end = Self::char_index(&self.lines[self.cursor_line], before);
            let killed = self.lines[self.cursor_line][start..end].to_string();
            self.lines[self.cursor_line].replace_range(start..end, "");
            self.kill_ring = Some(killed);
        }
    }

    /// Yank the kill ring at the cursor (`ctrl+y`).
    pub fn yank(&mut self) {
        if let Some(text) = self.kill_ring.clone() {
            self.insert_text(&text);
        }
    }

    /// Undo the last edit (`ctrl+-`).
    pub fn undo(&mut self) {
        if let Some((lines, cursor_line, cursor_col)) = self.undo_stack.pop() {
            self.lines = lines;
            self.cursor_line = cursor_line;
            self.cursor_col = cursor_col;
            self.last_snapshot_dirty = false;
        }
    }

    /// Render the visible lines given a viewport, returning the lines and the
    /// cursor position within the rendered text.
    pub fn visible_lines(&self, view_top: usize, height: usize) -> (&[String], usize, usize) {
        let top = view_top.min(self.lines.len());
        let end = (top + height).min(self.lines.len());
        let visible = &self.lines[top..end];
        let cursor = self.cursor_line.saturating_sub(top);
        (visible, cursor, self.cursor_col)
    }

    /// The viewport top that keeps the cursor visible within `height` rows.
    pub fn viewport_for_cursor(&self, view_top: usize, height: usize) -> usize {
        if self.cursor_line < view_top {
            self.cursor_line
        } else if self.cursor_line >= view_top + height {
            self.cursor_line.saturating_sub(height - 1)
        } else {
            view_top
        }
    }

    /// Width of the widest visible line (for rendering hints).
    pub fn max_line_width(&self, start: usize, end: usize) -> usize {
        self.lines[start..end.min(self.lines.len())]
            .iter()
            .map(|line| UnicodeWidthStr::width(line.as_str()))
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typing_and_navigation() {
        let mut editor = Editor::new();
        editor.insert_text("hello");
        editor.newline();
        editor.insert_text("world");
        assert_eq!(editor.text(), "hello\nworld");
        assert_eq!(editor.cursor_line(), 1);
        assert_eq!(editor.cursor_col(), 5);

        editor.move_up();
        assert_eq!(editor.cursor_line(), 0);
        editor.line_end();
        assert_eq!(editor.cursor_col(), 5);
        editor.backspace();
        assert_eq!(editor.text(), "hell\nworld");
    }

    #[test]
    fn kill_yank_undo() {
        let mut editor = Editor::new();
        editor.insert_text("hello world");
        editor.line_start();
        editor.kill_to_end();
        assert_eq!(editor.text(), "");
        editor.yank();
        assert_eq!(editor.text(), "hello world");
        // Undo reverts the whole edit burst to the last snapshot (pre-insert).
        editor.undo();
        assert_eq!(editor.text(), "");
    }

    #[test]
    fn word_navigation() {
        let mut editor = Editor::new();
        editor.insert_text("foo bar baz");
        editor.line_end();
        editor.word_left();
        editor.word_left();
        assert_eq!(editor.cursor_col(), 4);
        editor.word_right();
        assert_eq!(editor.cursor_col(), 7);
    }

    #[test]
    fn viewport_keeps_cursor_visible() {
        let mut editor = Editor::new();
        editor.set_text(
            &(0..10)
                .map(|i| format!("line {i}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        editor.cursor_line = 9;
        // Cursor below the window: window moves down.
        let top = editor.viewport_for_cursor(0, 5);
        assert_eq!(top, 5);
        // Cursor already visible: window stays.
        let top2 = editor.viewport_for_cursor(8, 5);
        assert_eq!(top2, 8);
    }
}
