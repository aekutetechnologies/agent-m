//! Task planning: extract a numbered `Plan:` list from an assistant reply and
//! track `[DONE:n]` completion markers (pi's plan-mode extension pattern).

/// One planned step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TodoItem {
    pub step: usize,
    pub text: String,
    pub completed: bool,
}

/// Parse a numbered plan out of an assistant reply. The plan starts at a line
/// that reads `Plan:` (possibly a markdown heading or bold) and continues
/// while lines are numbered (`1. step`, `2) step`). Stops at the first
/// non-numbered, non-blank line (the next section).
pub fn parse_plan(text: &str) -> Vec<TodoItem> {
    let mut items: Vec<TodoItem> = Vec::new();
    let mut in_plan = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if !in_plan {
            if is_plan_header(trimmed) {
                in_plan = true;
            }
            continue;
        }
        if trimmed.is_empty() {
            continue; // blank lines inside the plan are tolerated
        }
        match parse_numbered(trimmed) {
            Some(rest) => items.push(TodoItem {
                step: items.len() + 1,
                text: rest.to_string(),
                completed: false,
            }),
            None => break, // next section started
        }
    }
    items
}

fn is_plan_header(line: &str) -> bool {
    let stripped = line
        .trim()
        .trim_start_matches('#')
        .trim()
        .trim_matches('*')
        .trim()
        .to_ascii_lowercase();
    stripped == "plan:" || stripped == "plan"
}

fn parse_numbered(line: &str) -> Option<&str> {
    let digits: usize = line.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits == 0 {
        return None;
    }
    let after = line[digits..].trim_start();
    let sep = after.chars().next()?;
    if sep != '.' && sep != ')' {
        return None;
    }
    let rest = after[sep.len_utf8()..].trim_start();
    if rest.is_empty() { None } else { Some(rest) }
}

/// Mark todos completed whose step number appears as `[DONE:n]` (case
/// insensitive, optional space after the colon). Returns whether anything
/// changed.
pub fn apply_done_markers(text: &str, todos: &mut [TodoItem]) -> bool {
    let lower = text.to_lowercase();
    let mut changed = false;
    for todo in todos.iter_mut() {
        if todo.completed {
            continue;
        }
        let with_space = format!("[done: {}]", todo.step);
        let tight = format!("[done:{}]", todo.step);
        if lower.contains(&with_space) || lower.contains(&tight) {
            todo.completed = true;
            changed = true;
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_plan() {
        let plan = parse_plan(
            "I'll plan this out.\n\nPlan:\n1. Read the client\n2. Find the button\n3. Change color\n\nThen I will execute.",
        );
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].text, "Read the client");
        assert_eq!(plan[2].step, 3);
        assert!(!plan[0].completed);
    }

    #[test]
    fn parses_heading_and_parenthesized_markers() {
        let plan = parse_plan("## Plan\n1) inspect\n2) edit\n");
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].text, "inspect");
        assert_eq!(plan[1].text, "edit");
    }

    #[test]
    fn stops_at_next_section() {
        let plan = parse_plan("Plan:\n1. step one\n\nSummary: done");
        assert_eq!(plan.len(), 1);
    }

    #[test]
    fn done_markers_update_steps() {
        let mut todos = vec![
            TodoItem {
                step: 1,
                text: "a".into(),
                completed: false,
            },
            TodoItem {
                step: 2,
                text: "b".into(),
                completed: false,
            },
        ];
        assert!(apply_done_markers("Step 1 complete [DONE:1]", &mut todos));
        assert!(todos[0].completed);
        assert!(!todos[1].completed);
        assert!(!apply_done_markers("nothing", &mut todos));
    }
}
