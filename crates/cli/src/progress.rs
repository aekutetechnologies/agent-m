use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

pub struct TurnProgress {
    pb: Option<ProgressBar>,
}

impl TurnProgress {
    pub fn new() -> Self {
        Self { pb: None }
    }

    pub fn start_tool(&mut self, name: &str, args_summary: &str) {
        let pb = ProgressBar::new_spinner();
        if let Ok(style) = ProgressStyle::default_spinner()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
            .template("{spinner:.green} [{msg}]")
        {
            pb.set_style(style);
        }
        let summary_truncated: String = args_summary.chars().take(80).collect();
        pb.set_message(format!("Tool: {} {}", name, summary_truncated));
        pb.enable_steady_tick(Duration::from_millis(80));
        self.pb = Some(pb);
    }

    pub fn finish_tool(&mut self, is_error: bool, summary: &str) {
        if let Some(pb) = self.pb.take() {
            if is_error {
                let first_line = summary.lines().next().unwrap_or(summary);
                let truncated: String = first_line.chars().take(80).collect();
                pb.finish_with_message(format!("✖ [Tool Error] {}", truncated));
            } else {
                pb.finish_and_clear();
            }
        }
    }
}
