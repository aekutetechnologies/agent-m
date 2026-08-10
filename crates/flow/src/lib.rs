//! agent-m-flow: Devin-style YAML flow engine for agent-m.
//!
//! A flow is an ordered list of steps (prompt / ask / tool / condition /
//! phase / verify) that share a serializable `FlowContext`. The design
//! borrows the GSD (Git. Ship. Done.) phase-loop ideas: fresh-context agent
//! work per step, state artifacts, and verify-before-done.

pub mod executor;
pub mod model;

pub use executor::{
    FlowDeps, FlowProgress, FlowRun, StepRecord, StepStatus, load_flow, run_flow, should_compact,
};
pub use model::{Flow, FlowContext, FlowStep};
