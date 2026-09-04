//! Module for `planner` state machines.
pub mod modes;
pub mod sm;

pub use sm::{
    DoneCallback, ErrorCallback, Event, MAX_PLAN_STEPS, MAX_SEQ, PlanDone, PlanDoneEvent,
    PlanError, PlanErrorEvent, PlanMode, PlanOutput, PlanRequest, PlanResult, Planner,
    PlannerError, SEQ_WORDS,
};

/// Namespace aliases mirroring the pinned request event layout.
pub mod event {
    pub use super::{DoneCallback, ErrorCallback, Event, PlanMode, PlanRequest};
}

/// Namespace aliases mirroring the pinned completion event layout.
pub mod events {
    pub use super::{PlanDone, PlanDoneEvent, PlanError, PlanErrorEvent};
}
