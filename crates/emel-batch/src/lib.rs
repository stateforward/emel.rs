//! Batch planning and deterministic request scheduling.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// An identifier assigned to a batch by the planner.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BatchId(pub u64);
pub mod planner;

pub use planner::sm::{
    DoneCallback, EqualStrategy, ErrorCallback, Event, MAX_PLAN_STEPS, MAX_SEQ, MaskWordCount,
    OutputMask, OutputSelection, PlanConfig, PlanDone, PlanDoneEvent, PlanError, PlanErrorEvent,
    PlanMode, PlanOutput, PlanRequest, PlanResult, Planner, PlannerError, SEQ_WORDS,
    SequenceConfig, StepSize,
};

/// Request-event namespace matching the maintained planner facade.
pub mod event {
    pub use crate::planner::event::*;
}

/// Completion-event namespace matching the maintained planner facade.
pub mod events {
    pub use crate::planner::events::*;
}
