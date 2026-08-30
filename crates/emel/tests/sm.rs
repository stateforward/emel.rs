#![allow(missing_docs, unused_crate_dependencies)]

use emel::sm::{
    CompletionSource, DispatchScope, FifoScheduler, FixedTaskAllocator, ForkJoinGroup,
    ForkJoinStartGate, InlineScheduler, QueueFull,
};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::thread;

#[test]
fn inline_scheduler_and_bounded_fifo_are_synchronous_and_bounded() {
    let scheduler = InlineScheduler;
    assert_eq!(scheduler.run(|| 7), 7);

    let mut fifo = FifoScheduler::<u8, 1>::new();
    assert_eq!(fifo.push(3), Ok(()));
    assert_eq!(fifo.push(4), Err(QueueFull));
    assert_eq!(fifo.pop(), Some(3));
    assert!(fifo.is_empty());
}

#[test]
fn fixed_allocator_reuses_slots_and_external_sources_validate_indices() {
    let mut allocator = FixedTaskAllocator::<u8, 1>::new();
    let slot = allocator.allocate(9).expect("first slot available");
    assert_eq!(allocator.allocate(10), Err(QueueFull));
    assert_eq!(allocator.deallocate(slot), Some(9));
    assert_eq!(allocator.allocate(10), Ok(slot));

    let mut completions = emel::sm::ExternalCompletionScheduler::<2>::new();
    let source = CompletionSource::new(1);
    completions.complete(source).expect("source is in range");
    assert_eq!(completions.take(source), Ok(true));
    assert_eq!(completions.take(source), Ok(false));
    assert!(completions.complete(CompletionSource::new(2)).is_err());
}

#[test]
fn fork_join_and_start_gate_join_before_observation() {
    let group = Arc::new(ForkJoinGroup::new());
    group.start_one();
    let worker_group = Arc::clone(&group);
    let worker = thread::spawn(move || worker_group.complete_one());
    assert!(group.wait());
    worker.join().expect("worker completes");

    let gate = Arc::new(ForkJoinStartGate::new());
    let worker_gate = Arc::clone(&gate);
    let worker = thread::spawn(move || worker_gate.arrive_and_wait());
    gate.open_after_arrivals(1);
    worker.join().expect("worker passes gate");
}

#[test]
fn dispatch_scope_enforces_single_writer() {
    let active = AtomicBool::new(false);
    let first = DispatchScope::try_enter(&active);
    assert!(first.acquired());
    let second = DispatchScope::try_enter(&active);
    assert!(!second.acquired());
    drop(first);
    let third = DispatchScope::try_enter(&active);
    assert!(third.acquired());
}
