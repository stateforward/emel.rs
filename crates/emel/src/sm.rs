//! Safe, run-to-completion state-machine infrastructure shared by emel actors.
//!
//! This module mirrors the reusable surface of the pinned C++
//! `src/emel/sm.hpp`.  Scheduler and coroutine internals are represented by
//! explicit safe policies; no task or event can escape the owning dispatch
//! boundary.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::thread;

/// A completed boolean task returned by the inline dispatch policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoolTask {
    value: bool,
}

impl BoolTask {
    /// Creates an already-completed task.
    #[must_use]
    pub const fn from_value(value: bool) -> Self {
        Self { value }
    }

    /// Returns the completed task value.
    #[must_use]
    pub const fn value(self) -> bool {
        self.value
    }
}

impl Future for BoolTask {
    type Output = bool;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        Poll::Ready(self.value)
    }
}

/// A scheduler that executes work immediately on the calling thread.
#[derive(Debug, Default, Clone, Copy)]
pub struct InlineScheduler;

impl InlineScheduler {
    /// Runs `task` to completion without deferring it.
    pub fn run<F, R>(&self, task: F) -> R
    where
        F: FnOnce() -> R,
    {
        task()
    }
}

/// Error returned when a bounded FIFO has no capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueFull;

/// A bounded FIFO for explicitly managed events.
#[derive(Debug)]
pub struct FifoScheduler<T, const CAPACITY: usize> {
    queue: VecDeque<T>,
}

impl<T, const CAPACITY: usize> Default for FifoScheduler<T, CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const CAPACITY: usize> FifoScheduler<T, CAPACITY> {
    /// Creates an empty bounded FIFO.
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: VecDeque::with_capacity(CAPACITY),
        }
    }

    /// Returns the maximum number of events this FIFO stores.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        CAPACITY
    }

    /// Returns the number of events currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns whether no event is currently stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Appends an event, reporting a full queue instead of allocating.
    ///
    /// # Errors
    ///
    /// Returns [`QueueFull`] when the FIFO already contains `CAPACITY` events.
    pub fn push(&mut self, event: T) -> Result<(), QueueFull> {
        if self.queue.len() == CAPACITY {
            Err(QueueFull)
        } else {
            self.queue.push_back(event);
            Ok(())
        }
    }

    /// Removes the oldest event.
    pub fn pop(&mut self) -> Option<T> {
        self.queue.pop_front()
    }

    /// Removes all queued events.
    pub fn clear(&mut self) {
        self.queue.clear();
    }
}

/// An explicit completion source used to classify externally observed work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompletionSource {
    index: usize,
}

impl CompletionSource {
    /// Creates a source identified by `index`.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        Self { index }
    }

    /// Returns the source index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }
}

/// A bounded collection of completion sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalCompletionScheduler<const SOURCES: usize> {
    completed: [bool; SOURCES],
}

impl<const SOURCES: usize> Default for ExternalCompletionScheduler<SOURCES> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const SOURCES: usize> ExternalCompletionScheduler<SOURCES> {
    /// Creates a scheduler with no completed sources.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            completed: [false; SOURCES],
        }
    }

    /// Marks a valid source as completed.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidSource`] when the source index is outside this policy.
    pub fn complete(&mut self, source: CompletionSource) -> Result<(), InvalidSource> {
        self.completed.get_mut(source.index).map_or(
            Err(InvalidSource {
                index: source.index,
            }),
            |slot| {
                *slot = true;
                Ok(())
            },
        )
    }

    /// Consumes and reports a completed source.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidSource`] when the source index is outside this policy.
    pub fn take(&mut self, source: CompletionSource) -> Result<bool, InvalidSource> {
        self.completed.get_mut(source.index).map_or(
            Err(InvalidSource {
                index: source.index,
            }),
            |slot| {
                let value = *slot;
                *slot = false;
                Ok(value)
            },
        )
    }
}

/// Error returned when a completion source index is outside its policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidSource {
    /// The rejected source index.
    pub index: usize,
}

/// A safe fixed-capacity task allocator.
#[derive(Debug)]
pub struct FixedTaskAllocator<T, const CAPACITY: usize> {
    slots: Vec<Option<T>>,
}

impl<T, const CAPACITY: usize> Default for FixedTaskAllocator<T, CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const CAPACITY: usize> FixedTaskAllocator<T, CAPACITY> {
    /// Creates an allocator with all slots available.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: (0..CAPACITY).map(|_| None).collect(),
        }
    }

    /// Attempts to store a task and returns its stable slot index.
    ///
    /// # Errors
    ///
    /// Returns [`QueueFull`] when all allocator slots are occupied.
    pub fn allocate(&mut self, task: T) -> Result<usize, QueueFull> {
        if let Some((index, slot)) = self
            .slots
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.is_none())
        {
            *slot = Some(task);
            Ok(index)
        } else {
            Err(QueueFull)
        }
    }

    /// Removes and returns a task by slot index.
    pub fn deallocate(&mut self, index: usize) -> Option<T> {
        self.slots.get_mut(index).and_then(Option::take)
    }

    /// Returns the number of occupied slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }

    /// Returns whether no allocator slots are occupied.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A bounded fork/join group for synchronous RTC work.
#[derive(Debug)]
pub struct ForkJoinGroup {
    remaining: AtomicUsize,
    accepted: AtomicBool,
}

impl Default for ForkJoinGroup {
    fn default() -> Self {
        Self::new()
    }
}

impl ForkJoinGroup {
    /// Creates an empty group.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            remaining: AtomicUsize::new(0),
            accepted: AtomicBool::new(true),
        }
    }

    /// Registers one unit of work.
    pub fn start_one(&self) {
        self.remaining.fetch_add(1, Ordering::AcqRel);
    }

    /// Records a rejected unit and completes it.
    pub fn reject_one(&self) {
        self.accepted.store(false, Ordering::Release);
        self.complete_one();
    }

    /// Marks the group rejected without registering work.
    pub fn reject(&self) {
        self.accepted.store(false, Ordering::Release);
    }

    /// Completes one registered unit.
    pub fn complete_one(&self) {
        let _ = self
            .remaining
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_sub(1)
            });
    }

    /// Waits until all registered work has completed.
    #[must_use]
    pub fn wait(&self) -> bool {
        while self.remaining.load(Ordering::Acquire) != 0 {
            thread::yield_now();
        }
        self.accepted.load(Ordering::Acquire)
    }
}

/// A reusable start gate for a bounded fork/join phase.
#[derive(Debug, Default)]
pub struct ForkJoinStartGate {
    arrived: AtomicUsize,
    open: AtomicBool,
}

impl ForkJoinStartGate {
    /// Creates a closed gate.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            arrived: AtomicUsize::new(0),
            open: AtomicBool::new(false),
        }
    }

    /// Waits until the gate is open.
    pub fn wait(&self) {
        while !self.open.load(Ordering::Acquire) {
            thread::yield_now();
        }
    }

    /// Records an arrival and waits for opening.
    pub fn arrive_and_wait(&self) {
        self.arrived.fetch_add(1, Ordering::AcqRel);
        self.wait();
    }

    /// Opens the gate.
    pub fn open(&self) {
        self.open.store(true, Ordering::Release);
    }

    /// Opens the gate after `expected_arrivals` have arrived.
    pub fn open_after_arrivals(&self, expected_arrivals: usize) {
        while self.arrived.load(Ordering::Acquire) < expected_arrivals {
            thread::yield_now();
        }
        self.open();
    }
}

/// Scope enforcing single-writer dispatch for one actor.
#[derive(Debug)]
pub struct DispatchScope<'a> {
    active: &'a AtomicBool,
    acquired: bool,
}

impl<'a> DispatchScope<'a> {
    /// Attempts to acquire the actor's dispatch slot.
    #[must_use]
    pub fn try_enter(active: &'a AtomicBool) -> Self {
        Self {
            acquired: active
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            active,
        }
    }

    /// Returns whether this scope acquired the dispatch slot.
    #[must_use]
    pub const fn acquired(&self) -> bool {
        self.acquired
    }
}

impl Drop for DispatchScope<'_> {
    fn drop(&mut self) {
        if self.acquired {
            self.active.store(false, Ordering::Release);
        }
    }
}

/// Safe process support that immediately invokes an owning actor.
#[derive(Debug)]
pub struct ProcessSupport<'a, Owner> {
    owner: &'a mut Owner,
}

impl<'a, Owner> ProcessSupport<'a, Owner> {
    /// Creates a process support handle for the current RTC scope.
    pub const fn new(owner: &'a mut Owner) -> Self {
        Self { owner }
    }

    /// Immediately processes one event before the current RTC boundary ends.
    pub fn push<Event>(&mut self, event: Event)
    where
        Owner: ProcessEvent<Event>,
    {
        let _ = self.owner.process_event(event);
    }
}

/// Trait used by [`ProcessSupport`] for immediate event delivery.
pub trait ProcessEvent<Event> {
    /// Processes an event synchronously.
    fn process_event(&mut self, event: Event) -> bool;
}

/// A no-op marker retained for callers that only need the infrastructure namespace.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Infrastructure;
