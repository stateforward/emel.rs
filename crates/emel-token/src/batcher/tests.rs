use allocation_counter::measure;

use super::{
    BatchDone, BatchError, BatchFailure, BatchOutputs, BatchRequest, MAX_SEQ, MAX_TOKENS,
    PositionSeedError, SEQ_WORDS, TokenBatcher,
};

static DONE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static ERROR_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn on_done(done: BatchDone) {
    assert_eq!(done.token_count, 1);
    assert_eq!(done.outputs_total, 1);
    DONE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn on_error(error: BatchFailure) {
    assert_eq!(error.error, BatchError::InvalidRequest);
    assert_eq!(error.token_count, 0);
    ERROR_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[allow(clippy::unnecessary_wraps)]
fn seed_zero(
    context: &super::PositionSeedContext<'_>,
    _sequence_id: i32,
) -> Result<i32, PositionSeedError> {
    Ok(context.seeds[0])
}

#[allow(
    clippy::too_many_arguments,
    reason = "fixture construction mirrors the explicit batch request fields"
)]
fn request<'a>(
    token_ids: &'a [i32],
    vocab_size: i32,
    seq_masks: Option<&'a [u64]>,
    seq_mask_words: usize,
    seq_primary_ids: Option<&'a [i32]>,
    positions: Option<&'a [i32]>,
    seq_primary_ids_out: &'a mut [i32],
    seq_masks_out: &'a mut [u64],
    positions_out: &'a mut [i32],
    output_mask: &'a mut [i8],
) -> BatchRequest<'a> {
    BatchRequest {
        token_ids,
        vocab_size,
        seq_masks,
        seq_mask_words,
        seq_primary_ids,
        positions,
        output_mask_input: None,
        output_all: false,
        enforce_single_output_per_seq: false,
        resolve_position_seed: None,
        seq_mask_words_out: None,
        positions_count_out: None,
        outputs_total_out: None,
        on_done: None,
        on_error: None,
        outputs: BatchOutputs {
            seq_primary_ids: seq_primary_ids_out,
            seq_masks: seq_masks_out,
            positions: positions_out,
            output_mask,
        },
    }
}

#[test]
fn default_sequence_normalization_is_real_and_reusable() {
    let token_ids = [4, 5];
    let mut primary_out = [-1; 2];
    let mut masks_out = [u64::MAX; 2];
    let mut positions_out = [0; 2];
    let mut output_mask = [0; 2];
    let mut batcher = TokenBatcher::new();

    let result = batcher.process_event(request(
        &token_ids,
        32,
        None,
        0,
        None,
        None,
        &mut primary_out,
        &mut masks_out,
        &mut positions_out,
        &mut output_mask,
    ));

    assert_eq!(result.unwrap().seq_mask_words, 1);
    assert_eq!(primary_out, [0, 0]);
    assert_eq!(masks_out, [1, 1]);

    let primary_ids = [3, 7];
    let result = batcher.process_event(request(
        &token_ids,
        32,
        None,
        1,
        Some(&primary_ids),
        None,
        &mut primary_out,
        &mut masks_out,
        &mut positions_out,
        &mut output_mask,
    ));
    assert_eq!(result.unwrap().seq_mask_words, 1);
    assert_eq!(primary_out, primary_ids);
    assert_eq!(masks_out, [1 << 3, 1 << 7]);
}

#[test]
fn masks_are_copied_and_primary_ids_are_first_set_bits() {
    let token_ids = [9, 10];
    let input_masks = [0x5, 0, 0, 0, 0, 1 << 6, 0, 0];
    let input_primary_ids = [0, 70];
    let mut primary_out = [-1; 2];
    let mut masks_out = [u64::MAX; 8];
    let mut positions_out = [0; 2];
    let mut output_mask = [0; 2];

    let result = TokenBatcher::new().process_event(request(
        &token_ids,
        32,
        Some(&input_masks),
        SEQ_WORDS,
        Some(&input_primary_ids),
        None,
        &mut primary_out,
        &mut masks_out,
        &mut positions_out,
        &mut output_mask,
    ));

    assert_eq!(
        result.unwrap(),
        super::super::batcher::BatchResult {
            token_count: 2,
            seq_mask_words: SEQ_WORDS,
            positions_count: 2,
            outputs_total: 1,
        }
    );
    assert_eq!(primary_out, [0, 70]);
    assert_eq!(masks_out, input_masks);
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the fixture table keeps each invalid contract case explicit"
)]
fn invalid_counts_capacities_vocab_and_positions_fail_explicitly() {
    let mut batcher = TokenBatcher::new();
    let mut primary_out = [0; 1];
    let mut masks_out = [0; 1];
    let mut positions_out = [0; 1];
    let mut output_mask = [0; 1];

    assert_eq!(
        batcher.process_event(request(
            &[],
            32,
            None,
            1,
            None,
            None,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        )),
        Err(BatchError::InvalidRequest)
    );
    let too_many = vec![0; MAX_TOKENS + 1];
    assert_eq!(
        batcher.process_event(request(
            &too_many,
            0,
            None,
            1,
            None,
            None,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        )),
        Err(BatchError::InvalidRequest)
    );
    assert_eq!(
        batcher.process_event(request(
            &[1],
            32,
            None,
            1,
            None,
            None,
            &mut [],
            &mut [],
            &mut [],
            &mut [],
        )),
        Err(BatchError::InvalidRequest)
    );
    assert_eq!(
        batcher.process_event(request(
            &[32],
            32,
            None,
            1,
            None,
            None,
            &mut primary_out,
            &mut masks_out,
            &mut positions_out,
            &mut output_mask,
        )),
        Err(BatchError::InvalidRequest)
    );
    assert_eq!(
        batcher.process_event(request(
            &[-1],
            32,
            None,
            1,
            None,
            None,
            &mut primary_out,
            &mut masks_out,
            &mut positions_out,
            &mut output_mask,
        )),
        Err(BatchError::InvalidRequest)
    );
    assert_eq!(
        batcher.process_event(request(
            &[1],
            -1,
            None,
            1,
            None,
            None,
            &mut primary_out,
            &mut masks_out,
            &mut positions_out,
            &mut output_mask,
        )),
        Err(BatchError::InvalidRequest)
    );
    assert_eq!(
        batcher.process_event(request(
            &[1],
            32,
            None,
            1,
            None,
            Some(&[]),
            &mut primary_out,
            &mut masks_out,
            &mut positions_out,
            &mut output_mask,
        )),
        Err(BatchError::InvalidRequest)
    );
}

#[test]
fn three_lane_positions_shape_participates_in_capacity_validation() {
    let token_ids = [1, 2];
    let positions_input = [0, 1, 0, 0, 1, 1];
    let mut primary_out = [0; 2];
    let mut masks_out = [0; 2];
    let mut positions_out = [0; 6];
    let mut output_mask = [0; 2];

    let result = TokenBatcher::new().process_event(request(
        &token_ids,
        32,
        None,
        1,
        None,
        Some(&positions_input),
        &mut primary_out,
        &mut masks_out,
        &mut positions_out,
        &mut output_mask,
    ));

    assert_eq!(result.unwrap().token_count, 2);
}

#[test]
fn sequence_payload_checks_cover_masks_primary_ids_and_short_inputs() {
    let mut primary_out = [0; 1];
    let mut masks_out = [0; 1];
    let mut positions_out = [0; 1];
    let mut output_mask = [0; 1];
    let mut batcher = TokenBatcher::new();

    assert_eq!(
        batcher.process_event(request(
            &[1],
            32,
            Some(&[0]),
            1,
            None,
            None,
            &mut primary_out,
            &mut masks_out,
            &mut positions_out,
            &mut output_mask,
        )),
        Err(BatchError::InvalidRequest)
    );
    assert_eq!(
        batcher.process_event(request(
            &[1],
            32,
            None,
            1,
            Some(&[64]),
            None,
            &mut primary_out,
            &mut masks_out,
            &mut positions_out,
            &mut output_mask,
        )),
        Err(BatchError::InvalidRequest)
    );
    assert_eq!(
        batcher.process_event(request(
            &[1],
            32,
            Some(&[1]),
            1,
            Some(&[1]),
            None,
            &mut primary_out,
            &mut masks_out,
            &mut positions_out,
            &mut output_mask,
        )),
        Err(BatchError::InvalidRequest)
    );
    assert_eq!(
        batcher.process_event(request(
            &[1],
            32,
            Some(&[1]),
            SEQ_WORDS + 1,
            None,
            None,
            &mut primary_out,
            &mut masks_out,
            &mut positions_out,
            &mut output_mask,
        )),
        Err(BatchError::InvalidRequest)
    );

    // C++ has a pointer-plus-count input contract: a short optional input is
    // not an active mode, so this request follows the default branch.
    let short_masks = [u64::MAX];
    let result = batcher.process_event(request(
        &[1, 2],
        32,
        Some(&short_masks),
        1,
        None,
        None,
        &mut [0; 2],
        &mut [0; 2],
        &mut [0; 2],
        &mut [0; 2],
    ));
    assert_eq!(result.unwrap().seq_mask_words, 1);
}

#[test]
fn zero_vocab_matches_source_non_enforcing_mode_and_unexpected_is_explicit() {
    let mut primary_out = [0; 1];
    let mut masks_out = [0; 1];
    let mut positions_out = [0; 1];
    let mut output_mask = [0; 1];
    let mut batcher = TokenBatcher::new();

    let result = batcher.process_event(request(
        &[-1],
        0,
        None,
        1,
        None,
        None,
        &mut primary_out,
        &mut masks_out,
        &mut positions_out,
        &mut output_mask,
    ));
    assert!(result.is_ok());
    assert_eq!(
        batcher.process_unexpected(),
        Err(BatchError::UnexpectedEvent)
    );
}

#[test]
fn request_dispatch_allocates_nothing() {
    let mut batcher = TokenBatcher::new();
    let token_ids = [1, 2];
    let mut primary_out = [0; 2];
    let mut masks_out = [0; 2];
    let mut positions_out = [0; 2];
    let mut output_mask = [0; 2];

    let allocation = measure(|| {
        assert!(
            batcher
                .process_event(request(
                    &token_ids,
                    32,
                    None,
                    1,
                    None,
                    None,
                    &mut primary_out,
                    &mut masks_out,
                    &mut positions_out,
                    &mut output_mask,
                ))
                .is_ok()
        );
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(MAX_SEQ / 64, SEQ_WORDS);
}

#[test]
fn full_outputs_and_stride_three_are_published() {
    let ids = [1, 2];
    let mut primary = [0; 2];
    let mut masks = [0; 2];
    let mut positions = [0; 6];
    let input_positions = [1, 2, 3, 4, 5, 6];
    let mut output = [0; 2];
    let mut words = 0;
    let mut position_count = 0;
    let mut total = 0;
    let result = TokenBatcher::new()
        .process_event(BatchRequest {
            token_ids: &ids,
            vocab_size: 10,
            seq_masks: None,
            seq_mask_words: 1,
            seq_primary_ids: None,
            positions: Some(&input_positions),
            output_mask_input: None,
            output_all: true,
            enforce_single_output_per_seq: false,
            resolve_position_seed: None,
            seq_mask_words_out: Some(&mut words),
            positions_count_out: Some(&mut position_count),
            outputs_total_out: Some(&mut total),
            on_done: None,
            on_error: None,
            outputs: BatchOutputs {
                seq_primary_ids: &mut primary,
                seq_masks: &mut masks,
                positions: &mut positions,
                output_mask: &mut output,
            },
        })
        .unwrap();
    assert_eq!(result.positions_count, 6);
    assert_eq!(positions, input_positions);
    assert_eq!(output, [1, 1]);
    assert_eq!((words, position_count, total), (1, 6, 2));
}

#[test]
fn seeded_generation_and_output_mask_copy_are_source_branches() {
    let ids = [1, 2];
    let sequences = [0, 0];
    let mut primary = [0; 2];
    let mut masks = [0; 2];
    let mut positions = [0; 2];
    let mut output = [0; 2];
    let mut words = 0;
    let mut position_count = 0;
    let mut total = 0;
    let result = TokenBatcher::new()
        .process_event(BatchRequest {
            token_ids: &ids,
            vocab_size: 10,
            seq_masks: None,
            seq_mask_words: 1,
            seq_primary_ids: Some(&sequences),
            positions: None,
            output_mask_input: Some(&[1, 0]),
            output_all: false,
            enforce_single_output_per_seq: true,
            resolve_position_seed: Some(super::PositionSeedResolver {
                context: super::PositionSeedContext { seeds: &[10] },
                resolve: seed_zero,
            }),
            seq_mask_words_out: Some(&mut words),
            positions_count_out: Some(&mut position_count),
            outputs_total_out: Some(&mut total),
            on_done: None,
            on_error: None,
            outputs: BatchOutputs {
                seq_primary_ids: &mut primary,
                seq_masks: &mut masks,
                positions: &mut positions,
                output_mask: &mut output,
            },
        })
        .unwrap();
    assert_eq!(positions, [10, 11]);
    assert_eq!(output, [1, 0]);
    assert_eq!(
        (result.outputs_total, words, position_count, total),
        (1, 1, 2, 1)
    );
}

#[test]
fn oversized_output_buffers_are_logically_bounded() {
    let ids = [1, 2];
    let mut primary = [0; 2];
    let mut masks = [0; 2];
    let mut positions = [0; 2];
    let mut output = [9, 9, 9, 9];
    let mut total = 0;
    let result = TokenBatcher::new()
        .process_event(BatchRequest {
            token_ids: &ids,
            vocab_size: 10,
            seq_masks: None,
            seq_mask_words: 1,
            seq_primary_ids: None,
            positions: None,
            output_mask_input: Some(&[1, 0]),
            output_all: false,
            enforce_single_output_per_seq: false,
            resolve_position_seed: None,
            seq_mask_words_out: None,
            positions_count_out: None,
            outputs_total_out: Some(&mut total),
            on_done: None,
            on_error: None,
            outputs: BatchOutputs {
                seq_primary_ids: &mut primary,
                seq_masks: &mut masks,
                positions: &mut positions,
                output_mask: &mut output,
            },
        })
        .unwrap();
    assert_eq!(result.outputs_total, 1);
    assert_eq!(total, 1);
    assert_eq!(output, [1, 0, 9, 9]);
}

#[test]
fn maximum_seed_is_rejected_before_increment() {
    #[allow(clippy::unnecessary_wraps)]
    fn max_seed(
        _context: &super::PositionSeedContext<'_>,
        _sequence_id: i32,
    ) -> Result<i32, PositionSeedError> {
        Ok(i32::MAX)
    }
    let ids = [1];
    let mut primary = [0; 1];
    let mut masks = [0; 1];
    let mut positions = [0; 1];
    let mut output = [0; 1];
    let result = TokenBatcher::new().process_event(BatchRequest {
        token_ids: &ids,
        vocab_size: 10,
        seq_masks: None,
        seq_mask_words: 1,
        seq_primary_ids: None,
        positions: None,
        output_mask_input: None,
        output_all: false,
        enforce_single_output_per_seq: false,
        resolve_position_seed: Some(super::PositionSeedResolver {
            context: super::PositionSeedContext { seeds: &[] },
            resolve: max_seed,
        }),
        seq_mask_words_out: None,
        positions_count_out: None,
        outputs_total_out: None,
        on_done: None,
        on_error: None,
        outputs: BatchOutputs {
            seq_primary_ids: &mut primary,
            seq_masks: &mut masks,
            positions: &mut positions,
            output_mask: &mut output,
        },
    });
    assert_eq!(result, Err(BatchError::InvalidRequest));
}

#[test]
fn output_last_does_not_touch_oversized_tail() {
    let ids = [1, 2];
    let mut primary = [0; 2];
    let mut masks = [0; 2];
    let mut positions = [0; 2];
    let mut output = [7, 7, 7];
    let result = TokenBatcher::new()
        .process_event(BatchRequest {
            token_ids: &ids,
            vocab_size: 10,
            seq_masks: None,
            seq_mask_words: 1,
            seq_primary_ids: None,
            positions: None,
            output_mask_input: None,
            output_all: false,
            enforce_single_output_per_seq: false,
            resolve_position_seed: None,
            seq_mask_words_out: None,
            positions_count_out: None,
            outputs_total_out: None,
            on_done: None,
            on_error: None,
            outputs: BatchOutputs {
                seq_primary_ids: &mut primary,
                seq_masks: &mut masks,
                positions: &mut positions,
                output_mask: &mut output,
            },
        })
        .unwrap();
    assert_eq!(result.outputs_total, 1);
    assert_eq!(output, [0, 1, 7]);
}

#[test]
fn callbacks_are_synchronous_and_non_retained() {
    DONE_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    ERROR_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
    let ids = [1];
    let mut primary = [0];
    let mut masks = [0];
    let mut positions = [0];
    let mut output = [0];
    let mut batcher = TokenBatcher::new();
    let request = BatchRequest {
        token_ids: &ids,
        vocab_size: 10,
        seq_masks: None,
        seq_mask_words: 1,
        seq_primary_ids: None,
        positions: None,
        output_mask_input: None,
        output_all: false,
        enforce_single_output_per_seq: false,
        resolve_position_seed: None,
        seq_mask_words_out: None,
        positions_count_out: None,
        outputs_total_out: None,
        on_done: Some(on_done),
        on_error: Some(on_error),
        outputs: BatchOutputs {
            seq_primary_ids: &mut primary,
            seq_masks: &mut masks,
            positions: &mut positions,
            output_mask: &mut output,
        },
    };
    assert!(batcher.process_event(request).is_ok());
    assert_eq!(DONE_COUNT.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(ERROR_COUNT.load(std::sync::atomic::Ordering::Relaxed), 0);
    let mut primary = [];
    let mut masks = [];
    let mut positions = [];
    let mut output = [];
    let request = BatchRequest {
        token_ids: &[],
        vocab_size: 10,
        seq_masks: None,
        seq_mask_words: 1,
        seq_primary_ids: None,
        positions: None,
        output_mask_input: None,
        output_all: false,
        enforce_single_output_per_seq: false,
        resolve_position_seed: None,
        seq_mask_words_out: None,
        positions_count_out: None,
        outputs_total_out: None,
        on_done: Some(on_done),
        on_error: Some(on_error),
        outputs: BatchOutputs {
            seq_primary_ids: &mut primary,
            seq_masks: &mut masks,
            positions: &mut positions,
            output_mask: &mut output,
        },
    };
    assert_eq!(
        batcher.process_event(request),
        Err(BatchError::InvalidRequest)
    );
    assert_eq!(ERROR_COUNT.load(std::sync::atomic::Ordering::Relaxed), 1);
}

#[test]
fn output_last_and_single_output_rejection_are_explicit() {
    let ids = [1, 2];
    let sequences = [0, 0];
    let mut primary = [0; 2];
    let mut masks = [0; 2];
    let mut positions = [0; 2];
    let mut output = [0; 2];
    let result = TokenBatcher::new()
        .process_event(BatchRequest {
            token_ids: &ids,
            vocab_size: 10,
            seq_masks: None,
            seq_mask_words: 1,
            seq_primary_ids: Some(&sequences),
            positions: None,
            output_mask_input: None,
            output_all: false,
            enforce_single_output_per_seq: true,
            resolve_position_seed: None,
            seq_mask_words_out: None,
            positions_count_out: None,
            outputs_total_out: None,
            on_done: None,
            on_error: None,
            outputs: BatchOutputs {
                seq_primary_ids: &mut primary,
                seq_masks: &mut masks,
                positions: &mut positions,
                output_mask: &mut output,
            },
        })
        .unwrap();
    assert_eq!(output, [0, 1]);
    assert_eq!(result.outputs_total, 1);

    let mut primary = [0; 2];
    let mut masks = [0; 2];
    let mut positions = [0; 2];
    let mut output = [0; 2];
    let mut batcher = TokenBatcher::new();
    let result = batcher.process_event(BatchRequest {
        token_ids: &ids,
        vocab_size: 10,
        seq_masks: None,
        seq_mask_words: 1,
        seq_primary_ids: Some(&sequences),
        positions: None,
        output_mask_input: Some(&[1, 1]),
        output_all: false,
        enforce_single_output_per_seq: true,
        resolve_position_seed: None,
        seq_mask_words_out: None,
        positions_count_out: None,
        outputs_total_out: None,
        on_done: None,
        on_error: None,
        outputs: BatchOutputs {
            seq_primary_ids: &mut primary,
            seq_masks: &mut masks,
            positions: &mut positions,
            output_mask: &mut output,
        },
    });
    assert_eq!(result, Err(BatchError::InvalidRequest));
}
