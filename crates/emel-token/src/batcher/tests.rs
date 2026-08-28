use allocation_counter::measure;

use super::{BatchError, BatchOutputs, BatchRequest, MAX_SEQ, MAX_TOKENS, SEQ_WORDS, TokenBatcher};

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
