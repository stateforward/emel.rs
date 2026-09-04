#![allow(missing_docs)]

use sml as _;

use emel_memory::view::{
    DEFAULT_BLOCK_TOKENS, INVALID_KV_BLOCK, MAX_BLOCKS_PER_SEQUENCE, MAX_SEQUENCES, Snapshot,
    blocks_for_tokens, positions_capacity_for, resolved_or_default,
};

#[test]
fn geometry_defaults_and_boundaries_match_pinned_contract() {
    assert_eq!(DEFAULT_BLOCK_TOKENS, 16);
    assert_eq!(
        resolved_or_default(0, DEFAULT_BLOCK_TOKENS),
        DEFAULT_BLOCK_TOKENS
    );
    assert_eq!(
        resolved_or_default(-3, DEFAULT_BLOCK_TOKENS),
        DEFAULT_BLOCK_TOKENS
    );
    assert_eq!(resolved_or_default(7, DEFAULT_BLOCK_TOKENS), 7);

    assert_eq!(blocks_for_tokens(16, -1), 0);
    assert_eq!(blocks_for_tokens(16, 0), 0);
    assert_eq!(blocks_for_tokens(16, 1), 1);
    assert_eq!(blocks_for_tokens(16, 16), 1);
    assert_eq!(blocks_for_tokens(16, 17), 2);
    assert_eq!(blocks_for_tokens(0, 17), 0);
    assert_eq!(blocks_for_tokens(-1, 17), 0);

    assert_eq!(positions_capacity_for(16, 17), 32);
    assert_eq!(positions_capacity_for(16, 0), 0);
    assert_eq!(positions_capacity_for(2, i32::MAX), -1);
    assert_eq!(positions_capacity_for(i32::MAX, i32::MAX), i32::MAX);
}

#[test]
fn snapshot_lookup_rejects_invalid_sequence_position_and_block() {
    let mut snapshot = Snapshot::default();
    assert_eq!(snapshot.max_sequences, 0);
    assert_eq!(snapshot.block_tokens, DEFAULT_BLOCK_TOKENS);
    assert_eq!(snapshot.sequence_active.len(), MAX_SEQUENCES);
    assert_eq!(snapshot.sequence_kv_blocks.len(), MAX_SEQUENCES);
    assert_eq!(
        snapshot.sequence_kv_blocks[0].len(),
        MAX_BLOCKS_PER_SEQUENCE
    );
    assert_eq!(snapshot.lookup_kv_block(-1, 0), -1);
    assert!(!snapshot.is_sequence_active(0));
    assert_eq!(snapshot.sequence_length(0), 0);
    assert_eq!(snapshot.lookup_recurrent_slot(0), -1);

    snapshot.max_sequences = 1;
    snapshot.sequence_active[0] = 1;
    snapshot.sequence_length_values[0] = 17;
    snapshot.sequence_kv_block_count[0] = 2;
    snapshot.sequence_kv_blocks[0][0] = 41;
    snapshot.sequence_kv_blocks[0][1] = 42;
    snapshot.sequence_recurrent_slot[0] = 7;

    assert!(snapshot.valid_seq_id(0));
    assert!(!snapshot.valid_seq_id(-1));
    assert!(!snapshot.valid_seq_id(1));
    assert_eq!(snapshot.sequence_length(0), 17);
    assert_eq!(snapshot.lookup_kv_block(0, 0), 41);
    assert_eq!(snapshot.lookup_kv_block(0, 15), 41);
    assert_eq!(snapshot.lookup_kv_block(0, 16), 42);
    assert_eq!(snapshot.lookup_kv_block(0, 17), -1);
    assert_eq!(snapshot.lookup_kv_block(0, -1), -1);
    assert_eq!(snapshot.lookup_recurrent_slot(0), 7);

    snapshot.sequence_kv_blocks[0][1] = INVALID_KV_BLOCK;
    assert_eq!(snapshot.lookup_kv_block(0, 16), -1);
    snapshot.sequence_kv_block_count[0] = i32::try_from(MAX_BLOCKS_PER_SEQUENCE).unwrap() + 1;
    assert_eq!(snapshot.lookup_kv_block(0, 0), -1);
}
