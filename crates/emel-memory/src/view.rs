#![allow(clippy::cast_possible_truncation)]

//! Public, owned snapshots and bounded KV block geometry.

pub const MAX_SEQUENCES: usize = 256;
pub const MAX_BLOCKS_PER_SEQUENCE: usize = 4_096;
pub const INVALID_KV_BLOCK: u16 = u16::MAX;
pub const DEFAULT_BLOCK_TOKENS: i32 = 16;
pub const MAX_BLOCKS: usize = 32_768;

const MAX_BLOCKS_PER_SEQUENCE_I32: i32 = 4_096;

#[must_use]
pub const fn resolved_or_default(value: i32, fallback: i32) -> i32 {
    if value > 0 { value } else { fallback }
}

#[must_use]
pub const fn blocks_for_tokens(block_tokens: i32, token_count: i32) -> i32 {
    if block_tokens <= 0 || token_count <= 0 {
        return 0;
    }

    ((token_count as i64 + block_tokens as i64 - 1) / block_tokens as i64) as i32
}

#[must_use]
pub const fn positions_capacity_for(block_tokens: i32, token_count: i32) -> i32 {
    if block_tokens <= 0 || token_count <= 0 {
        return 0;
    }

    let capacity = blocks_for_tokens(block_tokens, token_count) as i64 * block_tokens as i64;
    if capacity > i32::MAX as i64 {
        -1
    } else {
        capacity as i32
    }
}

/// An owned view of bounded memory state.
///
/// Each KV block row has its own heap allocation. This keeps the potentially
/// large `MAX_SEQUENCES x MAX_BLOCKS_PER_SEQUENCE` map out of stack frames,
/// including while constructing a default snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Snapshot {
    pub max_sequences: i32,
    pub block_tokens: i32,
    pub sequence_active: Box<[u8]>,
    pub sequence_length_values: Box<[i32]>,
    pub sequence_kv_block_count: Box<[i32]>,
    pub sequence_kv_blocks: Box<[Box<[u16; MAX_BLOCKS_PER_SEQUENCE]>]>,
    pub sequence_recurrent_slot: Box<[i32]>,
}

impl Default for Snapshot {
    fn default() -> Self {
        let sequence_kv_blocks = (0..MAX_SEQUENCES)
            .map(|_| Box::new([0; MAX_BLOCKS_PER_SEQUENCE]))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            max_sequences: 0,
            block_tokens: DEFAULT_BLOCK_TOKENS,
            sequence_active: vec![0; MAX_SEQUENCES].into_boxed_slice(),
            sequence_length_values: vec![0; MAX_SEQUENCES].into_boxed_slice(),
            sequence_kv_block_count: vec![0; MAX_SEQUENCES].into_boxed_slice(),
            sequence_kv_blocks,
            sequence_recurrent_slot: vec![0; MAX_SEQUENCES].into_boxed_slice(),
        }
    }
}

impl Snapshot {
    fn sequence_index(&self, seq_id: i32) -> Option<usize> {
        (seq_id >= 0 && seq_id < self.max_sequences)
            .then(|| usize::try_from(seq_id).ok())
            .flatten()
            .filter(|&index| index < MAX_SEQUENCES)
    }

    #[must_use]
    pub fn valid_seq_id(&self, seq_id: i32) -> bool {
        self.sequence_index(seq_id).is_some()
    }

    #[must_use]
    pub fn is_sequence_active(&self, seq_id: i32) -> bool {
        self.sequence_index(seq_id)
            .is_some_and(|index| self.sequence_active[index] != 0)
    }

    #[must_use]
    pub fn sequence_length(&self, seq_id: i32) -> i32 {
        self.sequence_index(seq_id)
            .filter(|&index| self.sequence_active[index] != 0)
            .map_or(0, |index| self.sequence_length_values[index])
    }

    #[must_use]
    pub fn lookup_kv_block(&self, seq_id: i32, pos: i32) -> i32 {
        if !self.is_sequence_active(seq_id) || pos < 0 || self.block_tokens <= 0 {
            return -1;
        }

        let Some(sequence_index) = self.sequence_index(seq_id) else {
            return -1;
        };
        let length = self.sequence_length_values[sequence_index];
        if pos >= length {
            return -1;
        }

        let block_count = self.sequence_kv_block_count[sequence_index];
        if !(1..=MAX_BLOCKS_PER_SEQUENCE_I32).contains(&block_count) {
            return -1;
        }

        let logical_block = usize::try_from(pos / self.block_tokens).ok();
        let Some(logical_block) =
            logical_block.filter(|&block| block < usize::try_from(block_count).unwrap_or(0))
        else {
            return -1;
        };

        let block = self.sequence_kv_blocks[sequence_index][logical_block];
        if block == INVALID_KV_BLOCK {
            -1
        } else {
            i32::from(block)
        }
    }

    #[must_use]
    pub fn lookup_recurrent_slot(&self, seq_id: i32) -> i32 {
        self.sequence_index(seq_id)
            .filter(|&index| self.sequence_active[index] != 0)
            .map_or(-1, |index| self.sequence_recurrent_slot[index])
    }
}

pub type View = Snapshot;
