//! Allocation-free generic vocabulary query child actor.

use crate::data::Vocab;

use super::event::{
    Error, ErrorKind, Info, Key, Phase, Token, WithCharmap, WithInfo, WithMerge, WithToken,
};
use super::{c_name, flag_at, flags, not_loaded, special_ids};

mod sm;

use sm::{VocabularyQueryStateMachine, VocabularyQueryStateMachineContext};

pub(super) trait Operation {
    fn index_is_valid(&self, vocab: &Vocab) -> bool;
    fn apply(&mut self, vocab: &Vocab);
    fn not_found(&mut self);
    fn error(&mut self, error: Error);
}

pub(super) struct QueryRequest<'data, O> {
    loaded: bool,
    vocab: &'data Vocab,
    operation: &'data mut O,
}

#[derive(Clone, Copy, Debug, Default)]
struct QueryContext;

pub(super) fn process<'data, O: Operation + 'data>(
    loaded: bool,
    vocab: &'data Vocab,
    operation: &'data mut O,
) {
    let mut request = QueryRequest {
        loaded,
        vocab,
        operation,
    };
    let mut machine = VocabularyQueryStateMachine::new(QueryContext);
    if machine.process_event(&mut request).is_err() {
        request.operation.error(internal_error());
    }
}

impl VocabularyQueryStateMachineContext for QueryContext {
    fn guard_not_loaded<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(!request.loaded)
    }
    fn guard_loaded_valid<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.loaded && request.operation.index_is_valid(request.vocab))
    }
    fn guard_loaded_out_of_range<'query, 'data: 'query, O: Operation + 'data>(
        &self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<bool, ()> {
        Ok(request.loaded && !request.operation.index_is_valid(request.vocab))
    }
    fn effect_not_loaded<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.error(not_loaded());
        Ok(())
    }
    fn effect_apply<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.apply(request.vocab);
        Ok(())
    }
    fn effect_not_found<'query, 'data: 'query, O: Operation + 'data>(
        &mut self,
        request: &'query mut QueryRequest<'data, O>,
    ) -> Result<(), ()> {
        request.operation.not_found();
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

impl<F, R> Operation for WithInfo<F, R>
where
    F: for<'a> FnMut(Info<'a>) -> R,
{
    fn index_is_valid(&self, vocab: &Vocab) -> bool {
        vocab.tokenizer_pre_id.is_some()
    }
    fn apply(&mut self, vocab: &Vocab) {
        let Some(pre) = vocab.tokenizer_pre_id else {
            self.result = Err(internal_error());
            return;
        };
        self.result = Ok(Some((self.callback)(Info {
            model: vocab.tokenizer_model_id,
            pre,
            model_name: c_name(&vocab.tokenizer_model_name),
            pre_name: c_name(&vocab.tokenizer_pre_name),
            token_count: vocab.n_tokens,
            token_type_count: vocab.n_token_types,
            token_bytes: vocab.token_bytes_used,
            merge_count: vocab.n_merges,
            merge_bytes: vocab.merge_bytes_used,
            charmap_bytes: vocab.precompiled_charsmap_size,
            special_ids: special_ids(vocab),
            flags: flags(vocab),
        })));
    }
    fn not_found(&mut self) {
        self.result = Err(internal_error());
    }
    fn error(&mut self, error: Error) {
        self.result = Err(error);
    }
}

impl<F, R> Operation for WithToken<F, R>
where
    F: for<'a> FnMut(Token<'a>) -> R,
{
    fn index_is_valid(&self, vocab: &Vocab) -> bool {
        usize::try_from(self.index).is_ok_and(|index| index < vocab.n_tokens as usize)
    }
    fn apply(&mut self, vocab: &Vocab) {
        let index = usize::try_from(self.index).expect("query guard validated token index");
        let entry = vocab.entries[index];
        let start = entry.text_offset as usize;
        let end = start + entry.text_length as usize;
        self.result = Ok(Some((self.callback)(Token {
            text: &vocab.token_storage[start..end],
            score: entry.score,
            r#type: entry.r#type,
            lstrip: flag_at(&vocab.lstrip_flags, index),
            rstrip: flag_at(&vocab.rstrip_flags, index),
        })));
    }
    fn not_found(&mut self) {
        self.result = Ok(None);
    }
    fn error(&mut self, error: Error) {
        self.result = Err(error);
    }
}

impl<F, R> Operation for WithMerge<F, R>
where
    F: for<'a> FnMut(&'a [u8]) -> R,
{
    fn index_is_valid(&self, vocab: &Vocab) -> bool {
        usize::try_from(self.index).is_ok_and(|index| index < vocab.n_merges as usize)
    }
    fn apply(&mut self, vocab: &Vocab) {
        let index = usize::try_from(self.index).expect("query guard validated merge index");
        let start = vocab.merge_offsets[index] as usize;
        let end = start + vocab.merge_lengths[index] as usize;
        self.result = Ok(Some((self.callback)(&vocab.merge_storage[start..end])));
    }
    fn not_found(&mut self) {
        self.result = Ok(None);
    }
    fn error(&mut self, error: Error) {
        self.result = Err(error);
    }
}

impl<F, R> Operation for WithCharmap<F, R>
where
    F: for<'a> FnMut(&'a [u8]) -> R,
{
    fn index_is_valid(&self, _vocab: &Vocab) -> bool {
        true
    }
    fn apply(&mut self, vocab: &Vocab) {
        self.result = Ok(Some((self.callback)(
            &vocab.precompiled_charsmap[..vocab.precompiled_charsmap_size as usize],
        )));
    }
    fn not_found(&mut self) {
        self.result = Err(internal_error());
    }
    fn error(&mut self, error: Error) {
        self.result = Err(error);
    }
}

const fn internal_error() -> Error {
    Error::new(Phase::Query, Key::None, ErrorKind::Internal)
}
