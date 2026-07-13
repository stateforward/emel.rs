#![allow(
    dead_code,
    reason = "shared helpers are compiled separately for each fuzz target"
)]

use emel_gguf::Loader;
use emel_gguf::event::{Bind, Error, Load, Parse, ParseDone, Probe, ProbeDone};

pub const MAX_INPUT_BYTES: usize = 64 * 1024;
const MAX_KV_ARENA_BYTES: usize = 64 * 1024;
const MAX_KV_ENTRIES: u32 = 512;
const MAX_TENSORS: u32 = 512;

pub fn requirements_are_bounded(requirements: ProbeDone) -> bool {
    requirements.metadata_count() <= MAX_KV_ENTRIES
        && requirements.tensor_count() <= MAX_TENSORS
        && requirements
            .required_metadata_bytes()
            .is_ok_and(|bytes| bytes <= MAX_KV_ARENA_BYTES)
}

pub fn validate_model(model: &ParseDone<'_>) {
    let requirements = model.probe();
    assert_eq!(
        model.metadata().len(),
        usize::try_from(requirements.metadata_count()).expect("u32 fits usize")
    );
    assert_eq!(
        model.tensors().len(),
        usize::try_from(requirements.tensor_count()).expect("u32 fits usize")
    );

    for entry in model.metadata() {
        assert!(!entry.key().is_empty());
        let _ = entry.value();
    }
    for tensor in model.tensors() {
        assert!(tensor.name().len() < 64);
        assert_eq!(tensor.data().len() as u64, tensor.data_size());
    }
}

pub trait LoaderExt {
    fn probe(&mut self, file_image: &[u8]) -> Result<ProbeDone, Error>;
    fn bind(&mut self) -> Result<(), Error>;
    fn bind_with_capacity(
        &mut self,
        metadata_bytes: usize,
        metadata_entries: usize,
        tensors: usize,
    ) -> Result<(), Error>;
    fn parse<'a>(&mut self, file_image: &'a [u8]) -> Result<ParseDone<'a>, Error>;
}

impl LoaderExt for Loader {
    fn probe(&mut self, file_image: &[u8]) -> Result<ProbeDone, Error> {
        self.process_event(Probe::new(file_image))
    }

    fn bind(&mut self) -> Result<(), Error> {
        self.process_event(Bind::exact())
    }

    fn bind_with_capacity(
        &mut self,
        metadata_bytes: usize,
        metadata_entries: usize,
        tensors: usize,
    ) -> Result<(), Error> {
        self.process_event(Bind::with_capacity(
            metadata_bytes,
            metadata_entries,
            tensors,
        ))
    }

    fn parse<'a>(&mut self, file_image: &'a [u8]) -> Result<ParseDone<'a>, Error> {
        self.process_event(Parse::new(file_image))
    }
}

pub fn load(file_image: &[u8]) -> Result<ParseDone<'_>, Error> {
    Loader::new().process_event(Load::new(file_image))
}
