//! Static child-actor capabilities accepted by the tensor store.

#![allow(clippy::assertions_on_constants)]

use emel_io::{mmap, read, staged_read};

/// Statically dispatched mapped-I/O capability.
pub trait Mapper {
    /// Whether this dependency provides a mapper actor.
    const AVAILABLE: bool;

    /// Dispatches one map request through the replacement actor.
    ///
    /// # Errors
    ///
    /// Returns the mapper actor's typed failure unchanged.
    fn map_tensor(
        &mut self,
        event: mmap::event::MapTensor,
    ) -> Result<mmap::event::MapDone, mmap::event::Error>;

    /// Dispatches one release request through the replacement actor.
    ///
    /// # Errors
    ///
    /// Returns the mapper actor's typed failure unchanged.
    fn release_mapping(
        &mut self,
        event: mmap::event::ReleaseMapping,
    ) -> Result<(), mmap::event::Error>;

    /// Dispatches one concrete synchronous access operation.
    ///
    /// # Errors
    ///
    /// Returns the mapper actor's typed failure unchanged.
    fn with_mapping<Operation>(
        &mut self,
        event: mmap::event::WithMapping<'_, Operation>,
    ) -> Result<Operation::Output, mmap::event::Error>
    where
        Operation: mmap::event::MappingOperation;
}

/// Statically dispatched read/copy capability.
pub trait Reader {
    /// Whether this dependency provides a reader actor.
    const AVAILABLE: bool;

    /// Dispatches one read request through the replacement actor.
    ///
    /// # Errors
    ///
    /// Returns the reader actor's typed failure unchanged.
    fn read_tensor(
        &mut self,
        event: read::event::ReadTensor<'_>,
    ) -> Result<read::event::ReadTensorDone, read::event::Error>;
}

/// Statically dispatched staged-copy capability.
pub trait Stager {
    /// Whether this dependency provides a staged-read actor.
    const AVAILABLE: bool;

    /// Dispatches one staged request through the replacement actor.
    ///
    /// # Errors
    ///
    /// Returns the staged-read actor's typed failure unchanged.
    fn stage_tensor(
        &mut self,
        event: staged_read::event::StageWindow<'_>,
    ) -> Result<staged_read::event::StageWindowDone, staged_read::event::Error>;
}

/// Complete static dependency contract required by tensor I/O events.
pub trait TensorDependencies: Mapper + Reader + Stager {}

impl<Dependency> TensorDependencies for Dependency where Dependency: Mapper + Reader + Stager {}

/// Production dependency bundle with independently replaceable actor types.
#[derive(Debug)]
pub struct Actors<MapperActor = (), ReaderActor = (), StagerActor = ()> {
    mapper: MapperActor,
    reader: ReaderActor,
    stager: StagerActor,
}

impl<MapperActor, ReaderActor, StagerActor> Actors<MapperActor, ReaderActor, StagerActor> {
    /// Creates a statically dispatched dependency bundle.
    #[must_use]
    pub const fn new(mapper: MapperActor, reader: ReaderActor, stager: StagerActor) -> Self {
        Self {
            mapper,
            reader,
            stager,
        }
    }
}

impl Mapper for mmap::Mapper {
    const AVAILABLE: bool = true;

    fn map_tensor(
        &mut self,
        event: mmap::event::MapTensor,
    ) -> Result<mmap::event::MapDone, mmap::event::Error> {
        self.process_event(event)
    }

    fn release_mapping(
        &mut self,
        event: mmap::event::ReleaseMapping,
    ) -> Result<(), mmap::event::Error> {
        self.process_event(event)
    }

    fn with_mapping<Operation>(
        &mut self,
        event: mmap::event::WithMapping<'_, Operation>,
    ) -> Result<Operation::Output, mmap::event::Error>
    where
        Operation: mmap::event::MappingOperation,
    {
        self.process_event(event)
    }
}

impl Reader for read::Reader {
    const AVAILABLE: bool = true;

    fn read_tensor(
        &mut self,
        event: read::event::ReadTensor<'_>,
    ) -> Result<read::event::ReadTensorDone, read::event::Error> {
        self.process_event(event)
    }
}

impl Stager for staged_read::Stager {
    const AVAILABLE: bool = true;

    fn stage_tensor(
        &mut self,
        event: staged_read::event::StageWindow<'_>,
    ) -> Result<staged_read::event::StageWindowDone, staged_read::event::Error> {
        self.process_event(event)
    }
}

impl Mapper for () {
    const AVAILABLE: bool = false;

    fn map_tensor(
        &mut self,
        _: mmap::event::MapTensor,
    ) -> Result<mmap::event::MapDone, mmap::event::Error> {
        Err(mmap::event::Error::UnsupportedPlatform)
    }

    fn release_mapping(
        &mut self,
        _: mmap::event::ReleaseMapping,
    ) -> Result<(), mmap::event::Error> {
        Err(mmap::event::Error::UnsupportedPlatform)
    }

    fn with_mapping<Operation>(
        &mut self,
        _: mmap::event::WithMapping<'_, Operation>,
    ) -> Result<Operation::Output, mmap::event::Error>
    where
        Operation: mmap::event::MappingOperation,
    {
        Err(mmap::event::Error::UnsupportedPlatform)
    }
}

impl Reader for () {
    const AVAILABLE: bool = false;

    fn read_tensor(
        &mut self,
        _: read::event::ReadTensor<'_>,
    ) -> Result<read::event::ReadTensorDone, read::event::Error> {
        Err(read::event::Error::UnsupportedPlatform)
    }
}

impl Stager for () {
    const AVAILABLE: bool = false;

    fn stage_tensor(
        &mut self,
        _: staged_read::event::StageWindow<'_>,
    ) -> Result<staged_read::event::StageWindowDone, staged_read::event::Error> {
        Err(staged_read::event::Error::UnsupportedPlatform)
    }
}

impl<MapperActor, ReaderActor, StagerActor> Mapper for Actors<MapperActor, ReaderActor, StagerActor>
where
    MapperActor: Mapper,
{
    const AVAILABLE: bool = MapperActor::AVAILABLE;

    fn map_tensor(
        &mut self,
        event: mmap::event::MapTensor,
    ) -> Result<mmap::event::MapDone, mmap::event::Error> {
        self.mapper.map_tensor(event)
    }

    fn release_mapping(
        &mut self,
        event: mmap::event::ReleaseMapping,
    ) -> Result<(), mmap::event::Error> {
        self.mapper.release_mapping(event)
    }

    fn with_mapping<Operation>(
        &mut self,
        event: mmap::event::WithMapping<'_, Operation>,
    ) -> Result<Operation::Output, mmap::event::Error>
    where
        Operation: mmap::event::MappingOperation,
    {
        self.mapper.with_mapping(event)
    }
}

impl<MapperActor, ReaderActor, StagerActor> Reader for Actors<MapperActor, ReaderActor, StagerActor>
where
    ReaderActor: Reader,
{
    const AVAILABLE: bool = ReaderActor::AVAILABLE;

    fn read_tensor(
        &mut self,
        event: read::event::ReadTensor<'_>,
    ) -> Result<read::event::ReadTensorDone, read::event::Error> {
        self.reader.read_tensor(event)
    }
}

impl<MapperActor, ReaderActor, StagerActor> Stager for Actors<MapperActor, ReaderActor, StagerActor>
where
    StagerActor: Stager,
{
    const AVAILABLE: bool = StagerActor::AVAILABLE;

    fn stage_tensor(
        &mut self,
        event: staged_read::event::StageWindow<'_>,
    ) -> Result<staged_read::event::StageWindowDone, staged_read::event::Error> {
        self.stager.stage_tensor(event)
    }
}

#[cfg(test)]
mod tests {
    use super::{Actors, Mapper, Reader, Stager};

    #[test]
    fn unavailable_unit_dependencies_fail_closed_and_static_flags_are_false() {
        assert!(!<() as Mapper>::AVAILABLE);
        assert!(!<() as Reader>::AVAILABLE);
        assert!(!<() as Stager>::AVAILABLE);
        let actors = Actors::new((), (), ());
        assert!(!<Actors as Mapper>::AVAILABLE);
        assert!(!<Actors as Reader>::AVAILABLE);
        assert!(!<Actors as Stager>::AVAILABLE);
        let _ = actors;
    }

    #[test]
    fn production_actor_flags_are_true() {
        assert!(<emel_io::mmap::Mapper as Mapper>::AVAILABLE);
        assert!(<emel_io::read::Reader as Reader>::AVAILABLE);
        assert!(<emel_io::staged_read::Stager as Stager>::AVAILABLE);
    }
}
