#![no_main]

use std::path::PathBuf;
use std::sync::OnceLock;

use emel_io::mmap::Mapper;
use emel_io::mmap::event::{
    AdviseDontNeed, AdviseSequential, AdviseWillNeed, MapTensor, MmapSource, ReleaseMapping,
    WithMapping,
};
use libfuzzer_sys::fuzz_target;

const FIXTURE_BYTES: usize = 65_536;
static FIXTURE: OnceLock<(PathBuf, MmapSource)> = OnceLock::new();

fuzz_target!(|data: &[u8]| {
    let file = mmap_source();
    let tensor_id = read_i32(data, 0);
    let requested_len = match data.get(4).map_or(0, |value| value % 4) {
        0 => 0,
        1 => u64::MAX,
        _ => u64::from(read_u16(data, 5)).clamp(1, FIXTURE_BYTES as u64),
    };
    let file_index = read_u16(data, 6);
    let offset = match data.get(8).map_or(0, |value| value % 3) {
        0 => 0,
        1 => 4_096,
        _ => read_u64(data, 9),
    };
    let mut mapper = Mapper::new();

    let request =
        MapTensor::new(tensor_id, file.clone(), offset, requested_len).with_file_index(file_index);
    let mapped = mapper.process_event(request).ok();
    if let Some(done) = mapped {
        let mut checksum = 0_u8;
        let mut observe = |bytes: &[u8]| {
            checksum = bytes.iter().fold(0, |value, byte| value ^ byte);
        };
        mapper
            .process_event(WithMapping::new(tensor_id, done.handle(), &mut observe))
            .expect("a committed fuzz mapping must remain accessible");
        std::hint::black_box(checksum);

        let wrong_owner = tensor_id.wrapping_add(1);
        let wrong_handle = done.handle().wrapping_add(read_u32(data, 17) | 1);
        let mut never_called = |_: &[u8]| panic!("invalid ownership invoked callback");
        let _ = mapper.process_event(WithMapping::new(
            wrong_owner,
            wrong_handle,
            &mut never_called,
        ));

        let advice_offset = read_u64(data, 21);
        let advice_len = read_u64(data, 29);
        match data.get(37).map_or(0, |value| value % 3) {
            0 => {
                let _ = mapper.process_event(AdviseSequential::new(
                    tensor_id,
                    done.handle(),
                    advice_offset,
                    advice_len,
                ));
            }
            1 => {
                let _ = mapper.process_event(AdviseWillNeed::new(
                    tensor_id,
                    done.handle(),
                    advice_offset,
                    advice_len,
                ));
            }
            _ => {
                let _ = mapper.process_event(AdviseDontNeed::new(
                    tensor_id,
                    done.handle(),
                    advice_offset,
                    advice_len,
                ));
            }
        }

        let _ = mapper.process_event(ReleaseMapping::new(wrong_owner, done.handle()));
        let _ = mapper.process_event(ReleaseMapping::new(tensor_id, wrong_handle));
        let _ = mapper.process_event(ReleaseMapping::new(tensor_id, done.handle()));
    }

    let recovery = mapper
        .process_event(MapTensor::new(0x5245_4356, file.clone(), 0, 4_096))
        .expect("the stable recovery fixture must map");
    mapper
        .process_event(ReleaseMapping::new(0x5245_4356, recovery.handle()))
        .expect("the recovery mapping must release");
});

#[allow(
    unsafe_code,
    reason = "the process-private fuzz fixture is initialized once and never mutated"
)]
fn mmap_source() -> &'static MmapSource {
    &FIXTURE
        .get_or_init(|| {
            let path =
                std::env::temp_dir().join(format!("emel-io-mmap-fuzz-{}.bin", std::process::id()));
            let bytes = (0_u8..=255).cycle().take(FIXTURE_BYTES).collect::<Vec<_>>();
            std::fs::write(&path, bytes).expect("write stable mmap fuzz fixture");
            // SAFETY: initialization completed before this call, this OnceLock
            // owns the process-private fixture, no code mutates it during or after
            // construction, and it outlives every mapping.
            let file = unsafe { MmapSource::open(&path) }.expect("open mmap fuzz fixture source");
            (path, file)
        })
        .1
}

fn read_u16(data: &[u8], offset: usize) -> u16 {
    let mut bytes = [0_u8; 2];
    if let Some(source) = data.get(offset..).and_then(|tail| tail.get(..2)) {
        bytes.copy_from_slice(source);
    }
    u16::from_le_bytes(bytes)
}

fn read_i32(data: &[u8], offset: usize) -> i32 {
    let mut bytes = [0_u8; 4];
    if let Some(source) = data.get(offset..).and_then(|tail| tail.get(..4)) {
        bytes.copy_from_slice(source);
    }
    i32::from_le_bytes(bytes)
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    let mut bytes = [0_u8; 4];
    if let Some(source) = data.get(offset..).and_then(|tail| tail.get(..4)) {
        bytes.copy_from_slice(source);
    }
    u32::from_le_bytes(bytes)
}

fn read_u64(data: &[u8], offset: usize) -> u64 {
    let mut bytes = [0_u8; 8];
    if let Some(source) = data.get(offset..).and_then(|tail| tail.get(..8)) {
        bytes.copy_from_slice(source);
    }
    u64::from_le_bytes(bytes)
}
