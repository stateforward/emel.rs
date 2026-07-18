use std::sync::Arc;

use emel_gguf::Loader;
use emel_gguf::event::{Bind, Parse, Probe, Storage};

pub const U8: u32 = 0;
pub const I8: u32 = 1;
pub const U16: u32 = 2;
pub const I16: u32 = 3;
pub const U32: u32 = 4;
pub const I32: u32 = 5;
pub const F32: u32 = 6;
pub const BOOL: u32 = 7;
pub const STRING: u32 = 8;
pub const ARRAY: u32 = 9;
pub const U64: u32 = 10;
pub const I64: u32 = 11;
pub const F64: u32 = 12;

pub struct Fixture {
    entries: Vec<(Vec<u8>, u32, Vec<u8>)>,
}
impl Fixture {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
    pub fn scalar(mut self, key: &[u8], kind: u32, payload: impl Into<Vec<u8>>) -> Self {
        self.entries.push((key.to_vec(), kind, payload.into()));
        self
    }
    pub fn string(self, key: &[u8], value: &[u8]) -> Self {
        self.scalar(key, STRING, string_payload(value))
    }
    pub fn array(self, key: &[u8], kind: u32, values: &[u8], count: u64) -> Self {
        self.scalar(key, ARRAY, array_payload(kind, count, values))
    }
    pub fn strings(self, key: &[u8], values: &[&[u8]]) -> Self {
        let mut p = Vec::new();
        for v in values {
            push_string(&mut p, v);
        }
        self.array(
            key,
            STRING,
            &p,
            u64::try_from(values.len()).expect("fixture string count fits u64"),
        )
    }
    pub fn repeated_strings(self, key: &[u8], value: &[u8], count: usize) -> Self {
        let mut payload = Vec::with_capacity(count.saturating_mul(value.len().saturating_add(8)));
        for _ in 0..count {
            push_string(&mut payload, value);
        }
        self.array(
            key,
            STRING,
            &payload,
            u64::try_from(count).expect("fixture string count fits u64"),
        )
    }
    pub fn build(self) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        push_u32(&mut b, 3);
        push_u64(&mut b, 0);
        push_u64(
            &mut b,
            u64::try_from(self.entries.len()).expect("fixture entry count fits u64"),
        );
        for (k, t, p) in self.entries {
            push_string(&mut b, &k);
            push_u32(&mut b, t);
            b.extend_from_slice(&p);
        }
        b
    }
}
fn push_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_u64(b: &mut Vec<u8>, v: u64) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_string(b: &mut Vec<u8>, v: &[u8]) {
    push_u64(
        b,
        u64::try_from(v.len()).expect("fixture string length fits u64"),
    );
    b.extend_from_slice(v);
}
fn string_payload(v: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    push_string(&mut b, v);
    b
}
fn array_payload(k: u32, n: u64, v: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    push_u32(&mut b, k);
    push_u64(&mut b, n);
    b.extend_from_slice(v);
    b
}
pub fn load(bytes: Vec<u8>) -> Loader {
    let mut g = Loader::new();
    let p = g.process_event(Probe::new(Arc::from(bytes))).unwrap();
    g.process_event(Bind::new(Storage::exact(p).unwrap()))
        .unwrap();
    g.process_event(Parse::new()).unwrap();
    g
}
