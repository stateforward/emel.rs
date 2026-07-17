#![no_main]

use emel_model_parity_inventory::{TsvSchema, validate_tsv};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((selector, payload)) = data.split_first() else {
        return;
    };
    let schema = match selector {
        b'I' => TsvSchema::Inventory,
        b'C' => TsvSchema::Coverage,
        b'G' => TsvSchema::Gap,
        _ => return,
    };
    let _ = validate_tsv(schema, payload);
});
