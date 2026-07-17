#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = emel_model_parity_inventory::validate_clang_ast_stream(data);
});
