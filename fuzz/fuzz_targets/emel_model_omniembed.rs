#![no_main]

use libfuzzer_sys::fuzz_target;

mod omniembed_support;

fuzz_target!(|input: &[u8]| omniembed_support::exercise(input));
