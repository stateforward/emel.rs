//! Public observer for the maintained `AArch64` target F32 duplication actor.

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use emel_kernels::aarch64::{Dup, Kernel};
    use std::fmt::Write as _;

    const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
    const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
    const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
    const AARCH64_GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
    const AARCH64_ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
    const AARCH64_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
    const SENTINEL: f32 = f32::from_bits(0x7fc0_1234);

    fn bits(values: &[f32]) -> String {
        let mut output = String::new();
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                output.push(',');
            }
            write!(&mut output, "{:08x}", value.to_bits()).expect("String write cannot fail");
        }
        output
    }

    fn print_header() {
        println!("kernel-aarch64-dup-f32-live/v1");
        println!("source_repository=stateforward/emel.cpp");
        println!("source_commit={SOURCE_COMMIT}");
        println!("source_kernel_events_blob={EVENTS_BLOB}");
        println!("source_kernel_detail_blob={DETAIL_BLOB}");
        println!("source_kernel_aarch64_guards_blob={AARCH64_GUARDS_BLOB}");
        println!("source_kernel_aarch64_actions_blob={AARCH64_ACTIONS_BLOB}");
        println!("source_kernel_aarch64_sm_blob={AARCH64_SM_BLOB}");
        println!(
            "source_detail_span=src/emel/kernel/detail.hpp:1535-1555,1700-1748,1920-1926,3788-3791"
        );
        println!("source_guard_span=src/emel/kernel/aarch64/guards.hpp:320-329,805-824");
        println!(
            "source_action_span=src/emel/kernel/aarch64/actions.hpp:887-930,2217-2237,8447-8479"
        );
        println!("source_transition_span=src/emel/kernel/aarch64/sm.hpp:30-43");
        println!("scope=dense_f32_dup_vector_tail_zero_count_metadata_rejection");
        println!(
            "metadata_semantics=cpp_tensor_view_rejects_unknown_dtype;rust_slice_api_unrepresentable"
        );
    }

    fn run_vector(kernel: &mut Kernel) {
        let input = [
            0.0_f32,
            -1.5,
            2.25,
            4.0,
            -8.0,
            16.0,
            32.0,
            -64.0,
            f32::from_bits(0x8000_0000),
        ];
        let mut output = [SENTINEL; 9];
        let result = kernel.process_event(Dup::new(&input), &mut output);
        assert!(result.is_ok());
        assert!(kernel.is_ready());
        assert_eq!(output.map(f32::to_bits), input.map(f32::to_bits));
        println!("case=vector_tail status=ok output_bits={}", bits(&output));
    }

    fn run_zero_count(kernel: &mut Kernel) {
        let output = [SENTINEL; 1];
        let result = kernel.process_event(Dup::new(&[]), &mut []);
        assert!(result.is_err());
        assert!(kernel.is_ready());
        assert_eq!(output[0].to_bits(), SENTINEL.to_bits());
        println!("case=zero_count status=error output_bits={}", bits(&output));
    }

    pub fn run() {
        print_header();
        let Some(mut kernel) = Kernel::try_new() else {
            eprintln!("AArch64 target kernel unavailable");
            std::process::exit(2);
        };
        run_vector(&mut kernel);
        run_zero_count(&mut kernel);
        println!("rust_metadata_case=unrepresentable_through_typed_dense_f32_slice_api");
    }
}

#[cfg(target_arch = "aarch64")]
fn main() {
    aarch64::run();
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 target duplication parity requires an AArch64 target");
    std::process::exit(2);
}
