//! Public observer for the maintained `AArch64` target `SiLU` actor.

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use emel_kernels::aarch64::{Kernel, UnarySilu};

    const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
    const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
    const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
    const AARCH64_GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
    const AARCH64_ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
    const AARCH64_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";

    fn print_bits(values: &[f32]) {
        for (index, value) in values.iter().enumerate() {
            if index != 0 {
                print!(",");
            }
            print!("{:08x}", value.to_bits());
        }
    }

    fn run_valid(name: &str, input: &[f32]) {
        let mut output = [0.0_f32; 5];
        let output = &mut output[..input.len()];
        let Some(mut kernel) = Kernel::try_new() else {
            println!("case={name} status=error output_bits=");
            return;
        };
        let status = kernel.process_event(UnarySilu::new(input), output).is_ok();
        print!(
            "case={name} status={} output_bits=",
            if status { "ok" } else { "error" }
        );
        print_bits(output);
        println!();
    }

    fn run_invalid() {
        let input = [1.0_f32, 2.0, 3.0];
        let sentinel = f32::from_bits(0x7fc0_1234);
        let mut output = [sentinel; 2];
        let Some(mut kernel) = Kernel::try_new() else {
            println!("case=invalid_shape status=error output_bits=");
            return;
        };
        let status = kernel
            .process_event(UnarySilu::new(&input), &mut output)
            .is_ok();
        assert!(kernel.is_ready(), "invalid dispatch must return to ready");
        assert!(
            output
                .iter()
                .all(|value| value.to_bits() == sentinel.to_bits())
        );
        print!(
            "case=invalid_shape status={} output_bits=",
            if status { "ok" } else { "error" }
        );
        print_bits(&output);
        println!();
    }

    pub fn run() {
        println!("kernel-aarch64-silu-live-parity/v1");
        println!("source_repository=stateforward/emel.cpp");
        println!("source_commit={SOURCE_COMMIT}");
        println!("source_kernel_events_blob={EVENTS_BLOB}");
        println!("source_kernel_detail_blob={DETAIL_BLOB}");
        println!("source_kernel_aarch64_guards_blob={AARCH64_GUARDS_BLOB}");
        println!("source_kernel_aarch64_actions_blob={AARCH64_ACTIONS_BLOB}");
        println!("source_kernel_aarch64_sm_blob={AARCH64_SM_BLOB}");
        println!("source_action_formula_span=src/emel/kernel/aarch64/actions.hpp:89-123,180-196");
        println!("source_guard_span=src/emel/kernel/aarch64/guards.hpp:826-886");
        println!("source_transition_span=src/emel/kernel/aarch64/sm.hpp:1161-1178");
        println!("scope=dense_f32_neon_silu_vector_and_scalar_tail_invalid_no_mutation");
        run_valid("vector", &[-4.0, -1.0, 0.0, 4.0]);
        run_valid("scalar_tail", &[1.0, 2.0, 3.0, -4.0, 0.5]);
        run_invalid();
    }
}

#[cfg(target_arch = "aarch64")]
fn main() {
    aarch64::run();
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 SiLU parity requires an AArch64 target");
    std::process::exit(2);
}
