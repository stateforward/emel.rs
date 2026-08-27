//! Public observer for the maintained `AArch64` target scalar F16 matmul actor.

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use emel_kernels::aarch64::{Kernel, OpScalarMulMatF16};
    use emel_kernels::any::f16_matmul::{F16MatmulError, F16View};
    use emel_kernels::any::tensor_view::{DType, Layout, TensorViewMut};
    use std::fmt::Write as _;

    const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
    const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
    const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
    const AARCH64_GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
    const AARCH64_ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
    const AARCH64_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
    const SENTINEL: f32 = f32::from_bits(0x7fc0_1234);

    const LHS: [u16; 6] = [0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600];
    const RHS: [u16; 6] = [0x3c00, 0x0000, 0x0000, 0x3c00, 0x3c00, 0x3c00];

    const fn f16_layout(ne: [u64; 4]) -> Layout {
        let mut nb = [2_u64; 4];
        nb[1] = 2 * ne[0];
        nb[2] = nb[1] * ne[1];
        nb[3] = nb[2] * ne[2];
        Layout::new(DType::F16, ne, nb)
    }

    fn f32_layout(ne: [u64; 4]) -> Layout {
        Layout::contiguous(DType::F32, ne).expect("fixture layout fits")
    }

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
        println!("kernel-aarch64-f16-matmul-live/v1");
        println!("source_repository=stateforward/emel.cpp");
        println!("source_commit={SOURCE_COMMIT}");
        println!("source_kernel_events_blob={EVENTS_BLOB}");
        println!("source_kernel_detail_blob={DETAIL_BLOB}");
        println!("source_kernel_aarch64_guards_blob={AARCH64_GUARDS_BLOB}");
        println!("source_kernel_aarch64_actions_blob={AARCH64_ACTIONS_BLOB}");
        println!("source_kernel_aarch64_sm_blob={AARCH64_SM_BLOB}");
        println!("source_detail_span=src/emel/kernel/detail.hpp:3874-3894,4198-4214,5089-5096");
        println!("source_guard_span=src/emel/kernel/aarch64/guards.hpp:275-285,973-994");
        println!("source_action_span=src/emel/kernel/aarch64/actions.hpp:7044-7064,9213-9216");
        println!("source_transition_span=src/emel/kernel/aarch64/sm.hpp:525-543");
        println!("scope=dense_f16_matmul_target_scalar_positive_shape_dtype_rejection");
    }

    fn run_matrix(kernel: &mut Kernel) {
        let mut output = [0.0_f32; 4];
        let result = kernel.process_f16_matmul(OpScalarMulMatF16::new(
            F16View::new(&LHS, f16_layout([3, 2, 1, 1])),
            F16View::new(&RHS, f16_layout([3, 2, 1, 1])),
            TensorViewMut::new(&mut output, f32_layout([2, 2, 1, 1])),
        ));
        println!(
            "case=matrix status={} output_bits={}",
            if result.is_ok() { "ok" } else { "error" },
            bits(&output)
        );
    }

    fn run_invalid_shape(kernel: &mut Kernel) {
        let mut output = [SENTINEL; 4];
        let result = kernel.process_f16_matmul(OpScalarMulMatF16::new(
            F16View::new(&LHS, f16_layout([3, 2, 1, 1])),
            F16View::new(&RHS[..4], f16_layout([2, 2, 1, 1])),
            TensorViewMut::new(&mut output, f32_layout([2, 2, 1, 1])),
        ));
        assert_eq!(result, Err(F16MatmulError::ShapeMismatch));
        assert!(
            output
                .iter()
                .all(|value| value.to_bits() == SENTINEL.to_bits())
        );
        println!(
            "case=invalid_shape status=error output_bits={}",
            bits(&output)
        );
    }

    fn run_invalid_dtype(kernel: &mut Kernel) {
        let mut output = [SENTINEL; 4];
        let result = kernel.process_f16_matmul(OpScalarMulMatF16::new(
            F16View::new(&LHS, f16_layout([3, 2, 1, 1])),
            F16View::new(&RHS, f16_layout([3, 2, 1, 1])),
            TensorViewMut::new(
                &mut output,
                Layout::new(DType::F16, [2, 2, 1, 1], [2, 4, 8, 8]),
            ),
        ));
        assert_eq!(result, Err(F16MatmulError::InvalidView));
        assert!(
            output
                .iter()
                .all(|value| value.to_bits() == SENTINEL.to_bits())
        );
        println!(
            "case=invalid_dtype status=error output_bits={}",
            bits(&output)
        );
    }

    pub fn run() {
        print_header();
        let Some(mut kernel) = Kernel::try_new() else {
            eprintln!("AArch64 target kernel unavailable");
            std::process::exit(2);
        };
        run_matrix(&mut kernel);
        run_invalid_shape(&mut kernel);
        run_invalid_dtype(&mut kernel);
        assert!(kernel.is_ready());
    }
}

#[cfg(target_arch = "aarch64")]
fn main() {
    aarch64::run();
}

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 F16 matmul parity requires an AArch64 target");
    std::process::exit(2);
}
