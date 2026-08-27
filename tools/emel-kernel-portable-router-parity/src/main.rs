//! Live observer for the public portable Kernel router.

use emel_kernels::Kernel;
use emel_kernels::any::event::{OpAdd, OpDup, OpSqr, UnaryError, UnexpectedUnary};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    let input = [0.0_f32, -1.5, 2.25, 4.0];
    let lhs = [8.0_f32, -9.0, 6.0, 4.0];
    let rhs = [2.0_f32, 3.0, -2.0, 4.0];
    let square_input = [-3.5_f32, -1.0, 0.5, 2.25];

    println!("kernel-portable-router-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("target_arch=portable");
    println!("scope=portable_router_dup_add_sqr_and_typed_unexpected_rejection");
    println!("execution=one_public_kernel_dispatching_three_live_events");

    let mut kernel = Kernel::new();
    let mut output = [0.0_f32; 4];
    assert!(kernel.is_ready());

    assert_eq!(
        kernel.process_event(OpDup::new(&input, &mut output)),
        Ok(())
    );
    println!("case=op_dup status=ok output_bits={}", bits(&output));

    assert_eq!(
        kernel.process_event(OpAdd::new(&lhs, &rhs, &mut output)),
        Ok(())
    );
    println!("case=op_add status=ok output_bits={}", bits(&output));

    assert_eq!(
        kernel.process_event(OpSqr::new(&square_input, &mut output)),
        Ok(())
    );
    println!("case=op_sqr status=ok output_bits={}", bits(&output));

    let rejected = [SENTINEL; 4];
    assert_eq!(
        kernel.process_event(UnexpectedUnary),
        Err(UnaryError::UnexpectedEvent)
    );
    assert!(
        rejected
            .iter()
            .all(|value| value.to_bits() == SENTINEL.to_bits())
    );
    println!(
        "case=typed_unexpected_event status=reject error=UnexpectedEvent output_bits={}",
        bits(&rejected)
    );
}
