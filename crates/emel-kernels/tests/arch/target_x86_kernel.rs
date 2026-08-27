#![allow(clippy::float_cmp)]
#![allow(missing_docs)]
#![cfg(target_arch = "x86_64")]

use allocation_counter::measure;
use emel_kernels::any::unary::UnaryError;
use emel_kernels::x86_64::F32FmaError;
use emel_kernels::x86_64::X86BroadcastF32Error;
use emel_kernels::x86_64::X86F32GemvError;
use emel_kernels::x86_64::{
    OpScalarUnaryElu, OpScalarUnaryExp, OpScalarUnaryGelu, OpScalarUnarySilu, OpScalarUnaryTanh,
    UnexpectedX86Kernel, X86BinaryAdd, X86BroadcastAdd, X86BroadcastMul, X86Dup, X86Fma, X86Gemv,
    X86Kernel, X86KernelError,
};

#[test]
fn router_is_ready_and_dispatches_dup() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    assert!(kernel.is_ready());

    let input = [1.0_f32, -2.5, 4.0, 8.25];
    let mut output = [0.0_f32; 4];
    assert_eq!(
        kernel.process_event(X86Dup::new(&input), &mut output),
        Ok(())
    );
    assert_eq!(output, input);
    assert!(kernel.is_ready());
}

#[test]
fn router_dispatches_binary_add() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    let lhs = [1.0_f32, -2.0, 5.5, 8.0];
    let rhs = [2.0_f32, 4.0, -0.5, 1.5];
    let mut output = [0.0_f32; 4];

    assert_eq!(
        kernel.process_event(X86BinaryAdd::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    assert_eq!(output, [3.0, 2.0, 5.0, 9.5]);
}

#[test]
fn router_dispatches_fma_and_single_rhs_gemv() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs_matrix = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut matrix_output = [f32::NAN; 4];
    assert_eq!(
        kernel.process_event(X86Fma::new(&lhs, &rhs_matrix, 2, 2, 3), &mut matrix_output),
        Ok(())
    );
    assert_eq!(matrix_output, [58.0, 64.0, 139.0, 154.0]);

    let rhs_vector = [7.0_f32, 8.0, 9.0];
    let mut vector_output = [f32::NAN; 2];
    assert_eq!(
        kernel.process_event(X86Gemv::new(&lhs, &rhs_vector, 2, 3), &mut vector_output),
        Ok(())
    );
    assert_eq!(vector_output, [50.0, 122.0]);
    assert!(kernel.is_ready());
}

#[test]
fn router_rejects_fma_vector_shape_and_gemv_invalid_shape_without_mutation() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    let mut output = [9.0_f32; 2];
    assert_eq!(
        kernel.process_event(
            X86Fma::new(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0], 2, 1, 2),
            &mut output,
        ),
        Err(F32FmaError::InvalidShape)
    );
    assert_eq!(output, [9.0; 2]);
    assert!(kernel.is_ready());

    assert_eq!(
        kernel.process_event(X86Gemv::new(&[1.0, 2.0], &[1.0], 2, 1), &mut output,),
        Err(X86F32GemvError::InvalidShape)
    );
    assert_eq!(output, [9.0; 2]);
    assert!(kernel.is_ready());

    assert_eq!(
        kernel.process_event(
            X86Gemv::new(&[1.0, 2.0, 3.0, 4.0], &[2.0, 3.0], 2, 2),
            &mut output,
        ),
        Ok(())
    );
    assert_eq!(output, [8.0, 18.0]);
}

#[test]
fn router_fma_and_gemv_dispatch_is_allocation_free() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    let lhs_matrix = [1.0_f32, 2.0, 3.0, 4.0];
    let rhs_matrix = [5.0_f32, 6.0, 7.0, 8.0];
    let lhs_vector = [1.0_f32, 2.0, 3.0, 4.0];
    let rhs_vector = [5.0_f32, 6.0];
    let mut matrix_output = [0.0_f32; 4];
    let mut vector_output = [0.0_f32; 2];
    let allocation = measure(|| {
        for _ in 0..64 {
            assert_eq!(
                kernel.process_event(
                    X86Fma::new(&lhs_matrix, &rhs_matrix, 2, 2, 2),
                    &mut matrix_output,
                ),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(
                    X86Gemv::new(&lhs_vector, &rhs_vector, 2, 2),
                    &mut vector_output,
                ),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(matrix_output, [19.0, 22.0, 43.0, 50.0]);
    assert_eq!(vector_output, [17.0, 39.0]);
}

#[test]
fn router_dispatches_binary_row_broadcast_add_and_mul() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    let input = [8.0_f32, -9.0, 6.0, 4.0, -2.0, 3.0, 7.0, 11.0];
    let row = [2.0_f32, 3.0, -2.0, 4.0];
    let mut output = [0.0_f32; 8];

    assert_eq!(
        kernel.process_event(X86BroadcastAdd::new(&input, &row), &mut output),
        Ok(())
    );
    assert_eq!(output, [10.0, -6.0, 4.0, 8.0, 0.0, 6.0, 5.0, 15.0]);

    assert_eq!(
        kernel.process_event(X86BroadcastMul::new(&input, &row), &mut output),
        Ok(())
    );
    assert_eq!(output, [16.0, -27.0, -12.0, 16.0, -4.0, 9.0, -14.0, 44.0]);
    assert!(kernel.is_ready());
}

#[test]
fn router_rejects_binary_row_broadcast_without_mutation_and_recovers() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    let mut output = [9.0_f32; 4];

    assert_eq!(
        kernel.process_event(
            X86BroadcastAdd::new(&[1.0, 2.0, 3.0], &[4.0, 5.0]),
            &mut output,
        ),
        Err(X86BroadcastF32Error::InvalidShape)
    );
    assert_eq!(output, [9.0; 4]);
    assert!(kernel.is_ready());

    assert_eq!(
        kernel.process_event(
            X86BroadcastMul::new(&[1.0, 2.0, 3.0, 4.0], &[2.0, 3.0]),
            &mut output,
        ),
        Ok(())
    );
    assert_eq!(output, [2.0, 6.0, 6.0, 12.0]);
}

#[test]
fn router_binary_row_broadcast_dispatch_is_allocation_free() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    let input = [1.0_f32; 12];
    let row = [2.0_f32, 3.0, 4.0];
    let mut output = [0.0_f32; 12];
    let allocation = measure(|| {
        for _ in 0..64 {
            assert_eq!(
                kernel.process_event(X86BroadcastAdd::new(&input, &row), &mut output),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(
        output,
        [3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn router_rejects_invalid_child_request_without_mutation() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    let mut output = [7.0_f32; 3];
    let before = output;

    assert!(
        kernel
            .process_event(X86Dup::new(&[1.0, 2.0]), &mut output)
            .is_err()
    );
    assert_eq!(output, before);
    assert!(kernel.is_ready());
}

#[test]
fn router_unexpected_event_is_explicit() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    assert_eq!(
        kernel.process_event(UnexpectedX86Kernel, &mut []),
        Err(X86KernelError::UnexpectedEvent)
    );
}

#[test]
fn router_dispatch_is_allocation_free_after_construction() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    let input = [1.0_f32; 17];
    let mut output = [0.0_f32; 17];
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(X86Dup::new(&input), &mut output),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
}

#[test]
fn router_dispatches_scalar_unary_slice() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    let input = [-1.0_f32, 0.0, 1.0];
    let mut output = [0.0_f32; 3];

    assert_eq!(
        kernel.process_event(OpScalarUnaryExp::new(&input), &mut output),
        Ok(())
    );
    assert!((output[0] - (-1.0_f32).exp()).abs() < f32::EPSILON);
    assert!((output[2] - 1.0_f32.exp()).abs() < f32::EPSILON);

    assert_eq!(
        kernel.process_event(OpScalarUnaryTanh::new(&input), &mut output),
        Ok(())
    );
    assert!((output[0] - (-1.0_f32).tanh()).abs() < f32::EPSILON);
    assert!((output[2] - 1.0_f32.tanh()).abs() < f32::EPSILON);

    assert_eq!(
        kernel.process_event(OpScalarUnaryElu::new(&input), &mut output),
        Ok(())
    );
    assert!((output[0] - (-1.0_f32).exp_m1()).abs() < f32::EPSILON);
    assert_eq!(output[2], 1.0);

    assert_eq!(
        kernel.process_event(OpScalarUnaryGelu::new(&input), &mut output),
        Ok(())
    );
    assert!(output[0].is_finite() && output[2].is_finite());

    assert_eq!(
        kernel.process_event(OpScalarUnarySilu::new(&input), &mut output),
        Ok(())
    );
    assert!((output[0] - (-1.0_f32 / (1.0 + 1.0_f32.exp()))).abs() < f32::EPSILON);
    assert!((output[2] - (1.0_f32 / (1.0 + (-1.0_f32).exp()))).abs() < f32::EPSILON);
    assert!(kernel.is_ready());
}

#[test]
fn scalar_unary_invalid_shape_does_not_mutate_and_recovers() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    let mut output = [7.0_f32; 2];
    let before = output;

    assert_eq!(
        kernel.process_event(OpScalarUnaryExp::new(&[1.0]), &mut output),
        Err(UnaryError::ShapeMismatch)
    );
    assert_eq!(output, before);

    assert_eq!(
        kernel.process_event(OpScalarUnaryTanh::new(&[]), &mut []),
        Err(UnaryError::InvalidView)
    );

    let input = [0.5_f32; 2];
    assert_eq!(
        kernel.process_event(OpScalarUnarySilu::new(&input), &mut output),
        Ok(())
    );
    assert!(kernel.is_ready());
}

#[test]
fn scalar_unary_dispatch_is_allocation_free_after_construction() {
    let Some(mut kernel) = X86Kernel::try_new() else {
        return;
    };
    let input = [0.25_f32; 17];
    let mut output = [0.0_f32; 17];
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpScalarUnaryExp::new(&input), &mut output),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(OpScalarUnaryTanh::new(&input), &mut output),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(OpScalarUnaryElu::new(&input), &mut output),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(OpScalarUnaryGelu::new(&input), &mut output),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(OpScalarUnarySilu::new(&input), &mut output),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
}

#[test]
fn scalar_unary_source_identity_is_pinned() {
    assert_eq!(
        emel_kernels::x86_64::PINNED_EMEL_CPP_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    assert_eq!(
        emel_kernels::x86_64::PINNED_DETAIL_SPAN,
        "src/emel/kernel/detail.hpp:2392-2411,3271-3325"
    );
    assert_eq!(
        emel_kernels::x86_64::PINNED_X86_GUARD_SPAN,
        "src/emel/kernel/x86_64/guards.hpp:237-241,255-264"
    );
    assert_eq!(
        emel_kernels::x86_64::PINNED_X86_GUARD_BLOB,
        "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf"
    );
    assert_eq!(
        emel_kernels::x86_64::PINNED_X86_ACTION_BLOB,
        "d45558f5eb96950f43c16a09d768cb4f382d6d61"
    );
    assert_eq!(
        emel_kernels::x86_64::PINNED_X86_ACTION_SPAN,
        "src/emel/kernel/x86_64/actions.hpp:2583-2600,2716-2720"
    );
    assert_eq!(
        emel_kernels::x86_64::PINNED_X86_TRANSITION_BLOB,
        "0b4d635ebbd0fbd52dbca8a2345547fb571205c8"
    );
    assert_eq!(
        emel_kernels::x86_64::PINNED_X86_TRANSITION_SPAN,
        "src/emel/kernel/x86_64/sm.hpp:1065-1093"
    );
}
