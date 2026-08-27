//! Public observer for the maintained portable F32 binary actor.

use emel_kernels::any::binary::{BinaryKernel, OpAdd, OpDiv, OpMul, OpSub};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const X86_ACTIONS_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
const X86_GUARDS_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{value:08x}", value = value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn layout(ne: [u64; 4], nb: [u64; 4]) -> Layout {
    Layout::new(DType::F32, ne, nb)
}

fn logical_values(data: &[f32], ne: [u64; 4], nb: [u64; 4]) -> Vec<f32> {
    let mut strides = nb;
    if strides[0] == 0 {
        strides[0] = 4;
        for dimension in 1..4 {
            strides[dimension] = strides[dimension - 1]
                .checked_mul(ne[dimension - 1])
                .expect("fixture stride fits");
        }
    }
    let count = ne
        .iter()
        .try_fold(1_usize, |count, extent| {
            count.checked_mul(usize::try_from(*extent).expect("fixture extent fits"))
        })
        .expect("fixture count fits");
    (0..count)
        .map(|ordinal| {
            let mut remaining = ordinal as u64;
            let mut byte_offset = 0_u64;
            for dimension in 0..4 {
                byte_offset += (remaining % ne[dimension]) * strides[dimension];
                remaining /= ne[dimension];
            }
            data[usize::try_from(byte_offset / 4).expect("fixture offset fits")]
        })
        .collect()
}

struct Case<'a, const N: usize> {
    name: &'a str,
    op: fn(
        &mut BinaryKernel,
        TensorView<'_>,
        TensorView<'_>,
        TensorViewMut<'_>,
    ) -> Result<(), emel_kernels::any::binary::BinaryError>,
    lhs: &'a [f32],
    lhs_layout: Layout,
    rhs: &'a [f32],
    rhs_layout: Layout,
    output: &'a mut [f32; N],
    output_layout: Layout,
}

fn run_case<const N: usize>(case: Case<'_, N>) {
    let mut actor = BinaryKernel::new();
    let result = (case.op)(
        &mut actor,
        TensorView::new(case.lhs, case.lhs_layout),
        TensorView::new(case.rhs, case.rhs_layout),
        TensorViewMut::new(case.output, case.output_layout),
    );
    match result {
        Ok(()) => println!(
            "case={} status=ok output_bits={}",
            case.name,
            bits(&logical_values(
                case.output,
                case.output_layout.ne(),
                case.output_layout.nb()
            ))
        ),
        Err(_) => println!(
            "case={} status=error output_bits={}",
            case.name,
            bits(&logical_values(
                case.output,
                case.output_layout.ne(),
                case.output_layout.nb()
            ))
        ),
    }
}

fn add(
    actor: &mut BinaryKernel,
    lhs: TensorView<'_>,
    rhs: TensorView<'_>,
    output: TensorViewMut<'_>,
) -> Result<(), emel_kernels::any::binary::BinaryError> {
    actor.process_event(OpAdd::new(lhs, rhs, output))
}

fn sub(
    actor: &mut BinaryKernel,
    lhs: TensorView<'_>,
    rhs: TensorView<'_>,
    output: TensorViewMut<'_>,
) -> Result<(), emel_kernels::any::binary::BinaryError> {
    actor.process_event(OpSub::new(lhs, rhs, output))
}

fn mul(
    actor: &mut BinaryKernel,
    lhs: TensorView<'_>,
    rhs: TensorView<'_>,
    output: TensorViewMut<'_>,
) -> Result<(), emel_kernels::any::binary::BinaryError> {
    actor.process_event(OpMul::new(lhs, rhs, output))
}

fn div(
    actor: &mut BinaryKernel,
    lhs: TensorView<'_>,
    rhs: TensorView<'_>,
    output: TensorViewMut<'_>,
) -> Result<(), emel_kernels::any::binary::BinaryError> {
    actor.process_event(OpDiv::new(lhs, rhs, output))
}

fn main() {
    println!("kernel-binary-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_sml_commit=49207123cd3f39767764bae774932cb48623f92f");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_x86_sm_blob={X86_SM_BLOB}");
    println!("source_kernel_x86_actions_blob={X86_ACTIONS_BLOB}");
    println!("source_kernel_x86_guards_blob={X86_GUARDS_BLOB}");
    println!("source_detail_guard_span=3794-3797");
    println!("source_detail_run_span=2366-2383");
    println!("scope=portable_f32_equal_count_dense_or_validated_strided_implicit");

    let lhs = [1.0_f32, 2.0, 3.0, 4.0];
    let rhs = [10.0_f32, 20.0, 30.0, 40.0];
    let dense = [4, 1, 1, 1];
    let dense_nb = [4, 16, 64, 64];
    let mut output = [0.0_f32; 4];
    run_case(Case {
        name: "dense_add",
        op: add,
        lhs: &lhs,
        lhs_layout: layout(dense, dense_nb),
        rhs: &rhs,
        rhs_layout: layout(dense, dense_nb),
        output: &mut output,
        output_layout: layout(dense, dense_nb),
    });
    output.fill(0.0);
    run_case(Case {
        name: "dense_sub",
        op: sub,
        lhs: &lhs,
        lhs_layout: layout(dense, dense_nb),
        rhs: &rhs,
        rhs_layout: layout(dense, dense_nb),
        output: &mut output,
        output_layout: layout(dense, dense_nb),
    });
    output.fill(0.0);
    run_case(Case {
        name: "dense_mul",
        op: mul,
        lhs: &lhs,
        lhs_layout: layout(dense, dense_nb),
        rhs: &rhs,
        rhs_layout: layout(dense, dense_nb),
        output: &mut output,
        output_layout: layout(dense, dense_nb),
    });
    output.fill(0.0);
    run_case(Case {
        name: "dense_div",
        op: div,
        lhs: &lhs,
        lhs_layout: layout(dense, dense_nb),
        rhs: &rhs,
        rhs_layout: layout(dense, dense_nb),
        output: &mut output,
        output_layout: layout(dense, dense_nb),
    });

    let sparse = [3, 1, 1, 1];
    let sparse_nb = [8, 24, 72, 72];
    let sparse_lhs = [1.0_f32, 99.0, 2.0, 99.0, 3.0];
    let sparse_rhs = [10.0_f32, 99.0, 20.0, 99.0, 30.0];
    let mut sparse_output = [0.0_f32; 5];
    run_case(Case {
        name: "strided_add",
        op: add,
        lhs: &sparse_lhs,
        lhs_layout: layout(sparse, sparse_nb),
        rhs: &sparse_rhs,
        rhs_layout: layout(sparse, sparse_nb),
        output: &mut sparse_output,
        output_layout: layout(sparse, sparse_nb),
    });

    let reshaped = [2, 2, 1, 1];
    let reshaped_nb = [4, 8, 16, 16];
    output.fill(0.0);
    run_case(Case {
        name: "count_equal_different_dimensions",
        op: add,
        lhs: &lhs,
        lhs_layout: layout(reshaped, reshaped_nb),
        rhs: &rhs,
        rhs_layout: layout(dense, dense_nb),
        output: &mut output,
        output_layout: layout(dense, dense_nb),
    });

    let implicit = [0, 0, 0, 0];
    output.fill(0.0);
    run_case(Case {
        name: "implicit_contiguous_mul",
        op: mul,
        lhs: &lhs,
        lhs_layout: layout(dense, implicit),
        rhs: &rhs,
        rhs_layout: layout(dense, implicit),
        output: &mut output,
        output_layout: layout(dense, implicit),
    });

    let short = [3, 1, 1, 1];
    let short_nb = [4, 12, 36, 36];
    let mut rejected = [9.0_f32; 3];
    run_case(Case {
        name: "count_mismatch_rejection",
        op: div,
        lhs: &lhs,
        lhs_layout: layout(dense, dense_nb),
        rhs: &rhs,
        rhs_layout: layout(dense, dense_nb),
        output: &mut rejected,
        output_layout: layout(short, short_nb),
    });
}
