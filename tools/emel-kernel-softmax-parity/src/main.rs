//! Public observer for the portable softmax kernel route.

use emel_kernels::Kernel;
use emel_kernels::any::reductions::{OpSoftMax, ReductionError};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const SML_COMMIT: &str = "49207123cd3f39767764bae774932cb48623f92f";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const SM_BLOB: &str = "cdce31c5f70501f9c26c66886c3348d774a8dc48";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| format!("{:08x}", v.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    let layout = Layout::contiguous(DType::F32, [3, 2, 1, 1]).expect("layout");
    let input = [1.0_f32, 2.0, 4.0, 2.0, 5.0, 8.0];
    println!(
        "kernel-softmax-live/v1\nsource_repository=stateforward/emel.cpp\nsource_commit={SOURCE_COMMIT}\nsource_sml_commit={SML_COMMIT}\nsource_kernel_events_blob={EVENTS_BLOB}\nsource_kernel_detail_blob={DETAIL_BLOB}\nsource_kernel_sm_blob={SM_BLOB}\nscope=portable_f32_op_soft_max_dense_and_contract_rejection\nexecution=public_emel_kernel_sm_process_event"
    );

    let mut kernel = Kernel::new();
    let mut output = [0.0_f32; 6];
    assert_eq!(
        kernel.process_event(OpSoftMax::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    println!("case=dense_rows status=ok output_bits={}", bits(&output));

    let implicit = Layout::new(DType::F32, [3, 2, 1, 1], [0, 0, 0, 0]);
    let mut implicit_output = [0.0_f32; 6];
    assert_eq!(
        kernel.process_event(OpSoftMax::new(
            TensorView::new(&input, implicit),
            TensorViewMut::new(&mut implicit_output, implicit)
        )),
        Ok(())
    );
    println!(
        "case=implicit_contiguous status=ok output_bits={}",
        bits(&implicit_output)
    );

    let typed = Layout::contiguous(DType::F16, [3, 2, 1, 1]).expect("layout");
    let mut typed_output = [SENTINEL; 6];
    assert_eq!(
        kernel.process_event(OpSoftMax::new(
            TensorView::new(&input, typed),
            TensorViewMut::new(&mut typed_output, typed)
        )),
        Err(ReductionError::InvalidView)
    );
    assert!(
        typed_output
            .iter()
            .all(|value| value.to_bits() == SENTINEL.to_bits())
    );

    let mismatch = Layout::contiguous(DType::F32, [2, 2, 1, 1]).expect("layout");
    let mut mismatch_output = [SENTINEL; 4];
    assert_eq!(
        kernel.process_event(OpSoftMax::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut mismatch_output, mismatch)
        )),
        Err(ReductionError::ShapeMismatch)
    );
    println!(
        "case=shape_mismatch status=reject error=ShapeMismatch output_bits={}",
        bits(&mismatch_output)
    );
}
