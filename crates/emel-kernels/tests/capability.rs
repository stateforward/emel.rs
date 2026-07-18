#![allow(missing_docs)]

use allocation_counter::measure;
use emel_kernels::capability::{Outcome, Resolver, Scope, event::Query};
use emel_tensor::dtype::SerializedType;
use sml as _;

const NATIVE: [SerializedType; 8] = [
    SerializedType::Q4_0,
    SerializedType::Q4_1,
    SerializedType::Q5_0,
    SerializedType::Q8_0,
    SerializedType::Q2K,
    SerializedType::Q3K,
    SerializedType::Q4K,
    SerializedType::Q6K,
];

const EXPLICIT_NO_CLAIM: [SerializedType; 25] = [
    SerializedType::F16,
    SerializedType::Q5_1,
    SerializedType::Q8_1,
    SerializedType::Q5K,
    SerializedType::Q8K,
    SerializedType::Iq2Xxs,
    SerializedType::Iq2Xs,
    SerializedType::Iq3Xxs,
    SerializedType::Iq1S,
    SerializedType::Iq4Nl,
    SerializedType::Iq3S,
    SerializedType::Iq2S,
    SerializedType::Iq4Xs,
    SerializedType::I8,
    SerializedType::I16,
    SerializedType::I32,
    SerializedType::I64,
    SerializedType::F64,
    SerializedType::Iq1M,
    SerializedType::Bf16,
    SerializedType::Tq1_0,
    SerializedType::Tq2_0,
    SerializedType::Mxfp4,
    SerializedType::Q4Kx8Bl4,
    SerializedType::Q4Kx8Bl8,
];

#[test]
fn public_query_and_error_surfaces_are_typed_and_stable() {
    let query = Query::new(Scope::MatrixWeightContract, SerializedType::Q6K);
    assert_eq!(query.scope(), Scope::MatrixWeightContract);
    assert_eq!(query.tensor_type(), SerializedType::Q6K);

    let resolver = Resolver::default();
    assert_eq!(format!("{resolver:?}"), "Resolver { .. }");
    assert_eq!(
        emel_kernels::capability::Error::UnexpectedEvent.to_string(),
        "unexpected capability resolver event"
    );
    assert_eq!(
        emel_kernels::capability::Error::Internal.to_string(),
        "internal capability resolver error"
    );
}

#[test]
fn every_scope_and_outcome_follows_the_contract_only_matrix() {
    assert_eq!(1 + NATIVE.len() + EXPLICIT_NO_CLAIM.len(), 34);
    let mut resolver = Resolver::new();
    assert_eq!(
        resolver.process_event(Query::new(
            Scope::VectorDequantContract,
            SerializedType::F32
        )),
        Ok(Outcome::ApprovedDenseF32ByContract)
    );
    assert_eq!(
        resolver.process_event(Query::new(Scope::MatrixWeightContract, SerializedType::F32)),
        Ok(Outcome::DisallowedFallback)
    );
    for tensor_type in NATIVE {
        assert_eq!(
            resolver.process_event(Query::new(Scope::VectorDequantContract, tensor_type)),
            Ok(Outcome::ApprovedDenseF32ByContract)
        );
        assert_eq!(
            resolver.process_event(Query::new(Scope::MatrixWeightContract, tensor_type)),
            Ok(Outcome::NativeQuantized)
        );
    }
    for tensor_type in EXPLICIT_NO_CLAIM {
        for scope in [Scope::VectorDequantContract, Scope::MatrixWeightContract] {
            assert_eq!(
                resolver.process_event(Query::new(scope, tensor_type)),
                Ok(Outcome::ExplicitNoClaim)
            );
        }
    }
}

#[test]
fn dispatch_allocates_nothing() {
    let mut resolver = Resolver::new();
    let allocation = measure(|| {
        for _ in 0..256 {
            assert_eq!(
                resolver.process_event(Query::new(
                    Scope::VectorDequantContract,
                    SerializedType::Q2K,
                )),
                Ok(Outcome::ApprovedDenseF32ByContract)
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}
