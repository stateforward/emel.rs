//! Public-actor capability parity observer and benchmark lane.

use std::env;
use std::hint::black_box;
use std::time::Instant;

use emel_kernels::capability::{Outcome, Resolver, Scope, event::Query};
use emel_model::generation_audit::TensorTypeLabel;
use emel_tensor::dtype::SerializedType;

const TYPES: [SerializedType; 34] = [
    SerializedType::F32,
    SerializedType::F16,
    SerializedType::Q4_0,
    SerializedType::Q4_1,
    SerializedType::Q5_0,
    SerializedType::Q5_1,
    SerializedType::Q8_0,
    SerializedType::Q8_1,
    SerializedType::Q2K,
    SerializedType::Q3K,
    SerializedType::Q4K,
    SerializedType::Q5K,
    SerializedType::Q6K,
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    match arguments.as_slice() {
        [] => write_parity()?,
        [mode] if mode == "--parity" => write_parity()?,
        [mode, iterations, runs, warmup] if mode == "--benchmark" => {
            write_benchmark(iterations.parse()?, runs.parse()?, warmup.parse()?)?;
        }
        _ => return Err(
            "usage: emel-kernel-capability-parity [--parity|--benchmark ITERATIONS RUNS WARMUP]"
                .into(),
        ),
    }
    Ok(())
}

fn write_parity() -> Result<(), emel_kernels::capability::Error> {
    println!("kernel-capability-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit=843a117386ef17dc5a50549bbfc821074c2141d6");
    println!("source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9");
    println!("source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0");
    println!("source_generation_header_blob=d521cf68e1bf52a2a193bbdb460741772199b318");
    println!("source_generation_blob=099058ccd441d1dc6bebbb0c4994070d2f533c47");
    for (name, label) in [
        ("f32", TensorTypeLabel::F32),
        ("q2_k", TensorTypeLabel::Q2_K),
        ("q3_k", TensorTypeLabel::Q3_K),
        ("q4_k", TensorTypeLabel::Q4_K),
        ("q6_k", TensorTypeLabel::Q6_K),
        ("q4_0", TensorTypeLabel::Q4_0),
        ("unknown", TensorTypeLabel::UNKNOWN),
    ] {
        println!("label={name} value={}", label.as_str());
    }
    let mut resolver = Resolver::new();
    for (scope_name, scope) in [
        ("vector", Scope::VectorDequantContract),
        ("matrix", Scope::MatrixWeightContract),
    ] {
        for tensor_type in TYPES {
            let outcome = resolver.process_event(Query::new(scope, tensor_type))?;
            println!(
                "scope={scope_name} code={} outcome={}",
                tensor_type.wire_code(),
                outcome_name(outcome)
            );
        }
    }
    Ok(())
}

fn write_benchmark(
    iterations: u32,
    runs: usize,
    warmup_iterations: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    if iterations == 0 || runs == 0 {
        return Err("benchmark iterations and runs must be positive".into());
    }
    let mut resolver = Resolver::new();
    run_queries(&mut resolver, warmup_iterations)?;
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        run_queries(&mut resolver, iterations)?;
        samples.push(started.elapsed().as_secs_f64() * 1_000_000_000.0 / f64::from(iterations));
    }
    samples.sort_by(f64::total_cmp);
    let median = if samples.len().is_multiple_of(2) {
        samples[samples.len() / 2 - 1].midpoint(samples[samples.len() / 2])
    } else {
        samples[samples.len() / 2]
    };
    println!(
        "case=matrix_q4_k rust_ns_per_op={median:.3} outcome=native_quantized iter={iterations} runs={runs}"
    );
    Ok(())
}

fn run_queries(
    resolver: &mut Resolver,
    iterations: u32,
) -> Result<(), emel_kernels::capability::Error> {
    for _ in 0..iterations {
        let outcome = resolver.process_event(Query::new(
            black_box(Scope::MatrixWeightContract),
            black_box(SerializedType::Q4K),
        ))?;
        assert_eq!(black_box(outcome), Outcome::NativeQuantized);
    }
    Ok(())
}

const fn outcome_name(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::NativeQuantized => "native_quantized",
        Outcome::ApprovedDenseF32ByContract => "approved_dense_f32_by_contract",
        Outcome::DisallowedFallback => "disallowed_fallback",
        Outcome::ExplicitNoClaim => "explicit_no_claim",
    }
}
