//! Public-actor observer for the maintained dense contiguous F32 kernel slice.

use emel_kernels::any::event;
use emel_kernels::{Error, Kernel};
use std::env;
use std::hint::black_box;
use std::time::Instant;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

const fn status(result: Result<(), Error>) -> &'static str {
    match result {
        Ok(()) => "ok",
        Err(Error::InvalidShape) => "invalid_shape",
        Err(Error::UnsupportedOperation(_) | Error::UnsupportedKernelKind(_)) => "unsupported",
        Err(Error::UnexpectedEvent | Error::Internal | Error::KernelUnavailable(_)) => {
            "internal_error"
        }
    }
}

fn main() {
    let arguments: Vec<String> = env::args().skip(1).collect();
    if arguments.is_empty() {
        write_parity();
    } else if arguments.len() == 4 && arguments[0] == "--benchmark" {
        let iterations = parse_positive(&arguments[1]);
        let runs = usize::try_from(parse_positive(&arguments[2])).unwrap_or_else(|_| {
            eprintln!("benchmark run count is too large");
            std::process::exit(2);
        });
        let warmup = parse_positive(&arguments[3]);
        write_benchmark(iterations, runs, warmup);
    } else {
        eprintln!("usage: emel-kernel-elementwise-parity [--benchmark ITERATIONS RUNS WARMUP]");
        std::process::exit(2);
    }
}

fn parse_positive(value: &str) -> u32 {
    let parsed = value.parse().unwrap_or_else(|_| {
        eprintln!("benchmark values must be positive integers");
        std::process::exit(2);
    });
    if parsed == 0 {
        eprintln!("benchmark values must be positive integers");
        std::process::exit(2);
    }
    parsed
}

fn write_parity() {
    println!("kernel-elementwise-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9");
    println!("source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0");
    println!("source_kernel_any_blob=a85d0942c81ef8f4d2dc293c5ceeaa63ac0e2c49");
    println!("scope=dense_contiguous_f32");

    let mut kernel = Kernel::new();
    let input = [1.5_f32, -2.0, 0.25];
    let other = [-0.5_f32, 4.0, 2.0];
    let mut output = [0.0_f32; 3];

    let result = kernel.process_event(event::OpDup::new(&input, &mut output));
    println!(
        "case=op_dup status={} output_bits={}",
        status(result),
        bits(&output)
    );

    let result = kernel.process_event(event::OpAdd::new(&input, &other, &mut output));
    println!(
        "case=op_add status={} output_bits={}",
        status(result),
        bits(&output)
    );

    let result = kernel.process_event(event::OpSub::new(&input, &other, &mut output));
    println!(
        "case=op_sub status={} output_bits={}",
        status(result),
        bits(&output)
    );

    let result = kernel.process_event(event::OpMul::new(&input, &other, &mut output));
    println!(
        "case=op_mul status={} output_bits={}",
        status(result),
        bits(&output)
    );

    let result = kernel.process_event(event::OpDiv::new(&input, &other, &mut output));
    println!(
        "case=op_div status={} output_bits={}",
        status(result),
        bits(&output)
    );

    let result = kernel.process_event(event::OpSqr::new(&input, &mut output));
    println!(
        "case=op_sqr status={} output_bits={}",
        status(result),
        bits(&output)
    );

    let sqrt_input = [0.0_f32, 4.0, 9.0];
    let result = kernel.process_event(event::OpSqrt::new(&sqrt_input, &mut output));
    println!(
        "case=op_sqrt status={} output_bits={}",
        status(result),
        bits(&output)
    );

    let mut short_output = [0.0_f32; 2];
    let result = kernel.process_event(event::OpDup::new(&input, &mut short_output));
    println!("case=invalid_shape status={}", status(result));
}

fn write_benchmark(iterations: u32, runs: usize, warmup: u32) {
    const COUNT: usize = 4096;
    let lhs = vec![1.25_f32; COUNT];
    let rhs = vec![0.75_f32; COUNT];
    let mut add_output = vec![0.0_f32; COUNT];
    let mut mul_output = vec![0.0_f32; COUNT];
    let mut add_kernel = Kernel::new();
    let mut mul_kernel = Kernel::new();
    let add = |kernel: &mut Kernel, output: &mut [f32], count: u32| {
        for _ in 0..count {
            let result = kernel.process_event(event::OpAdd::new(
                black_box(lhs.as_slice()),
                black_box(rhs.as_slice()),
                output,
            ));
            assert_eq!(black_box(result), Ok(()));
        }
        black_box(output[COUNT - 1]);
    };
    let mul = |kernel: &mut Kernel, output: &mut [f32], count: u32| {
        for _ in 0..count {
            let result = kernel.process_event(event::OpMul::new(
                black_box(lhs.as_slice()),
                black_box(rhs.as_slice()),
                output,
            ));
            assert_eq!(black_box(result), Ok(()));
        }
        black_box(output[COUNT - 1]);
    };
    add(&mut add_kernel, &mut add_output, warmup);
    mul(&mut mul_kernel, &mut mul_output, warmup);
    let add_samples = collect_samples(runs, iterations, || {
        let started = Instant::now();
        add(&mut add_kernel, &mut add_output, iterations);
        started.elapsed().as_secs_f64() * 1_000_000_000.0 / f64::from(iterations)
    });
    let mul_samples = collect_samples(runs, iterations, || {
        let started = Instant::now();
        mul(&mut mul_kernel, &mut mul_output, iterations);
        started.elapsed().as_secs_f64() * 1_000_000_000.0 / f64::from(iterations)
    });
    println!("kernel-elementwise-bench/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9");
    println!("source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0");
    println!("source_kernel_any_blob=a85d0942c81ef8f4d2dc293c5ceeaa63ac0e2c49");
    println!("benchmark_operand=dense_contiguous_f32,count=4096,lhs=1.25,rhs=0.75");
    println!(
        "case=op_add rust_ns_per_op={:.3} output=1.875 iter={iterations} runs={runs} warmup={warmup}",
        median(&add_samples)
    );
    println!(
        "case=op_mul rust_ns_per_op={:.3} output=0.9375 iter={iterations} runs={runs} warmup={warmup}",
        median(&mul_samples)
    );
}

fn collect_samples<F: FnMut() -> f64>(runs: usize, _iterations: u32, mut sample: F) -> Vec<f64> {
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        samples.push(sample());
    }
    samples
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len().is_multiple_of(2) {
        sorted[sorted.len() / 2 - 1].midpoint(sorted[sorted.len() / 2])
    } else {
        sorted[sorted.len() / 2]
    }
}
