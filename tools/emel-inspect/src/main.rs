//! Model metadata inspection command-line tool.

#![forbid(unsafe_code)]

fn main() {
    println!("emel-inspect: {:?}", emel::core::ModelFamily::Unknown);
}
