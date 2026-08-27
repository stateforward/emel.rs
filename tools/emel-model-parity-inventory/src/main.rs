//! Command-line entry point for the model parity inventory.

#[cfg(windows)]
use winapi_util as _;
#[cfg(windows)]
use windows_sys as _;
#[cfg(unix)]
use {cap_std as _, cap_tempfile as _, uuid as _};
use {
    proc_macro2 as _, quote as _, serde as _, serde_json as _, sha2 as _, shell_words as _,
    syn as _, tempfile as _,
};

fn main() {
    if let Err(error) = emel_model_parity_inventory::cli(std::env::args_os().skip(1)) {
        eprintln!("emel-model-parity-inventory: {error}");
        std::process::exit(2);
    }
}
