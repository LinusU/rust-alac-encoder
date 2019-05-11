use std::env;
use std::path::PathBuf;

use cc;
use bindgen;

fn main() {
    cc::Build::new()
        .file("vendor/codec/ag_enc.c")
        .file("vendor/codec/ALACBitUtilities.c")
        .file("vendor/codec/dp_enc.c")
        .file("vendor/codec/EndianPortable.c")
        .file("vendor/codec/matrix_enc.c")
        .include("vendor/codec")
        .warnings(false)
        .compile("alac");

    let bindings = bindgen::Builder::default()
        .clang_arg("-Ivendor/codec")
        .header("vendor/codec/aglib.h")
        .header("vendor/codec/ALACAudioTypes.h")
        .header("vendor/codec/ALACBitUtilities.h")
        .header("vendor/codec/dplib.h")
        .header("vendor/codec/matrixlib.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
