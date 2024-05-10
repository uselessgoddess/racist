use std::env;

fn kernel(path: &str) {
    println!("cargo:rerun-if-changed={path}");
    spirv_builder::SpirvBuilder::new(
        format!("{}/{path}", env!("CARGO_MANIFEST_DIR")),
        "spirv-unknown-vulkan1.2",
    )
    .build()
    .expect("Kernel failed to compile");
}

fn main() {
    kernel("simple");
}
