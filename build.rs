use std::{
    env,
    process::{Command, Stdio},
};

fn main() {
    // always rerun build.rs
    // cargo will care about caching
    // println!("cargo:rerun-if-changed=NULL");

    //let mut cargo = Command::new("cargo");
    //cargo.args(["run"]).env("RUSTUP_TOOLCHAIN", "nightly-2023-05-27").current_dir("./kernels");
    //
    //println!("{:?}", env::vars());
    //
    //let output = cargo.stderr(Stdio::inherit()).output().unwrap();
    //if !output.status.success() {
    //    panic!("--- build error ---");
    //}
}
