use std::{env, fs, path::Path};

fn present(path: &str) {
    fs::copy(env::var(format!("{path}.spv")).unwrap(), Path::new("src/k.gen").join(path)).unwrap();
}

fn main() {
    present("simple");
}
