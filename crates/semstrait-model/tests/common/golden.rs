use std::path::{Path, PathBuf};

use semstrait_common::diagnostic::{Diagnose, Diagnostic};

pub const GENERATOR: &str = "generate_goldens";

pub const SWITCH: &str = "SEMSTRAIT_GOLDENS";

#[derive(Debug)]
pub struct Golden {
    pub path: PathBuf,
    pub rendered: String,
}

impl Golden {
    pub fn message(case: &str, diagnostics: &[Diagnostic<impl Diagnose>]) -> Self {
        let rendered: Vec<String> = diagnostics.iter().map(ToString::to_string).collect();
        Self {
            path: Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                "tests/fixtures/{}/{case}.error.txt",
                env!("CARGO_CRATE_NAME")
            )),
            rendered: rendered.join("\n"),
        }
    }

    pub fn assert_committed(&self) {
        let path = self.path.display();
        let committed = std::fs::read_to_string(&self.path)
            .unwrap_or_else(|_| panic!("no golden at {path}: write it with `{}`", command()));
        assert_eq!(
            self.rendered,
            committed,
            "{path} is not what its gate renders: a change an ADR rules regenerates it with \
             `{}`, and its pull request names each class of change; any other change is a \
             defect",
            command()
        );
    }
}

fn command() -> String {
    format!(
        "{SWITCH}=1 cargo test -p {} --test {} {GENERATOR}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_CRATE_NAME")
    )
}

pub fn asked(mut args: impl Iterator<Item = String>, switch: bool) -> bool {
    switch && args.any(|arg| arg == GENERATOR)
}

pub fn generate(gates: impl FnOnce() -> Vec<Golden>) {
    let switch = std::env::var_os(SWITCH).is_some();
    generate_when(asked(std::env::args().skip(1), switch), gates);
}

pub fn generate_when(asked: bool, gates: impl FnOnce() -> Vec<Golden>) {
    if !asked {
        return;
    }
    for golden in gates() {
        if let Some(folder) = golden.path.parent() {
            std::fs::create_dir_all(folder).expect("create the golden's folder");
        }
        std::fs::write(&golden.path, &golden.rendered).expect("write the golden");
    }
}
