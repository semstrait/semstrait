#![cfg(feature = "api-cli")]

#[path = "../../semstrait-api/tests/common/mod.rs"]
mod common;

use std::io::empty;
use std::process::ExitCode;

use clap::Parser;
use common::{written, SALES};
use semstrait_api::cli::Cli;
use tempfile::tempdir;

#[tokio::test]
async fn the_api_cli_feature_forwards_to_the_cli_of_semstrait_api() {
    let dir = tempdir().expect("a temporary directory");
    let model = written(&dir, "sales.yaml", SALES);
    let args = [
        "semstrait",
        "validate",
        "--model",
        &model,
        "--select",
        "region",
    ];
    let (mut out, mut err) = (Vec::new(), Vec::new());

    let code = (Cli::try_parse_from(args).expect("the arguments parse"))
        .run(empty(), &mut out, &mut err)
        .await
        .expect("stdout and stderr take every write");

    assert_eq!((code, out), (ExitCode::SUCCESS, b"valid\n".to_vec()));
}
