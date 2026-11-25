use std::io::{stderr, stdin, stdout};
use std::process::ExitCode;

use clap::Parser;
use semstrait_api::cli::Cli;

#[tokio::main]
async fn main() -> ExitCode {
    let ran = Cli::parse()
        .run(stdin(), &mut stdout(), &mut stderr())
        .await;
    ran.unwrap_or_else(|error| {
        eprintln!("error: {error}");
        ExitCode::FAILURE
    })
}
