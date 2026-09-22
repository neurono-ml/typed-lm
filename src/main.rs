// Temporary: modules are wired incrementally per slice and the stub main
// references none of them yet. Remove this once the serve slice connects
// every module to the HTTP handlers.
// TODO(slice-6): remove the temporary dead_code allowance.
#![allow(dead_code)]

mod classifier;
mod cli;
mod context;
mod device;
mod evaluation_error;
mod evaluator;
mod jev;
mod model;
mod repository;

use clap::Parser;
use cli::{Cli, Command};

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    match args.command {
        Command::Serve(_serve) => anyhow::bail!("serve ainda não implementado (slice 6)"),
    }
}
