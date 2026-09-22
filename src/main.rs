mod api;
mod classifier;
mod cli;
mod context;
mod device;
mod evaluation_error;
mod evaluator;
mod jev;
mod model;
mod repository;
mod startup;

use clap::Parser;

use cli::Cli;

#[actix_web::main]
async fn main() -> anyhow::Result<()> {
    let command_line = Cli::parse();
    startup::run(command_line).await
}
