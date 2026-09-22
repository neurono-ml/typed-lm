mod health;
mod models;
mod routes;
mod state;
mod systemone;
#[cfg(test)]
mod tests;

// Temporary: the serve slice has not wired the HTTP handlers yet, so the
// public reexports are only reached by integration tests. Remove this once
// the serve slice connects every module to the HTTP handlers.
// TODO(slice-6): remove the temporary unused_imports allowance.
#[allow(unused_imports)]
pub use routes::configure;
#[allow(unused_imports)]
pub use state::SharedState;
