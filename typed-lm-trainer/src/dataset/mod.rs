//! Dataset discovery, record parsing, loading and collation.
//!
//! The pipeline is: [`discovery::discover_dataset_files`] finds the files,
//! [`loader::load_records`] parses them, [`record::expand_records`] turns each
//! answered question into a decision item, and [`collate::build_batches`]
//! tokenizes and batches those items for the training loop.

pub mod collate;
pub mod discovery;
pub mod loader;
pub mod record;
