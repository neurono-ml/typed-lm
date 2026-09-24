//! Dataset file discovery.
//!
//! A dataset is either a single `.jsonl`/`.json` file or a directory scanned
//! recursively for those extensions. Discovery is deterministic (files are
//! sorted) so a run over the same tree always yields the same record order.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Extensions recognized as dataset files.
const DATASET_EXTENSIONS: [&str; 2] = ["jsonl", "json"];

/// Whether a path carries a recognized dataset extension.
fn has_dataset_extension(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|extension| DATASET_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Discovers the dataset files reachable from `path`.
///
/// - A single `.jsonl`/`.json` file resolves to itself.
/// - A directory is scanned recursively; matches are sorted for determinism.
/// - A missing path, an unsupported file extension or an empty directory are
///   reported as errors with the offending path.
pub fn discover_dataset_files(path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if !path.exists() {
        return Err(anyhow::anyhow!(
            "dataset path does not exist: {}",
            path.display()
        ));
    }
    if path.is_file() {
        if !has_dataset_extension(path) {
            return Err(anyhow::anyhow!(
                "unsupported dataset file '{}': expected a .jsonl or .json file",
                path.display()
            ));
        }
        return Ok(vec![path.to_path_buf()]);
    }

    let mut files: Vec<PathBuf> = Vec::new();
    let mut pending_directories: Vec<PathBuf> = vec![path.to_path_buf()];
    while let Some(directory) = pending_directories.pop() {
        let entries = std::fs::read_dir(&directory).map_err(|error| {
            anyhow::anyhow!(
                "failed to read directory '{}': {error}",
                directory.display()
            )
        })?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                anyhow::anyhow!(
                    "failed to read an entry of '{}': {error}",
                    directory.display()
                )
            })?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                pending_directories.push(entry_path);
            } else if has_dataset_extension(&entry_path) {
                files.push(entry_path);
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(anyhow::anyhow!(
            "no .jsonl or .json dataset files found under '{}'",
            path.display()
        ));
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_jsonl_file_resolves_to_itself() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let file = directory.path().join("dataset.jsonl");
        std::fs::write(&file, b"")?;
        let discovered = discover_dataset_files(&file)?;
        assert_eq!(discovered, vec![file]);
        Ok(())
    }

    #[test]
    fn a_directory_scans_recursively_and_sorts() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let nested = directory.path().join("nested");
        std::fs::create_dir(&nested)?;
        std::fs::write(directory.path().join("b.jsonl"), b"")?;
        std::fs::write(nested.join("a.json"), b"{}")?;
        std::fs::write(directory.path().join("ignored.txt"), b"")?;
        let discovered = discover_dataset_files(directory.path())?;
        assert_eq!(discovered.len(), 2);
        // Sorted lexicographically by full path, so "b.jsonl" precedes the
        // nested "nested/a.json".
        assert!(discovered[0].ends_with("b.jsonl"));
        assert!(discovered[1].ends_with("nested/a.json"));
        Ok(())
    }

    #[test]
    fn a_missing_path_is_an_error() {
        let result = discover_dataset_files(Path::new("does-not-exist-dataset-98765"));
        assert!(result.is_err());
    }

    #[test]
    fn an_empty_directory_is_an_error() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let result = discover_dataset_files(directory.path());
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn an_unsupported_file_extension_is_an_error() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let file = directory.path().join("dataset.txt");
        std::fs::write(&file, b"")?;
        let result = discover_dataset_files(&file);
        assert!(result.is_err());
        Ok(())
    }
}
