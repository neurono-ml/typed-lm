//! Dataset loading from `.jsonl` and `.json` files.
//!
//! `.jsonl` files hold one record per line; `.json` files hold either a single
//! record object or an array of records. Failures name the file (and the line
//! for JSONL) so a malformed dataset is fixable without guesswork.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::dataset::record::{parse_record, TrainingRecord};

/// Whether a path is a JSONL file.
fn is_jsonl(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|extension| extension.eq_ignore_ascii_case("jsonl"))
        .unwrap_or(false)
}

/// Loads every record from the given files, in file order.
pub fn load_records(files: &[PathBuf]) -> anyhow::Result<Vec<TrainingRecord>> {
    let mut records: Vec<TrainingRecord> = Vec::new();
    for file in files {
        records.extend(load_file(file)?);
    }
    if records.is_empty() {
        return Err(anyhow::anyhow!(
            "no records found in {} dataset file(s)",
            files.len()
        ));
    }
    Ok(records)
}

/// Loads every record from one file.
pub fn load_file(path: &Path) -> anyhow::Result<Vec<TrainingRecord>> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| anyhow::anyhow!("failed to read '{}': {error}", path.display()))?;
    if is_jsonl(path) {
        parse_jsonl(path, &contents)
    } else {
        parse_json(path, &contents)
    }
}

/// Parses a JSONL document, one record per non-empty line.
fn parse_jsonl(path: &Path, contents: &str) -> anyhow::Result<Vec<TrainingRecord>> {
    let mut records: Vec<TrainingRecord> = Vec::new();
    for (line_index, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).map_err(|error| {
            anyhow::anyhow!(
                "{}:{}: invalid JSON: {error}",
                path.display(),
                line_index + 1
            )
        })?;
        let record = parse_record(&value)
            .map_err(|error| anyhow::anyhow!("{}:{}: {error}", path.display(), line_index + 1))?;
        records.push(record);
    }
    Ok(records)
}

/// Parses a JSON document holding one record object or an array of records.
fn parse_json(path: &Path, contents: &str) -> anyhow::Result<Vec<TrainingRecord>> {
    let value: Value = serde_json::from_str(contents)
        .map_err(|error| anyhow::anyhow!("{}: invalid JSON: {error}", path.display()))?;
    match value {
        Value::Array(entries) => {
            let mut records: Vec<TrainingRecord> = Vec::with_capacity(entries.len());
            for (entry_index, entry) in entries.iter().enumerate() {
                let record = parse_record(entry).map_err(|error| {
                    anyhow::anyhow!("{}:[{entry_index}]: {error}", path.display())
                })?;
                records.push(record);
            }
            Ok(records)
        }
        Value::Object(_) => {
            let record = parse_record(&value)
                .map_err(|error| anyhow::anyhow!("{}: {error}", path.display()))?;
            Ok(vec![record])
        }
        other => Err(anyhow::anyhow!(
            "{}: expected a JSON object or array of records, found {}",
            path.display(),
            json_kind(&other)
        )),
    }
}

/// Short description of a JSON value kind, for error messages.
fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECORD: &str = r#"{"state": "charged twice", "questions": {"refund": {"type": "noul", "instructions": "Refund?", "answer": "yes"}}}"#;

    #[test]
    fn loads_two_records_from_a_jsonl_file() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("dataset.jsonl");
        std::fs::write(&path, format!("{RECORD}\n{RECORD}\n"))?;
        let records = load_file(&path)?;
        assert_eq!(records.len(), 2);
        Ok(())
    }

    #[test]
    fn a_bad_jsonl_line_reports_its_number() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("dataset.jsonl");
        std::fs::write(&path, format!("{RECORD}\nnot json\n"))?;
        let result = load_file(&path);
        assert!(result.is_err());
        let Err(error) = result else {
            return Ok(());
        };
        assert!(error.to_string().contains(":2:"));
        Ok(())
    }

    #[test]
    fn loads_an_array_of_records_from_json() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("dataset.json");
        std::fs::write(&path, format!("[{RECORD}, {RECORD}]"))?;
        let records = load_file(&path)?;
        assert_eq!(records.len(), 2);
        Ok(())
    }

    #[test]
    fn loads_a_single_record_object_from_json() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("dataset.json");
        std::fs::write(&path, RECORD)?;
        let records = load_file(&path)?;
        assert_eq!(records.len(), 1);
        Ok(())
    }

    #[test]
    fn a_scalar_json_document_is_an_error() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("dataset.json");
        std::fs::write(&path, "42")?;
        assert!(load_file(&path).is_err());
        Ok(())
    }

    #[test]
    fn loading_no_records_is_an_error() {
        let result = load_records(&[]);
        assert!(result.is_err());
    }
}
