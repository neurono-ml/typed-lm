use std::path::{Path, PathBuf};

/// Source of the memory/context evaluated alongside each request.
///
/// The fixed system prompt was removed: the context is now loadable and
/// replaceable. Future implementations (e.g. RAG over the Rig vector store)
/// swap the provider without changing handlers, evaluator, or API.
pub trait ContextProvider: Send + Sync {
    /// Source name (for logs and /health).
    fn name(&self) -> String;
    /// Loads the full memory contents.
    fn load(&self) -> anyhow::Result<String>;
}

/// Reads the memory from a file (markdown, plain text, etc.).
pub struct FileContextProvider {
    path: PathBuf,
}

impl FileContextProvider {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }
}

impl ContextProvider for FileContextProvider {
    fn name(&self) -> String {
        format!("file:{}", self.path.display())
    }

    fn load(&self) -> anyhow::Result<String> {
        Ok(std::fs::read_to_string(&self.path)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn loads_memory_file_contents() -> anyhow::Result<()> {
        let mut tmp = tempfile_like()?;
        writeln!(tmp.0, "# Memory\n- Fact: the sky is blue.")?;
        let provider = FileContextProvider::new(&tmp.1);
        let content = provider.load()?;
        assert!(content.contains("the sky is blue"));
        Ok(())
    }

    #[test]
    fn name_identifies_the_source() {
        let provider = FileContextProvider::new(Path::new("resources/memory.md"));
        assert_eq!(provider.name(), "file:resources/memory.md");
    }

    #[test]
    fn missing_file_is_an_error_not_a_panic() {
        let provider = FileContextProvider::new(Path::new("does-not-exist-12345.md"));
        assert!(provider.load().is_err());
    }

    /// Minimal helper without a new dependency: unique temporary file.
    fn tempfile_like() -> anyhow::Result<(std::fs::File, PathBuf)> {
        let path = std::env::temp_dir().join(format!("typed-lm-ctx-{}.md", std::process::id()));
        let file = std::fs::File::create(&path)?;
        Ok((file, path))
    }
}
