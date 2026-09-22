use std::path::{Path, PathBuf};

/// Fonte da memória/contexto avaliado junto com cada request.
///
/// O system prompt fixo foi removido: o contexto agora é carregável e
/// substituível. Implementações futuras (ex.: RAG sobre vector store do
/// Rig) trocam o provider sem alterar handlers, evaluator ou API.
pub trait ContextProvider: Send + Sync {
    /// Nome da fonte (para logs e /health).
    fn name(&self) -> String;
    /// Carrega o conteúdo integral da memória.
    fn load(&self) -> anyhow::Result<String>;
}

/// Lê a memória de um arquivo (markdown, texto, etc.).
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
    fn loads_memory_file_contents() {
        let mut tmp = tempfile_like();
        writeln!(tmp.0, "# Memória\n- Fato: o céu é azul.").unwrap();
        let provider = FileContextProvider::new(&tmp.1);
        let content = provider.load().unwrap();
        assert!(content.contains("o céu é azul"));
    }

    #[test]
    fn name_identifies_the_source() {
        let provider = FileContextProvider::new(Path::new("resources/memory.md"));
        assert_eq!(provider.name(), "file:resources/memory.md");
    }

    #[test]
    fn missing_file_is_an_error_not_a_panic() {
        let provider = FileContextProvider::new(Path::new("nao-existe-12345.md"));
        assert!(provider.load().is_err());
    }

    /// Helper mínimo sem nova dependência: arquivo temporário único.
    fn tempfile_like() -> (std::fs::File, PathBuf) {
        let path = std::env::temp_dir().join(format!("manaca-ctx-{}.md", std::process::id()));
        let file = std::fs::File::create(&path).unwrap();
        (file, path)
    }
}
