use std::collections::HashMap;
use std::io::{self, ErrorKind};

pub trait SourceFs {
    fn read(&self, path: &str) -> Result<Vec<u8>, io::Error>;
}

#[derive(Debug, Clone, Default)]
pub struct LocalFs;

impl LocalFs {
    pub fn new() -> Self {
        Self
    }
}

impl SourceFs for LocalFs {
    fn read(&self, path: &str) -> Result<Vec<u8>, io::Error> {
        std::fs::read(path)
    }
}

#[derive(Debug, Clone, Default)]
pub struct InMemoryFs {
    files: HashMap<String, Vec<u8>>,
}

impl InMemoryFs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, path: impl Into<String>, contents: impl Into<Vec<u8>>) {
        self.files.insert(path.into(), contents.into());
    }

    pub fn with_file(mut self, path: impl Into<String>, contents: impl Into<Vec<u8>>) -> Self {
        self.insert(path, contents);
        self
    }
}

impl SourceFs for InMemoryFs {
    fn read(&self, path: &str) -> Result<Vec<u8>, io::Error> {
        self.files.get(path).cloned().ok_or_else(|| {
            io::Error::new(
                ErrorKind::NotFound,
                format!("InMemoryFs: no such path `{}`", path),
            )
        })
    }
}
