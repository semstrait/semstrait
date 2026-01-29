//! Error types for semstrait

use std::fmt;

/// Errors that can occur during parsing
#[derive(Debug)]
pub enum ParseError {
    /// IO error reading file
    Io {
        path: String,
        source: std::io::Error,
    },
    /// YAML deserialization error
    Yaml {
        source: serde_yaml::Error,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::Io { path, source } => {
                write!(f, "Failed to read '{}': {}", path, source)
            }
            ParseError::Yaml { source } => {
                write!(f, "Invalid YAML: {}", source)
            }
        }
    }
}

impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ParseError::Io { source, .. } => Some(source),
            ParseError::Yaml { source } => Some(source),
        }
    }
}

impl From<std::io::Error> for ParseError {
    fn from(err: std::io::Error) -> Self {
        ParseError::Io {
            path: String::new(),
            source: err,
        }
    }
}

impl From<serde_yaml::Error> for ParseError {
    fn from(err: serde_yaml::Error) -> Self {
        ParseError::Yaml { source: err }
    }
}
