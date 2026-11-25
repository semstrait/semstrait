use crate::catalog::CatalogsConfig;
use crate::error::build::ModelBuildErrorKind;
use crate::model::SemanticModel;
use crate::parse::{lower_source_into, parse_catalogs_with_source};
use crate::source_fs::{LocalFs, SourceFs};
use semstrait_common::diagnostic::{Diagnostic, Diagnostics, SourceId};

#[derive(Clone)]
enum LoadInput {
    Inline { yaml: String, source: SourceId },
    File { path: String, source: SourceId },
    InvalidUtf8 { source: SourceId },
}

#[derive(Clone)]
pub struct SemanticModelLoader<F: SourceFs = LocalFs> {
    fs: F,
    model_inputs: Vec<LoadInput>,
    catalogs_inputs: Vec<LoadInput>,
    catalogs_inline: Option<CatalogsConfig>,
    validate_pass: bool,
}

impl Default for SemanticModelLoader<LocalFs> {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticModelLoader<LocalFs> {
    pub fn new() -> Self {
        Self {
            fs: LocalFs,
            model_inputs: Vec::new(),
            catalogs_inputs: Vec::new(),
            catalogs_inline: None,
            validate_pass: true,
        }
    }
}

impl SemanticModel {
    pub fn loader() -> SemanticModelLoader<LocalFs> {
        SemanticModelLoader::new()
    }
}

impl<F: SourceFs> SemanticModelLoader<F> {
    pub fn with_fs<F2: SourceFs>(self, fs: F2) -> SemanticModelLoader<F2> {
        SemanticModelLoader {
            fs,
            model_inputs: self.model_inputs,
            catalogs_inputs: self.catalogs_inputs,
            catalogs_inline: self.catalogs_inline,
            validate_pass: self.validate_pass,
        }
    }

    pub fn from_yaml_str(mut self, yaml: impl Into<String>, source: SourceId) -> Self {
        self.model_inputs.push(LoadInput::Inline {
            yaml: yaml.into(),
            source,
        });
        self
    }

    pub fn from_yaml_bytes(mut self, bytes: &[u8], source: SourceId) -> Self {
        self.model_inputs.push(match std::str::from_utf8(bytes) {
            Ok(s) => LoadInput::Inline {
                yaml: s.to_owned(),
                source,
            },
            Err(_) => LoadInput::InvalidUtf8 { source },
        });
        self
    }

    pub fn from_yaml_file(mut self, path: impl Into<String>) -> Self {
        let path = path.into();
        let source = SourceId::named(path.clone());
        self.model_inputs.push(LoadInput::File { path, source });
        self
    }

    pub fn with_catalogs(mut self, c: CatalogsConfig) -> Self {
        self.catalogs_inline = Some(c);
        self.catalogs_inputs.clear();
        self
    }

    pub fn from_catalogs_yaml_str(mut self, yaml: impl Into<String>, source: SourceId) -> Self {
        self.catalogs_inputs.push(LoadInput::Inline {
            yaml: yaml.into(),
            source,
        });
        self
    }

    pub fn from_catalogs_yaml_file(mut self, path: impl Into<String>) -> Self {
        let path = path.into();
        let source = SourceId::named(path.clone());
        self.catalogs_inputs.push(LoadInput::File { path, source });
        self
    }

    pub fn skip_validate(mut self) -> Self {
        self.validate_pass = false;
        self
    }

    pub fn build(
        self,
    ) -> Result<(SemanticModel, Diagnostics<ModelBuildErrorKind>), Diagnostics<ModelBuildErrorKind>>
    {
        if self.model_inputs.is_empty() {
            let d = Diagnostic::new(ModelBuildErrorKind::NoSource);
            return Err(vec![d]);
        }

        let model_payloads = self.read_inputs(&self.model_inputs)?;
        let catalogs_payloads = self.read_inputs(&self.catalogs_inputs)?;

        let mut accumulated: Diagnostics<ModelBuildErrorKind> = Vec::new();
        let mut builder = SemanticModel::builder();
        let mut had_parse_error = false;
        for (yaml, source) in model_payloads {
            let (next, diags) = lower_source_into(&yaml, source.as_str(), builder);
            builder = next;
            if !diags.is_empty() {
                had_parse_error = true;
            }
            accumulated.extend(
                diags
                    .into_iter()
                    .map(|d| d.map_kind(ModelBuildErrorKind::Parse)),
            );
        }
        if had_parse_error {
            return Err(accumulated);
        }

        let mut had_catalogs_error = false;
        for (yaml, source) in catalogs_payloads {
            match parse_catalogs_with_source(&yaml, source.as_str()) {
                Ok((_, diags)) => accumulated.extend(
                    diags
                        .into_iter()
                        .map(|d| d.map_kind(ModelBuildErrorKind::CatalogsParse)),
                ),
                Err(diags) => {
                    accumulated.extend(
                        diags
                            .into_iter()
                            .map(|d| d.map_kind(ModelBuildErrorKind::CatalogsParse)),
                    );
                    had_catalogs_error = true;
                }
            }
        }
        if had_catalogs_error {
            return Err(accumulated);
        }

        if !self.validate_pass {
            return match builder.build() {
                Ok((m, diags)) => {
                    accumulated.extend(diags);
                    Ok((m, accumulated))
                }
                Err(_) => Err(accumulated),
            };
        }

        match builder.build() {
            Ok((m, diags)) => {
                accumulated.extend(diags);
                Ok((m, accumulated))
            }
            Err(diags) => {
                accumulated.extend(diags);
                Err(accumulated)
            }
        }
    }

    fn read_inputs(
        &self,
        inputs: &[LoadInput],
    ) -> Result<Vec<(String, SourceId)>, Diagnostics<ModelBuildErrorKind>> {
        let mut out = Vec::new();
        let mut errors = Vec::new();
        for input in inputs {
            match input {
                LoadInput::Inline { yaml, source } => {
                    out.push((yaml.clone(), source.clone()));
                }
                LoadInput::File { path, source } => match self.fs.read(path) {
                    Ok(bytes) => match String::from_utf8(bytes) {
                        Ok(s) => out.push((s, source.clone())),
                        Err(_) => {
                            errors.push(Diagnostic::new(ModelBuildErrorKind::SourceIo {
                                path: path.clone(),
                                error: std::io::ErrorKind::InvalidData,
                            }));
                        }
                    },
                    Err(e) => {
                        errors.push(Diagnostic::new(ModelBuildErrorKind::SourceIo {
                            path: path.clone(),
                            error: e.kind(),
                        }));
                    }
                },
                LoadInput::InvalidUtf8 { source } => {
                    errors.push(Diagnostic::new(ModelBuildErrorKind::SourceIo {
                        path: source.to_string(),
                        error: std::io::ErrorKind::InvalidData,
                    }));
                }
            }
        }
        if errors.is_empty() {
            Ok(out)
        } else {
            Err(errors)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::marker::PhantomData;

    use super::*;
    use crate::catalog::{CatalogAuthMethod, CatalogEntry, SecretKeyMapping};

    struct Probe<T>(PhantomData<T>);

    trait NotDebug {
        const DEBUG: bool = false;
    }

    impl<T> NotDebug for Probe<T> {}

    impl<T: Debug> Probe<T> {
        const DEBUG: bool = true;
    }

    #[test]
    fn only_the_types_that_hold_no_credential_are_debug() {
        let debug: Vec<&str> = [
            ("SecretKeyMapping", Probe::<SecretKeyMapping>::DEBUG),
            ("CatalogsConfig", Probe::<CatalogsConfig>::DEBUG),
            ("CatalogEntry", Probe::<CatalogEntry>::DEBUG),
            ("CatalogAuthMethod", Probe::<CatalogAuthMethod>::DEBUG),
            ("SemanticModelLoader", Probe::<SemanticModelLoader>::DEBUG),
            ("LoadInput", Probe::<LoadInput>::DEBUG),
        ]
        .into_iter()
        .filter_map(|(name, debug)| debug.then_some(name))
        .collect();

        assert_eq!(debug, ["SecretKeyMapping"]);
    }
}
