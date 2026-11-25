use semstrait_common::diagnostic::{Diagnose, Severity};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CatalogsParseErrorKind {
    YamlSyntax { message: String },
    UnsetEnvVar { var: String },
    CatalogMissingField { alias: String, field: String },
    UnknownField { field: String, parent: String },
}

impl Diagnose for CatalogsParseErrorKind {
    fn message(&self) -> String {
        use CatalogsParseErrorKind::*;
        match self {
            YamlSyntax { message } => format!("catalogs YAML syntax error: {}", message),
            UnsetEnvVar { var } => {
                format!("environment variable `{}` is not set", var)
            }
            CatalogMissingField { alias, field } => {
                format!("catalog `{}` is missing required field `{}`", alias, field)
            }
            UnknownField { field, parent } => {
                format!("unknown field `{}` in `{}`", field, parent)
            }
        }
    }

    fn severity_default(&self) -> Severity {
        Severity::Error
    }
}
