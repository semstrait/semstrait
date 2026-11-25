use std::fmt;

use semstrait_common::DataKindName;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FramePath(pub(crate) Vec<DataKindName>);

impl fmt::Display for FramePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.0.iter().map(DataKindName::as_str).collect();
        f.write_str(&names.join(" / "))
    }
}

pub(crate) fn at(path: &FramePath, body: impl fmt::Display) -> String {
    format!("at `{path}`: {body}")
}
