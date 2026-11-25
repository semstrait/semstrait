use std::fmt;

use semstrait_common::Dialect;
use serde::{Deserialize, Serialize};

pub mod datatype;
pub mod expr;
pub mod functions;
mod literal;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentDialect {
    Canonical,
    #[serde(untagged)]
    Dialect(Dialect),
}

impl fmt::Display for DocumentDialect {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.serialize(formatter)
    }
}

fn on_dialect<D>(
    canonical: &'static D,
    dialects: &'static [(Dialect, Option<D>)],
    dialect: Dialect,
) -> Option<&'static D> {
    dialects
        .iter()
        .find(|(entry, _)| *entry == dialect)
        .map_or(Some(canonical), |(_, declaration)| declaration.as_ref())
}
