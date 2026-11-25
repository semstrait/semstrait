use std::collections::BTreeMap;

use semstrait_common::Dialect;
use semstrait_sql::datatype::DataType;

use crate::compile::catalog::native_to_canonical;
use crate::manifest::read_type::{read_type, MappedSemantics};
use crate::manifest::semantic_definition::{is_count, SemanticDefinition};

#[derive(Debug, Default)]
pub(crate) struct ColumnTypes {
    observed: BTreeMap<String, DataType>,
    read: BTreeMap<String, DataType>,
}

impl ColumnTypes {
    pub(crate) fn new<'a>(
        native: Option<&semstrait_catalog::Schema>,
        mapped: impl IntoIterator<Item = (&'a str, &'a SemanticDefinition)>,
        dialect: Option<Dialect>,
    ) -> Self {
        let observed: BTreeMap<_, _> = (native.into_iter().flat_map(|schema| &schema.columns))
            .filter_map(|column| {
                Some((
                    column.name.clone(),
                    native_to_canonical(&column.data_type, dialect)?,
                ))
            })
            .collect();
        let mut by_column: BTreeMap<&str, Vec<MappedSemantics>> = BTreeMap::new();
        for (column, definition) in mapped {
            if !observed.contains_key(column) {
                by_column.entry(column).or_default().push(MappedSemantics {
                    name: &definition.name,
                    data_type: definition.data_type.as_ref(),
                    is_count: (definition.role_definition.aggregation())
                        .is_some_and(|(function, _)| is_count(function)),
                });
            }
        }
        let read = by_column
            .into_iter()
            .filter_map(|(column, semantics)| {
                Some((column.to_owned(), read_type(semantics)?.clone()))
            })
            .collect();
        Self { observed, read }
    }

    pub(crate) fn is_observed(&self, column: &str) -> bool {
        self.observed.contains_key(column)
    }

    pub(crate) fn read_type(&self, column: &str) -> Option<&DataType> {
        self.read.get(column)
    }

    pub(crate) fn of(&self, column: &str) -> DataType {
        (self.observed.get(column))
            .or_else(|| self.read_type(column))
            .cloned()
            .unwrap_or(DataType::Unknown)
    }
}
