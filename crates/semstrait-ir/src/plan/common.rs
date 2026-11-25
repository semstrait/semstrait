use semstrait_sql::expr::is_condition;

use crate::error::IrErrorKind;
use crate::expr::PhysicalExpr;
use crate::plan::trace::SemAnnotation;
use crate::types::{same_name, Schema, SchemaColumn};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Anchor(pub u128);

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct NodeCommon {
    pub anchor: Anchor,
    pub emit_form: Option<EmitForm>,
    pub hint: Option<Hint>,
    pub output_schema: Schema,
    pub semantic_meta: Vec<SemAnnotation>,
}

impl NodeCommon {
    pub(super) fn new(natural: Vec<SchemaColumn>) -> Self {
        Self {
            anchor: Anchor::default(),
            emit_form: None,
            hint: None,
            output_schema: Schema { columns: natural },
            semantic_meta: Vec::new(),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmitForm {
    pub output_mapping: Vec<usize>,
}

impl EmitForm {
    pub fn new(output_mapping: impl IntoIterator<Item = usize>) -> Self {
        Self {
            output_mapping: output_mapping.into_iter().collect(),
        }
    }

    pub(super) fn select(
        &self,
        natural: &[SchemaColumn],
    ) -> Result<Vec<SchemaColumn>, IrErrorKind> {
        (self.output_mapping.iter())
            .map(|&index| {
                natural
                    .get(index)
                    .cloned()
                    .ok_or(IrErrorKind::EmitFormIndexOutOfRange {
                        index,
                        fields: natural.len(),
                    })
            })
            .collect()
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Hint {
    pub stats: Option<Stats>,
    pub constraint: Option<RuntimeConstraint>,
    pub alias: Option<String>,
    pub output_names: Vec<String>,
    pub saved_computations: Vec<Computation>,
    pub loaded_computations: Vec<Computation>,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub row_count: f64,
    pub record_size: f64,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeConstraint {}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Computation {
    pub id: i32,
    pub computation_type: ComputationType,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputationType {
    Hashtable,
    BloomFilter,
    Unknown,
}

pub(super) fn unique_names<'n>(
    names: impl IntoIterator<Item = &'n str>,
) -> Result<(), IrErrorKind> {
    let mut seen: Vec<&str> = Vec::new();
    for name in names {
        if seen.iter().any(|earlier| same_name(earlier, name)) {
            return Err(IrErrorKind::RepeatedName {
                name: name.to_owned(),
            });
        }
        seen.push(name);
    }
    Ok(())
}

pub(super) fn condition(fields: &Schema, filter: &PhysicalExpr) -> Result<(), IrErrorKind> {
    let typed = fields.type_of(filter)?;
    if typed.data_type.is_valid() && is_condition(&typed.data_type, typed.family.as_ref()) {
        Ok(())
    } else {
        Err(IrErrorKind::NotACondition {
            got: typed.data_type,
        })
    }
}
