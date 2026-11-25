use crate::error::IrErrorKind;
use crate::plan::common::NodeCommon;
use crate::plan::node::PlanNode;
use crate::types::{same_name, Schema};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultisetOp {
    UnionAll,
    UnionDistinct,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct MultisetNode {
    pub common: NodeCommon,
    pub inputs: Vec<PlanNode>,
    pub op: MultisetOp,
}

impl MultisetNode {
    pub fn new(inputs: Vec<PlanNode>, op: MultisetOp) -> Result<Self, IrErrorKind> {
        let natural = union(&inputs)?;
        Ok(Self {
            common: NodeCommon::new(natural.columns),
            inputs,
            op,
        })
    }

    pub(super) fn natural_output(&self) -> Result<Schema, IrErrorKind> {
        union(&self.inputs)
    }
}

fn union(inputs: &[PlanNode]) -> Result<Schema, IrErrorKind> {
    let (first, others) = match inputs {
        [first, others @ ..] if !others.is_empty() => (first, others),
        _ => return Err(IrErrorKind::MultisetArity { got: inputs.len() }),
    };
    let mut columns = first.common().output_schema.columns.clone();
    for (branch, input) in (1..).zip(others) {
        let fields = &input.common().output_schema.columns;
        let agrees = |index: usize| {
            matches!(
                (columns.get(index), fields.get(index)),
                (Some(expected), Some(got))
                    if same_name(&expected.name, &got.name) && expected.data_type == got.data_type
            )
        };
        if let Some(index) = (0..columns.len().max(fields.len())).find(|&index| !agrees(index)) {
            return Err(IrErrorKind::BranchFieldMismatch { branch, index });
        }
        for (column, field) in columns.iter_mut().zip(fields) {
            column.nullable |= field.nullable;
        }
    }
    Ok(Schema { columns })
}
