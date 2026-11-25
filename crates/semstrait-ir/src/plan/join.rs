use crate::error::IrErrorKind;
use crate::expr::PhysicalExpr;
use crate::plan::common::{condition, NodeCommon};
use crate::plan::node::PlanNode;
use crate::primitives::JoinType;
use crate::types::{Schema, SchemaColumn};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct JoinNode {
    pub common: NodeCommon,
    pub left: Box<PlanNode>,
    pub right: Box<PlanNode>,
    pub join_type: JoinType,
    pub on: PhysicalExpr,
    pub post_join_filter: Option<PhysicalExpr>,
}

impl JoinNode {
    pub fn new(
        left: impl Into<PlanNode>,
        right: impl Into<PlanNode>,
        join_type: JoinType,
        on: PhysicalExpr,
        post_join_filter: Option<PhysicalExpr>,
    ) -> Result<Self, IrErrorKind> {
        let (left, right) = (Box::new(left.into()), Box::new(right.into()));
        condition(&concatenated(&left, &right), &on)?;
        let natural = natural_output(&left, &right, join_type);
        if let Some(filter) = &post_join_filter {
            condition(&natural, filter)?;
        }
        Ok(Self {
            common: NodeCommon::new(natural.columns),
            left,
            right,
            join_type,
            on,
            post_join_filter,
        })
    }

    pub fn natural_output(&self) -> Schema {
        natural_output(&self.left, &self.right, self.join_type)
    }

    pub fn concatenated(&self) -> Schema {
        concatenated(&self.left, &self.right)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct CrossNode {
    pub common: NodeCommon,
    pub left: Box<PlanNode>,
    pub right: Box<PlanNode>,
}

impl CrossNode {
    pub fn new(left: impl Into<PlanNode>, right: impl Into<PlanNode>) -> Self {
        let (left, right) = (Box::new(left.into()), Box::new(right.into()));
        Self {
            common: NodeCommon::new(concatenated(&left, &right).columns),
            left,
            right,
        }
    }

    pub(super) fn natural_output(&self) -> Schema {
        concatenated(&self.left, &self.right)
    }
}

fn natural_output(left: &PlanNode, right: &PlanNode, join_type: JoinType) -> Schema {
    let (left_outer, right_outer) = match join_type {
        JoinType::Inner => (false, false),
        JoinType::Left => (false, true),
        JoinType::Right => (true, false),
        JoinType::Full => (true, true),
    };
    let columns = ([(left, left_outer), (right, right_outer)].into_iter())
        .flat_map(|(input, outer)| {
            (input.common().output_schema.columns.iter()).map(move |field| SchemaColumn {
                nullable: field.nullable || outer,
                ..field.clone()
            })
        })
        .collect();
    Schema { columns }
}

fn concatenated(left: &PlanNode, right: &PlanNode) -> Schema {
    natural_output(left, right, JoinType::Inner)
}
