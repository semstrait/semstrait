use semstrait_common::{Tree, ValidateError};

use crate::error::IrErrorKind;
use crate::expr::PhysicalExpr;
use crate::plan::join::{CrossNode, JoinNode};
use crate::plan::multiset::MultisetNode;
use crate::plan::node::PlanNode;
use crate::plan::nodes::{AggregateNode, FetchNode, FilterNode, ProjectNode, SortNode, TopNNode};
use crate::plan::scan::ScanNode;
use crate::primitives::Name;

impl Tree for PlanNode {
    fn children(&self) -> Vec<&Self> {
        PlanNode::children(self)
    }

    fn with_new_children(mut self, children: Vec<Self>) -> Result<Self, ValidateError> {
        let (expected, got) = (self.children().len(), children.len());
        if expected != got {
            return Err(ValidateError::ChildCountMismatch { expected, got });
        }
        let names = self.computed_names()?;
        for (input, child) in self.children_mut().into_iter().zip(children) {
            *input = child;
        }
        Ok(self.rebuilt(names)?)
    }
}

impl PlanNode {
    pub(super) fn computed_names(&self) -> Result<Vec<Name>, IrErrorKind> {
        let (natural, computed) = self.natural_output()?;
        (natural.into_iter().enumerate().skip(computed.start))
            .map(|(index, field)| {
                Name::new(field.name).map_err(|_| IrErrorKind::UnnamedField { index })
            })
            .collect()
    }

    pub(super) fn rebuilt(self, names: Vec<Name>) -> Result<Self, IrErrorKind> {
        let named = |expressions: Vec<PhysicalExpr>| names.into_iter().zip(expressions).collect();
        let common = self.common().clone();
        let node: PlanNode = match self {
            Self::Scan(scan) => {
                let read = ScanNode::new(scan.reader, scan.schema)?;
                match scan.filter {
                    Some(filter) => read.with_filter(filter)?,
                    None => read,
                }
                .into()
            }
            Self::Filter(node) => FilterNode::new(*node.input, node.filter)?.into(),
            Self::Project(node) => ProjectNode::new(*node.input, named(node.expressions))?.into(),
            Self::Aggregate(node) => {
                AggregateNode::new(*node.input, named(node.groupings), node.aggregates)?.into()
            }
            Self::Sort(node) => SortNode::new(*node.input, node.order_by)?.into(),
            Self::Fetch(node) => FetchNode::new(*node.input, node.offset, node.limit)?.into(),
            Self::TopN(node) => TopNNode::new(
                *node.input,
                node.order_by,
                node.offset,
                node.limit,
                node.fetch_mode,
            )?
            .into(),
            Self::Join(node) => JoinNode::new(
                *node.left,
                *node.right,
                node.join_type,
                node.on,
                node.post_join_filter,
            )?
            .into(),
            Self::Cross(node) => CrossNode::new(*node.left, *node.right).into(),
            Self::Multiset(node) => MultisetNode::new(node.inputs, node.op)?.into(),
        };
        let mut node = match common.emit_form {
            Some(emit_form) => node.with_emit_form(emit_form)?,
            None => node,
        };
        let rebuilt = node.common_mut();
        rebuilt.anchor = common.anchor;
        rebuilt.hint = common.hint;
        rebuilt.semantic_meta = common.semantic_meta;
        Ok(node)
    }
}

impl From<IrErrorKind> for ValidateError {
    fn from(error: IrErrorKind) -> Self {
        Self::NodeRefused {
            reason: error.to_string(),
        }
    }
}
