use std::ops::Range;

use semstrait_common::diagnostic::{Diagnose, Diagnostic};

use crate::error::IrErrorKind;
use crate::plan::common::{unique_names, EmitForm, NodeCommon};
use crate::plan::join::{CrossNode, JoinNode};
use crate::plan::multiset::MultisetNode;
use crate::plan::nodes::{AggregateNode, FetchNode, FilterNode, ProjectNode, SortNode, TopNNode};
use crate::plan::scan::ScanNode;
use crate::plan::trace::SemAnnotation;
use crate::types::{Schema, SchemaColumn};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum PlanNode {
    Scan(ScanNode),
    Filter(FilterNode),
    Project(ProjectNode),
    Aggregate(AggregateNode),
    Sort(SortNode),
    Fetch(FetchNode),
    TopN(TopNNode),
    Join(JoinNode),
    Cross(CrossNode),
    Multiset(MultisetNode),
}

macro_rules! plan_nodes {
    ($($variant:ident($node:ident)),*) => {
        $(impl From<$node> for PlanNode {
            fn from(node: $node) -> Self {
                Self::$variant(node)
            }
        })*

        impl PlanNode {
            pub fn common(&self) -> &NodeCommon {
                match self {
                    $(Self::$variant(node) => &node.common,)*
                }
            }

            pub(super) fn common_mut(&mut self) -> &mut NodeCommon {
                match self {
                    $(Self::$variant(node) => &mut node.common,)*
                }
            }

            pub fn type_name(&self) -> &'static str {
                match self {
                    $(Self::$variant(_) => stringify!($node),)*
                }
            }
        }
    };
}

plan_nodes!(
    Scan(ScanNode),
    Filter(FilterNode),
    Project(ProjectNode),
    Aggregate(AggregateNode),
    Sort(SortNode),
    Fetch(FetchNode),
    TopN(TopNNode),
    Join(JoinNode),
    Cross(CrossNode),
    Multiset(MultisetNode)
);

impl PlanNode {
    pub fn children(&self) -> Vec<&Self> {
        match self {
            Self::Scan(_) => Vec::new(),
            Self::Filter(FilterNode { input, .. })
            | Self::Project(ProjectNode { input, .. })
            | Self::Aggregate(AggregateNode { input, .. })
            | Self::Sort(SortNode { input, .. })
            | Self::Fetch(FetchNode { input, .. })
            | Self::TopN(TopNNode { input, .. }) => vec![input.as_ref()],
            Self::Join(JoinNode { left, right, .. })
            | Self::Cross(CrossNode { left, right, .. }) => {
                vec![left.as_ref(), right.as_ref()]
            }
            Self::Multiset(MultisetNode { inputs, .. }) => inputs.iter().collect(),
        }
    }

    pub(super) fn children_mut(&mut self) -> Vec<&mut Self> {
        match self {
            Self::Scan(_) => Vec::new(),
            Self::Filter(FilterNode { input, .. })
            | Self::Project(ProjectNode { input, .. })
            | Self::Aggregate(AggregateNode { input, .. })
            | Self::Sort(SortNode { input, .. })
            | Self::Fetch(FetchNode { input, .. })
            | Self::TopN(TopNNode { input, .. }) => vec![input.as_mut()],
            Self::Join(JoinNode { left, right, .. })
            | Self::Cross(CrossNode { left, right, .. }) => {
                vec![left.as_mut(), right.as_mut()]
            }
            Self::Multiset(MultisetNode { inputs, .. }) => inputs.iter_mut().collect(),
        }
    }

    pub fn diagnostic<K: Diagnose>(&self, kind: K) -> Diagnostic<K> {
        let anchor = self.common().anchor.0;
        Diagnostic::new(kind).with_note(format!("node {anchor} ({})", self.type_name()))
    }

    pub fn with_emit_form(mut self, emit_form: EmitForm) -> Result<Self, IrErrorKind> {
        let (natural, computed) = self.natural_output()?;
        let output = emit_form.select(&natural)?;
        if let Some(index) = computed
            .clone()
            .find(|index| !emit_form.output_mapping.contains(index))
        {
            return Err(IrErrorKind::EmitFormLeavesOutComputedField { index });
        }
        let common = self.common_mut();
        common.output_schema = Schema { columns: output };
        common.emit_form = Some(emit_form);
        self.unique_output_names()?;
        Ok(self)
    }

    pub(super) fn unique_output_names(&self) -> Result<(), IrErrorKind> {
        match self {
            Self::Scan(_) => Ok(()),
            _ => unique_names(
                (self.common().output_schema.columns.iter()).map(|field| field.name.as_str()),
            ),
        }
    }

    pub fn with_trace(mut self, trace: impl IntoIterator<Item = SemAnnotation>) -> Self {
        let semantic_meta = &mut self.common_mut().semantic_meta;
        *semantic_meta = trace.into_iter().collect();
        semantic_meta.sort();
        self
    }

    pub(super) fn natural_output(&self) -> Result<(Vec<SchemaColumn>, Range<usize>), IrErrorKind> {
        let (passed, computed): (Vec<SchemaColumn>, usize) = match self {
            Self::Scan(scan) => (scan.schema.columns.clone(), 0),
            Self::Project(project) => (
                project.input.common().output_schema.columns.clone(),
                project.expressions.len(),
            ),
            Self::Aggregate(aggregate) => (
                Vec::new(),
                aggregate.groupings.len() + aggregate.aggregates.len(),
            ),
            Self::Filter(FilterNode { input, .. })
            | Self::Sort(SortNode { input, .. })
            | Self::Fetch(FetchNode { input, .. })
            | Self::TopN(TopNNode { input, .. }) => {
                (input.common().output_schema.columns.clone(), 0)
            }
            Self::Join(join) => (join.natural_output().columns, 0),
            Self::Cross(cross) => (cross.natural_output().columns, 0),
            Self::Multiset(multiset) => (multiset.natural_output()?.columns, 0),
        };
        let computed = passed.len()..passed.len() + computed;
        let common = self.common();
        let output_field = |index: usize| {
            let position = match &common.emit_form {
                None => index,
                Some(emit_form) => (emit_form.output_mapping.iter())
                    .position(|&i| i == index)
                    .ok_or(IrErrorKind::EmitFormLeavesOutComputedField { index })?,
            };
            (common.output_schema.columns.get(position).cloned())
                .ok_or(IrErrorKind::OutputSchemaMismatch { index: position })
        };
        let fields = (passed.into_iter().map(Ok))
            .chain(computed.clone().map(output_field))
            .collect::<Result<_, _>>()?;
        Ok((fields, computed))
    }
}

#[cfg(test)]
mod tests {
    use semstrait_sql::datatype::DataType;
    use semstrait_sql::expr::{Expr, Literal};

    use super::*;
    use crate::plan::multiset::MultisetOp;
    use crate::plan::nodes::FetchMode;
    use crate::plan::scan::VirtualTable;
    use crate::primitives::JoinType;

    fn relation(node: &PlanNode) -> &'static str {
        match node {
            PlanNode::Scan(_) => "ReadRel",
            PlanNode::Filter(_) => "FilterRel",
            PlanNode::Project(_) => "ProjectRel",
            PlanNode::Aggregate(_) => "AggregateRel",
            PlanNode::Sort(_) => "SortRel",
            PlanNode::Fetch(_) => "FetchRel",
            PlanNode::TopN(_) => "TopNRel",
            PlanNode::Join(_) => "JoinRel",
            PlanNode::Cross(_) => "CrossRel",
            PlanNode::Multiset(_) => "SetRel",
        }
    }

    #[test]
    fn the_plan_vocabulary_mirrors_ten_relations() -> Result<(), IrErrorKind> {
        let fields = Schema {
            columns: vec![SchemaColumn {
                name: "x".into(),
                data_type: DataType::BigInt,
                nullable: false,
            }],
        };
        let scan = || ScanNode::new(VirtualTable::new(Vec::new()), fields.clone());
        let holds = || Expr::Literal(Literal::Boolean(true));
        let nodes: Vec<PlanNode> = vec![
            scan()?.into(),
            FilterNode::new(scan()?, holds())?.into(),
            ProjectNode::new(scan()?, Vec::new())?.into(),
            AggregateNode::new(scan()?, Vec::new(), Vec::new())?.into(),
            SortNode::new(scan()?, Vec::new())?.into(),
            FetchNode::new(scan()?, 0, None)?.into(),
            TopNNode::new(scan()?, Vec::new(), 0, 1, FetchMode::RowsOnly)?.into(),
            JoinNode::new(scan()?, scan()?, JoinType::Inner, holds(), None)?.into(),
            CrossNode::new(scan()?, scan()?).into(),
            MultisetNode::new(vec![scan()?.into(), scan()?.into()], MultisetOp::UnionAll)?.into(),
        ];

        let relations: Vec<_> = nodes.iter().map(relation).collect();

        assert_eq!(
            relations,
            [
                "ReadRel",
                "FilterRel",
                "ProjectRel",
                "AggregateRel",
                "SortRel",
                "FetchRel",
                "TopNRel",
                "JoinRel",
                "CrossRel",
                "SetRel"
            ]
        );
        let inputs: Vec<_> = nodes.iter().map(|node| node.children().len()).collect();
        assert_eq!(inputs, [0, 1, 1, 1, 1, 1, 1, 2, 2, 2]);
        Ok(())
    }
}
