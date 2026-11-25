#![allow(
    clippy::result_large_err,
    reason = "every stage returns its Diagnostic envelope unboxed"
)]

use std::collections::HashSet;

use semstrait_common::diagnostic::Diagnostic;
use semstrait_common::Tree;
use semstrait_sql::expr::{Call, Expr, WindowBound, WindowFrameKind};

use crate::error::IrErrorKind;
use crate::expr::{Parameter, PhysicalExpr};
use crate::plan::common::Anchor;
use crate::plan::node::PlanNode;
use crate::plan::nodes::{SortNode, TopNNode};
use crate::plan::scan::Reader;
use crate::plan::semantic_plan::SemanticPlan;
use crate::primitives::Name;

impl SemanticPlan {
    pub fn validate(&self) -> Result<(), Diagnostic<IrErrorKind>> {
        unique_anchors(&self.root, &mut HashSet::new())?;
        check(&self.root)?;
        self.root_names().map_err(|kind| self.root.diagnostic(kind))
    }

    fn root_names(&self) -> Result<(), IrErrorKind> {
        let output_names: Vec<String> = (self.output_names.iter())
            .map(|name| name.as_str().to_owned())
            .collect();
        let fields: Vec<String> = (self.root.common().output_schema.columns.iter())
            .map(|field| field.name.clone())
            .collect();
        if output_names != fields {
            return Err(IrErrorKind::OutputNamesMismatch {
                output_names,
                fields,
            });
        }
        let reserved =
            (self.output_names.iter()).find(|name| Name::reserved_prefix(name.as_str()).is_some());
        match reserved {
            Some(name) => Err(IrErrorKind::ReservedRootName {
                name: name.as_str().to_owned(),
            }),
            None => Ok(()),
        }
    }
}

fn unique_anchors(
    node: &PlanNode,
    seen: &mut HashSet<Anchor>,
) -> Result<(), Diagnostic<IrErrorKind>> {
    let anchor = node.common().anchor;
    if !seen.insert(anchor) {
        return Err(node.diagnostic(IrErrorKind::RepeatedAnchor { anchor }));
    }
    (node.children().into_iter()).try_for_each(|input| unique_anchors(input, seen))
}

fn check(node: &PlanNode) -> Result<(), Diagnostic<IrErrorKind>> {
    node.children().into_iter().try_for_each(check)?;
    built(node)
        .and_then(|()| adaptable_node(node))
        .map_err(|kind| node.diagnostic(kind))
}

fn built(node: &PlanNode) -> Result<(), IrErrorKind> {
    let rebuilt = node.clone().rebuilt(node.computed_names()?)?;
    if let PlanNode::Aggregate(aggregate) = node {
        let ((derived, _), (stored, _)) = (rebuilt.natural_output()?, node.natural_output()?);
        let mut entries = (derived.into_iter().zip(stored)).skip(aggregate.groupings.len());
        if let Some((expected, got)) = entries.find(|(derived, stored)| derived != stored) {
            return Err(IrErrorKind::AggregateFieldMismatch { expected, got });
        }
    }
    let derived = &rebuilt.common().output_schema.columns;
    let stored = &node.common().output_schema.columns;
    if let Some(index) =
        (0..derived.len().max(stored.len())).find(|&i| derived.get(i) != stored.get(i))
    {
        return Err(IrErrorKind::OutputSchemaMismatch { index });
    }
    node.unique_output_names()
}

fn adaptable_node(node: &PlanNode) -> Result<(), IrErrorKind> {
    match node {
        PlanNode::TopN(_) => Err(IrErrorKind::TopNNotLowered),
        PlanNode::Scan(_)
        | PlanNode::Filter(_)
        | PlanNode::Project(_)
        | PlanNode::Aggregate(_)
        | PlanNode::Sort(_)
        | PlanNode::Fetch(_)
        | PlanNode::Join(_)
        | PlanNode::Cross(_)
        | PlanNode::Multiset(_) => {
            (expressions(node).into_iter()).try_for_each(|(expr, refusal)| adaptable(expr, refusal))
        }
    }
}

type WindowRefusal = fn(&'static str) -> IrErrorKind;

fn expressions(node: &PlanNode) -> Vec<(&PhysicalExpr, Option<WindowRefusal>)> {
    let in_filter: WindowRefusal = |function| IrErrorKind::WindowInFilter { function };
    match node {
        PlanNode::Scan(scan) => {
            let rows: &[Vec<PhysicalExpr>] = match &scan.reader {
                Reader::VirtualTable(table) => &table.expressions,
                Reader::FilesScan(_) | Reader::TableScan(_) => &[],
            };
            let in_rows: WindowRefusal = |function| IrErrorKind::WindowInReader { function };
            [
                each(&scan.filter, Some(in_filter)),
                each(rows.iter().flatten(), Some(in_rows)),
            ]
            .concat()
        }
        PlanNode::Filter(node) => each([&node.filter], Some(in_filter)),
        PlanNode::Project(node) => each(&node.expressions, None),
        PlanNode::Aggregate(node) => {
            let entries = node.aggregates.iter().flat_map(|(_, call)| {
                (call.args().iter().chain(call.filter()))
                    .chain(call.order_by().iter().map(|key| &key.expr))
            });
            each(
                node.groupings.iter().chain(entries),
                Some(|function| IrErrorKind::WindowInAggregate { function }),
            )
        }
        PlanNode::Sort(SortNode { order_by, .. }) | PlanNode::TopN(TopNNode { order_by, .. }) => {
            each(
                order_by.iter().map(|key| &key.expr),
                Some(|function| IrErrorKind::WindowInSortKey { function }),
            )
        }
        PlanNode::Join(node) => each(
            std::iter::once(&node.on).chain(&node.post_join_filter),
            Some(|function| IrErrorKind::WindowInJoin { function }),
        ),
        PlanNode::Fetch(_) | PlanNode::Cross(_) | PlanNode::Multiset(_) => Vec::new(),
    }
}

fn each<'e>(
    exprs: impl IntoIterator<Item = &'e PhysicalExpr>,
    refusal: Option<WindowRefusal>,
) -> Vec<(&'e PhysicalExpr, Option<WindowRefusal>)> {
    exprs.into_iter().map(|expr| (expr, refusal)).collect()
}

fn adaptable(expr: &PhysicalExpr, refusal: Option<WindowRefusal>) -> Result<(), IrErrorKind> {
    let refused = match expr {
        Expr::Leaf(Parameter { key, .. }) => Some(IrErrorKind::UnboundParameter { key: *key }),
        Expr::Literal(literal) => {
            (literal.check_fit().err()).map(|error| IrErrorKind::UnfitLiteral { error })
        }
        Expr::Function(call @ Call::Aggregate(..)) => Some(IrErrorKind::MisplacedAggregate {
            function: call.name(),
        }),
        Expr::Function(call @ Call::Window(windowed)) => {
            let function = call.name();
            let past_i64 = |bound| {
                matches!(bound, WindowBound::Preceding(n) | WindowBound::Following(n)
                    if i64::try_from(n).is_err())
            };
            let unfit = windowed.over().window_frame.is_some_and(|window_frame| {
                !matches!(
                    window_frame.kind,
                    WindowFrameKind::Rows | WindowFrameKind::Range
                ) || past_i64(window_frame.start)
                    || past_i64(window_frame.end)
            });
            match (windowed.nulls(), refusal) {
                (Some(_), _) => Some(IrErrorKind::CallWithNullTreatment { function }),
                (None, Some(refusal)) => Some(refusal(function)),
                (None, None) => unfit.then_some(IrErrorKind::UnfitWindowFrame { function }),
            }
        }
        _ => None,
    };
    match refused {
        Some(kind) => Err(kind),
        None => (expr.children().into_iter()).try_for_each(|operand| adaptable(operand, refusal)),
    }
}
