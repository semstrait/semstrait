use semstrait_common::SemanticsName;
use semstrait_common::{Tree, ValidateError};
use semstrait_ir::PhysicalExpr;
use semstrait_sql::expr::{
    AggregateAllowed, Call, Expr, ExprOperationExt, FunctionRef, Literal, NamedReference,
};
use semstrait_sql::functions::{registry, AggregateFunction};

use crate::formula::{of_the_row, ComputedFields};
use crate::steps::{qualified, reserved};
use crate::tree::column_leaves;

pub(crate) struct PartKey {
    pub(crate) columns: ComputedFields,
    pub(crate) per_part: PhysicalExpr,
}

pub(crate) fn part_key(
    body: &PhysicalExpr,
    values: &dyn Fn(&NamedReference) -> bool,
    metric: &SemanticsName,
    qualifier: Option<&str>,
) -> Result<PartKey, ValidateError> {
    let mut key = Key {
        values,
        metric,
        qualifier,
        cases: 0,
        pieces: 0,
        columns: Vec::new(),
    };
    let per_part = key.split(body, None, false)?;
    Ok(PartKey {
        columns: key.columns,
        per_part,
    })
}

pub(crate) fn summed() -> Result<(&'static AggregateFunction, bool), ValidateError> {
    match registry().get("sum") {
        Some(FunctionRef::Aggregate(function, AggregateAllowed)) => Ok((function, false)),
        _ => Err(ValidateError::CallRefused {
            reason: "the function catalog holds no `sum`".into(),
        }),
    }
}

struct Key<'a> {
    values: &'a dyn Fn(&NamedReference) -> bool,
    metric: &'a SemanticsName,
    qualifier: Option<&'a str>,
    cases: usize,
    pieces: usize,
    columns: ComputedFields,
}

impl Key<'_> {
    fn split(
        &mut self,
        node: &PhysicalExpr,
        reach: Option<&PhysicalExpr>,
        in_condition: bool,
    ) -> Result<PhysicalExpr, ValidateError> {
        match node {
            Expr::Function(Call::Aggregate(..)) => Ok(node.clone()),
            node if self.of_the_row_alone(node)
                && !node
                    .pre_order()
                    .any(|node| matches!(node, Expr::Case { .. })) =>
            {
                if in_condition || column_leaves(node).next().is_none() {
                    return Ok(node.clone());
                }
                self.pieces += 1;
                let name = reserved(format_args!("piece{}", self.pieces), self.metric);
                self.column(name, reached(reach, node.clone()))
            }
            Expr::Case { whens, else_ }
                if whens.iter().all(|(when, _)| self.of_the_row_alone(when)) =>
            {
                self.cases += 1;
                let name = reserved(format_args!("key{}", self.cases), self.metric);
                let mut positions =
                    (1_i64..).map(|position| Expr::Literal(Literal::BigInt(position)));
                let whens: Vec<_> = whens.iter().zip(&mut positions).collect();
                let else_ = else_.as_deref().zip(positions.next());
                let selects = (whens.iter())
                    .map(|((when, _), position)| (when.clone(), position.clone()))
                    .collect();
                let selected = Expr::Case {
                    whens: selects,
                    else_: else_
                        .as_ref()
                        .map(|(_, position)| Box::new(position.clone())),
                };
                let selected = reached(reach, selected);
                let key = self.column(name, selected.clone())?;
                let mut earlier = Vec::new();
                let mut on_part = Vec::new();
                for ((when, then), position) in whens {
                    self.split(when, after(reach, &earlier).as_ref(), true)?;
                    earlier.push(when.clone());
                    let reach = selected.clone().eq(position.clone());
                    let then = self.split(then, Some(&reach), in_condition)?;
                    on_part.push((key.clone().eq(position), then));
                }
                let else_ = match else_ {
                    Some((result, position)) => {
                        let reach = selected.clone().eq(position);
                        Some(Box::new(self.split(result, Some(&reach), in_condition)?))
                    }
                    None => None,
                };
                Ok(Expr::Case {
                    whens: on_part,
                    else_,
                })
            }
            node => {
                let children = (node.children().into_iter())
                    .map(|child| self.split(child, reach, in_condition))
                    .collect::<Result<_, _>>()?;
                node.clone().with_new_children(children)
            }
        }
    }

    fn of_the_row_alone(&self, node: &PhysicalExpr) -> bool {
        of_the_row(node) && !column_leaves(node).any(|column| (self.values)(column))
    }

    fn column(&mut self, name: String, tree: PhysicalExpr) -> Result<PhysicalExpr, ValidateError> {
        let field = qualified(self.qualifier, name)?;
        self.columns.push((field.clone(), tree));
        Ok(Expr::Column(field))
    }
}

fn after(reach: Option<&PhysicalExpr>, earlier: &[PhysicalExpr]) -> Option<PhysicalExpr> {
    if earlier.is_empty() {
        return reach.cloned();
    }
    let reached = reach
        .cloned()
        .unwrap_or(Expr::Literal(Literal::Boolean(true)));
    Some(Expr::Case {
        whens: (earlier.iter())
            .map(|when| (when.clone(), Expr::Literal(Literal::Boolean(false))))
            .collect(),
        else_: Some(Box::new(reached)),
    })
}

fn reached(reach: Option<&PhysicalExpr>, value: PhysicalExpr) -> PhysicalExpr {
    match reach {
        Some(reach) => Expr::Case {
            whens: vec![(reach.clone(), value)],
            else_: None,
        },
        None => value,
    }
}
