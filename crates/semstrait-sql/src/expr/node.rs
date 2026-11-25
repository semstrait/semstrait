use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use super::{
    AggregateAllowed, AggregateInRowExpr, Aggregation, BinaryOpKind, Call, LikeKind, Literal,
    NamedReference, NoAggregate, UnaryOpKind, UntypedLiteral,
};
use crate::datatype::DataType;

pub trait Leaf: Sized + Clone + Debug + PartialEq {
    fn untyped(&self) -> Option<&UntypedLiteral> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoLeaf {}

impl Leaf for NoLeaf {}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Expr<L: Leaf = NoLeaf, A: Aggregation = AggregateAllowed> {
    Literal(Literal),
    Column(NamedReference),
    Leaf(L),
    BinaryOp {
        op: BinaryOpKind,
        left: Box<Self>,
        right: Box<Self>,
    },
    UnaryOp {
        op: UnaryOpKind,
        operand: Box<Self>,
    },
    Cast {
        input: Box<Self>,
        target: DataType,
        on_failure: CastFailure,
    },
    Case {
        whens: Vec<(Self, Self)>,
        else_: Option<Box<Self>>,
    },
    InList {
        value: Box<Self>,
        list: Vec<Self>,
        negated: bool,
    },
    Between {
        value: Box<Self>,
        low: Box<Self>,
        high: Box<Self>,
        negated: bool,
    },
    Like {
        value: Box<Self>,
        pattern: Box<Self>,
        kind: LikeKind,
    },
    IsNull(Box<Self>),
    Function(Call<L, A>),
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CastFailure {
    Error,
    Null,
}

pub type RowExpr<L> = Expr<L, NoAggregate>;

pub(super) fn narrow_all<L: Leaf>(
    exprs: Vec<Expr<L>>,
) -> Result<Vec<RowExpr<L>>, AggregateInRowExpr> {
    exprs.into_iter().map(RowExpr::try_from).collect()
}

impl<L: Leaf> TryFrom<Expr<L>> for RowExpr<L> {
    type Error = AggregateInRowExpr;

    fn try_from(expr: Expr<L>) -> Result<Self, Self::Error> {
        let one = |expr: Box<Expr<L>>| Self::try_from(*expr).map(Box::new);
        Ok(match expr {
            Expr::Literal(literal) => Self::Literal(literal),
            Expr::Column(column) => Self::Column(column),
            Expr::Leaf(leaf) => Self::Leaf(leaf),
            Expr::BinaryOp { op, left, right } => Self::BinaryOp {
                op,
                left: one(left)?,
                right: one(right)?,
            },
            Expr::UnaryOp { op, operand } => Self::UnaryOp {
                op,
                operand: one(operand)?,
            },
            Expr::Cast {
                input,
                target,
                on_failure,
            } => Self::Cast {
                input: one(input)?,
                target,
                on_failure,
            },
            Expr::Case { whens, else_ } => Self::Case {
                whens: whens
                    .into_iter()
                    .map(|(when, then)| Ok((when.try_into()?, then.try_into()?)))
                    .collect::<Result<_, _>>()?,
                else_: else_.map(one).transpose()?,
            },
            Expr::InList {
                value,
                list,
                negated,
            } => Self::InList {
                value: one(value)?,
                list: narrow_all(list)?,
                negated,
            },
            Expr::Between {
                value,
                low,
                high,
                negated,
            } => Self::Between {
                value: one(value)?,
                low: one(low)?,
                high: one(high)?,
                negated,
            },
            Expr::Like {
                value,
                pattern,
                kind,
            } => Self::Like {
                value: one(value)?,
                pattern: one(pattern)?,
                kind,
            },
            Expr::IsNull(operand) => Self::IsNull(one(operand)?),
            Expr::Function(call) => Self::Function(call.try_into()?),
        })
    }
}

impl<L: Leaf, A: Aggregation> Expr<L, A> {
    pub fn literal(&self) -> Option<&Literal> {
        match self {
            Self::Literal(literal) => Some(literal),
            Self::Leaf(leaf) => leaf.untyped().map(|UntypedLiteral(literal)| literal),
            _ => None,
        }
    }
}
