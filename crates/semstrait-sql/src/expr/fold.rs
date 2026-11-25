use std::borrow::Cow;
use std::cmp::Ordering;
use std::ops::Not;

use super::{Aggregation, BinaryOpKind, Expr, Leaf, Literal, OperationKind, UnaryOpKind};
use crate::datatype::DataType;
use crate::functions::Nullability;

impl<L: Leaf, A: Aggregation> Expr<L, A> {
    pub fn fold(self) -> Self {
        self.fold_with(&|_| None)
    }

    pub fn fold_with<'p>(mut self, literal_of: &dyn Fn(&L) -> Option<Cow<'p, Literal>>) -> Self {
        self.fold_in_place(literal_of);
        self
    }

    fn fold_in_place<'p>(&mut self, literal_of: &dyn Fn(&L) -> Option<Cow<'p, Literal>>) {
        for operand in self.children_mut() {
            operand.fold_in_place(literal_of);
        }
        if let Some(value) = self.value(literal_of) {
            *self = value;
        }
    }

    fn value<'p>(&mut self, literal_of: &dyn Fn(&L) -> Option<Cow<'p, Literal>>) -> Option<Self> {
        let value = match self {
            Self::Case { whens, else_ } => return Self::chosen(whens, else_, literal_of),
            Self::BinaryOp { op, left, right } => {
                let (left, right) = (left.stands_for(literal_of), right.stands_for(literal_of));
                let (left, right) = (left.as_deref(), right.as_deref());
                match op {
                    BinaryOpKind::And => Truth::all([Truth::of(left), Truth::of(right)]).literal(),
                    BinaryOpKind::Or => Truth::any([Truth::of(left), Truth::of(right)]).literal(),
                    BinaryOpKind::Add
                    | BinaryOpKind::Subtract
                    | BinaryOpKind::Multiply
                    | BinaryOpKind::Divide
                    | BinaryOpKind::SafeDivide
                    | BinaryOpKind::Mod => arithmetic(*op, left, right),
                    BinaryOpKind::Eq
                    | BinaryOpKind::NotEq
                    | BinaryOpKind::Lt
                    | BinaryOpKind::LtEq
                    | BinaryOpKind::Gt
                    | BinaryOpKind::GtEq => compared(*op, left, right).literal(),
                    BinaryOpKind::IsNotDistinctFrom => None,
                }
            }
            Self::UnaryOp { op, operand } => {
                let operand = operand.stands_for(literal_of);
                match op {
                    UnaryOpKind::Negate => negation(operand.as_deref()),
                    UnaryOpKind::Not => (!Truth::of(operand.as_deref())).literal(),
                }
            }
            Self::Between {
                value,
                low,
                high,
                negated,
            } => {
                let (value, low, high) = (
                    value.stands_for(literal_of),
                    low.stands_for(literal_of),
                    high.stands_for(literal_of),
                );
                Truth::all([
                    compared(BinaryOpKind::GtEq, value.as_deref(), low.as_deref()),
                    compared(BinaryOpKind::LtEq, value.as_deref(), high.as_deref()),
                ])
                .negated_if(*negated)
                .literal()
            }
            Self::InList {
                value,
                list,
                negated,
            } => {
                let value = value.stands_for(literal_of);
                Truth::any(list.iter().map(|item| {
                    let item = item.stands_for(literal_of);
                    compared(BinaryOpKind::Eq, value.as_deref(), item.as_deref())
                }))
                .negated_if(*negated)
                .literal()
            }
            Self::Like {
                value,
                pattern,
                kind,
            } => {
                let (value, pattern) =
                    (value.stands_for(literal_of), pattern.stands_for(literal_of));
                gives_null(
                    OperationKind::Like(*kind),
                    &[value.as_deref(), pattern.as_deref()],
                )
                .then_some(Literal::Null(DataType::Boolean))
            }
            Self::IsNull(operand) => operand
                .stands_for(literal_of)
                .map(|literal| Literal::Boolean(matches!(*literal, Literal::Null(_)))),
            Self::Cast { input, target, .. } => input
                .stands_for(literal_of)
                .and_then(|literal| literal.cast(target)),
            Self::Literal(_) | Self::Column(_) | Self::Leaf(_) | Self::Function(_) => None,
        };
        value.map(Self::Literal)
    }

    fn chosen<'p>(
        whens: &mut Vec<(Self, Self)>,
        else_: &mut Option<Box<Self>>,
        literal_of: &dyn Fn(&L) -> Option<Cow<'p, Literal>>,
    ) -> Option<Self> {
        let first = whens
            .iter()
            .map(|(when, _)| Truth::of(when.stands_for(literal_of).as_deref()))
            .enumerate()
            .find(|(_, truth)| !matches!(truth, Truth::False | Truth::Null));
        match first {
            Some((index, Truth::True)) => Some(whens.swap_remove(index).1),
            Some(_) => None,
            None => else_.take().map(|else_| *else_).or_else(|| {
                let then = whens
                    .iter()
                    .find_map(|(_, then)| then.stands_for(literal_of))?;
                Some(Self::Literal(Literal::Null(then.data_type())))
            }),
        }
    }

    fn stands_for<'s, 'p: 's>(
        &'s self,
        literal_of: &dyn Fn(&L) -> Option<Cow<'p, Literal>>,
    ) -> Option<Cow<'s, Literal>> {
        match self {
            Self::Literal(literal) => Some(Cow::Borrowed(literal)),
            Self::Leaf(leaf) => literal_of(leaf),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Truth {
    True,
    False,
    Null,
    Undecided,
}

impl Truth {
    fn of(literal: Option<&Literal>) -> Self {
        match literal {
            Some(Literal::Boolean(value)) => Self::from(*value),
            Some(Literal::Null(_)) => Self::Null,
            _ => Self::Undecided,
        }
    }

    fn all(truths: impl IntoIterator<Item = Self>) -> Self {
        truths
            .into_iter()
            .fold(Self::True, |all, truth| match (all, truth) {
                (Self::False, _) | (_, Self::False) => Self::False,
                (Self::Undecided, _) | (_, Self::Undecided) => Self::Undecided,
                (Self::Null, _) | (_, Self::Null) => Self::Null,
                (Self::True, Self::True) => Self::True,
            })
    }

    fn negated_if(self, negated: bool) -> Self {
        if negated {
            !self
        } else {
            self
        }
    }

    fn any(truths: impl IntoIterator<Item = Self>) -> Self {
        !Self::all(truths.into_iter().map(Not::not))
    }

    fn literal(self) -> Option<Literal> {
        match self {
            Self::True => Some(Literal::Boolean(true)),
            Self::False => Some(Literal::Boolean(false)),
            Self::Null => Some(Literal::Null(DataType::Boolean)),
            Self::Undecided => None,
        }
    }
}

impl From<bool> for Truth {
    fn from(value: bool) -> Self {
        if value {
            Self::True
        } else {
            Self::False
        }
    }
}

impl Not for Truth {
    type Output = Self;

    fn not(self) -> Self {
        match self {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Null | Self::Undecided => self,
        }
    }
}

fn gives_null(kind: OperationKind, operands: &[Option<&Literal>]) -> bool {
    kind.operation().canonical.nullability == Nullability::Mirror
        && operands
            .iter()
            .any(|operand| matches!(operand, Some(Literal::Null(_))))
}

fn compared(op: BinaryOpKind, left: Option<&Literal>, right: Option<&Literal>) -> Truth {
    if gives_null(OperationKind::Binary(op), &[left, right]) {
        return Truth::Null;
    }
    let ordering = left.zip(right).and_then(|(left, right)| order(left, right));
    let Some(ordering) = ordering else {
        return Truth::Undecided;
    };
    Truth::from(match op {
        BinaryOpKind::Eq => ordering.is_eq(),
        BinaryOpKind::NotEq => ordering.is_ne(),
        BinaryOpKind::Lt => ordering.is_lt(),
        BinaryOpKind::LtEq => ordering.is_le(),
        BinaryOpKind::Gt => ordering.is_gt(),
        BinaryOpKind::GtEq => ordering.is_ge(),
        BinaryOpKind::Add
        | BinaryOpKind::Subtract
        | BinaryOpKind::Multiply
        | BinaryOpKind::Divide
        | BinaryOpKind::SafeDivide
        | BinaryOpKind::Mod
        | BinaryOpKind::IsNotDistinctFrom
        | BinaryOpKind::And
        | BinaryOpKind::Or => return Truth::Undecided,
    })
}

fn arithmetic(
    op: BinaryOpKind,
    left: Option<&Literal>,
    right: Option<&Literal>,
) -> Option<Literal> {
    let data_type = integer_type(left?)?.wider(integer_type(right?)?)?;
    if gives_null(OperationKind::Binary(op), &[left, right]) {
        return Some(Literal::Null(data_type));
    }
    let (left, right) = (integer(left?)?, integer(right?)?);
    let value = match op {
        BinaryOpKind::Add => left.checked_add(right),
        BinaryOpKind::Subtract => left.checked_sub(right),
        BinaryOpKind::Multiply => left.checked_mul(right),
        BinaryOpKind::Divide | BinaryOpKind::SafeDivide => left.checked_div(right),
        BinaryOpKind::Mod => left.checked_rem(right),
        BinaryOpKind::Eq
        | BinaryOpKind::NotEq
        | BinaryOpKind::Lt
        | BinaryOpKind::LtEq
        | BinaryOpKind::Gt
        | BinaryOpKind::GtEq
        | BinaryOpKind::IsNotDistinctFrom
        | BinaryOpKind::And
        | BinaryOpKind::Or => None,
    }?;
    Literal::BigInt(value).read_as(&data_type).ok()
}

fn integer_type(literal: &Literal) -> Option<DataType> {
    let data_type = literal.data_type();
    data_type
        .declarations()?
        .canonical
        .digits
        .map(|_| data_type)
}

fn negation(operand: Option<&Literal>) -> Option<Literal> {
    let data_type = integer_type(operand?)?;
    if gives_null(OperationKind::Unary(UnaryOpKind::Negate), &[operand]) {
        return Some(Literal::Null(data_type));
    }
    let value = integer(operand?)?.checked_neg()?;
    Literal::BigInt(value).read_as(&data_type).ok()
}

fn order(left: &Literal, right: &Literal) -> Option<Ordering> {
    match (left, right) {
        (Literal::Boolean(left), Literal::Boolean(right)) => Some(left.cmp(right)),
        (Literal::Date(left), Literal::Date(right)) => Some(left.cmp(right)),
        (
            Literal::String(left) | Literal::VarChar { value: left, .. },
            Literal::String(right) | Literal::VarChar { value: right, .. },
        ) => Some(left.cmp(right)),
        _ => Some(integer(left)?.cmp(&integer(right)?)),
    }
}

fn integer(literal: &Literal) -> Option<i64> {
    match *literal {
        Literal::TinyInt(value) => Some(value.into()),
        Literal::SmallInt(value) => Some(value.into()),
        Literal::Integer(value) => Some(value.into()),
        Literal::BigInt(value) => Some(value),
        _ => None,
    }
}
