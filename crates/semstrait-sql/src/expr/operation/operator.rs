use crate::expr::{Aggregation, BinaryOpKind, Expr, Leaf, UnaryOpKind};

macro_rules! operators {
    ($($binary:ident $binary_method:ident => $binary_op:ident),*; $($unary:ident $unary_method:ident => $unary_op:ident),*) => {
        $(impl<L: Leaf, A: Aggregation> std::ops::$binary for Expr<L, A> {
            type Output = Self;

            fn $binary_method(self, right: Self) -> Self {
                Self::BinaryOp { op: BinaryOpKind::$binary_op, left: Box::new(self), right: Box::new(right) }
            }
        })*
        $(impl<L: Leaf, A: Aggregation> std::ops::$unary for Expr<L, A> {
            type Output = Self;

            fn $unary_method(self) -> Self {
                Self::UnaryOp { op: UnaryOpKind::$unary_op, operand: Box::new(self) }
            }
        })*
    };
}

operators!(
    Add add => Add, Sub sub => Subtract, Mul mul => Multiply, Div div => Divide, Rem rem => Mod,
    BitAnd bitand => And, BitOr bitor => Or;
    Neg neg => Negate, Not not => Not
);

pub trait ExprOperationExt: Sized {
    fn eq(self, other: Self) -> Self;
    fn neq(self, other: Self) -> Self;
    fn lt(self, other: Self) -> Self;
    fn lt_eq(self, other: Self) -> Self;
    fn gt(self, other: Self) -> Self;
    fn gt_eq(self, other: Self) -> Self;
    #[allow(clippy::wrong_self_convention)]
    fn is_not_distinct_from(self, other: Self) -> Self;
    #[allow(clippy::wrong_self_convention)]
    fn is_null(self) -> Self;
}

macro_rules! comparisons {
    ($($method:ident => $op:ident),*) => {
        impl<L: Leaf, A: Aggregation> ExprOperationExt for Expr<L, A> {
            $(fn $method(self, other: Self) -> Self {
                Self::BinaryOp { op: BinaryOpKind::$op, left: Box::new(self), right: Box::new(other) }
            })*

            fn is_null(self) -> Self {
                Self::IsNull(Box::new(self))
            }
        }
    };
}

comparisons!(
    eq => Eq, neq => NotEq, lt => Lt, lt_eq => LtEq, gt => Gt, gt_eq => GtEq,
    is_not_distinct_from => IsNotDistinctFrom
);
