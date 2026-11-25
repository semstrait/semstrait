use std::iter;

use semstrait_common::{Tree, ValidateError};

use super::{Aggregation, Expr, Leaf};

impl<L: Leaf, A: Aggregation> Tree for Expr<L, A> {
    fn children(&self) -> Vec<&Self> {
        match self {
            Self::Literal(_) | Self::Column(_) | Self::Leaf(_) => Vec::new(),
            Self::UnaryOp { operand, .. }
            | Self::Cast { input: operand, .. }
            | Self::IsNull(operand) => {
                vec![operand]
            }
            Self::BinaryOp { left, right, .. }
            | Self::Like {
                value: left,
                pattern: right,
                ..
            } => vec![left, right],
            Self::Between {
                value, low, high, ..
            } => vec![value, low, high],
            Self::InList { value, list, .. } => iter::once(&**value).chain(list).collect(),
            Self::Case { whens, else_ } => whens
                .iter()
                .flat_map(|(when, then)| [when, then])
                .chain(else_.as_deref())
                .collect(),
            Self::Function(call) => call.children(),
        }
    }

    fn with_new_children(mut self, children: Vec<Self>) -> Result<Self, ValidateError> {
        let slots = self.children_mut();
        if slots.len() != children.len() {
            return Err(ValidateError::ChildCountMismatch {
                expected: slots.len(),
                got: children.len(),
            });
        }
        for (slot, child) in slots.into_iter().zip(children) {
            *slot = child;
        }
        self.checked()
    }
}

impl<L: Leaf, A: Aggregation> Expr<L, A> {
    pub fn checked(self) -> Result<Self, ValidateError> {
        match &self {
            Self::InList { list, .. } if list.is_empty() => Err(ValidateError::EmptyInList),
            Self::Case { whens, .. } if whens.is_empty() => Err(ValidateError::EmptyCase),
            Self::Function(call) => match call.check() {
                Ok(()) => Ok(self),
                Err(error) => Err(ValidateError::CallRefused {
                    reason: error.to_string(),
                }),
            },
            _ => Ok(self),
        }
    }

    pub(super) fn children_mut(&mut self) -> Vec<&mut Self> {
        match self {
            Self::Literal(_) | Self::Column(_) | Self::Leaf(_) => Vec::new(),
            Self::UnaryOp { operand, .. }
            | Self::Cast { input: operand, .. }
            | Self::IsNull(operand) => {
                vec![operand]
            }
            Self::BinaryOp { left, right, .. }
            | Self::Like {
                value: left,
                pattern: right,
                ..
            } => vec![left, right],
            Self::Between {
                value, low, high, ..
            } => vec![value, low, high],
            Self::InList { value, list, .. } => iter::once(&mut **value).chain(list).collect(),
            Self::Case { whens, else_ } => whens
                .iter_mut()
                .flat_map(|(when, then)| [when, then])
                .chain(else_.as_deref_mut())
                .collect(),
            Self::Function(call) => call.children_mut(),
        }
    }
}
