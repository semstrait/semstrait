use std::fmt::Debug;

use super::{AggregateFunction, Function, Scalar, Signature, Window};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionRef<A: Aggregation> {
    Scalar(&'static Function<Scalar>),
    Aggregate(&'static AggregateFunction, A),
    Window(&'static Function<Window>),
}

impl<A: Aggregation> FunctionRef<A> {
    pub fn canonical(self) -> (&'static str, &'static Signature) {
        match self {
            Self::Scalar(function) => (function.canonical.name, &function.canonical.signature),
            Self::Aggregate(function, _) => {
                let canonical = &function.declarations.canonical;
                (canonical.name, &canonical.signature)
            }
            Self::Window(function) => (function.canonical.name, &function.canonical.signature),
        }
    }
}

pub trait Aggregation: sealed::Sealed + Debug + Clone + PartialEq {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AggregateAllowed;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoAggregate {}

impl Aggregation for AggregateAllowed {}
impl Aggregation for NoAggregate {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::AggregateAllowed {}
    impl Sealed for super::NoAggregate {}
}
