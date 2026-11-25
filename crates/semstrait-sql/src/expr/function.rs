use std::collections::BTreeMap;

use super::call::{bind, bind_named, with_defaults};
use super::{
    AggregateAllowed, AggregateInRowExpr, Aggregation, CallError, Expr, FunctionRef, Leaf,
    NoAggregate,
};
use crate::functions::ParameterDecl;

impl<A: Aggregation> FunctionRef<A> {
    pub fn parameters<L: Leaf>(
        self,
        args: &[Expr<L, A>],
    ) -> Result<Vec<&'static ParameterDecl>, CallError> {
        let (name, signature) = self.canonical();
        let binding = bind(name, signature, args)?;
        Ok(binding
            .into_iter()
            .map(|(_, parameter)| parameter)
            .collect())
    }

    pub fn parameter(self, name: &str) -> Result<&'static ParameterDecl, CallError> {
        let (function, signature) = self.canonical();
        (signature.parameter(name))
            .map(|(_, parameter)| parameter)
            .ok_or_else(|| CallError::UnknownParameter {
                function,
                name: name.to_owned(),
            })
    }

    pub fn arguments_by_position<L: Leaf>(self, args: Vec<Expr<L, A>>) -> Vec<Expr<L, A>> {
        let (function, signature) = self.canonical();
        with_defaults(function, signature, args)
    }

    pub fn arguments_by_name<L: Leaf>(
        self,
        named: BTreeMap<&str, Expr<L, A>>,
    ) -> Result<Vec<Expr<L, A>>, CallError> {
        let (function, signature) = self.canonical();
        bind_named(function, signature, named)
    }
}

impl TryFrom<FunctionRef<AggregateAllowed>> for FunctionRef<NoAggregate> {
    type Error = AggregateInRowExpr;

    fn try_from(function: FunctionRef<AggregateAllowed>) -> Result<Self, Self::Error> {
        match function {
            FunctionRef::Scalar(function) => Ok(Self::Scalar(function)),
            FunctionRef::Aggregate(..) => Err(AggregateInRowExpr),
            FunctionRef::Window(function) => Ok(Self::Window(function)),
        }
    }
}
