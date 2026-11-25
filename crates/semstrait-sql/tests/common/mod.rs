#![allow(dead_code)]

use semstrait_sql::expr::{
    AggregateAllowed, Expr, FunctionRef, Leaf, NullOrdering, SortDir, SortKey,
};
use semstrait_sql::functions::{registry, AggregateFunction, Function, Scalar, Window};

pub fn function(name: &str) -> FunctionRef<AggregateAllowed> {
    registry()
        .get(name)
        .unwrap_or_else(|| panic!("the catalog declares `{name}`"))
}

pub fn scalar(name: &str) -> &'static Function<Scalar> {
    match function(name) {
        FunctionRef::Scalar(function) => function,
        other => panic!("{name} is no scalar function of the catalog: {other:?}"),
    }
}

pub fn aggregate_function(name: &str) -> &'static AggregateFunction {
    match function(name) {
        FunctionRef::Aggregate(function, AggregateAllowed) => function,
        other => panic!("{name} is no aggregate function of the catalog: {other:?}"),
    }
}

pub fn window_function(name: &str) -> &'static Function<Window> {
    match function(name) {
        FunctionRef::Window(function) => function,
        other => panic!("{name} is no window function of the catalog: {other:?}"),
    }
}

pub fn ascending<L: Leaf>(expr: Expr<L>) -> SortKey<L, AggregateAllowed> {
    SortKey {
        expr,
        dir: SortDir::Asc {
            nulls: NullOrdering::Unspecified,
        },
    }
}
