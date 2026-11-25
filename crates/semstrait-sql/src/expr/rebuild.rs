use std::vec;

use semstrait_common::Tree;

use super::{
    AggregateAllowed, AggregateCall, Aggregation, Call, CallCheck, Expr, Leaf, Over, RebuildError,
    ScalarCall, SortKey, WindowCall,
};

pub fn map_nodes<L: Leaf, A: Aggregation, To: Leaf, E>(
    expr: &Expr<L, A>,
    node: &mut impl FnMut(&Expr<L, A>) -> Option<Result<Expr<To>, E>>,
    refused: &impl Fn(RebuildError) -> E,
) -> Result<Expr<To>, E> {
    if let Some(mapped) = node(expr) {
        return mapped;
    }
    let children = expr
        .children()
        .into_iter()
        .map(|child| map_nodes(child, node, refused))
        .collect::<Result<_, _>>()?;
    expr.rebuilt(children, CallCheck::Full).map_err(refused)
}

impl<L: Leaf, A: Aggregation> Expr<L, A> {
    pub fn rebuilt<To: Leaf>(
        &self,
        children: Vec<Expr<To>>,
        check: CallCheck,
    ) -> Result<Expr<To>, RebuildError> {
        let mut feed = Feed(children.into_iter());
        let built = match self {
            Self::Literal(literal) => Expr::Literal(literal.clone()),
            Self::Column(column) => Expr::Column(column.clone()),
            Self::Leaf(_) => return Err(RebuildError::Leaf),
            Self::BinaryOp { op, .. } => Expr::BinaryOp {
                op: *op,
                left: feed.boxed()?,
                right: feed.boxed()?,
            },
            Self::UnaryOp { op, .. } => Expr::UnaryOp {
                op: *op,
                operand: feed.boxed()?,
            },
            Self::Cast {
                target, on_failure, ..
            } => Expr::Cast {
                input: feed.boxed()?,
                target: target.clone(),
                on_failure: *on_failure,
            },
            Self::Case { whens, else_ } => Expr::Case {
                whens: (0..whens.len())
                    .map(|_| Ok((feed.one()?, feed.one()?)))
                    .collect::<Result<_, _>>()?,
                else_: else_.as_ref().map(|_| feed.boxed()).transpose()?,
            },
            Self::InList { list, negated, .. } => Expr::InList {
                value: feed.boxed()?,
                list: feed.many(list.len())?,
                negated: *negated,
            },
            Self::Between { negated, .. } => Expr::Between {
                value: feed.boxed()?,
                low: feed.boxed()?,
                high: feed.boxed()?,
                negated: *negated,
            },
            Self::Like { kind, .. } => Expr::Like {
                value: feed.boxed()?,
                pattern: feed.boxed()?,
                kind: *kind,
            },
            Self::IsNull(_) => Expr::IsNull(feed.boxed()?),
            Self::Function(call) => Expr::Function(call.rebuilt(&mut feed, check)?),
        };
        match feed.0.next() {
            Some(_) => Err(RebuildError::ChildCount),
            None => Ok(built),
        }
    }
}

impl<L: Leaf, A: Aggregation> Call<L, A> {
    fn rebuilt<To: Leaf>(
        &self,
        feed: &mut Feed<Expr<To>>,
        check: CallCheck,
    ) -> Result<Call<To, AggregateAllowed>, RebuildError> {
        let options = self.options().to_vec();
        let args = feed.many(self.args().len())?;
        match self {
            Self::Scalar(call) => ScalarCall::builder(call.function(), args)
                .options(options)
                .check(check)
                .build()
                .map(Call::Scalar)
                .map_err(RebuildError::Call),
            Self::Aggregate(call, _) => {
                let filter = call.filter().map(|_| feed.one()).transpose()?;
                let order_by = feed.keys(call.order_by())?;
                AggregateCall::builder(call.function(), args)
                    .options(options)
                    .distinct(call.distinct())
                    .maybe_filter(filter)
                    .order_by(order_by)
                    .maybe_nulls(call.nulls())
                    .check(check)
                    .build()
                    .map(|call| Call::Aggregate(call, AggregateAllowed))
                    .map_err(RebuildError::Call)
            }
            Self::Window(call) => {
                let over = call.over();
                let over = Over {
                    partition_by: feed.many(over.partition_by.len())?,
                    order_by: feed.keys(&over.order_by)?,
                    window_frame: over.window_frame,
                };
                WindowCall::builder(call.function(), args)
                    .options(options)
                    .maybe_nulls(call.nulls())
                    .over(over)
                    .check(check)
                    .build()
                    .map(Call::Window)
                    .map_err(RebuildError::Call)
            }
        }
    }
}

struct Feed<T>(vec::IntoIter<T>);

impl<T> Feed<T> {
    fn one(&mut self) -> Result<T, RebuildError> {
        self.0.next().ok_or(RebuildError::ChildCount)
    }

    fn boxed(&mut self) -> Result<Box<T>, RebuildError> {
        self.one().map(Box::new)
    }

    fn many(&mut self, n: usize) -> Result<Vec<T>, RebuildError> {
        (0..n).map(|_| self.one()).collect()
    }
}

impl<To: Leaf> Feed<Expr<To>> {
    fn keys<L: Leaf, A: Aggregation>(
        &mut self,
        keys: &[SortKey<L, A>],
    ) -> Result<Vec<SortKey<To, AggregateAllowed>>, RebuildError> {
        keys.iter()
            .map(|key| {
                Ok(SortKey {
                    expr: self.one()?,
                    dir: key.dir,
                })
            })
            .collect()
    }
}
