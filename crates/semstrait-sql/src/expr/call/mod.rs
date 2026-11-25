mod check;

use bon::bon;
use serde::{Deserialize, Serialize};

use super::{
    narrow_all, AggregateAllowed, AggregateInRowExpr, Aggregation, CallError, Expr, FunctionRef,
    Leaf, NamedReference, NoAggregate, NoLeaf, Over, SortKey,
};
use crate::functions::{AggregateFunction, ClauseRule, Function, Scalar, Window};

pub(super) use check::{bind, bind_named, with_defaults};
pub use check::{CallCheck, CallClause};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Call<L: Leaf, A: Aggregation> {
    Scalar(ScalarCall<L, A>),
    Aggregate(AggregateCall<L, A>, A),
    Window(WindowCall<L, A>),
}

#[derive(Debug, Clone)]
pub struct ScalarCall<L: Leaf, A: Aggregation> {
    function: &'static Function<Scalar>,
    args: Vec<Expr<L, A>>,
    options: Vec<OptionSetting>,
    check: CallCheck,
}

#[derive(Debug, Clone)]
pub struct AggregateCall<L: Leaf, A: Aggregation> {
    function: &'static AggregateFunction,
    args: Vec<Expr<L, A>>,
    options: Vec<OptionSetting>,
    distinct: bool,
    filter: Option<Box<Expr<L, A>>>,
    order_by: Vec<SortKey<L, A>>,
    nulls: Option<NullTreatment>,
    check: CallCheck,
}

#[derive(Debug, Clone)]
pub struct WindowCall<L: Leaf, A: Aggregation> {
    function: &'static Function<Window>,
    args: Vec<Expr<L, A>>,
    options: Vec<OptionSetting>,
    nulls: Option<NullTreatment>,
    over: Over<L, A>,
    check: CallCheck,
}

pub(super) struct CallOperands<'call, L: Leaf, A: Aggregation> {
    pub(super) args: Vec<&'call mut Expr<L, A>>,
    pub(super) filter: Option<&'call mut Expr<L, A>>,
    pub(super) partition_by: Vec<&'call mut Expr<L, A>>,
    pub(super) order_by: Vec<&'call mut Expr<L, A>>,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NullTreatment {
    IgnoreNulls,
    RespectNulls,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OptionSetting {
    pub name: String,
    pub value: String,
}

impl<L: Leaf, A: Aggregation> Call<L, A> {
    pub fn new(function: FunctionRef<A>, args: Vec<Expr<L, A>>) -> Result<Self, CallError> {
        Ok(match function {
            FunctionRef::Scalar(function) => {
                Self::Scalar(ScalarCall::builder(function, args).build()?)
            }
            FunctionRef::Aggregate(function, aggregation) => {
                Self::Aggregate(AggregateCall::builder(function, args).build()?, aggregation)
            }
            FunctionRef::Window(function) => {
                Self::Window(WindowCall::builder(function, args).build()?)
            }
        })
    }

    pub(crate) fn function(&self) -> FunctionRef<A> {
        match self {
            Self::Scalar(call) => FunctionRef::Scalar(call.function),
            Self::Aggregate(call, aggregation) => {
                FunctionRef::Aggregate(call.function, aggregation.clone())
            }
            Self::Window(call) => FunctionRef::Window(call.function),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Scalar(call) => call.function.canonical.name,
            Self::Aggregate(call, _) => call.function.declarations.canonical.name,
            Self::Window(call) => call.function.canonical.name,
        }
    }

    pub fn args(&self) -> &[Expr<L, A>] {
        match self {
            Self::Scalar(call) => call.args(),
            Self::Aggregate(call, _) => call.args(),
            Self::Window(call) => call.args(),
        }
    }

    pub fn argument(&self, parameter: &str) -> Option<&Expr<L, A>> {
        let (function, signature) = self.function().canonical();
        let binding = bind(function, signature, self.args()).ok()?;
        (binding.into_iter().zip(self.args()))
            .find_map(|((_, declared), arg)| (declared.name == parameter).then_some(arg))
    }

    pub fn options(&self) -> &[OptionSetting] {
        match self {
            Self::Scalar(call) => call.options(),
            Self::Aggregate(call, _) => call.options(),
            Self::Window(call) => call.options(),
        }
    }

    pub fn check_options(&self) -> Result<(), CallError> {
        let name = self.name();
        match self {
            Self::Scalar(call) => check::check_options(
                name,
                call.function.canonical.specific.options,
                call.options(),
            ),
            Self::Aggregate(call, _) => check::check_options(
                name,
                call.function.declarations.canonical.specific.options,
                call.options(),
            ),
            Self::Window(call) => check::check_options(
                name,
                call.function.canonical.specific.options,
                call.options(),
            ),
        }
    }
}

macro_rules! parts {
    ($($call:ident: $function:ty),*) => {$(
        impl<L: Leaf, A: Aggregation> $call<L, A> {
            pub fn function(&self) -> &'static $function {
                self.function
            }

            pub fn args(&self) -> &[Expr<L, A>] {
                &self.args
            }

            pub fn options(&self) -> &[OptionSetting] {
                &self.options
            }
        }
    )*};
}

parts!(
    ScalarCall: Function<Scalar>,
    AggregateCall: AggregateFunction,
    WindowCall: Function<Window>
);

macro_rules! equal_but_its_check {
    ($($call:ident { $($field:ident),* }),*) => {$(
        impl<L: Leaf, A: Aggregation> PartialEq for $call<L, A> {
            fn eq(&self, other: &Self) -> bool {
                let Self { $($field,)* check: _ } = self;
                $(*$field == other.$field)&&*
            }
        }
    )*};
}

equal_but_its_check!(
    ScalarCall {
        function,
        args,
        options
    },
    AggregateCall {
        function,
        args,
        options,
        distinct,
        filter,
        order_by,
        nulls
    },
    WindowCall {
        function,
        args,
        options,
        nulls,
        over
    }
);

#[bon]
impl<L: Leaf, A: Aggregation> ScalarCall<L, A> {
    #[builder]
    pub fn new(
        #[builder(start_fn)] function: &'static Function<Scalar>,
        #[builder(start_fn)] args: Vec<Expr<L, A>>,
        #[builder(default)] options: Vec<OptionSetting>,
        #[builder(default = CallCheck::Full)] check: CallCheck,
    ) -> Result<Self, CallError> {
        let canonical = &function.canonical;
        let call = Self {
            function,
            args,
            options: check::held_options(canonical.name, canonical.specific.options, options)?,
            check,
        };
        call.check().map(|()| call)
    }

    fn check(&self) -> Result<(), CallError> {
        check::check_call(&self.function.canonical, &self.args, self.check, &[])
    }
}

#[bon]
impl<L: Leaf, A: Aggregation> AggregateCall<L, A> {
    #[builder]
    pub fn new(
        #[builder(start_fn)] function: &'static AggregateFunction,
        #[builder(start_fn)] args: Vec<Expr<L, A>>,
        #[builder(default)] options: Vec<OptionSetting>,
        #[builder(default)] distinct: bool,
        filter: Option<Expr<L, A>>,
        #[builder(default)] order_by: Vec<SortKey<L, A>>,
        nulls: Option<NullTreatment>,
        #[builder(default = CallCheck::Full)] check: CallCheck,
    ) -> Result<Self, CallError> {
        let canonical = &function.declarations.canonical;
        let call = Self {
            function,
            args,
            options: check::held_options(canonical.name, canonical.specific.options, options)?,
            distinct,
            filter: filter.map(Box::new),
            order_by,
            nulls,
            check,
        };
        call.check().map(|()| call)
    }

    fn check(&self) -> Result<(), CallError> {
        let declaration = &self.function.declarations.canonical;
        let rules = &declaration.specific.clauses;
        let within_group = rules.within_group != ClauseRule::Refused;
        let ordered = !self.order_by.is_empty();
        check::check_call(
            declaration,
            &self.args,
            self.check,
            &[
                (CallClause::Distinct, self.distinct, rules.distinct),
                (CallClause::Filter, self.filter.is_some(), rules.filter),
                (
                    CallClause::OrderBy,
                    ordered && !within_group,
                    rules.order_by,
                ),
                (
                    CallClause::WithinGroup,
                    ordered && within_group,
                    rules.within_group,
                ),
                (CallClause::Nulls, self.nulls.is_some(), rules.nulls),
            ],
        )?;
        let function = declaration.name;
        check::check_nesting(self.operands(), &|nested| match nested {
            Call::Scalar(_) => None,
            Call::Aggregate(..) => Some(CallError::AggregateInAggregate {
                function,
                nested: nested.name(),
            }),
            Call::Window(_) => Some(CallError::WindowInAggregate {
                function,
                nested: nested.name(),
            }),
        })
    }

    fn operands(&self) -> impl Iterator<Item = &Expr<L, A>> {
        (self.args.iter())
            .chain(self.filter.as_deref())
            .chain(self.order_by.iter().map(|key| &key.expr))
    }

    pub fn distinct(&self) -> bool {
        self.distinct
    }

    pub fn filter(&self) -> Option<&Expr<L, A>> {
        self.filter.as_deref()
    }

    pub fn order_by(&self) -> &[SortKey<L, A>] {
        &self.order_by
    }

    pub fn nulls(&self) -> Option<NullTreatment> {
        self.nulls
    }
}

impl AggregateFunction {
    pub fn check_value_call(&'static self, distinct: bool) -> Result<(), CallError> {
        let value = NamedReference::unqualified("value").map(Expr::<NoLeaf>::Column);
        AggregateCall::builder(self, value.into_iter().collect())
            .distinct(distinct)
            .build()
            .map(drop)
    }
}

#[bon]
impl<L: Leaf, A: Aggregation> WindowCall<L, A> {
    #[builder]
    pub fn new(
        #[builder(start_fn)] function: &'static Function<Window>,
        #[builder(start_fn)] args: Vec<Expr<L, A>>,
        #[builder(default)] options: Vec<OptionSetting>,
        nulls: Option<NullTreatment>,
        #[builder(default)] over: Over<L, A>,
        #[builder(default = CallCheck::Full)] check: CallCheck,
    ) -> Result<Self, CallError> {
        let canonical = &function.canonical;
        let call = Self {
            function,
            args,
            options: check::held_options(canonical.name, canonical.specific.options, options)?,
            nulls,
            over,
            check,
        };
        call.check().map(|()| call)
    }

    fn check(&self) -> Result<(), CallError> {
        let rules = &self.function.canonical.specific.clauses;
        let over = &self.over;
        check::check_call(
            &self.function.canonical,
            &self.args,
            self.check,
            &[
                (CallClause::Nulls, self.nulls.is_some(), rules.nulls),
                (
                    CallClause::PartitionBy,
                    !over.partition_by.is_empty(),
                    rules.partition_by,
                ),
                (
                    CallClause::OrderBy,
                    !over.order_by.is_empty(),
                    rules.order_by,
                ),
                (
                    CallClause::WindowFrame,
                    over.window_frame.is_some(),
                    rules.window_frame,
                ),
            ],
        )?;
        let function = self.function.canonical.name;
        over.check()
            .map_err(|error| CallError::Over { function, error })?;
        check::check_nesting(self.operands(), &|nested| match nested {
            Call::Scalar(_) | Call::Aggregate(..) => None,
            Call::Window(_) => Some(CallError::WindowInWindow {
                function,
                nested: nested.name(),
            }),
        })
    }

    fn operands(&self) -> impl Iterator<Item = &Expr<L, A>> {
        (self.args.iter())
            .chain(&self.over.partition_by)
            .chain(self.over.order_by.iter().map(|key| &key.expr))
    }

    pub fn nulls(&self) -> Option<NullTreatment> {
        self.nulls
    }

    pub fn over(&self) -> &Over<L, A> {
        &self.over
    }
}

impl<L: Leaf> TryFrom<Call<L, AggregateAllowed>> for Call<L, NoAggregate> {
    type Error = AggregateInRowExpr;

    fn try_from(call: Call<L, AggregateAllowed>) -> Result<Self, Self::Error> {
        Ok(match call {
            Call::Scalar(ScalarCall {
                function,
                args,
                options,
                check,
            }) => Self::Scalar(ScalarCall {
                function,
                args: narrow_all(args)?,
                options,
                check,
            }),
            Call::Aggregate(..) => return Err(AggregateInRowExpr),
            Call::Window(WindowCall {
                function,
                args,
                options,
                nulls,
                over,
                check,
            }) => Self::Window(WindowCall {
                function,
                args: narrow_all(args)?,
                options,
                nulls,
                check,
                over: Over {
                    partition_by: narrow_all(over.partition_by)?,
                    order_by: over
                        .order_by
                        .into_iter()
                        .map(|SortKey { expr, dir }| {
                            Ok(SortKey {
                                expr: expr.try_into()?,
                                dir,
                            })
                        })
                        .collect::<Result<_, _>>()?,
                    window_frame: over.window_frame,
                },
            }),
        })
    }
}

impl<L: Leaf, A: Aggregation> Call<L, A> {
    pub(super) fn children(&self) -> Vec<&Expr<L, A>> {
        match self {
            Self::Scalar(call) => call.args.iter().collect(),
            Self::Aggregate(call, _) => call.operands().collect(),
            Self::Window(call) => call.operands().collect(),
        }
    }

    pub(super) fn children_mut(&mut self) -> Vec<&mut Expr<L, A>> {
        let CallOperands {
            args,
            filter,
            partition_by,
            order_by,
        } = self.operands_mut();
        (args.into_iter().chain(filter))
            .chain(partition_by)
            .chain(order_by)
            .collect()
    }

    pub(super) fn operands_mut(&mut self) -> CallOperands<'_, L, A> {
        match self {
            Self::Scalar(call) => CallOperands {
                args: call.args.iter_mut().collect(),
                filter: None,
                partition_by: Vec::new(),
                order_by: Vec::new(),
            },
            Self::Aggregate(call, _) => CallOperands {
                args: call.args.iter_mut().collect(),
                filter: call.filter.as_deref_mut(),
                partition_by: Vec::new(),
                order_by: call.order_by.iter_mut().map(|key| &mut key.expr).collect(),
            },
            Self::Window(call) => CallOperands {
                args: call.args.iter_mut().collect(),
                filter: None,
                partition_by: call.over.partition_by.iter_mut().collect(),
                order_by: (call.over.order_by.iter_mut())
                    .map(|key| &mut key.expr)
                    .collect(),
            },
        }
    }

    pub(super) fn insert_argument(
        &mut self,
        position: usize,
        arg: Expr<L, A>,
    ) -> Result<(), CallError> {
        let args = match self {
            Self::Scalar(call) => &mut call.args,
            Self::Aggregate(call, _) => &mut call.args,
            Self::Window(call) => &mut call.args,
        };
        args.insert(position.min(args.len()), arg);
        self.check()
    }

    pub(super) fn check(&self) -> Result<(), CallError> {
        match self {
            Self::Scalar(call) => call.check(),
            Self::Aggregate(call, _) => call.check(),
            Self::Window(call) => call.check(),
        }
    }
}

#[cfg(test)]
mod tests;
