use std::borrow::Cow;
use std::collections::HashMap;
use std::iter;
use std::ops::RangeInclusive;
use std::sync::OnceLock;

use semstrait_common::diagnostic::Diagnose;
use semstrait_common::{Dialect, Release};

use super::handle::{AggregateAllowed, FunctionRef};
use super::operation::OperationKind;
use super::{
    catalog, AggregateFunction, Category, Condition, Constraint, Declaration, Function,
    FunctionType, Nullability, OptionDecl, OptionName, ParameterDecl, ParameterType, Presence,
    ReturnRule, Scalar, SealFault, Window,
};
use crate::DocumentDialect;

#[derive(Debug)]
pub struct FunctionCatalog {
    pub scalar: &'static [(Category, &'static [&'static Function<Scalar>])],
    pub aggregate: &'static [&'static AggregateFunction],
    pub window: &'static [&'static Function<Window>],
}

#[derive(Debug)]
pub struct Registry {
    functions: Vec<FunctionRef<AggregateAllowed>>,
    names: HashMap<&'static str, FunctionRef<AggregateAllowed>>,
    dialect_names: HashMap<(Dialect, &'static str), FunctionRef<AggregateAllowed>>,
}

impl Registry {
    pub fn get(&self, name: &str) -> Option<FunctionRef<AggregateAllowed>> {
        let folded: Cow<'_, str> = if name.bytes().any(|byte| byte.is_ascii_uppercase()) {
            Cow::Owned(name.to_ascii_lowercase())
        } else {
            Cow::Borrowed(name)
        };
        self.names.get(&*folded).copied()
    }

    pub fn all(&self) -> impl Iterator<Item = FunctionRef<AggregateAllowed>> + '_ {
        self.functions.iter().copied()
    }

    pub fn find(
        &self,
        dialect: DocumentDialect,
        name: &str,
    ) -> Option<FunctionRef<AggregateAllowed>> {
        let declared = match dialect {
            DocumentDialect::Canonical => None,
            DocumentDialect::Dialect(dialect) => self.dialect_names.get(&(dialect, name)),
        };
        declared
            .or_else(|| self.names.get(name))
            .copied()
            .filter(|function| function.name_in(dialect) == Some(name))
    }

    fn add<F: FunctionType>(
        &mut self,
        handle: FunctionRef<AggregateAllowed>,
        function: &'static Function<F>,
        file: Option<Category>,
    ) -> Vec<SealFault> {
        self.functions.push(handle);
        self.dialect_names.extend(
            function
                .dialects
                .iter()
                .filter_map(|(dialect, declaration)| {
                    declaration
                        .as_ref()
                        .map(|declaration| ((*dialect, declaration.name), handle))
                }),
        );
        let canonical = &function.canonical;
        let names = iter::once(canonical.name)
            .chain(canonical.aliases.iter().copied())
            .filter_map(|name| {
                if OperationKind::from_name(name).is_some() {
                    return Some(SealFault::OperationName { name });
                }
                self.names
                    .insert(name, handle)
                    .map(|_| SealFault::SharedName { name })
            })
            .collect::<Vec<_>>();
        let declarations = iter::once(canonical)
            .chain(
                function
                    .dialects
                    .iter()
                    .filter_map(|(_, declaration)| declaration.as_ref()),
            )
            .chain(
                function
                    .releases
                    .iter()
                    .filter_map(|(_, _, declaration)| declaration.as_ref()),
            )
            .flat_map(|declaration| declaration_faults(canonical.name, declaration, file));
        names
            .into_iter()
            .chain(dialect_faults(function))
            .chain(release_faults(function))
            .chain(declarations)
            .collect()
    }
}

fn dialect_faults<F: FunctionType>(function: &Function<F>) -> Vec<SealFault> {
    let canonical = &function.canonical;
    function
        .dialects
        .iter()
        .enumerate()
        .flat_map(|(index, (dialect, declaration))| {
            let dialect = *dialect;
            let twice = function
                .dialects
                .iter()
                .take(index)
                .any(|(earlier, _)| *earlier == dialect)
                .then_some(SealFault::TwoDialectDeclarations {
                    function: canonical.name,
                    dialect,
                });
            let aliases = declaration
                .as_ref()
                .filter(|declaration| !declaration.aliases.is_empty())
                .map(|_| SealFault::DialectAliases {
                    function: canonical.name,
                    dialect,
                });
            let repeated = declaration
                .as_ref()
                .filter(
                    |Declaration {
                         name,
                         aliases: _,
                         signature,
                         returns,
                         nullability,
                         volatility,
                         specific,
                     }| {
                        (name, signature, returns, nullability, volatility, specific)
                            == (
                                &canonical.name,
                                &canonical.signature,
                                &canonical.returns,
                                &canonical.nullability,
                                &canonical.volatility,
                                &canonical.specific,
                            )
                    },
                )
                .map(|_| SealFault::DialectEqualsCanonical {
                    function: canonical.name,
                    dialect,
                });
            twice.into_iter().chain(aliases).chain(repeated)
        })
        .collect()
}

fn ranges_overlap(a: &RangeInclusive<Release>, b: &RangeInclusive<Release>) -> bool {
    a.start() <= b.end() && b.start() <= a.end()
}

fn release_faults<F: FunctionType>(function: &Function<F>) -> Vec<SealFault> {
    let canonical = &function.canonical;
    function
        .releases
        .iter()
        .enumerate()
        .flat_map(|(index, (dialect, range, declaration))| {
            let dialect = *dialect;
            let overlapping = function
                .releases
                .iter()
                .take(index)
                .any(|(earlier, earlier_range, _)| {
                    *earlier == dialect && ranges_overlap(earlier_range, range)
                })
                .then_some(SealFault::OverlappingReleases {
                    function: canonical.name,
                    dialect,
                });
            let holds_reference = dialect
                .reference_release()
                .filter(|reference| range.contains(reference))
                .map(|_| SealFault::ReleaseHoldsReference {
                    function: canonical.name,
                    dialect,
                });
            let aliases = declaration
                .as_ref()
                .filter(|declaration| !declaration.aliases.is_empty())
                .map(|_| SealFault::DialectAliases {
                    function: canonical.name,
                    dialect,
                });
            overlapping
                .into_iter()
                .chain(holds_reference)
                .chain(aliases)
        })
        .collect()
}

fn declaration_faults<F: FunctionType>(
    function: &'static str,
    declaration: &Declaration<F>,
    file: Option<Category>,
) -> Vec<SealFault> {
    let scalar = declaration.specific.scalar();
    let category = scalar
        .map(|scalar| scalar.category)
        .zip(file)
        .filter(|(category, file)| category != file)
        .map(|(category, file)| SealFault::CategoryNotItsFile {
            function,
            category,
            file,
        });
    let signature = &declaration.signature;
    let parameters = signature
        .parameters
        .iter()
        .chain(&signature.variadic)
        .collect::<Vec<_>>();
    let first_optional = parameters
        .iter()
        .position(|parameter| parameter.presence != Presence::Required);
    let default_fault = |parameter: &ParameterDecl| {
        (parameter.default())
            .filter(|default| !parameter.ty.takes(&default.data_type()))
            .map(|_| SealFault::DefaultNotOfItsType {
                function,
                parameter: parameter.name,
            })
    };
    let order = parameters
        .iter()
        .enumerate()
        .filter_map(|(index, parameter)| match &parameter.presence {
            Presence::Required => first_optional.filter(|first| *first < index).map(|_| {
                SealFault::OptionalBeforeRequired {
                    function,
                    parameter: parameter.name,
                }
            }),
            Presence::Optional => {
                (index + 1 < parameters.len()).then_some(SealFault::OptionalWithNoDefault {
                    function,
                    parameter: parameter.name,
                })
            }
            Presence::Defaulted(_) => default_fault(parameter),
            Presence::When {
                condition: Condition::Type { argument, .. } | Condition::Value { argument, .. },
                ..
            } => (*argument >= index)
                .then_some(SealFault::ConditionNamesLaterArgument {
                    function,
                    parameter: parameter.name,
                })
                .or_else(|| default_fault(parameter)),
        });
    let returned = match declaration.returns {
        ReturnRule::SameAs(index) => Some(index),
        _ => None,
    };
    let constrained = signature
        .constraints
        .iter()
        .filter_map(|constraint| match constraint {
            Constraint::EnumRefused {
                enumeration,
                argument,
                ..
            } => Some([*enumeration, *argument]),
            Constraint::OneFamily | Constraint::Families(_) => None,
        })
        .flatten();
    let nullabilities = nested(&declaration.nullability);
    let mirrored = nullabilities
        .iter()
        .flat_map(|nullability| match nullability {
            Nullability::MirrorArgs(indexes) => indexes,
            _ => &[][..],
        })
        .copied();
    let out_of_range = returned
        .into_iter()
        .chain(constrained)
        .chain(mirrored)
        .filter(|index| *index >= parameters.len())
        .map(|index| SealFault::ArgumentIndexOutOfRange { function, index });
    let unbound = (declaration.returns == ReturnRule::Var
        && !parameters
            .iter()
            .any(|parameter| matches!(parameter.ty, ParameterType::Var(_))))
    .then_some(SealFault::UnboundTypeVariable { function });
    let aggregate = (scalar.is_some()
        && nullabilities
            .iter()
            .any(|nullability| **nullability == Nullability::EmptyGroupNull))
    .then_some(SealFault::AggregateNullability { function });
    category
        .into_iter()
        .chain(declaration.specific.option_faults(function))
        .chain(order)
        .chain(out_of_range)
        .chain(unbound)
        .chain(aggregate)
        .collect()
}

fn nested(nullability: &Nullability) -> Vec<&Nullability> {
    let mut all = vec![nullability];
    if let Nullability::ByOption(by_value) = nullability {
        all.extend(
            by_value
                .iter()
                .flat_map(|(_, nullability)| nested(nullability)),
        );
    }
    all
}

pub(super) fn option_faults<N: OptionName>(
    function: &'static str,
    options: &[OptionDecl<N>],
) -> Vec<SealFault> {
    options
        .iter()
        .flat_map(|declared| {
            let option = declared.name.name();
            let outside = declared
                .values
                .iter()
                .filter(|value| !declared.name.values().contains(value))
                .map(move |&value| SealFault::OptionValueOutsideOption {
                    function,
                    option,
                    value,
                });
            let canonical = (!declared.values.contains(&declared.canonical))
                .then_some(SealFault::CanonicalValueNotAmongValues { function, option });
            outside.chain(canonical)
        })
        .collect()
}

impl FunctionRef<AggregateAllowed> {
    pub(crate) fn name_in(self, dialect: DocumentDialect) -> Option<&'static str> {
        match self {
            Self::Scalar(function) => function.name_in(dialect),
            Self::Aggregate(function, _) => function.declarations.name_in(dialect),
            Self::Window(function) => function.name_in(dialect),
        }
    }
}

pub fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        seal(&catalog::CATALOG).unwrap_or_else(|faults| {
            let messages = faults
                .iter()
                .map(Diagnose::message)
                .collect::<Vec<_>>()
                .join("; ");
            panic!("the function catalog does not seal: {messages}")
        })
    })
}

pub fn seal(catalog: &FunctionCatalog) -> Result<Registry, Vec<SealFault>> {
    let mut registry = Registry {
        functions: Vec::new(),
        names: HashMap::new(),
        dialect_names: HashMap::new(),
    };
    let mut faults = Vec::new();
    for (file, functions) in catalog.scalar {
        for function in *functions {
            faults.extend(registry.add(FunctionRef::Scalar(function), function, Some(*file)));
        }
    }
    for function in catalog.aggregate {
        faults.extend(registry.add(
            FunctionRef::Aggregate(function, AggregateAllowed),
            &function.declarations,
            None,
        ));
    }
    for function in catalog.window {
        faults.extend(registry.add(FunctionRef::Window(function), function, None));
    }
    faults.extend(registry.all().flat_map(|function| {
        registry
            .dialect_names
            .iter()
            .filter(move |(&(dialect, name), other)| {
                **other != function
                    && function.name_in(DocumentDialect::Dialect(dialect)) == Some(name)
            })
            .map(|(&(dialect, name), _)| SealFault::SharedDialectName { dialect, name })
    }));
    if faults.is_empty() {
        Ok(registry)
    } else {
        Err(faults)
    }
}
