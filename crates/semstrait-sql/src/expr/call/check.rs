use std::borrow::Borrow;
use std::collections::{BTreeMap, HashSet};
use std::fmt;

use semstrait_common::Tree;

use super::OptionSetting;
use crate::expr::{Aggregation, Call, CallError, Expr, Leaf, Literal};
use crate::functions::{
    ArgumentForm, ClauseRule, Condition, Declaration, FunctionType, OptionDecl, OptionName,
    ParameterDecl, Presence, Signature,
};

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallCheck {
    Full,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallClause {
    Distinct,
    Filter,
    OrderBy,
    WithinGroup,
    Nulls,
    PartitionBy,
    WindowFrame,
}

impl fmt::Display for CallClause {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Distinct => "DISTINCT",
            Self::Filter => "FILTER",
            Self::OrderBy => "ORDER BY",
            Self::WithinGroup => "WITHIN GROUP",
            Self::Nulls => "IGNORE NULLS or RESPECT NULLS",
            Self::PartitionBy => "PARTITION BY",
            Self::WindowFrame => "window frame",
        })
    }
}

pub(super) fn check_call<F: FunctionType, L: Leaf, A: Aggregation>(
    declaration: &Declaration<F>,
    args: &[Expr<L, A>],
    check: CallCheck,
    clauses: &[(CallClause, bool, ClauseRule)],
) -> Result<(), CallError> {
    let function = declaration.name;
    check_arguments(function, &declaration.signature, args, check)?;
    clauses
        .iter()
        .try_for_each(|&(clause, written, rule)| match (written, rule) {
            (true, ClauseRule::Refused | ClauseRule::NoEffect) => {
                Err(CallError::ClauseRefused { function, clause })
            }
            (false, ClauseRule::Required) => Err(CallError::ClauseRequired { function, clause }),
            _ => Ok(()),
        })
}

pub(super) fn check_nesting<'call, L: Leaf + 'call, A: Aggregation + 'call>(
    operands: impl IntoIterator<Item = &'call Expr<L, A>>,
    refused: &impl Fn(&Call<L, A>) -> Option<CallError>,
) -> Result<(), CallError> {
    operands.into_iter().try_for_each(|operand| {
        if let Expr::Function(call) = operand {
            refused(call).map_or(Ok(()), Err)?;
        }
        check_nesting(operand.children(), refused)
    })
}

pub(super) fn check_options<N: OptionName>(
    function: &'static str,
    declared: &[OptionDecl<N>],
    options: &[OptionSetting],
) -> Result<(), CallError> {
    check_each_option_once(function, options)?;
    options.iter().try_for_each(|setting| {
        let option = declared
            .iter()
            .find(|option| option.name.name() == setting.name)
            .ok_or_else(|| CallError::UnknownOption {
                function,
                option: setting.name.clone(),
            })?;
        if option.values.contains(&setting.value.as_str()) {
            Ok(())
        } else {
            Err(CallError::OptionValue {
                function,
                option: option.name.name(),
                value: setting.value.clone(),
            })
        }
    })
}

fn check_each_option_once(
    function: &'static str,
    options: &[OptionSetting],
) -> Result<(), CallError> {
    let mut names = HashSet::new();
    options
        .iter()
        .find(|setting| !names.insert(setting.name.as_str()))
        .map_or(Ok(()), |setting| {
            Err(CallError::RepeatedOption {
                function,
                option: setting.name.clone(),
            })
        })
}

pub(super) fn held_options<N: OptionName>(
    function: &'static str,
    declared: &[OptionDecl<N>],
    options: Vec<OptionSetting>,
) -> Result<Vec<OptionSetting>, CallError> {
    check_each_option_once(function, &options)?;
    let canonical = |setting: &OptionSetting| {
        (declared.iter())
            .any(|option| option.name.name() == setting.name && option.canonical == setting.value)
    };
    Ok(options
        .into_iter()
        .filter(|setting| !canonical(setting))
        .collect())
}

fn check_arguments<L: Leaf, A: Aggregation>(
    function: &'static str,
    signature: &Signature,
    args: &[Expr<L, A>],
    check: CallCheck,
) -> Result<(), CallError> {
    let binding = bind(function, signature, args)?;
    match check {
        CallCheck::Full => args
            .iter()
            .zip(binding)
            .try_for_each(|(arg, (_, parameter))| check_form(function, parameter, arg)),
    }
}

pub(crate) fn bind<'d, L: Leaf, A: Aggregation>(
    function: &'static str,
    signature: &'d Signature,
    args: &[Expr<L, A>],
) -> Result<Vec<(usize, &'d ParameterDecl)>, CallError> {
    let (min, max) = signature.argument_count();
    if args.len() < min {
        return Err(CallError::TooFewArguments { function, min, max });
    }
    let parameters = signature.parameters;
    let mut bound: Vec<Option<&Expr<L, A>>> = Vec::with_capacity(parameters.len());
    let mut binding = Vec::with_capacity(args.len());
    let mut rest = args;
    for (index, parameter) in parameters.iter().enumerate() {
        let taken_later = || {
            parameters[index + 1..]
                .iter()
                .filter(|later| written(later, &bound) == Some(true))
                .count()
        };
        if !written(parameter, &bound).unwrap_or_else(|| rest.len() > taken_later()) {
            bound.push(None);
            continue;
        }
        let Some((arg, others)) = rest.split_first() else {
            return Err(CallError::MissingArgument {
                function,
                parameter: parameter.name,
            });
        };
        bound.push(Some(arg));
        binding.push((index, parameter));
        rest = others;
    }
    if signature.variadic.is_none() && !rest.is_empty() {
        return Err(CallError::TooManyArguments {
            function,
            got: args.len(),
        });
    }
    binding.extend(signature.tail().take(rest.len()));
    Ok(binding)
}

pub(crate) fn with_defaults<L: Leaf, A: Aggregation>(
    function: &'static str,
    signature: &Signature,
    mut args: Vec<Expr<L, A>>,
) -> Vec<Expr<L, A>> {
    while let Err(CallError::MissingArgument { parameter, .. }) = bind(function, signature, &args) {
        let Some(default) =
            (signature.parameter(parameter)).and_then(|(_, declared)| declared.default())
        else {
            break;
        };
        args.push(Expr::Literal(default));
    }
    args
}

pub(crate) fn bind_named<L: Leaf, A: Aggregation>(
    function: &'static str,
    signature: &Signature,
    named: BTreeMap<&str, Expr<L, A>>,
) -> Result<Vec<Expr<L, A>>, CallError> {
    let parameters = signature.parameters;
    let mut args = vec![None; parameters.len()];
    for (name, arg) in named {
        let (index, _) =
            (signature.parameter(name)).ok_or_else(|| CallError::UnknownParameter {
                function,
                name: name.to_owned(),
            })?;
        args[index] = Some(arg);
    }
    let last = args.iter().rposition(Option::is_some).unwrap_or(0);
    for (arg, parameter) in args[..last].iter_mut().zip(parameters) {
        if arg.is_none() && matches!(parameter.presence, Presence::Defaulted(_)) {
            *arg = parameter.default().map(Expr::Literal);
        }
    }
    let mut filled = Vec::with_capacity(parameters.len());
    for (parameter, arg) in parameters.iter().zip(args) {
        let arg = match (written(parameter, &filled), arg) {
            (Some(true), None) => {
                let default = parameter.default().ok_or(CallError::MissingParameter {
                    function,
                    parameter: parameter.name,
                })?;
                Some(Expr::Literal(default))
            }
            (Some(false), Some(_)) => {
                return Err(CallError::UnwrittenParameter {
                    function,
                    parameter: parameter.name,
                })
            }
            (_, arg) => arg,
        };
        filled.push(arg);
    }
    let (placed, args): (Vec<_>, Vec<_>) = (filled.into_iter().enumerate())
        .filter_map(|(index, arg)| Some((index, arg?)))
        .unzip();
    let binding = bind(function, signature, &args)?;
    (placed.into_iter().zip(binding))
        .find(|(placed, (bound, _))| placed != bound)
        .map_or(Ok(args), |(index, _)| {
            Err(CallError::UnwrittenParameter {
                function,
                parameter: parameters[index].name,
            })
        })
}

fn written<L: Leaf, A: Aggregation>(
    parameter: &ParameterDecl,
    bound: &[Option<impl Borrow<Expr<L, A>>>],
) -> Option<bool> {
    match &parameter.presence {
        Presence::Required => Some(true),
        Presence::When {
            condition: Condition::Value { argument, value },
            ..
        } => match bound.get(*argument)? {
            None => Some(false),
            Some(arg) => (arg.borrow().literal())
                .map(|literal| matches!(literal, Literal::String(text) if text == value)),
        },
        Presence::Optional
        | Presence::Defaulted(_)
        | Presence::When {
            condition: Condition::Type { .. },
            ..
        } => None,
    }
}

fn check_form<L: Leaf, A: Aggregation>(
    function: &'static str,
    parameter: &ParameterDecl,
    arg: &Expr<L, A>,
) -> Result<(), CallError> {
    let name = parameter.name;
    match (&parameter.form, arg.literal()) {
        (ArgumentForm::Value, _) | (ArgumentForm::Constant, Some(_)) => Ok(()),
        (ArgumentForm::Enum(values), Some(Literal::String(value)))
            if values.contains(&value.as_str()) =>
        {
            Ok(())
        }
        (ArgumentForm::DialectEnum(names), Some(Literal::String(value)))
            if names.iter().any(|(canonical, _)| canonical == value) =>
        {
            Ok(())
        }
        (ArgumentForm::Enum(_) | ArgumentForm::DialectEnum(_), Some(value)) => {
            Err(CallError::EnumValue {
                function,
                parameter: name,
                value: value.clone(),
            })
        }
        (ArgumentForm::Constant | ArgumentForm::Enum(_) | ArgumentForm::DialectEnum(_), None) => {
            Err(CallError::NotALiteral {
                function,
                parameter: name,
            })
        }
    }
}
