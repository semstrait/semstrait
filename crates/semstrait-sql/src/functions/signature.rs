use super::{
    ArgumentRefusal, Condition, Constraint, Nullability, OptionDecl, OptionName, ParameterDecl,
    ParameterType, Presence, ReturnRule,
};
use crate::datatype::family::one_family;
use crate::datatype::type_system::{arithmetic, first, promoted};
use crate::datatype::{DataType, Decimal, TypeFamily, TypeOrFamily, TypeSystem};
use crate::literal::Literal;

#[non_exhaustive]
#[derive(Debug, PartialEq)]
pub struct Signature {
    pub parameters: &'static [ParameterDecl],
    pub variadic: Option<ParameterDecl>,
    pub constraints: &'static [Constraint],
}

impl Signature {
    pub fn parameter(&self, name: &str) -> Option<(usize, &'static ParameterDecl)> {
        (self.parameters.iter().enumerate()).find(|(_, parameter)| parameter.name == name)
    }

    pub fn argument_count(&self) -> (usize, Option<usize>) {
        let fewest = (self.parameters.iter().chain(&self.variadic))
            .filter(|parameter| parameter.presence == Presence::Required)
            .count();
        let most = self.variadic.is_none().then_some(self.parameters.len());
        (fewest, most)
    }

    pub(crate) fn tail(&self) -> impl Iterator<Item = (usize, &ParameterDecl)> + '_ {
        let index = self.parameters.len();
        self.variadic.iter().map(move |tail| (index, tail)).cycle()
    }
}

pub(crate) fn option_values<'s, N: OptionName>(
    options: &'static [OptionDecl<N>],
    setting: impl Fn(&str) -> Option<&'s str>,
) -> Vec<&'static str> {
    options
        .iter()
        .filter_map(|option| match setting(option.name.name()) {
            None => Some(option.canonical),
            Some(set) => option.values.iter().copied().find(|value| *value == set),
        })
        .collect()
}

pub(crate) fn nullable(
    rule: &Nullability,
    options: &[&str],
    binding: &[(usize, &'static ParameterDecl)],
    arguments: &[bool],
) -> bool {
    match rule {
        Nullability::Mirror | Nullability::Kleene => arguments.contains(&true),
        Nullability::MirrorArgs(indexes) => (binding.iter().zip(arguments))
            .any(|((index, _), nullable)| *nullable && indexes.contains(index)),
        Nullability::AllArgs => arguments.iter().all(|nullable| *nullable),
        Nullability::Never => false,
        Nullability::ByOption(rows) => rows
            .iter()
            .find(|(value, _)| options.contains(value))
            .map_or(true, |(_, rule)| {
                nullable(rule, options, binding, arguments)
            }),
        Nullability::Always | Nullability::EmptyGroupNull => true,
    }
}

pub(crate) fn derives_from_operands(returns: &ReturnRule) -> bool {
    match returns {
        ReturnRule::SameAs(_)
        | ReturnRule::Promote
        | ReturnRule::Sum
        | ReturnRule::Product
        | ReturnRule::Quotient
        | ReturnRule::Mod => true,
        ReturnRule::Fixed(_)
        | ReturnRule::Var
        | ReturnRule::SumOf
        | ReturnRule::AvgOf
        | ReturnRule::DecimalScaleZero
        | ReturnRule::RoundOf => false,
    }
}

pub(crate) fn at_t(parameter: &ParameterDecl) -> bool {
    matches!(parameter.ty, ParameterType::Var(_))
}

fn named((data_type, family): &(DataType, Option<TypeFamily>)) -> TypeOrFamily {
    TypeOrFamily::of(data_type, family.as_ref())
}

pub(crate) fn one_exact_type<'t>(
    types: impl IntoIterator<Item = &'t (DataType, Option<TypeFamily>)>,
) -> Result<DataType, (TypeOrFamily, TypeOrFamily)> {
    let typed = |(data_type, _): &(DataType, Option<TypeFamily>)| *data_type != DataType::Unknown;
    let (first, unknown) = (types.into_iter()).try_fold(
        (None::<&(DataType, Option<TypeFamily>)>, false),
        |(first, unknown), value| match first {
            _ if !typed(value) && value.1.is_none() => Ok((first, true)),
            Some(first)
                if first.1 != value.1 || (typed(first) && typed(value) && first.0 != value.0) =>
            {
                Err((named(first), named(value)))
            }
            Some(first) if typed(first) => Ok((Some(first), unknown || !typed(value))),
            _ => Ok((Some(value), unknown || !typed(value))),
        },
    )?;
    Ok(match first {
        Some((data_type, _)) if !unknown => data_type.clone(),
        _ => DataType::Unknown,
    })
}

pub(crate) fn check_typed_arguments(
    signature: &Signature,
    binding: &[(usize, &'static ParameterDecl)],
    arguments: &[(DataType, Option<TypeFamily>)],
    strings: &[Option<&str>],
) -> Result<(DataType, Vec<(usize, Literal)>), ArgumentRefusal> {
    let misfit = (binding.iter().zip(arguments)).find(|((_, parameter), (data_type, family))| {
        !parameter.ty.takes_value(data_type, family.as_ref())
    });
    if let Some(((_, parameter), argument)) = misfit {
        return Err(ArgumentRefusal::Argument {
            parameter: parameter.name,
            takes: &parameter.ty,
            got: named(argument),
        });
    }
    let t_types = (binding.iter().zip(arguments))
        .filter(|((_, parameter), _)| at_t(parameter))
        .map(|(_, argument)| argument);
    let t = one_exact_type(t_types)
        .map_err(|(bound, got)| ArgumentRefusal::TypeVariable { bound, got })?;
    let fits = |constraint: &Constraint| match constraint {
        Constraint::OneFamily => {
            let mut known = arguments.iter().filter_map(|(_, family)| family.as_ref());
            known
                .next()
                .map_or(true, |first| known.all(|family| family == first))
        }
        Constraint::Families(rows) => rows.iter().any(|row| {
            row.len() == arguments.len()
                && (row.iter().zip(arguments))
                    .all(|(want, (_, family))| family.as_ref().map_or(true, |got| got == want))
        }),
        Constraint::EnumRefused { .. } => true,
    };
    if !signature.constraints.iter().all(fits) {
        return Err(ArgumentRefusal::Arguments {
            types: arguments.iter().map(named).collect(),
        });
    }
    let at = |index: usize| {
        (binding.iter().zip(strings).zip(arguments))
            .find(|(((at, _), _), _)| *at == index)
            .map(|(((_, parameter), string), typed)| (*parameter, *string, typed))
    };
    let refused = signature.constraints.iter().find_map(|constraint| {
        let Constraint::EnumRefused {
            enumeration,
            values,
            argument,
            ty,
        } = constraint
        else {
            return None;
        };
        let (parameter, Some(value), _) = at(*enumeration)? else {
            return None;
        };
        let (_, _, got) = at(*argument)?;
        (values.contains(&value) && ty.takes(&got.0)).then(|| ArgumentRefusal::EnumRefused {
            parameter: parameter.name,
            value: value.to_owned(),
            got: named(got),
        })
    });
    if let Some(refusal) = refused {
        return Err(refusal);
    }
    let defaults = (signature.parameters.iter().enumerate())
        .filter_map(|(index, parameter)| {
            let Presence::When {
                condition: Condition::Type { argument, ty },
                ..
            } = &parameter.presence
            else {
                return None;
            };
            let (_, _, (got, _)) =
                at(*argument).filter(|(_, _, (got, _))| *got != DataType::Unknown)?;
            let written = binding.iter().any(|(at, _)| *at == index);
            match (written, ty.takes(got), parameter.default()) {
                (true, false, _) => Some(Err(ArgumentRefusal::ParameterRefused {
                    parameter: parameter.name,
                })),
                (false, true, Some(default)) => Some(Ok((index, default))),
                (false, true, None) => Some(Err(ArgumentRefusal::ParameterRequired {
                    parameter: parameter.name,
                })),
                (true, true, _) | (false, false, _) => None,
            }
        })
        .collect::<Result<_, _>>()?;
    Ok((t, defaults))
}

pub(crate) fn given<'t>(
    known: impl IntoIterator<Item = Option<&'t DataType>>,
    binds: impl Fn(&DataType) -> bool,
    context: Option<&DataType>,
) -> Option<DataType> {
    let mut known = known.into_iter().flatten().peekable();
    if known.peek().is_none() {
        return context.cloned();
    }
    known
        .find(|data_type| **data_type != DataType::Unknown)
        .filter(|data_type| binds(data_type))
        .cloned()
}

pub(crate) fn result_type(
    system: &dyn TypeSystem,
    returns: &ReturnRule,
    binding: &[(usize, &'static ParameterDecl)],
    operands: Vec<(DataType, Option<TypeFamily>)>,
    t: DataType,
) -> (DataType, Option<TypeFamily>) {
    let of_decimal = |data_type: DataType, rule: &dyn Fn(Decimal) -> Decimal| match data_type {
        DataType::Decimal { precision, scale } => rule(Decimal { precision, scale }).into(),
        other => other,
    };
    match returns {
        ReturnRule::Fixed(exact) => (exact.data_type(), None),
        ReturnRule::Var => {
            let t_families = (binding.iter().zip(&operands))
                .filter(|((_, parameter), _)| at_t(parameter))
                .map(|(_, (_, family))| family.as_ref());
            (t, one_family(t_families))
        }
        ReturnRule::SameAs(at) => (binding.iter().zip(operands))
            .find(|((index, _), _)| index == at)
            .map_or((DataType::Unknown, None), |(_, operand)| operand),
        ReturnRule::Sum => arithmetic(operands, |a, b| system.decimal_sum(a, b)),
        ReturnRule::Product => arithmetic(operands, |a, b| system.decimal_product(a, b)),
        ReturnRule::Quotient => arithmetic(operands, |a, b| system.decimal_quotient(a, b)),
        ReturnRule::Mod => arithmetic(operands, |a, b| system.decimal_mod(a, b)),
        ReturnRule::Promote => promoted(operands, |a, b| system.decimal_union(a, b)),
        ReturnRule::SumOf => first(operands, |arg| system.sum_of(&arg)),
        ReturnRule::AvgOf => first(operands, |arg| system.avg_of(&arg)),
        ReturnRule::DecimalScaleZero => first(operands, |arg| {
            of_decimal(arg, &|arg| system.decimal_scale_zero(arg))
        }),
        ReturnRule::RoundOf => first(operands, |arg| of_decimal(arg, &|arg| system.round_of(arg))),
    }
}
