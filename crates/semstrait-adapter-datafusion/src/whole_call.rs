use std::collections::BTreeMap;

use semstrait_adapter::{Extensions, FunctionName, InvalidRewrite};
use semstrait_common::{Dialect, Release};
use semstrait_ir::{Parameter, PhysicalExpr};
use semstrait_sql::expr::{AggregateAllowed, CallError, Expr, FunctionRef, Literal, ScalarCall};
use semstrait_sql::functions::ArgumentForm;
use substrait::proto::expression::cast::FailureBehavior;
use substrait::proto::expression::literal::{IntervalDayToSecond, LiteralType};
use substrait::proto::expression::{self, Cast, RexType, ScalarFunction};
use substrait::proto::function_argument::ArgType;
use substrait::proto::r#type::{Kind, PrecisionTimestampTz};
use substrait::proto::{Expression, FunctionArgument, Type};

const DATAFUSION: &str = "extension:semstrait:datafusion";

type Bound<'call> = BTreeMap<&'static str, (&'call PhysicalExpr, &'call FunctionArgument)>;

pub(crate) fn whole_call(
    call: &ScalarCall<Parameter, AggregateAllowed>,
    arguments: &[FunctionArgument],
    output_type: &Type,
    extensions: &mut Extensions,
    release: Release,
) -> Result<Option<Expression>, InvalidRewrite> {
    let Some(declaration) = call.function().on_release(Dialect::DataFusion, release) else {
        return Ok(None);
    };
    let name = declaration.name;
    let canonical = call.function().canonical.name;
    let shape = || InvalidRewrite::CallShape {
        function: canonical,
    };
    match canonical {
        "regexp_like" | "regexp_instr" => {
            scalar(extensions, name, arguments.to_vec(), output_type.clone())
        }
        "regexp_replace" => {
            let arguments = [arguments, &[string("g")]].concat();
            scalar(extensions, name, arguments, output_type.clone())
        }
        "date_trunc" => {
            let bound = bound(call, arguments)?;
            let granularity = text(&bound, "granularity").ok_or_else(shape)?;
            let value = lowered(&bound, "value").ok_or_else(shape)?.clone();
            let date = matches!(output_type.kind, Some(Kind::Date(_)));
            let (value, truncated) = if date {
                (value, output_type.clone())
            } else {
                let zone = match bound.get("timezone") {
                    Some((Expr::Literal(Literal::Null(_)), _)) => {
                        return Ok(Some(literal(LiteralType::Null(output_type.clone()))));
                    }
                    Some(_) => text(&bound, "timezone").ok_or_else(shape)?.to_owned(),
                    None => match function(call)
                        .parameter("timezone")
                        .map_err(unbound)?
                        .default()
                    {
                        Some(Literal::String(zone)) => zone,
                        _ => return Err(shape()),
                    },
                };
                let (instant, unit) = zoned(output_type).ok_or_else(shape)?;
                let target = string(&format!("Timestamp({unit}, \"{zone}\")"));
                let arguments = vec![argument(cast(value, instant.clone())), target];
                let moved = scalar(extensions, "arrow_cast", arguments, instant.clone())?;
                (moved, instant)
            };
            let sunday = text(&bound, "week_start") == Some("sunday");
            let shift = |extensions: &mut Extensions, direction, value| {
                if !sunday {
                    return Ok(value);
                }
                let arguments = vec![argument(value), day()];
                scalar(extensions, direction, arguments, truncated.clone())
            };
            let value = shift(extensions, "add", value)?;
            let arguments = vec![string(granularity), argument(value)];
            let bucket = scalar(extensions, name, arguments, truncated.clone())?;
            let bucket = shift(extensions, "subtract", bucket)?;
            if date {
                Ok(cast(bucket, output_type.clone()))
            } else {
                let arguments = vec![argument(bucket)];
                scalar(extensions, "to_local_time", arguments, output_type.clone())
            }
        }
        "extract" => {
            let bound = bound(call, arguments)?;
            let component = text(&bound, "component").ok_or_else(shape)?;
            let part = (declaration.signature.parameters.iter()).find_map(|parameter| {
                match parameter.form {
                    ArgumentForm::DialectEnum(parts) => (parts.iter())
                        .find_map(|&(canonical, part)| (canonical == component).then_some(part)),
                    _ => None,
                }
            });
            let value = lowered(&bound, "value").ok_or_else(shape)?.clone();
            let out = output_type.clone();
            let int = |n| argument(literal(LiteralType::I32(n)));
            let extracted = match part {
                Some(part) => scalar(
                    extensions,
                    name,
                    vec![string(part), argument(value)],
                    out.clone(),
                )?,
                None if component == "day_of_week" => {
                    let dow = scalar(
                        extensions,
                        name,
                        vec![string("dow"), argument(value)],
                        out.clone(),
                    )?;
                    let shifted =
                        scalar(extensions, "add", vec![argument(dow), int(6)], out.clone())?;
                    let wrapped = scalar(
                        extensions,
                        "modulus",
                        vec![argument(shifted), int(7)],
                        out.clone(),
                    )?;
                    scalar(extensions, "add", vec![argument(wrapped), int(1)], out)?
                }
                None => return Err(shape()),
            };
            Ok(cast(extracted, output_type.clone()))
        }
        _ => return Ok(None),
    }
    .map(Some)
}

fn bound<'call>(
    call: &'call ScalarCall<Parameter, AggregateAllowed>,
    arguments: &'call [FunctionArgument],
) -> Result<Bound<'call>, InvalidRewrite> {
    let parameters = function(call).parameters(call.args()).map_err(unbound)?;
    let names = parameters.into_iter().map(|parameter| parameter.name);
    Ok(names.zip(call.args().iter().zip(arguments)).collect())
}

fn function(call: &ScalarCall<Parameter, AggregateAllowed>) -> FunctionRef<AggregateAllowed> {
    FunctionRef::Scalar(call.function())
}

fn unbound(error: CallError) -> InvalidRewrite {
    InvalidRewrite::UnboundCall { error }
}

fn text<'call>(bound: &Bound<'call>, parameter: &str) -> Option<&'call str> {
    match bound.get(parameter)? {
        (Expr::Literal(Literal::String(text)), _) => Some(text),
        _ => None,
    }
}

fn lowered<'call>(bound: &Bound<'call>, parameter: &str) -> Option<&'call Expression> {
    match &bound.get(parameter)?.1.arg_type {
        Some(ArgType::Value(value)) => Some(value),
        _ => None,
    }
}

fn zoned(timestamp: &Type) -> Option<(Type, &'static str)> {
    let Some(Kind::PrecisionTimestamp(timestamp)) = &timestamp.kind else {
        return None;
    };
    let unit = match timestamp.precision {
        0 => "Second",
        3 => "Millisecond",
        6 => "Microsecond",
        9 => "Nanosecond",
        _ => return None,
    };
    let instant = Kind::PrecisionTimestampTz(PrecisionTimestampTz {
        precision: timestamp.precision,
        type_variation_reference: timestamp.type_variation_reference,
        nullability: timestamp.nullability,
    });
    Some((
        Type {
            kind: Some(instant),
        },
        unit,
    ))
}

fn scalar(
    extensions: &mut Extensions,
    name: &'static str,
    arguments: Vec<FunctionArgument>,
    output_type: Type,
) -> Result<Expression, InvalidRewrite> {
    let function_reference = extensions.declare(FunctionName {
        urn: DATAFUSION,
        name,
    })?;
    Ok(Expression {
        rex_type: Some(RexType::ScalarFunction(ScalarFunction {
            function_reference,
            arguments,
            output_type: Some(output_type),
            ..ScalarFunction::default()
        })),
    })
}

fn cast(input: Expression, to: Type) -> Expression {
    Expression {
        rex_type: Some(RexType::Cast(Box::new(Cast {
            r#type: Some(to),
            input: Some(Box::new(input)),
            failure_behavior: FailureBehavior::ThrowException.into(),
        }))),
    }
}

fn argument(value: Expression) -> FunctionArgument {
    FunctionArgument {
        arg_type: Some(ArgType::Value(value)),
    }
}

fn literal(literal_type: LiteralType) -> Expression {
    let literal = expression::Literal {
        nullable: matches!(literal_type, LiteralType::Null(_)),
        literal_type: Some(literal_type),
        ..expression::Literal::default()
    };
    Expression {
        rex_type: Some(RexType::Literal(literal)),
    }
}

fn string(value: &str) -> FunctionArgument {
    argument(literal(LiteralType::String(value.to_owned())))
}

fn day() -> FunctionArgument {
    let day = IntervalDayToSecond {
        days: 1,
        ..IntervalDayToSecond::default()
    };
    argument(literal(LiteralType::IntervalDayToSecond(day)))
}
