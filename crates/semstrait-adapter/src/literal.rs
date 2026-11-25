use semstrait_sql::expr::{HeldValue, Literal, LiteralError};
use substrait::proto::expression::literal::interval_day_to_second::PrecisionMode;
use substrait::proto::expression::literal::{self, LiteralType};
use substrait::proto::r#type::Nullability;

use crate::error::{InvalidRewrite, VocabularyPart};
use crate::rules::SubstraitRules;
use crate::types::{list, proto_type};

pub(crate) fn literal(
    value: &Literal,
    rules: &dyn SubstraitRules,
) -> Result<substrait::proto::expression::Literal, InvalidRewrite> {
    let width = |length: u32| usize::try_from(length).map_err(|_| unfit(LiteralError::OutOfRange));
    let literal_type = match value.held_value().map_err(unfit)? {
        Some(held) => held_literal(held)?,
        None => match value {
            Literal::Boolean(value) => LiteralType::Boolean(*value),
            Literal::TinyInt(value) => LiteralType::I8((*value).into()),
            Literal::SmallInt(value) => LiteralType::I16((*value).into()),
            Literal::Integer(value) => LiteralType::I32(*value),
            Literal::BigInt(value) => LiteralType::I64(*value),
            Literal::Real(value) => LiteralType::Fp32(*value),
            Literal::Double(value) => LiteralType::Fp64(*value),
            Literal::String(value) => LiteralType::String(value.clone()),
            Literal::VarChar { value, length } => LiteralType::VarChar(literal::VarChar {
                value: value.clone(),
                length: *length,
            }),
            Literal::FixedChar { value, length } => {
                LiteralType::FixedChar(format!("{value:<width$}", width = width(*length)?))
            }
            Literal::Binary(value) => LiteralType::Binary(value.clone()),
            Literal::FixedBinary { value, length } => {
                let mut bytes = value.clone();
                bytes.resize(width(*length)?, 0);
                LiteralType::FixedBinary(bytes)
            }
            Literal::Array { values, .. } if !values.is_empty() => {
                LiteralType::List(literal::List {
                    values: (values.iter())
                        .map(|value| literal(value, rules))
                        .collect::<Result<_, _>>()?,
                })
            }
            Literal::Array { element, .. } => LiteralType::EmptyList(list(
                element,
                rules.type_variation(&value.data_type()),
                Nullability::Required.into(),
                rules,
            )?),
            Literal::Null(data_type) => LiteralType::Null(proto_type(data_type, true, rules)?),
            _ => {
                return Err(InvalidRewrite::UnknownForm {
                    part: VocabularyPart::Literal,
                })
            }
        },
    };
    Ok(substrait::proto::expression::Literal {
        nullable: matches!(value, Literal::Null(_)),
        type_variation_reference: rules.type_variation(&value.data_type()),
        literal_type: Some(literal_type),
    })
}

fn held_literal(held: HeldValue) -> Result<LiteralType, InvalidRewrite> {
    Ok(match held {
        HeldValue::Date(days) => LiteralType::Date(days),
        HeldValue::Time { units, precision } => {
            LiteralType::PrecisionTime(literal::PrecisionTime {
                precision: precision.into(),
                value: units,
            })
        }
        HeldValue::Timestamp { units, precision } => {
            LiteralType::PrecisionTimestamp(literal::PrecisionTimestamp {
                precision: precision.into(),
                value: units,
            })
        }
        HeldValue::Decimal {
            unscaled,
            precision,
            scale,
        } => LiteralType::Decimal(literal::Decimal {
            value: unscaled.to_le_bytes().to_vec(),
            precision: precision.into(),
            scale: scale.into(),
        }),
        HeldValue::Interval {
            months,
            days,
            nanoseconds,
        } => LiteralType::IntervalCompound(literal::IntervalCompound {
            interval_year_to_month: Some(year_month(months)),
            interval_day_to_second: Some(day_second(days, nanoseconds, 9).map_err(unfit)?),
        }),
        HeldValue::IntervalYearMonth { months } => {
            LiteralType::IntervalYearToMonth(year_month(months))
        }
        HeldValue::IntervalDaySecond {
            days,
            units,
            precision,
        } => LiteralType::IntervalDayToSecond(day_second(days, units, precision).map_err(unfit)?),
        HeldValue::Uuid(bits) => LiteralType::Uuid(bits.to_be_bytes().to_vec()),
        _ => {
            return Err(InvalidRewrite::UnknownForm {
                part: VocabularyPart::Literal,
            })
        }
    })
}

fn unfit(error: LiteralError) -> InvalidRewrite {
    InvalidRewrite::UnfitLiteral { error }
}

fn year_month(months: i32) -> literal::IntervalYearToMonth {
    literal::IntervalYearToMonth {
        years: months / 12,
        months: months % 12,
    }
}

fn day_second(
    days: i32,
    units: i64,
    precision: u8,
) -> Result<literal::IntervalDayToSecond, LiteralError> {
    let out_of_range = LiteralError::OutOfRange;
    let unit = 10_i64.checked_pow(precision.into()).ok_or(out_of_range)?;
    Ok(literal::IntervalDayToSecond {
        days,
        seconds: i32::try_from(units / unit).map_err(|_| out_of_range)?,
        subseconds: units % unit,
        precision_mode: Some(PrecisionMode::Precision(precision.into())),
    })
}
