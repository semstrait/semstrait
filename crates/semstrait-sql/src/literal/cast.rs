use super::parse::Number;
use super::Literal;
use crate::datatype::DataType;

impl Literal {
    pub(crate) fn cast(&self, to: &DataType) -> Option<Literal> {
        match (self, Numeric::of(self)) {
            (Self::Null(_) | Self::String(_), _) => self.read_as(to).ok(),
            (_, Some(number)) => number.cast(to),
            _ => (self.data_type() == *to).then(|| self.clone()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Numeric {
    Whole(i128),
    Float(f64),
    Scaled(i128, u32),
}

impl Numeric {
    fn of(literal: &Literal) -> Option<Self> {
        Some(match *literal {
            Literal::TinyInt(value) => Self::Whole(value.into()),
            Literal::SmallInt(value) => Self::Whole(value.into()),
            Literal::Integer(value) => Self::Whole(value.into()),
            Literal::BigInt(value) => Self::Whole(value.into()),
            Literal::Real(value) => Self::Float(value.into()),
            Literal::Double(value) => Self::Float(value),
            Literal::Decimal {
                ref value, scale, ..
            } => {
                let scale = u32::try_from(scale).ok()?;
                Self::Scaled(Number::read(value).ok()?.unscaled(scale)?, scale)
            }
            _ => return None,
        })
    }

    fn cast(self, to: &DataType) -> Option<Literal> {
        let text = match *to {
            DataType::TinyInt | DataType::SmallInt | DataType::Integer | DataType::BigInt => {
                match self {
                    Self::Whole(value) => value,
                    Self::Float(value) => whole(value.trunc())?,
                    Self::Scaled(digits, scale) => digits / 10_i128.checked_pow(scale)?,
                }
                .to_string()
            }
            DataType::Decimal { scale, .. } => {
                let scale = u32::try_from(scale).ok()?;
                let digits = match self {
                    Self::Whole(value) => value.checked_mul(10_i128.checked_pow(scale)?)?,
                    Self::Float(value) => {
                        whole((10_f64.powi(i32::try_from(scale).ok()?) * value).round())?
                    }
                    Self::Scaled(digits, from) => rescaled(digits, from, scale)?,
                };
                decimal_text(digits, scale)?
            }
            DataType::Double => {
                let value = match self {
                    Self::Whole(value) => value as f64,
                    Self::Float(value) => value,
                    Self::Scaled(digits, scale) => descaled(digits, scale)?,
                };
                return value.is_finite().then_some(Literal::Double(value));
            }
            DataType::Real => {
                let value = match self {
                    Self::Whole(value) => value as f32,
                    Self::Float(value) if value.abs() > f64::from(f32::MAX) => return None,
                    Self::Float(value) => value as f32,
                    Self::Scaled(digits, scale) => descaled(digits, scale)? as f32,
                };
                return value.is_finite().then_some(Literal::Real(value));
            }
            _ => return None,
        };
        Literal::parse(&text, to).ok()
    }
}

fn whole(value: f64) -> Option<i128> {
    value.is_finite().then_some(value as i128)
}

fn rescaled(digits: i128, from: u32, to: u32) -> Option<i128> {
    if to >= from {
        return digits.checked_mul(10_i128.checked_pow(to - from)?);
    }
    let divisor = 10_i128.checked_pow(from - to)?;
    let rounds_away = (digits % divisor).unsigned_abs() >= divisor.unsigned_abs() / 2;
    Some(digits / divisor + digits.signum() * i128::from(rounds_away))
}

fn descaled(digits: i128, scale: u32) -> Option<f64> {
    Some(digits as f64 / 10_f64.powi(i32::try_from(scale).ok()?))
}

fn decimal_text(digits: i128, scale: u32) -> Option<String> {
    let divisor = 10_u128.checked_pow(scale)?;
    let magnitude = digits.unsigned_abs();
    let sign = if digits < 0 { "-" } else { "" };
    let whole = magnitude / divisor;
    let width = usize::try_from(scale).ok()?;
    let fraction = format!("{:0width$}", magnitude % divisor);
    Some(match fraction.trim_end_matches('0') {
        "" => format!("{sign}{whole}"),
        fraction => format!("{sign}{whole}.{fraction}"),
    })
}
