use std::str::FromStr;

use super::Literal;
use super::LiteralError::{self, CutsADigit, Malformed, NoReading, OutOfRange, Zoned};
use crate::datatype::DataType;

impl Literal {
    pub fn parse(text: &str, to: &DataType) -> Result<Self, LiteralError> {
        let owned = || text.to_owned();
        let literal = match *to {
            DataType::Boolean => Self::Boolean(match text {
                "true" => true,
                "false" => false,
                _ => return Err(Malformed),
            }),
            DataType::TinyInt => Self::TinyInt(integer(text)?),
            DataType::SmallInt => Self::SmallInt(integer(text)?),
            DataType::Integer => Self::Integer(integer(text)?),
            DataType::BigInt => Self::BigInt(integer(text)?),
            DataType::Real => Self::Real(float(text)?),
            DataType::Double => Self::Double(float(text)?),
            DataType::Decimal { precision, scale } => Self::Decimal {
                value: owned(),
                precision,
                scale,
            },
            DataType::String => Self::String(owned()),
            DataType::VarChar { length } => Self::VarChar {
                value: owned(),
                length,
            },
            DataType::FixedChar { length } => Self::FixedChar {
                value: owned(),
                length,
            },
            DataType::Date => Self::Date(owned()),
            DataType::Time { precision } => Self::Time {
                value: owned(),
                precision,
            },
            DataType::Timestamp { precision } => Self::Timestamp {
                value: owned(),
                precision,
            },
            DataType::Interval => Self::Interval(owned()),
            DataType::IntervalYearMonth => Self::IntervalYearMonth(owned()),
            DataType::IntervalDaySecond { precision } => Self::IntervalDaySecond {
                value: owned(),
                precision,
            },
            DataType::Uuid => Self::Uuid(owned()),
            DataType::Binary | DataType::FixedBinary { .. } | DataType::Array(_) => {
                return Err(NoReading)
            }
            DataType::Unknown => return Err(LiteralError::InvalidType),
        };
        literal.check_fit()?;
        Ok(literal)
    }

    pub fn held_value(&self) -> Result<Option<HeldValue>, LiteralError> {
        self.check_fit()?;
        Ok(Some(match self {
            Self::Date(value) => HeldValue::Date(narrow(date(value)?)?),
            Self::Time { value, precision } => {
                let (seconds, fraction) = time(value)?;
                HeldValue::Time {
                    units: units(1, seconds.into(), fraction, *precision)?,
                    precision: *precision,
                }
            }
            Self::Timestamp { value, precision } => HeldValue::Timestamp {
                units: timestamp(value, *precision)?,
                precision: *precision,
            },
            Self::Decimal {
                value,
                precision,
                scale,
            } => {
                let digits = u32::try_from(*scale).map_err(|_| CutsADigit)?;
                HeldValue::Decimal {
                    unscaled: Number::read(value)?.unscaled(digits).ok_or(CutsADigit)?,
                    precision: *precision,
                    scale: *scale,
                }
            }
            Self::Interval(value) => {
                let (months, days, nanoseconds) = interval(value, &self.data_type())?;
                HeldValue::Interval {
                    months,
                    days,
                    nanoseconds,
                }
            }
            Self::IntervalYearMonth(value) => HeldValue::IntervalYearMonth {
                months: interval(value, &self.data_type())?.0,
            },
            Self::IntervalDaySecond { value, precision } => {
                let (_, days, units) = interval(value, &self.data_type())?;
                HeldValue::IntervalDaySecond {
                    days,
                    units,
                    precision: *precision,
                }
            }
            Self::Uuid(value) => HeldValue::Uuid(uuid(value)?),
            Self::Boolean(_)
            | Self::TinyInt(_)
            | Self::SmallInt(_)
            | Self::Integer(_)
            | Self::BigInt(_)
            | Self::Real(_)
            | Self::Double(_)
            | Self::String(_)
            | Self::VarChar { .. }
            | Self::FixedChar { .. }
            | Self::Binary(_)
            | Self::FixedBinary { .. }
            | Self::Array { .. }
            | Self::Null(_) => return Ok(None),
        }))
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeldValue {
    Date(i32),
    Time {
        units: i64,
        precision: u8,
    },
    Timestamp {
        units: i64,
        precision: u8,
    },
    Decimal {
        unscaled: i128,
        precision: u8,
        scale: i8,
    },
    Interval {
        months: i32,
        days: i32,
        nanoseconds: i64,
    },
    IntervalYearMonth {
        months: i32,
    },
    IntervalDaySecond {
        days: i32,
        units: i64,
        precision: u8,
    },
    Uuid(u128),
}

#[derive(Debug)]
pub(super) struct Number<'a> {
    negative: bool,
    integer: &'a str,
    fraction: &'a str,
}

impl<'a> Number<'a> {
    pub(super) fn read(text: &'a str) -> Result<Self, LiteralError> {
        let (negative, unsigned) = match text.strip_prefix('-') {
            Some(unsigned) => (true, unsigned),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };
        let (integer, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
        if integer.len() + fraction.len() == 0 || !digits(integer) || !digits(fraction) {
            return Err(Malformed);
        }
        Ok(Self {
            negative,
            integer: integer.trim_start_matches('0'),
            fraction: fraction.trim_end_matches('0'),
        })
    }

    pub(super) fn data_type(&self) -> DataType {
        let digits = self.integer.len() + self.fraction.len();
        DataType::Decimal {
            precision: u8::try_from(digits.max(1)).unwrap_or(u8::MAX),
            scale: i8::try_from(self.fraction.len()).unwrap_or(i8::MAX),
        }
    }

    pub(super) fn unscaled(&self, scale: u32) -> Option<i128> {
        let padding = usize::try_from(scale)
            .ok()?
            .checked_sub(self.fraction.len())?;
        let digits = format!("0{}{}{}", self.integer, self.fraction, "0".repeat(padding));
        let magnitude: i128 = digits.parse().ok()?;
        Some(if self.negative { -magnitude } else { magnitude })
    }

    pub(super) fn fits(&self, precision: u8, scale: i8) -> Result<(), LiteralError> {
        let scale = usize::try_from(scale).map_err(|_| CutsADigit)?;
        ensure(
            self.fraction.len() <= scale && self.integer.len() + scale <= usize::from(precision),
            CutsADigit,
        )
    }
}

fn units(sign: i128, seconds: i128, fraction: &str, precision: u8) -> Result<i64, LiteralError> {
    fraction_fits(fraction, precision)?;
    let subsecond = Number {
        negative: false,
        integer: "",
        fraction,
    };
    let fraction = subsecond.unscaled(precision.into()).ok_or(OutOfRange)?;
    (10_i128.checked_pow(u32::from(precision)))
        .and_then(|unit| seconds.checked_mul(unit)?.checked_add(fraction))
        .and_then(|units| i64::try_from(sign * units).ok())
        .ok_or(OutOfRange)
}

fn narrow(number: impl TryInto<i32>) -> Result<i32, LiteralError> {
    number.try_into().map_err(|_| OutOfRange)
}

pub(super) fn ensure(holds: bool, error: LiteralError) -> Result<(), LiteralError> {
    if holds {
        Ok(())
    } else {
        Err(error)
    }
}

fn integer<Integer: TryFrom<i128>>(text: &str) -> Result<Integer, LiteralError> {
    let number = Number::read(text)?;
    ensure(number.fraction.is_empty(), CutsADigit)?;
    let integer = number.unscaled(0).ok_or(OutOfRange)?;
    Integer::try_from(integer).map_err(|_| OutOfRange)
}

fn float<Float: FromStr + ToString>(text: &str) -> Result<Float, LiteralError> {
    let mantissa = text
        .split_once(['e', 'E'])
        .map_or(text, |(mantissa, _)| mantissa);
    let number = Number::read(mantissa)?;
    let value: Float = text.parse().map_err(|_| Malformed)?;
    let significant = |number: &Number| {
        [number.integer, number.fraction]
            .concat()
            .trim_matches('0')
            .to_owned()
    };
    match Number::read(&value.to_string()) {
        Ok(read) if significant(&read) == significant(&number) => Ok(value),
        Ok(_) => Err(CutsADigit),
        Err(_) => Err(OutOfRange),
    }
}

fn digits(text: &str) -> bool {
    text.bytes().all(|byte| byte.is_ascii_digit())
}

fn fields(text: &str, separator: char, widths: &[usize]) -> Option<Vec<u32>> {
    let parts: Vec<&str> = text.split(separator).collect();
    (parts.len() == widths.len()).then_some(())?;
    parts
        .iter()
        .zip(widths)
        .map(|(part, &width)| (part.len() == width && digits(part)).then(|| part.parse().ok())?)
        .collect()
}

pub(super) fn date(text: &str) -> Result<i64, LiteralError> {
    let Some(&[year, month, day]) = fields(text, '-', &[4, 2, 2]).as_deref() else {
        return Err(Malformed);
    };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        2 => 28 + u32::from(leap),
        4 | 6 | 9 | 11 => 30,
        1..=12 => 31,
        _ => 0,
    };
    ensure(year > 0 && (1..=days_in_month).contains(&day), Malformed)?;
    let (year, month, day) = (
        i64::from(year) - i64::from(month <= 2),
        i64::from(month),
        i64::from(day),
    );
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    Ok(era * 146_097 + day_of_era - 719_468)
}

fn split(text: &str, separator: char) -> Result<(&str, &str), LiteralError> {
    match text.split_once(separator) {
        Some((_, "")) => Err(Malformed),
        Some(parts) => Ok(parts),
        None => Ok((text, "")),
    }
}

pub(super) fn time(text: &str) -> Result<(i64, &str), LiteralError> {
    let (clock, fraction) = split(text, '.')?;
    let Some(&[hour, minute, second]) = fields(clock, ':', &[2, 2, 2]).as_deref() else {
        return Err(Malformed);
    };
    ensure(
        hour < 24 && minute < 60 && second < 60 && digits(fraction),
        Malformed,
    )?;
    Ok((
        i64::from(hour * 3_600 + minute * 60 + second),
        fraction.trim_end_matches('0'),
    ))
}

pub(super) fn fraction_fits(fraction: &str, precision: u8) -> Result<(), LiteralError> {
    ensure(fraction.len() <= usize::from(precision), CutsADigit)
}

pub(super) fn timestamp(text: &str, precision: u8) -> Result<i64, LiteralError> {
    let (Some(day), Some(rest)) = (text.get(..10), text.get(10..)) else {
        return Err(Malformed);
    };
    let days = date(day)?;
    let (seconds, fraction) = match rest.strip_prefix(['T', ' ']) {
        None if rest.is_empty() => (0, ""),
        None => return Err(Malformed),
        Some(clock) if clock.contains(['Z', 'z', '+', '-']) => return Err(Zoned),
        Some(clock) => time(clock)?,
    };
    units(
        1,
        i128::from(days) * 86_400 + i128::from(seconds),
        fraction,
        precision,
    )
}

pub(super) fn interval(text: &str, data_type: &DataType) -> Result<(i32, i32, i64), LiteralError> {
    let (year_month, day_time, precision) = match *data_type {
        DataType::Interval => (true, true, 9),
        DataType::IntervalYearMonth => (true, false, 0),
        DataType::IntervalDaySecond { precision } => (false, true, precision),
        _ => (false, false, 0),
    };
    let (sign, body) = match text.strip_prefix('-') {
        Some(body) => (-1_i128, body),
        None => (1, text),
    };
    let (date_part, time_part) = split(body.strip_prefix('P').ok_or(Malformed)?, 'T')?;
    let [years, months, days] = designated(date_part, ['Y', 'M', 'D'])?;
    let [hours, minutes, seconds] = designated(time_part, ['H', 'M', 'S'])?;
    let (seconds, fraction) = match seconds {
        Some(seconds) => split(seconds, '.').map(|(whole, fraction)| (Some(whole), fraction))?,
        None => (None, ""),
    };
    let numbers = [years, months, days, hours, minutes, seconds];
    let has_year_month = numbers[..2].iter().any(Option::is_some);
    let has_day_time = numbers[2..].iter().any(Option::is_some);
    let whole = |number: &&str| !number.is_empty() && digits(number);
    ensure(
        (has_year_month || has_day_time)
            && numbers.iter().flatten().all(whole)
            && digits(fraction)
            && (year_month || !has_year_month)
            && (day_time || !has_day_time),
        Malformed,
    )?;
    let count = |number: Option<&str>| {
        number.map_or(Ok(0), |number| {
            number
                .parse::<i64>()
                .map(i128::from)
                .map_err(|_| OutOfRange)
        })
    };
    let clock = count(hours)? * 3_600 + count(minutes)? * 60 + count(seconds)?;
    narrow(sign * clock)?;
    Ok((
        narrow(sign * (count(years)? * 12 + count(months)?))?,
        narrow(sign * count(days)?)?,
        units(sign, clock, fraction.trim_end_matches('0'), precision)?,
    ))
}

fn designated<const N: usize>(
    mut text: &str,
    designators: [char; N],
) -> Result<[Option<&str>; N], LiteralError> {
    let numbers = designators.map(|designator| {
        let (number, rest) = text.split_once(designator)?;
        text = rest;
        Some(number)
    });
    ensure(text.is_empty(), Malformed).map(|()| numbers)
}

pub(super) fn uuid(text: &str) -> Result<u128, LiteralError> {
    ensure(
        text.split('-').map(str::len).eq([8, 4, 4, 4, 12])
            && text
                .bytes()
                .all(|byte| byte == b'-' || byte.is_ascii_hexdigit()),
        Malformed,
    )?;
    u128::from_str_radix(&text.replace('-', ""), 16).map_err(|_| Malformed)
}
