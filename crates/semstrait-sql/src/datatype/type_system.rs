use super::family::one_family;
use super::{DataType, Limit, TypeFamily, Values};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Decimal {
    pub precision: u8,
    pub scale: i8,
}

impl From<Decimal> for DataType {
    fn from(Decimal { precision, scale }: Decimal) -> Self {
        Self::Decimal { precision, scale }
    }
}

const LEAST_SCALE: i8 = 6;

fn greatest_precision() -> u8 {
    match DataType::decimal()
        .declarations()
        .and_then(|declarations| declarations.canonical.precision)
    {
        Some(Limit {
            values: Values::Range(_, greatest),
            ..
        }) => u8::try_from(greatest).unwrap_or(u8::MAX),
        _ => u8::MAX,
    }
}

fn widest_decimal(scale: i8) -> DataType {
    DataType::Decimal {
        precision: greatest_precision(),
        scale,
    }
}

impl Decimal {
    pub fn of(data_type: &DataType) -> Option<Self> {
        match *data_type {
            DataType::Decimal { precision, scale } => Some(Self { precision, scale }),
            _ => Some(Self {
                precision: data_type.declarations()?.canonical.digits?,
                scale: 0,
            }),
        }
    }

    fn parts(self) -> (i16, i16) {
        (self.precision.into(), self.scale.into())
    }

    fn capped(precision: i16, scale: i16) -> Self {
        let greatest = i16::from(greatest_precision());
        let excess = (precision - greatest).max(0);
        Self {
            precision: u8::try_from(precision.min(greatest)).unwrap_or_default(),
            scale: i8::try_from((scale - excess).max(scale.min(LEAST_SCALE.into())))
                .unwrap_or_default(),
        }
    }
}

pub trait TypeSystem {
    fn decimal_union(&self, a: Decimal, b: Decimal) -> Decimal {
        let ((p1, s1), (p2, s2)) = (a.parts(), b.parts());
        let scale = s1.max(s2);
        Decimal::capped((p1 - s1).max(p2 - s2) + scale, scale)
    }

    fn decimal_sum(&self, a: Decimal, b: Decimal) -> Decimal {
        let ((p1, s1), (p2, s2)) = (a.parts(), b.parts());
        let scale = s1.max(s2);
        Decimal::capped(scale + (p1 - s1).max(p2 - s2) + 1, scale)
    }

    fn decimal_product(&self, a: Decimal, b: Decimal) -> Decimal {
        let ((p1, s1), (p2, s2)) = (a.parts(), b.parts());
        Decimal::capped(p1 + p2 + 1, s1 + s2)
    }

    fn decimal_quotient(&self, a: Decimal, b: Decimal) -> Decimal {
        let ((p1, s1), (p2, _)) = (a.parts(), b.parts());
        let scale = (s1 + p2 + 1).max(LEAST_SCALE.into());
        Decimal::capped(p1 - s1 + p2 + scale, scale)
    }

    fn decimal_mod(&self, a: Decimal, b: Decimal) -> Decimal {
        let ((p1, s1), (p2, s2)) = (a.parts(), b.parts());
        let scale = s1.max(s2);
        Decimal::capped((p1 - s1).min(p2 - s2) + scale, scale)
    }

    fn sum_of(&self, arg: &DataType) -> DataType {
        match *arg {
            DataType::TinyInt | DataType::SmallInt | DataType::Integer | DataType::BigInt => {
                DataType::BigInt
            }
            DataType::Real | DataType::Double => DataType::Double,
            DataType::Decimal { scale, .. } => widest_decimal(scale),
            _ => arg.clone(),
        }
    }

    fn avg_of(&self, arg: &DataType) -> DataType {
        match *arg {
            DataType::TinyInt
            | DataType::SmallInt
            | DataType::Integer
            | DataType::BigInt
            | DataType::Real
            | DataType::Double => DataType::Double,
            DataType::Decimal { scale, .. } => widest_decimal(scale.max(LEAST_SCALE)),
            _ => arg.clone(),
        }
    }

    fn decimal_scale_zero(&self, arg: Decimal) -> Decimal {
        let (precision, scale) = arg.parts();
        Decimal::capped(precision - scale + 1, 0)
    }

    fn round_of(&self, arg: Decimal) -> Decimal {
        Decimal {
            precision: arg.precision.saturating_add(1).min(greatest_precision()),
            ..arg
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CanonicalTypeSystem;

impl TypeSystem for CanonicalTypeSystem {}

#[derive(Debug, Clone, Copy)]
pub struct SubstraitTypeSystem;

impl TypeSystem for SubstraitTypeSystem {
    fn avg_of(&self, arg: &DataType) -> DataType {
        match *arg {
            DataType::Decimal { scale, .. } => widest_decimal(scale),
            _ => arg.clone(),
        }
    }
}

pub fn promote(system: &dyn TypeSystem, types: &[DataType]) -> DataType {
    let types = (types.iter())
        .map(|data_type| (data_type.clone(), None))
        .collect();
    promoted(types, |a, b| system.decimal_union(a, b)).0
}

pub(crate) fn first(
    arguments: Vec<(DataType, Option<TypeFamily>)>,
    derive: impl FnOnce(DataType) -> DataType,
) -> (DataType, Option<TypeFamily>) {
    arguments
        .into_iter()
        .next()
        .map_or((DataType::Unknown, None), |(data_type, family)| {
            (derive(data_type), family)
        })
}

pub(crate) fn arithmetic(
    operands: Vec<(DataType, Option<TypeFamily>)>,
    decimal_rule: impl Fn(Decimal, Decimal) -> Decimal,
) -> (DataType, Option<TypeFamily>) {
    let temporal = |(_, family): &&(DataType, Option<TypeFamily>)| {
        matches!(
            family,
            Some(TypeFamily::DateAndTimestamp | TypeFamily::Time)
        )
    };
    match operands.iter().find(temporal) {
        Some(operand) => operand.clone(),
        None => promoted(operands, decimal_rule),
    }
}

pub(crate) fn promoted(
    arguments: Vec<(DataType, Option<TypeFamily>)>,
    decimal_rule: impl Fn(Decimal, Decimal) -> Decimal,
) -> (DataType, Option<TypeFamily>) {
    let family = one_family(arguments.iter().map(|(_, family)| family.as_ref()));
    let decimal = |data_type: &DataType| matches!(data_type, DataType::Decimal { .. });
    let data_type = arguments
        .into_iter()
        .map(|(data_type, _)| data_type)
        .reduce(
            |left, right| match (Decimal::of(&left), Decimal::of(&right)) {
                (Some(a), Some(b)) if decimal(&left) || decimal(&right) => {
                    decimal_rule(a, b).into()
                }
                _ => left.wider(right).unwrap_or(DataType::Unknown),
            },
        )
        .unwrap_or(DataType::Unknown);
    (data_type, family)
}
