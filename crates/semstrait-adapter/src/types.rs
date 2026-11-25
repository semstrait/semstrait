use semstrait_ir::Schema;
use semstrait_sql::datatype::DataType;
use substrait::proto::r#type::{self, Kind, Nullability};
use substrait::proto::{NamedStruct, Type};

use crate::error::InvalidRewrite;
use crate::rules::SubstraitRules;

pub(crate) fn named_struct(
    schema: &Schema,
    rules: &dyn SubstraitRules,
) -> Result<NamedStruct, InvalidRewrite> {
    let types = (schema.columns.iter())
        .map(|field| proto_type(&field.data_type, field.nullable, rules))
        .collect::<Result<_, _>>()?;
    Ok(NamedStruct {
        names: (schema.columns.iter())
            .map(|field| field.name.clone())
            .collect(),
        r#struct: Some(r#type::Struct {
            types,
            type_variation_reference: 0,
            nullability: Nullability::Required.into(),
        }),
    })
}

pub(crate) fn proto_type(
    data_type: &DataType,
    nullable: bool,
    rules: &dyn SubstraitRules,
) -> Result<Type, InvalidRewrite> {
    let type_variation_reference = rules.type_variation(data_type);
    let nullability = if nullable {
        Nullability::Nullable
    } else {
        Nullability::Required
    }
    .into();
    let no_form = || InvalidRewrite::NoSubstraitType {
        data_type: data_type.clone(),
    };
    let length = |length: &u32| i32::try_from(*length).map_err(|_| no_form());
    macro_rules! kind {
        ($kind:ident($message:ident { $($field:ident: $value:expr),* })) => {
            Kind::$kind(r#type::$message {
                $($field: $value,)*
                type_variation_reference,
                nullability,
            })
        };
    }
    let kind = match data_type {
        DataType::Boolean => kind!(Bool(Boolean {})),
        DataType::TinyInt => kind!(I8(I8 {})),
        DataType::SmallInt => kind!(I16(I16 {})),
        DataType::Integer => kind!(I32(I32 {})),
        DataType::BigInt => kind!(I64(I64 {})),
        DataType::Real => kind!(Fp32(Fp32 {})),
        DataType::Double => kind!(Fp64(Fp64 {})),
        DataType::Decimal { precision, scale } => kind!(Decimal(Decimal {
            precision: (*precision).into(),
            scale: (*scale).into()
        })),
        DataType::String => kind!(String(String {})),
        DataType::VarChar { length: n } => kind!(Varchar(VarChar { length: length(n)? })),
        DataType::FixedChar { length: n } => kind!(FixedChar(FixedChar { length: length(n)? })),
        DataType::Binary => kind!(Binary(Binary {})),
        DataType::FixedBinary { length: n } => {
            kind!(FixedBinary(FixedBinary { length: length(n)? }))
        }
        DataType::Date => kind!(Date(Date {})),
        DataType::Time { precision } => kind!(PrecisionTime(PrecisionTime {
            precision: (*precision).into()
        })),
        DataType::Timestamp { precision } => kind!(PrecisionTimestamp(PrecisionTimestamp {
            precision: (*precision).into()
        })),
        DataType::Interval => kind!(IntervalCompound(IntervalCompound { precision: 9 })),
        DataType::IntervalYearMonth => kind!(IntervalYear(IntervalYear {})),
        DataType::IntervalDaySecond { precision } => kind!(IntervalDay(IntervalDay {
            precision: Some((*precision).into())
        })),
        DataType::Uuid => kind!(Uuid(Uuid {})),
        DataType::Array(element) => Kind::List(Box::new(list(
            element,
            type_variation_reference,
            nullability,
            rules,
        )?)),
        _ => return Err(no_form()),
    };
    Ok(Type { kind: Some(kind) })
}

pub(crate) fn list(
    element: &DataType,
    type_variation_reference: u32,
    nullability: i32,
    rules: &dyn SubstraitRules,
) -> Result<r#type::List, InvalidRewrite> {
    Ok(r#type::List {
        r#type: Some(Box::new(proto_type(element, true, rules)?)),
        type_variation_reference,
        nullability,
    })
}

pub(crate) fn zoned(timestamp: Type) -> Type {
    let kind = match timestamp.kind {
        Some(Kind::PrecisionTimestamp(r#type::PrecisionTimestamp {
            precision,
            type_variation_reference,
            nullability,
        })) => Some(Kind::PrecisionTimestampTz(r#type::PrecisionTimestampTz {
            precision,
            type_variation_reference,
            nullability,
        })),
        kind => kind,
    };
    Type { kind }
}
