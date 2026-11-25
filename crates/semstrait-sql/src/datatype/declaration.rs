use semstrait_common::Dialect;

use super::DataType;
use crate::DocumentDialect;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TypeDeclarations {
    pub canonical: TypeDeclaration,
    pub dialects: &'static [(Dialect, Option<TypeDeclaration>)],
}

impl TypeDeclarations {
    pub fn in_document(
        &'static self,
        dialect: DocumentDialect,
    ) -> Option<&'static TypeDeclaration> {
        match dialect {
            DocumentDialect::Canonical => Some(&self.canonical),
            DocumentDialect::Dialect(dialect) => self.on(dialect),
        }
    }

    pub fn on(&'static self, dialect: Dialect) -> Option<&'static TypeDeclaration> {
        crate::on_dialect(&self.canonical, self.dialects, dialect)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeDeclaration {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub precision: Option<Limit>,
    pub scale: Option<Limit>,
    pub length: Option<Limit>,
    pub digits: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Limit {
    pub values: Values,
    pub default: Option<u32>,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Values {
    Range(u32, u32),
    OneOf(&'static [u32]),
}

impl Values {
    pub(super) fn contains(self, value: u32) -> bool {
        match self {
            Self::Range(low, high) => (low..=high).contains(&value),
            Self::OneOf(values) => values.contains(&value),
        }
    }
}

impl TypeDeclaration {
    pub fn limits(&self) -> impl Iterator<Item = (&'static str, &'static str, Limit)> {
        [
            ("p", "precision", self.precision),
            ("s", "scale", self.scale),
            ("n", "length", self.length),
        ]
        .into_iter()
        .filter_map(|(letter, key, limit)| Some((letter, key, limit?)))
    }

    const fn in_dialect(self, name: &'static str) -> Self {
        Self {
            name,
            aliases: &[],
            ..self
        }
    }
}

const fn named(name: &'static str, aliases: &'static [&'static str]) -> TypeDeclaration {
    TypeDeclaration {
        name,
        aliases,
        precision: None,
        scale: None,
        length: None,
        digits: None,
    }
}

const fn limit(values: Values, default: u32) -> Option<Limit> {
    Some(Limit {
        values,
        default: Some(default),
    })
}

const UNITS: Values = Values::OneOf(&[0, 3, 6, 9]);

const LENGTH: Option<Limit> = Some(Limit {
    values: Values::Range(1, 2_147_483_647),
    default: None,
});

macro_rules! declarations {
    (@on $canonical:expr, none) => {
        None
    };
    (@on $canonical:expr, $name:literal) => {
        Some($canonical.in_dialect($name))
    };
    (@on $canonical:expr, ($name:literal, $precisions:expr, $default:expr)) => {
        Some(TypeDeclaration {
            precision: limit($precisions, $default),
            ..$canonical.in_dialect($name)
        })
    };
    ($canonical:expr $(, $dialect:ident: $name:tt)*) => {
        TypeDeclarations {
            canonical: $canonical,
            dialects: &[$((Dialect::$dialect, declarations!(@on $canonical, $name))),*],
        }
    };
}

impl DataType {
    const BOOLEAN: TypeDeclarations = declarations!(named("boolean", &["bool"]));
    const TINYINT: TypeDeclarations = declarations!(
        TypeDeclaration { digits: Some(3), ..named("tinyint", &["byte", "i8"]) },
        Substrait: "i8", Iceberg: none
    );
    const SMALLINT: TypeDeclarations = declarations!(
        TypeDeclaration { digits: Some(5), ..named("smallint", &["short", "int16", "i16"]) },
        Substrait: "i16", Iceberg: none
    );
    const INTEGER: TypeDeclarations = declarations!(
        TypeDeclaration { digits: Some(10), ..named("integer", &["int", "int32", "i32"]) },
        Substrait: "i32", Iceberg: "int"
    );
    const BIGINT: TypeDeclarations = declarations!(
        TypeDeclaration { digits: Some(19), ..named("bigint", &["long", "int64", "i64"]) },
        Substrait: "i64", Iceberg: "long"
    );
    const REAL: TypeDeclarations = declarations!(
        named("real", &["float", "float32", "f32"]),
        Substrait: "fp32", Iceberg: "float"
    );
    const DOUBLE: TypeDeclarations =
        declarations!(named("double", &["float64", "f64"]), Substrait: "fp64");
    const DECIMAL: TypeDeclarations = declarations!(TypeDeclaration {
        precision: limit(Values::Range(1, 38), 38),
        scale: limit(Values::Range(0, 38), 9),
        ..named("decimal", &["numeric"])
    });
    const STRING: TypeDeclarations =
        declarations!(named("string", &["text", "utf8", "large_utf8"]));
    const VARCHAR: TypeDeclarations = declarations!(
        TypeDeclaration { length: LENGTH, ..named("varchar", &[]) },
        Iceberg: none
    );
    const FIXEDCHAR: TypeDeclarations = declarations!(
        TypeDeclaration { length: LENGTH, ..named("char", &[]) },
        Substrait: "fixedchar", Iceberg: none
    );
    const BINARY: TypeDeclarations =
        declarations!(named("binary", &["blob", "bytes", "varbinary"]));
    const FIXEDBINARY: TypeDeclarations = declarations!(
        TypeDeclaration { length: LENGTH, ..named("binary", &[]) },
        Substrait: "fixedbinary", Iceberg: "fixed"
    );
    const DATE: TypeDeclarations = declarations!(named("date", &["date32", "date64"]));
    const TIME: TypeDeclarations = declarations!(
        TypeDeclaration { precision: limit(UNITS, 9), ..named("time", &[]) },
        Substrait: "precision_time", Iceberg: ("time", Values::OneOf(&[6]), 6)
    );
    const TIMESTAMP: TypeDeclarations = declarations!(
        TypeDeclaration { precision: limit(UNITS, 6), ..named("timestamp", &["datetime"]) },
        Substrait: "precision_timestamp", Iceberg: ("timestamp", Values::OneOf(&[6, 9]), 6)
    );
    const INTERVAL: TypeDeclarations = declarations!(
        named("interval", &[]),
        Substrait: "interval_compound", Iceberg: none
    );
    const INTERVAL_YEAR_MONTH: TypeDeclarations = declarations!(
        named("interval_year_month", &[]),
        Substrait: "interval_year", Iceberg: none
    );
    const INTERVAL_DAY_SECOND: TypeDeclarations = declarations!(
        TypeDeclaration {
            precision: limit(Values::Range(0, 9), 9),
            ..named("interval_day_second", &[])
        },
        Substrait: "interval_day", Iceberg: none
    );
    const UUID: TypeDeclarations = declarations!(named("uuid", &[]));
    const ARRAY: TypeDeclarations = declarations!(
        named("array", &[]),
        Substrait: "list", Iceberg: "list"
    );

    pub(super) fn array_declaration(dialect: DocumentDialect) -> Option<&'static TypeDeclaration> {
        Self::Array(Box::new(Self::Unknown))
            .declarations()?
            .in_document(dialect)
    }

    #[deny(clippy::wildcard_enum_match_arm)]
    pub fn declarations(&self) -> Option<&'static TypeDeclarations> {
        Some(match self {
            Self::Boolean => &Self::BOOLEAN,
            Self::TinyInt => &Self::TINYINT,
            Self::SmallInt => &Self::SMALLINT,
            Self::Integer => &Self::INTEGER,
            Self::BigInt => &Self::BIGINT,
            Self::Real => &Self::REAL,
            Self::Double => &Self::DOUBLE,
            Self::Decimal { .. } => &Self::DECIMAL,
            Self::String => &Self::STRING,
            Self::VarChar { .. } => &Self::VARCHAR,
            Self::FixedChar { .. } => &Self::FIXEDCHAR,
            Self::Binary => &Self::BINARY,
            Self::FixedBinary { .. } => &Self::FIXEDBINARY,
            Self::Date => &Self::DATE,
            Self::Time { .. } => &Self::TIME,
            Self::Timestamp { .. } => &Self::TIMESTAMP,
            Self::Interval => &Self::INTERVAL,
            Self::IntervalYearMonth => &Self::INTERVAL_YEAR_MONTH,
            Self::IntervalDaySecond { .. } => &Self::INTERVAL_DAY_SECOND,
            Self::Uuid => &Self::UUID,
            Self::Array(_) => &Self::ARRAY,
            Self::Unknown => return None,
        })
    }
}
