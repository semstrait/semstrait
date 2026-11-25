use semstrait_common::Dialect;
use semstrait_sql::datatype::{
    DataType, Limit, TypeDeclaration, TypeDeclarations, TypeFamily, Values,
};
use semstrait_sql::expr::check_cast;

fn every_type() -> Vec<DataType> {
    vec![
        DataType::Boolean,
        DataType::TinyInt,
        DataType::SmallInt,
        DataType::Integer,
        DataType::BigInt,
        DataType::Real,
        DataType::Double,
        DataType::Decimal {
            precision: 18,
            scale: 2,
        },
        DataType::String,
        DataType::VarChar { length: 10 },
        DataType::FixedChar { length: 5 },
        DataType::Binary,
        DataType::FixedBinary { length: 16 },
        DataType::Date,
        DataType::Time { precision: 9 },
        DataType::Timestamp { precision: 6 },
        DataType::Interval,
        DataType::IntervalYearMonth,
        DataType::IntervalDaySecond { precision: 9 },
        DataType::Uuid,
        DataType::Array(Box::new(DataType::String)),
    ]
}

fn canonical(data_type: &DataType) -> &'static TypeDeclaration {
    &data_type
        .declarations()
        .unwrap_or_else(|| panic!("{data_type:?} has declarations"))
        .canonical
}

fn on(data_type: &DataType, dialect: Dialect) -> &'static TypeDeclaration {
    data_type
        .declarations()
        .and_then(|declarations| declarations.on(dialect))
        .unwrap_or_else(|| panic!("{dialect:?} has {data_type:?}"))
}

#[test]
fn substrait_names_each_type_of_rule_6s_table_in_its_own_words() {
    let table = [
        (DataType::TinyInt, "i8"),
        (DataType::SmallInt, "i16"),
        (DataType::Integer, "i32"),
        (DataType::BigInt, "i64"),
        (DataType::Real, "fp32"),
        (DataType::Double, "fp64"),
        (DataType::FixedChar { length: 5 }, "fixedchar"),
        (DataType::FixedBinary { length: 16 }, "fixedbinary"),
        (DataType::Time { precision: 9 }, "precision_time"),
        (DataType::Timestamp { precision: 6 }, "precision_timestamp"),
        (DataType::Interval, "interval_compound"),
        (DataType::IntervalYearMonth, "interval_year"),
        (DataType::IntervalDaySecond { precision: 9 }, "interval_day"),
        (DataType::Array(Box::new(DataType::String)), "list"),
    ];
    for (data_type, name) in table {
        let substrait = on(&data_type, Dialect::Substrait);
        let canonical = canonical(&data_type);
        assert_eq!(substrait.name, name, "{data_type:?}");
        assert!(
            substrait.aliases.is_empty(),
            "{data_type:?}: a dialect declaration has no alias"
        );
        assert_eq!(
            (
                substrait.precision,
                substrait.scale,
                substrait.length,
                substrait.digits
            ),
            (
                canonical.precision,
                canonical.scale,
                canonical.length,
                canonical.digits
            ),
            "{data_type:?}: Substrait takes every canonical value, so it gives a name only",
        );
    }
}

#[test]
fn substrait_gives_the_canonical_declaration_for_every_type_its_table_leaves_out() {
    let left_out = [
        DataType::Boolean,
        DataType::Decimal {
            precision: 18,
            scale: 2,
        },
        DataType::String,
        DataType::VarChar { length: 10 },
        DataType::Binary,
        DataType::Date,
        DataType::Uuid,
    ];
    for data_type in left_out {
        assert_eq!(
            on(&data_type, Dialect::Substrait),
            canonical(&data_type),
            "{data_type:?}"
        );
    }
}

#[test]
fn datafusion_declares_no_data_type_so_it_gives_the_canonical_declaration_for_every_type() {
    for data_type in every_type() {
        assert_eq!(
            on(&data_type, Dialect::DataFusion),
            canonical(&data_type),
            "{data_type:?}"
        );
    }
}

#[test]
fn on_gives_no_declaration_where_the_dialect_has_no_such_type() {
    let declarations: &'static TypeDeclarations = Box::leak(Box::new(TypeDeclarations {
        canonical: *canonical(&DataType::Uuid),
        dialects: &[(Dialect::Substrait, None)],
    }));
    assert_eq!(declarations.on(Dialect::Substrait), None);
    assert_eq!(
        declarations.on(Dialect::DataFusion),
        Some(&declarations.canonical)
    );
}

const fn limit(values: Values, default: Option<u32>) -> Option<Limit> {
    Some(Limit { values, default })
}

const UNITS: Values = Values::OneOf(&[0, 3, 6, 9]);

#[test]
fn each_precision_scale_and_length_takes_rule_2s_values_and_bare_default() {
    let decimal = canonical(&DataType::decimal());
    assert_eq!(decimal.precision, limit(Values::Range(1, 38), Some(38)));
    assert_eq!(decimal.scale, limit(Values::Range(0, 38), Some(9)));
    assert_eq!(decimal.length, None);

    let precisions = [
        (DataType::time(), limit(UNITS, Some(9))),
        (DataType::timestamp(), limit(UNITS, Some(6))),
        (
            DataType::interval_day_second(),
            limit(Values::Range(0, 9), Some(9)),
        ),
    ];
    for (data_type, precision) in precisions {
        let declaration = canonical(&data_type);
        assert_eq!(declaration.precision, precision, "{data_type:?}");
        assert_eq!((declaration.scale, declaration.length), (None, None));
    }

    let length = limit(Values::Range(1, 2_147_483_647), None);
    for data_type in [
        DataType::VarChar { length: 10 },
        DataType::FixedChar { length: 5 },
        DataType::FixedBinary { length: 16 },
    ] {
        let declaration = canonical(&data_type);
        assert_eq!(declaration.length, length, "{data_type:?}");
        assert_eq!((declaration.precision, declaration.scale), (None, None));
    }
}

#[test]
fn a_bare_constructor_gives_the_type_at_its_declarations_default() {
    assert_eq!(
        DataType::decimal(),
        DataType::Decimal {
            precision: 38,
            scale: 9,
        },
    );
    assert_eq!(DataType::time(), DataType::Time { precision: 9 });
    assert_eq!(DataType::timestamp(), DataType::Timestamp { precision: 6 });
    assert_eq!(
        DataType::interval_day_second(),
        DataType::IntervalDaySecond { precision: 9 },
    );
}

#[test]
fn an_integer_has_fixed_digits_and_no_other_type_has_any() {
    let integers = [
        (DataType::TinyInt, 3),
        (DataType::SmallInt, 5),
        (DataType::Integer, 10),
        (DataType::BigInt, 19),
    ];
    for data_type in every_type() {
        let digits = integers
            .iter()
            .find_map(|(integer, digits)| (*integer == data_type).then_some(*digits));
        assert_eq!(canonical(&data_type).digits, digits, "{data_type:?}");
    }
}

#[test]
fn a_value_inside_its_limit_is_valid() {
    let edges = [
        DataType::Decimal {
            precision: 1,
            scale: 0,
        },
        DataType::Decimal {
            precision: 38,
            scale: 38,
        },
        DataType::Time { precision: 0 },
        DataType::Time { precision: 3 },
        DataType::Timestamp { precision: 0 },
        DataType::Timestamp { precision: 9 },
        DataType::IntervalDaySecond { precision: 0 },
        DataType::VarChar { length: 1 },
        DataType::FixedChar {
            length: 2_147_483_647,
        },
        DataType::Array(Box::new(DataType::Timestamp { precision: 3 })),
    ];
    for data_type in edges.into_iter().chain(every_type()) {
        assert!(data_type.is_valid(), "{data_type:?}");
    }
}

#[test]
fn a_value_outside_its_limit_is_not_valid() {
    let outside = [
        DataType::Timestamp { precision: 4 },
        DataType::Time { precision: 12 },
        DataType::Decimal {
            precision: 40,
            scale: 2,
        },
        DataType::Decimal {
            precision: 0,
            scale: 0,
        },
        DataType::Decimal {
            precision: 10,
            scale: -1,
        },
        DataType::IntervalDaySecond { precision: 10 },
        DataType::VarChar { length: 0 },
        DataType::FixedChar {
            length: 2_147_483_648,
        },
        DataType::FixedBinary { length: 0 },
        DataType::Array(Box::new(DataType::Timestamp { precision: 4 })),
    ];
    for data_type in outside {
        assert!(!data_type.is_valid(), "{data_type:?}");
    }
}

#[test]
fn a_scale_above_its_precision_is_not_valid() {
    let at = DataType::Decimal {
        precision: 5,
        scale: 5,
    };
    let above = DataType::Decimal {
        precision: 5,
        scale: 6,
    };
    assert!(at.is_valid());
    assert!(!above.is_valid());
}

#[test]
fn a_type_is_valid_on_a_dialect_where_it_is_valid_and_the_dialect_takes_its_values() {
    for data_type in every_type() {
        assert!(data_type.is_valid_on(Dialect::Substrait), "{data_type:?}");
        assert!(data_type.is_valid_on(Dialect::DataFusion), "{data_type:?}");
    }
    let outside = DataType::Timestamp { precision: 4 };
    assert!(!outside.is_valid_on(Dialect::Substrait));
    assert!(!outside.is_valid_on(Dialect::DataFusion));
}

#[test]
fn each_type_writes_its_canonical_name() {
    let names = [
        (DataType::Boolean, "boolean"),
        (DataType::TinyInt, "tinyint"),
        (DataType::SmallInt, "smallint"),
        (DataType::Integer, "integer"),
        (DataType::BigInt, "bigint"),
        (DataType::Real, "real"),
        (DataType::Double, "double"),
        (
            DataType::Decimal {
                precision: 18,
                scale: 2,
            },
            "decimal(18,2)",
        ),
        (DataType::String, "string"),
        (DataType::VarChar { length: 10 }, "varchar(10)"),
        (DataType::FixedChar { length: 5 }, "char(5)"),
        (DataType::Binary, "binary"),
        (DataType::FixedBinary { length: 16 }, "binary(16)"),
        (DataType::Date, "date"),
        (DataType::Time { precision: 9 }, "time(9)"),
        (DataType::Timestamp { precision: 6 }, "timestamp(6)"),
        (DataType::Interval, "interval"),
        (DataType::IntervalYearMonth, "interval_year_month"),
        (
            DataType::IntervalDaySecond { precision: 9 },
            "interval_day_second(9)",
        ),
        (DataType::Uuid, "uuid"),
        (DataType::Array(Box::new(DataType::String)), "array<string>"),
        (
            DataType::Array(Box::new(DataType::Decimal {
                precision: 10,
                scale: 2,
            })),
            "array<decimal(10,2)>",
        ),
    ];
    assert_eq!(names.len(), every_type().len() + 1);
    for (data_type, name) in names {
        assert_eq!(data_type.to_string(), name);
        assert_eq!(DataType::from_name(name), Some(data_type), "{name}");
    }
}

#[test]
fn a_bare_spelling_reads_as_its_type_at_the_declarations_default() {
    let bare = [
        ("decimal", DataType::decimal()),
        ("time", DataType::time()),
        ("timestamp", DataType::timestamp()),
        ("interval_day_second", DataType::interval_day_second()),
        ("binary", DataType::Binary),
    ];
    for (name, data_type) in bare {
        assert_eq!(DataType::from_name(name), Some(data_type), "{name}");
    }
}

#[test]
fn each_alias_of_rule_7s_table_reads_as_its_type() {
    let table: [(DataType, &[&str]); 12] = [
        (DataType::Boolean, &["bool"]),
        (DataType::TinyInt, &["byte", "i8"]),
        (DataType::SmallInt, &["short", "int16", "i16"]),
        (DataType::Integer, &["int", "int32", "i32"]),
        (DataType::BigInt, &["long", "int64", "i64"]),
        (DataType::Real, &["float", "float32", "f32"]),
        (DataType::Double, &["float64", "f64"]),
        (DataType::decimal(), &["numeric"]),
        (DataType::String, &["text", "utf8", "large_utf8"]),
        (DataType::Binary, &["blob", "bytes", "varbinary"]),
        (DataType::Date, &["date32", "date64"]),
        (DataType::timestamp(), &["datetime"]),
    ];
    for (data_type, aliases) in &table {
        assert_eq!(canonical(data_type).aliases, *aliases, "{data_type:?}");
        for alias in *aliases {
            assert_eq!(
                DataType::from_name(alias),
                Some(data_type.clone()),
                "{alias}"
            );
        }
    }
    for data_type in every_type() {
        let declaration = canonical(&data_type);
        if !table
            .iter()
            .any(|(aliased, _)| canonical(aliased) == declaration)
        {
            assert!(declaration.aliases.is_empty(), "{data_type:?}");
        }
    }
}

#[test]
fn an_alias_takes_what_its_declaration_takes() {
    let spelled = [
        (
            "numeric(18,2)",
            DataType::Decimal {
                precision: 18,
                scale: 2,
            },
        ),
        ("datetime(3)", DataType::Timestamp { precision: 3 }),
        ("array<long>", DataType::Array(Box::new(DataType::BigInt))),
    ];
    for (name, data_type) in spelled {
        assert_eq!(DataType::from_name(name), Some(data_type), "{name}");
    }
    assert_eq!(DataType::from_name("varbinary(16)"), None);
}

#[test]
fn a_spelling_that_names_another_type_or_fixes_a_value_reads_as_no_type() {
    let refused = [
        "int8",
        "number",
        "number(10,2)",
        "varchar",
        "char",
        "timestamp_ms",
        "timestamp_us",
        "timestamp_ns",
    ];
    for name in refused {
        assert_eq!(DataType::from_name(name), None, "{name}");
    }
}

#[test]
fn a_spelling_outside_its_limits_reads_as_its_type_which_is_not_valid() {
    let outside = [
        ("timestamp(4)", DataType::Timestamp { precision: 4 }),
        (
            "decimal(40,2)",
            DataType::Decimal {
                precision: 40,
                scale: 2,
            },
        ),
        (
            "decimal(5,6)",
            DataType::Decimal {
                precision: 5,
                scale: 6,
            },
        ),
        ("varchar(0)", DataType::VarChar { length: 0 }),
        (
            "array<timestamp(4)>",
            DataType::Array(Box::new(DataType::Timestamp { precision: 4 })),
        ),
    ];
    for (name, data_type) in outside {
        assert!(!data_type.is_valid(), "{name}");
        assert_eq!(DataType::from_name(name), Some(data_type), "{name}");
    }
}

#[test]
fn a_spelling_with_values_that_its_type_cannot_hold_reads_as_no_type() {
    for name in [
        "decimal(18)",
        "bigint(19)",
        "array<varchar>",
        "timestamp(256)",
    ] {
        assert_eq!(DataType::from_name(name), None, "{name}");
    }
}

#[test]
fn the_name_function_reads_a_name_as_written_so_casing_counts() {
    for name in [
        "BIGINT",
        "Boolean",
        "Decimal(18,2)",
        "ARRAY<string>",
        "array<INT>",
    ] {
        assert_eq!(DataType::from_name(name), None, "{name}");
    }
}

#[test]
fn a_spelling_may_space_its_values() {
    assert_eq!(
        DataType::from_name("decimal(18, 2)"),
        Some(DataType::Decimal {
            precision: 18,
            scale: 2,
        }),
    );
}

#[test]
fn a_name_nests_128_arrays_at_most() {
    let nested = |layers| format!("{}string{}", "array<".repeat(layers), ">".repeat(layers));
    let deepest = (0..128).fold(DataType::String, |element, _| {
        DataType::Array(Box::new(element))
    });
    assert_eq!(DataType::from_name(&nested(128)), Some(deepest));
    for layers in [129, 100_000] {
        assert_eq!(
            DataType::from_name(&nested(layers)),
            None,
            "{layers} layers"
        );
    }
}

#[test]
fn a_native_type_reads_through_the_canonical_step_then_icebergs_dialect() {
    let through_iceberg = [
        ("time", DataType::Time { precision: 6 }),
        ("timestamp", DataType::Timestamp { precision: 6 }),
        ("timestamptz", DataType::Timestamp { precision: 6 }),
        ("timestamp_ns", DataType::Timestamp { precision: 9 }),
        ("timestamptz_ns", DataType::Timestamp { precision: 9 }),
        ("fixed[16]", DataType::FixedBinary { length: 16 }),
        ("fixed[ 16 ]", DataType::FixedBinary { length: 16 }),
    ];
    for (native, data_type) in through_iceberg {
        assert_eq!(
            DataType::from_native_name(native, Some(Dialect::Iceberg)),
            Some(data_type),
            "{native}"
        );
        assert_eq!(
            DataType::from_native_name(native, None),
            None,
            "{native}: with no dialect named, the first step alone gives no type"
        );
    }

    let through_canonical = [
        ("boolean", DataType::Boolean),
        ("int", DataType::Integer),
        ("long", DataType::BigInt),
        ("float", DataType::Real),
        ("double", DataType::Double),
        (
            "decimal(9,2)",
            DataType::Decimal {
                precision: 9,
                scale: 2,
            },
        ),
        ("date", DataType::Date),
        ("string", DataType::String),
        ("uuid", DataType::Uuid),
        ("binary", DataType::Binary),
    ];
    for (native, data_type) in through_canonical {
        assert_eq!(
            DataType::from_native_name(native, Some(Dialect::Iceberg)),
            Some(data_type.clone()),
            "{native}"
        );
        assert_eq!(
            DataType::from_native_name(native, None),
            Some(data_type),
            "{native}: the first step alone already gives this type"
        );
    }
}

#[test]
fn only_iceberg_has_a_dialect_spelling_of_its_own() {
    for native in ["time", "timestamp", "timestamptz", "fixed[16]"] {
        for dialect in [Dialect::Substrait, Dialect::DataFusion] {
            assert_eq!(
                DataType::from_native_name(native, Some(dialect)),
                None,
                "{native} under {dialect:?}"
            );
        }
    }
}

#[test]
fn a_spelling_with_no_iceberg_type_reads_as_no_type() {
    for native in [
        "decimal",
        "Timestamp",
        "time_ns",
        "timetz",
        "fixed[0]",
        "fixed[2147483648]",
        "decimal(40,2)",
        "variant",
        "unknown",
        "geometry(srid:4326)",
        "{\"type\":\"list\",\"element-id\":1,\"element\":\"string\"}",
        "{\"type\":\"struct\",\"fields\":[]}",
        "{\"type\":\"map\",\"key\":\"string\",\"value\":\"string\"}",
    ] {
        assert_eq!(
            DataType::from_native_name(native, Some(Dialect::Iceberg)),
            None,
            "{native}"
        );
    }
}

#[test]
fn icebergs_declarations_name_its_own_types_and_mark_the_rest_absent() {
    let named = [
        (DataType::BigInt, "long"),
        (DataType::FixedBinary { length: 16 }, "fixed"),
    ];
    for (data_type, name) in named {
        assert_eq!(on(&data_type, Dialect::Iceberg).name, name, "{data_type:?}");
    }
    assert_eq!(
        DataType::TinyInt
            .declarations()
            .and_then(|declarations| declarations.on(Dialect::Iceberg)),
        None,
        "Iceberg has no `tinyint`"
    );

    let valid = [
        DataType::Timestamp { precision: 6 },
        DataType::Timestamp { precision: 9 },
        DataType::Time { precision: 6 },
        DataType::FixedBinary { length: 16 },
        DataType::Decimal {
            precision: 9,
            scale: 2,
        },
    ];
    for data_type in valid {
        assert!(
            data_type.is_valid_on(Dialect::Iceberg),
            "{data_type:?} is valid on Iceberg"
        );
    }
    let not_valid = [
        DataType::Timestamp { precision: 3 },
        DataType::Time { precision: 9 },
        DataType::TinyInt,
        DataType::VarChar { length: 10 },
        DataType::Interval,
    ];
    for data_type in not_valid {
        assert!(
            !data_type.is_valid_on(Dialect::Iceberg),
            "{data_type:?} is not valid on Iceberg"
        );
    }
}

fn named(name: &str) -> DataType {
    DataType::from_name(name).unwrap_or_else(|| panic!("`{name}` names a type"))
}

#[test]
fn each_type_is_in_its_family() {
    let table = [
        (
            TypeFamily::Number,
            &[
                "tinyint",
                "smallint",
                "integer",
                "bigint",
                "real",
                "double",
                "decimal(18,2)",
            ][..],
        ),
        (TypeFamily::String, &["string", "varchar(10)", "char(5)"]),
        (TypeFamily::Boolean, &["boolean"]),
        (TypeFamily::Binary, &["binary", "binary(16)"]),
        (TypeFamily::DateAndTimestamp, &["date", "timestamp(3)"]),
        (TypeFamily::Time, &["time(6)"]),
        (TypeFamily::Interval, &["interval"]),
        (TypeFamily::YearMonthInterval, &["interval_year_month"]),
        (TypeFamily::DayTimeInterval, &["interval_day_second(3)"]),
        (TypeFamily::Uuid, &["uuid"]),
    ];
    for (family, names) in table {
        for name in names {
            assert_eq!(named(name).family(), Some(family.clone()), "{name}");
        }
    }
}

#[test]
fn an_arrays_family_comes_from_its_element() {
    let bigints = named("array<bigint>").family();
    assert_eq!(
        bigints,
        Some(TypeFamily::Array(Box::new(TypeFamily::Number)))
    );
    assert_eq!(named("array<integer>").family(), bigints);
    assert_ne!(named("array<string>").family(), bigints);
}

#[test]
fn each_widening_step_of_rule_9s_table_widens() {
    let steps = [
        ("smallint", "integer"),
        ("smallint", "bigint"),
        ("integer", "bigint"),
        ("integer", "double"),
        ("integer", "decimal(10,2)"),
        ("bigint", "double"),
        ("bigint", "decimal(38,0)"),
        ("real", "double"),
        ("decimal(10,2)", "decimal(12,2)"),
        ("char(5)", "varchar(5)"),
        ("char(5)", "varchar(8)"),
        ("varchar(5)", "varchar(8)"),
        ("char(5)", "string"),
        ("varchar(5)", "string"),
        ("binary(16)", "binary"),
        ("interval_day_second(3)", "interval_day_second(6)"),
    ];
    for (from, to) in steps {
        assert!(named(from).widens_to(&named(to)), "{from} to {to}");
    }
}

#[test]
fn no_type_widens_where_rule_9s_table_names_no_step() {
    let not_steps = [
        ("integer", "smallint"),
        ("bigint", "integer"),
        ("double", "real"),
        ("real", "bigint"),
        ("smallint", "double"),
        ("double", "decimal(38,9)"),
        ("decimal(10,2)", "decimal(12,3)"),
        ("decimal(12,2)", "decimal(10,2)"),
        ("decimal(10,2)", "double"),
        ("varchar(8)", "varchar(5)"),
        ("varchar(5)", "char(5)"),
        ("char(5)", "char(8)"),
        ("string", "varchar(10)"),
        ("binary", "binary(16)"),
        ("binary(8)", "binary(16)"),
        ("interval_day_second(6)", "interval_day_second(3)"),
        ("interval_year_month", "interval"),
        ("date", "timestamp(6)"),
        ("timestamp(3)", "timestamp(6)"),
        ("integer", "string"),
    ];
    for (from, to) in not_steps {
        assert!(!named(from).widens_to(&named(to)), "{from} to {to}");
    }
}

#[test]
fn tinyint_widens_to_nothing() {
    for to in every_type() {
        assert!(!DataType::TinyInt.widens_to(&to), "tinyint to {to}");
    }
}

#[test]
fn no_array_widens_to_another() {
    for (from, to) in [
        ("array<integer>", "array<bigint>"),
        ("array<char(5)>", "array<string>"),
    ] {
        assert!(!named(from).widens_to(&named(to)), "{from} to {to}");
    }
}

#[test]
fn no_type_widens_to_itself() {
    for data_type in every_type() {
        assert!(!data_type.widens_to(&data_type), "{data_type}");
    }
}

#[derive(Clone, Copy)]
enum Cast {
    Yes,
    No,
    TimestampToTime,
    ByElement,
}

use Cast::{ByElement, No, TimestampToTime, Yes};

const FAMILIES: [&[&str]; 11] = [
    &["tinyint", "integer", "bigint", "double", "decimal(10,2)"],
    &["string", "varchar(10)", "char(5)"],
    &["boolean"],
    &["binary", "binary(16)"],
    &["date", "timestamp(6)"],
    &["time(9)"],
    &["interval"],
    &["interval_year_month"],
    &["interval_day_second(9)"],
    &["uuid"],
    &["array<integer>", "array<string>", "array<boolean>"],
];

#[rustfmt::skip]
const CASTS: [[Cast; 11]; 11] = [
    [Yes, Yes, No,  No,  No,  No,              No,  No,  No,  No,  No],
    [Yes, Yes, Yes, Yes, Yes, Yes,             Yes, Yes, Yes, Yes, No],
    [No,  Yes, Yes, No,  No,  No,              No,  No,  No,  No,  No],
    [No,  Yes, No,  Yes, No,  No,              No,  No,  No,  No,  No],
    [No,  Yes, No,  No,  Yes, TimestampToTime, No,  No,  No,  No,  No],
    [No,  Yes, No,  No,  No,  Yes,             No,  No,  No,  No,  No],
    [No,  Yes, No,  No,  No,  No,              Yes, No,  No,  No,  No],
    [No,  Yes, No,  No,  No,  No,              Yes, Yes, No,  No,  No],
    [No,  Yes, No,  No,  No,  No,              Yes, No,  Yes, No,  No],
    [No,  Yes, No,  No,  No,  No,              No,  No,  No,  Yes, No],
    [No,  No,  No,  No,  No,  No,              No,  No,  No,  No,  ByElement],
];

fn family_row(name: &str) -> usize {
    FAMILIES
        .iter()
        .position(|members| members.contains(&name))
        .unwrap_or_else(|| panic!("`{name}` is a member of the table"))
}

fn table_says(from: &str, to: &str) -> bool {
    match CASTS[family_row(from)][family_row(to)] {
        Yes => true,
        No => false,
        TimestampToTime => from.starts_with("timestamp"),
        ByElement => table_says(element(from), element(to)),
    }
}

fn element(array: &str) -> &str {
    &array["array<".len()..array.len() - 1]
}

#[test]
fn every_cell_of_rule_10s_cast_table() {
    for from in FAMILIES.iter().flat_map(|members| members.iter()) {
        for to in FAMILIES.iter().flat_map(|members| members.iter()) {
            assert_eq!(
                named(from).casts_to(&named(to)),
                table_says(from, to),
                "{from} to {to}",
            );
        }
    }
}

#[test]
fn a_value_with_no_type_casts_where_some_type_of_its_family_casts() {
    let members = || FAMILIES.iter().flat_map(|members| members.iter());
    for family in members().filter_map(|member| named(member).family()) {
        let kin = || members().filter(|member| named(member).family().as_ref() == Some(&family));
        for to in members() {
            assert_eq!(
                check_cast(&DataType::Unknown, Some(&family), &named(to)).is_ok(),
                kin().any(|from| table_says(from, to)),
                "{family} to {to}",
            );
        }
    }
}

fn unknowns() -> [DataType; 2] {
    [
        DataType::Unknown,
        DataType::Array(Box::new(DataType::Unknown)),
    ]
}

#[test]
fn unknown_has_no_declaration() {
    assert_eq!(DataType::Unknown.declarations(), None);
}

#[test]
fn unknown_writes_unknown_for_a_diagnostic() {
    assert_eq!(DataType::Unknown.to_string(), "unknown");
}

#[test]
fn unknown_and_an_array_of_it_have_no_family_and_are_not_valid() {
    for unknown in unknowns() {
        assert_eq!(unknown.family(), None, "{unknown:?}");
        assert!(!unknown.is_valid(), "{unknown:?}");
        assert!(!unknown.is_valid_on(Dialect::Substrait), "{unknown:?}");
        assert!(!unknown.is_valid_on(Dialect::DataFusion), "{unknown:?}");
    }
}

#[test]
fn serde_refuses_unknown_and_an_array_of_it_in_both_directions() {
    for unknown in unknowns() {
        assert!(serde_json::to_string(&unknown).is_err(), "{unknown:?}");
    }
    for written in [r#""unknown""#, r#""array<unknown>""#] {
        assert!(
            serde_json::from_str::<DataType>(written).is_err(),
            "{written}"
        );
    }
}

fn stored_types() -> Vec<DataType> {
    let nested = DataType::Array(Box::new(DataType::Array(Box::new(DataType::Decimal {
        precision: 18,
        scale: 2,
    }))));
    every_type().into_iter().chain([nested]).collect()
}

const NOT_CANONICAL: [&str; 6] = [
    "long",
    "i64",
    "int",
    "BIGINT",
    "Decimal(18,2)",
    "array<long>",
];

#[test]
fn serde_writes_the_canonical_name_and_reads_it_back() {
    for data_type in stored_types() {
        let written = serde_json::to_string(&data_type).expect("a valid type serializes");
        assert_eq!(written, format!("\"{data_type}\""));
        assert_eq!(
            serde_json::from_str::<DataType>(&written).expect("its name reads back"),
            data_type,
        );
    }
}

#[test]
fn serde_refuses_an_alias_or_another_case_and_names_the_spelling() {
    for spelling in NOT_CANONICAL {
        let refused = serde_json::from_value::<DataType>(serde_json::json!(spelling))
            .expect_err(spelling)
            .to_string();
        assert!(refused.contains(&format!("`{spelling}`")), "{refused}");
    }
}

#[test]
fn serde_refuses_a_type_that_is_not_valid_in_both_directions() {
    for name in ["timestamp(4)", "decimal(5,6)", "array<timestamp(4)>"] {
        let outside = named(name);
        assert!(serde_json::to_string(&outside).is_err(), "{name}");
        assert!(
            serde_json::from_str::<DataType>(&format!("\"{name}\"")).is_err(),
            "{name}"
        );
    }
}

#[cfg(feature = "schemars")]
#[test]
fn the_json_schema_of_a_data_type_gives_the_verdict_of_its_reader() {
    let schema = serde_json::to_value(schemars::schema_for!(DataType)).expect("a JSON schema");
    let validator = jsonschema::validator_for(&schema).expect("a valid JSON schema");
    let stored = stored_types()
        .into_iter()
        .map(|data_type| (serde_json::json!(data_type.to_string()), true));
    let refused = NOT_CANONICAL
        .into_iter()
        .map(|spelling| serde_json::json!(spelling))
        .chain([serde_json::json!({"decimal": {"precision": 18, "scale": 2}})])
        .map(|written| (written, false));
    for (written, reads) in stored.chain(refused) {
        assert_eq!(validator.is_valid(&written), reads, "{written}");
        assert_eq!(
            serde_json::from_value::<DataType>(written.clone()).is_ok(),
            reads,
            "{written}"
        );
    }
}

#[test]
#[deny(unreachable_patterns)]
fn data_type_is_non_exhaustive() {
    for data_type in every_type().iter().chain(&unknowns()) {
        match data_type {
            DataType::Boolean
            | DataType::TinyInt
            | DataType::SmallInt
            | DataType::Integer
            | DataType::BigInt
            | DataType::Real
            | DataType::Double
            | DataType::Decimal { .. }
            | DataType::String
            | DataType::VarChar { .. }
            | DataType::FixedChar { .. }
            | DataType::Binary
            | DataType::FixedBinary { .. }
            | DataType::Date
            | DataType::Time { .. }
            | DataType::Timestamp { .. }
            | DataType::Interval
            | DataType::IntervalYearMonth
            | DataType::IntervalDaySecond { .. }
            | DataType::Uuid
            | DataType::Array(_)
            | DataType::Unknown => {}
            _ => panic!("{data_type:?} takes no named arm"),
        }
    }
}
