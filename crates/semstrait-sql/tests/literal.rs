use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{HeldValue, Literal, LiteralError};

fn text(value: &str) -> String {
    value.to_owned()
}

fn decimal(value: &str, precision: u8, scale: i8) -> Literal {
    Literal::Decimal {
        value: text(value),
        precision,
        scale,
    }
}

fn time(value: &str, precision: u8) -> Literal {
    Literal::Time {
        value: text(value),
        precision,
    }
}

fn timestamp(value: &str, precision: u8) -> Literal {
    Literal::Timestamp {
        value: text(value),
        precision,
    }
}

fn day_second(value: &str, precision: u8) -> Literal {
    Literal::IntervalDaySecond {
        value: text(value),
        precision,
    }
}

#[test]
fn a_value_that_its_type_holds_fits() {
    let cases = [
        Literal::BigInt(i64::MAX),
        Literal::Double(0.1),
        decimal("1.5", 38, 1),
        decimal("-123.45", 5, 2),
        decimal("0.001", 3, 3),
        decimal("1.50", 18, 1),
        decimal("12345678901234567890123456789012345678", 38, 0),
        Literal::VarChar {
            value: text("abc"),
            length: 3,
        },
        Literal::FixedChar {
            value: text("ab"),
            length: 5,
        },
        Literal::FixedBinary {
            value: vec![1, 2],
            length: 16,
        },
        Literal::Date(text("2024-02-29")),
        Literal::Date(text("0001-01-01")),
        time("23:59:59.123", 3),
        time("10:00:00", 0),
        timestamp("9999-12-31", 6),
        timestamp("2026-05-19T10:00:00.123456789", 9),
        timestamp("2026-05-19 10:00:00", 0),
        timestamp("1677-09-21T00:12:43.145224192", 9),
        timestamp("2262-04-11T23:47:16.854775807", 9),
        Literal::Interval(text("P1Y2M3DT4H5M6.5S")),
        Literal::Interval(text("-P1D")),
        Literal::IntervalYearMonth(text("P1Y2M")),
        day_second("P1DT1.5S", 6),
        day_second("PT36H", 0),
        day_second("PT2147483647S", 0),
        day_second("-PT2147483648S", 0),
        day_second("-PT2147483648.5S", 1),
        Literal::Interval(text("PT596523H14M7.999999999S")),
        Literal::Uuid(text("550e8400-e29b-41d4-A716-446655440000")),
        Literal::Array {
            element: DataType::Date,
            values: vec![
                Literal::Date(text("2026-05-19")),
                Literal::Null(DataType::Date),
            ],
        },
        Literal::Null(DataType::BigInt),
    ];
    for literal in cases {
        assert_eq!(literal.check_fit(), Ok(()), "{literal:?}");
    }
}

#[test]
fn a_value_that_its_type_cannot_hold_is_refused() {
    let cases = [
        (
            decimal("1234567890123456789012345678901234567890", 38, 0),
            LiteralError::TooManyDigits,
        ),
        (
            decimal("0.123456789012345678901234567890123456789", 38, 39),
            LiteralError::TooManyDigits,
        ),
        (decimal("123.45", 4, 2), LiteralError::CutsADigit),
        (decimal("1.125", 18, 2), LiteralError::CutsADigit),
        (time("10:00:00.1234", 3), LiteralError::CutsADigit),
        (
            timestamp("2026-05-19T10:00:00.1234567", 6),
            LiteralError::CutsADigit,
        ),
        (day_second("PT1.1234567S", 6), LiteralError::CutsADigit),
        (timestamp("9999-12-31", 9), LiteralError::OutOfRange),
        (
            timestamp("1677-09-21T00:12:43.145224191", 9),
            LiteralError::OutOfRange,
        ),
        (
            Literal::IntervalYearMonth(text("P178956971Y")),
            LiteralError::OutOfRange,
        ),
        (day_second("PT2147483648S", 0), LiteralError::OutOfRange),
        (day_second("-PT2147483649S", 0), LiteralError::OutOfRange),
        (day_second("PT1000000H", 0), LiteralError::OutOfRange),
        (
            Literal::Interval(text("PT600000H")),
            LiteralError::OutOfRange,
        ),
        (
            Literal::VarChar {
                value: text("abcd"),
                length: 3,
            },
            LiteralError::TooLong,
        ),
        (
            Literal::FixedChar {
                value: text("abc"),
                length: 2,
            },
            LiteralError::TooLong,
        ),
        (
            Literal::FixedBinary {
                value: vec![1, 2, 3],
                length: 2,
            },
            LiteralError::TooLong,
        ),
        (
            timestamp("2026-05-19T10:00:00+02:00", 6),
            LiteralError::Zoned,
        ),
        (timestamp("2026-05-19T10:00:00Z", 6), LiteralError::Zoned),
    ];
    for (literal, refusal) in cases {
        assert_eq!(literal.check_fit(), Err(refusal), "{literal:?}");
    }
}

#[test]
fn a_literal_whose_type_is_not_valid_is_refused() {
    let cases = [
        decimal("1.5", 40, 2),
        time("10:00:00", 4),
        Literal::VarChar {
            value: String::new(),
            length: 0,
        },
        Literal::Null(DataType::Unknown),
        Literal::Array {
            element: DataType::Unknown,
            values: vec![],
        },
    ];
    for literal in cases {
        assert_eq!(
            literal.check_fit(),
            Err(LiteralError::InvalidType),
            "{literal:?}"
        );
    }
}

#[test]
fn a_value_that_is_not_text_of_its_type_is_refused() {
    let cases = [
        decimal("1.2.3", 38, 1),
        decimal("1e5", 38, 0),
        Literal::Date(text("2026-02-30")),
        Literal::Date(text("2026-5-19")),
        Literal::Date(text("0000-01-01")),
        time("24:00:00", 9),
        time("10:00:00.", 9),
        timestamp("2026-05-19T10:00", 6),
        Literal::Interval(text("1D")),
        Literal::Interval(text("P")),
        Literal::Interval(text("P1DT")),
        Literal::Interval(text("P1D2Y")),
        Literal::IntervalYearMonth(text("P1D")),
        day_second("P1Y", 9),
        Literal::Uuid(text("550e8400e29b41d4a716446655440000")),
        Literal::Uuid(text("550e8400-e29b-41d4-a716-44665544000g")),
    ];
    for literal in cases {
        assert_eq!(
            literal.check_fit(),
            Err(LiteralError::Malformed),
            "{literal:?}"
        );
    }
}

#[test]
fn an_array_value_of_another_type_than_its_element_is_refused() {
    let array = Literal::Array {
        element: DataType::String,
        values: vec![Literal::BigInt(1)],
    };
    assert_eq!(array.check_fit(), Err(LiteralError::ElementType));
}

#[test]
fn a_literal_s_variant_gives_its_data_type_with_its_length_precision_and_scale() {
    let cases = [
        (Literal::Boolean(true), DataType::Boolean),
        (Literal::TinyInt(1), DataType::TinyInt),
        (Literal::SmallInt(1), DataType::SmallInt),
        (Literal::Integer(1), DataType::Integer),
        (Literal::BigInt(1), DataType::BigInt),
        (Literal::Real(1.5), DataType::Real),
        (Literal::Double(1.5), DataType::Double),
        (
            Literal::Decimal {
                value: text("1.5"),
                precision: 18,
                scale: 2,
            },
            DataType::Decimal {
                precision: 18,
                scale: 2,
            },
        ),
        (Literal::String(text("a")), DataType::String),
        (
            Literal::VarChar {
                value: text("abc"),
                length: 10,
            },
            DataType::VarChar { length: 10 },
        ),
        (
            Literal::FixedChar {
                value: text("ab"),
                length: 2,
            },
            DataType::FixedChar { length: 2 },
        ),
        (Literal::Binary(vec![1, 2]), DataType::Binary),
        (
            Literal::FixedBinary {
                value: vec![1, 2],
                length: 16,
            },
            DataType::FixedBinary { length: 16 },
        ),
        (Literal::Date(text("2026-05-19")), DataType::Date),
        (
            Literal::Time {
                value: text("10:00:00"),
                precision: 3,
            },
            DataType::Time { precision: 3 },
        ),
        (
            Literal::Timestamp {
                value: text("2026-05-19T10:00:00"),
                precision: 6,
            },
            DataType::Timestamp { precision: 6 },
        ),
        (Literal::Interval(text("P1D")), DataType::Interval),
        (
            Literal::IntervalYearMonth(text("P1Y2M")),
            DataType::IntervalYearMonth,
        ),
        (
            Literal::IntervalDaySecond {
                value: text("P1DT1.5S"),
                precision: 6,
            },
            DataType::IntervalDaySecond { precision: 6 },
        ),
        (
            Literal::Uuid(text("550e8400-e29b-41d4-a716-446655440000")),
            DataType::Uuid,
        ),
        (
            Literal::Array {
                element: DataType::String,
                values: vec![Literal::String(text("a"))],
            },
            DataType::Array(Box::new(DataType::String)),
        ),
        (Literal::Null(DataType::BigInt), DataType::BigInt),
    ];
    for (literal, data_type) in cases {
        assert_eq!(literal.data_type(), data_type, "{literal:?}");
    }
}

#[test]
fn a_float_literal_s_nan_is_not_equal_to_itself() {
    assert_ne!(Literal::Real(f32::NAN), Literal::Real(f32::NAN));
    assert_ne!(Literal::Double(f64::NAN), Literal::Double(f64::NAN));
}

const DECIMAL_18_2: DataType = DataType::Decimal {
    precision: 18,
    scale: 2,
};

#[test]
fn an_untyped_literal_reads_as_its_context_s_type_with_no_loss() {
    let cases = [
        (Literal::Double(1.5), DECIMAL_18_2, decimal("1.5", 18, 2)),
        (
            Literal::String(text("2026-05-19")),
            DataType::Date,
            Literal::Date(text("2026-05-19")),
        ),
        (
            Literal::Null(DataType::String),
            DataType::BigInt,
            Literal::Null(DataType::BigInt),
        ),
        (
            Literal::String(text("5")),
            DECIMAL_18_2,
            decimal("5", 18, 2),
        ),
        (Literal::BigInt(1), DataType::Integer, Literal::Integer(1)),
        (Literal::BigInt(0), DataType::Integer, Literal::Integer(0)),
        (Literal::BigInt(1), DataType::BigInt, Literal::BigInt(1)),
    ];
    for (literal, context, read) in cases {
        assert_eq!(
            literal.read_as(&context),
            Ok(read),
            "{literal:?} as {context}"
        );
    }
}

#[test]
fn an_untyped_literal_that_would_lose_part_of_its_value_is_refused() {
    let cases = [
        (
            Literal::Double(1.5),
            DataType::Integer,
            LiteralError::CutsADigit,
        ),
        (
            Literal::String(text("2026-05-19T10:00:00+02:00")),
            DataType::timestamp(),
            LiteralError::Zoned,
        ),
        (
            Literal::Double(1.125),
            DECIMAL_18_2,
            LiteralError::CutsADigit,
        ),
    ];
    for (literal, context, refusal) in cases {
        assert_eq!(
            literal.read_as(&context),
            Err(refusal),
            "{literal:?} as {context}"
        );
    }
}

#[test]
fn a_number_reads_through_its_text_and_a_string_as_text_of_its_type() {
    let reads = [
        (Literal::Double(2.0), DataType::Integer, Literal::Integer(2)),
        (Literal::Double(0.1), DataType::Real, Literal::Real(0.1)),
        (
            Literal::String(text("1.1")),
            DataType::Double,
            Literal::Double(1.1),
        ),
        (
            decimal("123", 38, 0),
            DataType::BigInt,
            Literal::BigInt(123),
        ),
        (Literal::BigInt(-5), DECIMAL_18_2, decimal("-5", 18, 2)),
        (
            Literal::String(text("true")),
            DataType::Boolean,
            Literal::Boolean(true),
        ),
        (
            Literal::String(text("abc")),
            DataType::VarChar { length: 3 },
            Literal::VarChar {
                value: text("abc"),
                length: 3,
            },
        ),
        (
            Literal::String(text("P1Y")),
            DataType::IntervalYearMonth,
            Literal::IntervalYearMonth(text("P1Y")),
        ),
        (
            Literal::Date(text("2026-05-19")),
            DataType::Date,
            Literal::Date(text("2026-05-19")),
        ),
    ];
    for (literal, to, read) in reads {
        assert_eq!(literal.read_as(&to), Ok(read), "{literal:?} as {to}");
    }
}

#[test]
fn a_floats_text_may_write_an_exponent() {
    for (written, to, read) in [
        ("1e3", DataType::Double, Ok(Literal::Double(1000.0))),
        ("-2.5E-3", DataType::Double, Ok(Literal::Double(-0.0025))),
        ("1.5e+2", DataType::Real, Ok(Literal::Real(150.0))),
        (
            "0.123456789012345678e1",
            DataType::Double,
            Err(LiteralError::CutsADigit),
        ),
        ("1e400", DataType::Double, Err(LiteralError::OutOfRange)),
        ("1e", DataType::Double, Err(LiteralError::Malformed)),
        ("nan", DataType::Double, Err(LiteralError::Malformed)),
        ("inf", DataType::Real, Err(LiteralError::Malformed)),
        ("1e3", DataType::BigInt, Err(LiteralError::Malformed)),
        ("1e3", DECIMAL_18_2, Err(LiteralError::Malformed)),
    ] {
        assert_eq!(
            Literal::String(text(written)).read_as(&to),
            read,
            "`{written}` as {to}"
        );
    }
}

#[test]
fn a_value_s_text_parses_as_the_literal_of_a_type_that_holds_it() {
    for (written, to, parsed) in [
        ("1.5", DECIMAL_18_2, Ok(decimal("1.5", 18, 2))),
        ("42", DataType::Integer, Ok(Literal::Integer(42))),
        ("1e3", DataType::Double, Ok(Literal::Double(1000.0))),
        ("false", DataType::Boolean, Ok(Literal::Boolean(false))),
        (
            "2026-05-19",
            DataType::Date,
            Ok(Literal::Date(text("2026-05-19"))),
        ),
        ("1.125", DECIMAL_18_2, Err(LiteralError::CutsADigit)),
        ("2026-02-30", DataType::Date, Err(LiteralError::Malformed)),
        ("300", DataType::TinyInt, Err(LiteralError::OutOfRange)),
        (
            "abc",
            DataType::VarChar { length: 2 },
            Err(LiteralError::TooLong),
        ),
        ("ab", DataType::Binary, Err(LiteralError::NoReading)),
        ("1", DataType::Unknown, Err(LiteralError::InvalidType)),
    ] {
        assert_eq!(Literal::parse(written, &to), parsed, "`{written}` as {to}");
    }
}

#[test]
fn a_value_with_no_lossless_reading_as_the_type_is_refused() {
    let refusals = [
        (
            Literal::BigInt(5),
            DataType::String,
            LiteralError::NoReading,
        ),
        (
            Literal::Boolean(true),
            DataType::Integer,
            LiteralError::NoReading,
        ),
        (
            Literal::String(text("ab")),
            DataType::Binary,
            LiteralError::NoReading,
        ),
        (
            Literal::BigInt(300),
            DataType::TinyInt,
            LiteralError::OutOfRange,
        ),
        (
            Literal::BigInt(16_777_217),
            DataType::Real,
            LiteralError::CutsADigit,
        ),
        (
            Literal::String(text("x")),
            DECIMAL_18_2,
            LiteralError::Malformed,
        ),
        (
            Literal::String(text("abc")),
            DataType::VarChar { length: 2 },
            LiteralError::TooLong,
        ),
        (
            Literal::Null(DataType::String),
            DataType::Unknown,
            LiteralError::InvalidType,
        ),
    ];
    for (literal, to, refusal) in refusals {
        assert_eq!(literal.read_as(&to), Err(refusal), "{literal:?} as {to}");
    }
}

#[test]
fn a_literal_round_trips_under_its_canonical_tag() {
    let cases = [
        (Literal::Boolean(true), r#"{"boolean":true}"#),
        (Literal::TinyInt(-1), r#"{"tinyint":-1}"#),
        (Literal::SmallInt(2), r#"{"smallint":2}"#),
        (Literal::Integer(3), r#"{"integer":3}"#),
        (Literal::BigInt(4), r#"{"bigint":4}"#),
        (Literal::Real(1.5), r#"{"real":"1.5"}"#),
        (Literal::Double(0.1), r#"{"double":"0.1"}"#),
        (Literal::Double(-0.0), r#"{"double":"-0"}"#),
        (
            decimal("1.5", 18, 2),
            r#"{"decimal":{"value":"1.5","precision":18,"scale":2}}"#,
        ),
        (Literal::String(text("a")), r#"{"string":"a"}"#),
        (
            Literal::VarChar {
                value: text("abc"),
                length: 10,
            },
            r#"{"varchar":{"value":"abc","length":10}}"#,
        ),
        (
            Literal::FixedChar {
                value: text("ab"),
                length: 2,
            },
            r#"{"char":{"value":"ab","length":2}}"#,
        ),
        (Literal::Binary(vec![1, 2]), r#"{"binary":[1,2]}"#),
        (
            Literal::FixedBinary {
                value: vec![1, 2],
                length: 16,
            },
            r#"{"binary":{"value":[1,2],"length":16}}"#,
        ),
        (
            Literal::Date(text("2026-05-19")),
            r#"{"date":"2026-05-19"}"#,
        ),
        (
            time("10:00:00", 3),
            r#"{"time":{"value":"10:00:00","precision":3}}"#,
        ),
        (
            timestamp("2026-05-19T10:00:00", 6),
            r#"{"timestamp":{"value":"2026-05-19T10:00:00","precision":6}}"#,
        ),
        (Literal::Interval(text("P1D")), r#"{"interval":"P1D"}"#),
        (
            Literal::IntervalYearMonth(text("P1Y2M")),
            r#"{"interval_year_month":"P1Y2M"}"#,
        ),
        (
            day_second("P1DT1.5S", 6),
            r#"{"interval_day_second":{"value":"P1DT1.5S","precision":6}}"#,
        ),
        (
            Literal::Uuid(text("550e8400-e29b-41d4-a716-446655440000")),
            r#"{"uuid":"550e8400-e29b-41d4-a716-446655440000"}"#,
        ),
        (
            Literal::Array {
                element: DataType::String,
                values: vec![Literal::String(text("a")), Literal::Null(DataType::String)],
            },
            r#"{"array":{"element":"string","values":[{"string":"a"},{"null":"string"}]}}"#,
        ),
        (Literal::Null(DataType::BigInt), r#"{"null":"bigint"}"#),
    ];
    for (literal, form) in cases {
        assert_eq!(serde_json::to_string(&literal).expect("writes"), form);
        assert_eq!(
            serde_json::from_str::<Literal>(form).expect("reads"),
            literal,
            "{form}"
        );
    }
}

#[test]
fn a_literal_s_form_refuses_what_no_literal_writes() {
    let forms = [
        r#"{"big_int":1}"#,
        r#"{"long":1}"#,
        r#"{"bigint":1,"string":"a"}"#,
        r#"{"null":"bigint","bigint":1}"#,
        r#"{"numeric":{"value":"1.5","precision":18,"scale":2}}"#,
        r#"{}"#,
        r#"{"decimal":{"value":"1.5","precision":18}}"#,
        r#"{"varchar":{"value":"a","length":1,"precision":1}}"#,
        r#"{"double":"one"}"#,
        r#"{"null":"unknown"}"#,
        r#""a""#,
    ];
    for form in forms {
        assert!(serde_json::from_str::<Literal>(form).is_err(), "{form}");
    }
    assert!(serde_json::to_string(&Literal::Null(DataType::Unknown)).is_err());
}

#[test]
fn a_literal_s_form_refuses_a_value_that_its_type_cannot_hold() {
    let cases = [
        (
            decimal("1.5", 40, 1),
            r#"{"decimal":{"value":"1.5","precision":40,"scale":1}}"#,
            LiteralError::InvalidType,
        ),
        (
            decimal("abc", 18, 0),
            r#"{"decimal":{"value":"abc","precision":18,"scale":0}}"#,
            LiteralError::Malformed,
        ),
        (
            decimal("123.45", 4, 2),
            r#"{"decimal":{"value":"123.45","precision":4,"scale":2}}"#,
            LiteralError::CutsADigit,
        ),
        (
            Literal::VarChar {
                value: text("abcd"),
                length: 2,
            },
            r#"{"varchar":{"value":"abcd","length":2}}"#,
            LiteralError::TooLong,
        ),
        (
            Literal::Date(text("2026-02-30")),
            r#"{"date":"2026-02-30"}"#,
            LiteralError::Malformed,
        ),
        (
            timestamp("9999-12-31", 9),
            r#"{"timestamp":{"value":"9999-12-31","precision":9}}"#,
            LiteralError::OutOfRange,
        ),
        (
            Literal::Array {
                element: DataType::BigInt,
                values: vec![Literal::String(text("a"))],
            },
            r#"{"array":{"element":"bigint","values":[{"string":"a"}]}}"#,
            LiteralError::ElementType,
        ),
    ];
    for (literal, form, refusal) in cases {
        let written = serde_json::to_string(&literal).expect_err(form);
        let read = serde_json::from_str::<Literal>(form).expect_err(form);
        for error in [written.to_string(), read.to_string()] {
            assert!(error.starts_with(&refusal.to_string()), "{form}: {error}");
        }
    }
}

#[test]
fn a_literal_holds_its_value_as_the_numbers_of_its_type() {
    let cases = [
        (Literal::Date(text("1970-01-02")), HeldValue::Date(1)),
        (Literal::Date(text("1969-12-31")), HeldValue::Date(-1)),
        (
            time("01:00:00.5", 3),
            HeldValue::Time {
                units: 3_600_500,
                precision: 3,
            },
        ),
        (
            timestamp("2000-01-01T00:00:01.25", 3),
            HeldValue::Timestamp {
                units: 946_684_801_250,
                precision: 3,
            },
        ),
        (
            decimal("-123.45", 5, 2),
            HeldValue::Decimal {
                unscaled: -12_345,
                precision: 5,
                scale: 2,
            },
        ),
        (
            decimal("1.5", 18, 2),
            HeldValue::Decimal {
                unscaled: 150,
                precision: 18,
                scale: 2,
            },
        ),
        (
            Literal::Interval(text("P1Y2M3DT4H5M6.5S")),
            HeldValue::Interval {
                months: 14,
                days: 3,
                nanoseconds: 14_706_500_000_000,
            },
        ),
        (
            Literal::IntervalYearMonth(text("-P1Y2M")),
            HeldValue::IntervalYearMonth { months: -14 },
        ),
        (
            day_second("P1DT1.5S", 6),
            HeldValue::IntervalDaySecond {
                days: 1,
                units: 1_500_000,
                precision: 6,
            },
        ),
        (
            day_second("-PT2147483648.5S", 1),
            HeldValue::IntervalDaySecond {
                days: 0,
                units: -21_474_836_485,
                precision: 1,
            },
        ),
        (
            Literal::Uuid(text("550e8400-e29b-41d4-A716-446655440000")),
            HeldValue::Uuid(0x550e_8400_e29b_41d4_a716_4466_5544_0000),
        ),
    ];
    for (literal, held) in cases {
        assert_eq!(literal.held_value(), Ok(Some(held)), "{literal:?}");
    }
    assert_eq!(Literal::BigInt(5).held_value(), Ok(None));
    assert_eq!(
        Literal::Date(text("2026-02-30")).held_value(),
        Err(LiteralError::Malformed)
    );
    assert_eq!(
        day_second("PT9223372036.854775808S", 9).held_value(),
        Err(LiteralError::OutOfRange)
    );
}
