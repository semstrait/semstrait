use semstrait_sql::datatype::{
    CanonicalTypeSystem, DataType, Decimal, SubstraitTypeSystem, TypeSystem,
};

const QTY: DataType = DataType::Integer;
const UNITS: DataType = DataType::BigInt;
const PRICE: DataType = decimal_type(18, 2);
const RATE: DataType = decimal_type(10, 4);

fn operand(data_type: &DataType) -> Decimal {
    Decimal::of(data_type).unwrap_or_else(|| panic!("{data_type} enters a decimal rule"))
}

const fn decimal(precision: u8, scale: i8) -> Decimal {
    Decimal { precision, scale }
}

const fn decimal_type(precision: u8, scale: i8) -> DataType {
    DataType::Decimal { precision, scale }
}

#[test]
fn an_integer_enters_a_decimal_rule_as_its_digits_at_scale_0() {
    let entries = [
        (DataType::TinyInt, decimal(3, 0)),
        (DataType::SmallInt, decimal(5, 0)),
        (DataType::Integer, decimal(10, 0)),
        (DataType::BigInt, decimal(19, 0)),
        (PRICE, decimal(18, 2)),
    ];
    for (data_type, entry) in entries {
        assert_eq!(Decimal::of(&data_type), Some(entry), "{data_type}");
    }
    for data_type in [
        DataType::Real,
        DataType::Double,
        DataType::String,
        DataType::Unknown,
    ] {
        assert_eq!(Decimal::of(&data_type), None, "{data_type}");
    }
}

#[test]
fn decimal_sum_gives_the_scale_and_the_most_digits_before_the_point_and_one() {
    let sums = [
        (QTY, PRICE, decimal(19, 2)),
        (PRICE, RATE, decimal(21, 4)),
        (UNITS, PRICE, decimal(22, 2)),
    ];
    for (a, b, sum) in sums {
        assert_eq!(
            CanonicalTypeSystem.decimal_sum(operand(&a), operand(&b)),
            sum,
            "{a} + {b}"
        );
    }
}

#[test]
fn decimal_product_gives_the_sum_of_the_scales_and_of_the_precisions_and_one() {
    let products = [
        (UNITS, PRICE, decimal(38, 2)),
        (PRICE, RATE, decimal(29, 6)),
        (PRICE, PRICE, decimal(37, 4)),
    ];
    for (a, b, product) in products {
        assert_eq!(
            CanonicalTypeSystem.decimal_product(operand(&a), operand(&b)),
            product,
            "{a} * {b}"
        );
    }
}

#[test]
fn decimal_quotient_above_precision_38_lowers_its_scale() {
    assert_eq!(
        CanonicalTypeSystem.decimal_quotient(operand(&PRICE), operand(&RATE)),
        decimal(38, 12),
    );
}

#[test]
fn above_precision_38_a_decimal_rule_lowers_the_scale_to_no_less_than_6_or_its_own() {
    let widest = decimal(38, 10);
    let cases = [
        (CanonicalTypeSystem.decimal_product(widest, widest), (38, 6)),
        (CanonicalTypeSystem.decimal_sum(widest, widest), (38, 9)),
        (
            CanonicalTypeSystem.decimal_sum(decimal(38, 2), decimal(38, 2)),
            (38, 2),
        ),
    ];
    for (result, (precision, scale)) in cases {
        assert_eq!(result, decimal(precision, scale));
    }
}

#[test]
fn decimal_union_of_greatest_gives_the_bigger_scale_and_the_most_digits_before_the_point() {
    assert_eq!(
        CanonicalTypeSystem.decimal_union(operand(&PRICE), operand(&RATE)),
        decimal(20, 4),
    );
}

#[test]
fn decimal_mod_gives_the_scale_and_the_fewest_digits_before_the_point() {
    assert_eq!(
        CanonicalTypeSystem.decimal_mod(operand(&PRICE), operand(&RATE)),
        decimal(10, 4),
    );
    assert_eq!(
        CanonicalTypeSystem.decimal_mod(operand(&UNITS), operand(&PRICE)),
        decimal(18, 2),
    );
}

const INTEGERS: [DataType; 4] = [
    DataType::TinyInt,
    DataType::SmallInt,
    DataType::Integer,
    DataType::BigInt,
];

#[test]
fn sum_of_an_integer_is_bigint_of_a_real_or_a_double_double_and_of_a_decimal_precision_38() {
    for integer in INTEGERS {
        assert_eq!(
            CanonicalTypeSystem.sum_of(&integer),
            DataType::BigInt,
            "{integer}"
        );
    }
    for float in [DataType::Real, DataType::Double] {
        assert_eq!(
            CanonicalTypeSystem.sum_of(&float),
            DataType::Double,
            "{float}"
        );
    }
    assert_eq!(CanonicalTypeSystem.sum_of(&PRICE), decimal_type(38, 2),);
}

#[test]
fn avg_of_an_integer_a_real_or_a_double_is_double_and_of_a_decimal_keeps_6_digits_or_more() {
    for number in INTEGERS
        .into_iter()
        .chain([DataType::Real, DataType::Double])
    {
        assert_eq!(
            CanonicalTypeSystem.avg_of(&number),
            DataType::Double,
            "{number}"
        );
    }
    let averages = [(PRICE, (38, 6)), (decimal_type(10, 8), (38, 8))];
    for (data_type, (precision, scale)) in averages {
        assert_eq!(
            CanonicalTypeSystem.avg_of(&data_type),
            decimal_type(precision, scale),
            "{data_type}"
        );
    }
}

#[test]
fn decimal_scale_zero_of_ceil_and_floor_keeps_the_digits_before_the_point_and_one() {
    assert_eq!(
        CanonicalTypeSystem.decimal_scale_zero(operand(&PRICE)),
        decimal(17, 0)
    );
    assert_eq!(
        CanonicalTypeSystem.decimal_scale_zero(decimal(38, 0)),
        decimal(38, 0)
    );
}

#[test]
fn round_of_gives_one_more_digit_up_to_precision_38_and_keeps_the_scale() {
    assert_eq!(
        CanonicalTypeSystem.round_of(operand(&PRICE)),
        decimal(19, 2)
    );
    assert_eq!(
        CanonicalTypeSystem.round_of(decimal(38, 10)),
        decimal(38, 10)
    );
}

#[test]
fn a_rule_of_a_functions_result_type_gives_any_type_it_does_not_name_itself() {
    for data_type in [DataType::String, DataType::Date, DataType::Unknown] {
        assert_eq!(CanonicalTypeSystem.sum_of(&data_type), data_type);
        assert_eq!(CanonicalTypeSystem.avg_of(&data_type), data_type);
    }
}

#[test]
fn substrait_avg_of_an_integer_a_real_or_a_double_keeps_its_type_and_of_a_decimal_its_scale() {
    for number in INTEGERS
        .into_iter()
        .chain([DataType::Real, DataType::Double])
    {
        assert_eq!(SubstraitTypeSystem.avg_of(&number), number, "{number}");
    }
    assert_eq!(SubstraitTypeSystem.avg_of(&PRICE), decimal_type(38, 2),);
}

#[test]
fn substrait_gives_the_canonical_result_of_every_rule_but_avg_of() {
    let (price, rate) = (operand(&PRICE), operand(&RATE));
    let substrait = SubstraitTypeSystem;
    let decimals = [
        (substrait.decimal_union(price, rate), (20, 4)),
        (substrait.decimal_sum(price, rate), (21, 4)),
        (substrait.decimal_product(price, rate), (29, 6)),
        (substrait.decimal_quotient(price, rate), (38, 12)),
        (substrait.decimal_mod(price, rate), (10, 4)),
        (substrait.decimal_scale_zero(price), (17, 0)),
        (substrait.round_of(price), (19, 2)),
    ];
    for (result, (precision, scale)) in decimals {
        assert_eq!(result, decimal(precision, scale));
    }
    assert_eq!(substrait.sum_of(&QTY), DataType::BigInt);
    assert_eq!(substrait.sum_of(&PRICE), decimal_type(38, 2),);
}
