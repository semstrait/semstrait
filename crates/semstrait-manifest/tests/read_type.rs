use semstrait_common::SemanticsName;
use semstrait_manifest::{read_type, MappedSemantics};
use semstrait_sql::datatype::DataType;

type Mapped = (SemanticsName, Option<DataType>, bool);

fn named(name: &str) -> DataType {
    DataType::from_name(name).unwrap_or_else(|| panic!("`{name}` names a type"))
}

fn value(name: &str, data_type: &str) -> Mapped {
    (name.into(), Some(named(data_type)), false)
}

fn count(name: &str, data_type: &str) -> Mapped {
    (name.into(), Some(named(data_type)), true)
}

fn read_type_of(mapped: &[Mapped]) -> Option<DataType> {
    read_type(
        mapped
            .iter()
            .map(|(name, data_type, is_count)| MappedSemantics {
                name,
                data_type: data_type.as_ref(),
                is_count: *is_count,
            }),
    )
    .cloned()
}

#[test]
fn orders_daily_reads_amount_as_the_double_of_revenue() {
    assert_eq!(
        read_type_of(&[value("revenue", "double")]),
        Some(named("double"))
    );
}

#[test]
fn orders_daily_reads_order_id_as_the_bigint_of_order_count() {
    assert_eq!(
        read_type_of(&[count("order_count", "bigint")]),
        Some(named("bigint"))
    );
}

#[test]
fn date_and_timestamp_on_created_at_read_as_the_wider_timestamp() {
    let mapped = [value("order_date", "date"), value("order_ts", "timestamp")];
    assert_eq!(read_type_of(&mapped), Some(named("timestamp")));
}

#[test]
fn a_count_on_order_id_takes_no_part_beside_the_string_of_order_ref() {
    let mapped = [count("order_count", "bigint"), value("order_ref", "string")];
    assert_eq!(read_type_of(&mapped), Some(named("string")));
}

#[test]
fn user_ref_and_users_on_user_id_read_as_the_string_of_user_ref() {
    let mapped = [count("users", "bigint"), value("user_ref", "string")];
    assert_eq!(read_type_of(&mapped), Some(named("string")));
}

#[test]
fn an_approx_count_distinct_alone_on_user_id_gives_its_bigint() {
    assert_eq!(
        read_type_of(&[count("users", "bigint")]),
        Some(named("bigint"))
    );
}

#[test]
fn bigint_and_string_give_the_type_of_the_first_by_name() {
    let mapped = [
        value("account_ref", "string"),
        value("account_id", "bigint"),
    ];
    assert_eq!(read_type_of(&mapped), Some(named("bigint")));
}

#[test]
fn decimals_with_different_scales_give_the_type_of_the_first_by_name() {
    let mapped = [
        value("price", "decimal(18,2)"),
        value("cost", "decimal(10,4)"),
    ];
    assert_eq!(read_type_of(&mapped), Some(named("decimal(10,4)")));
}

#[test]
fn a_column_with_no_semantics_that_gives_a_type_has_no_read_type() {
    assert_eq!(read_type_of(&[]), None);
    assert_eq!(read_type_of(&[("big_orders".into(), None, false)]), None);
}

#[test]
fn the_type_that_each_other_type_widens_to_wins() {
    for (narrower, wider) in [
        ("smallint", "integer"),
        ("smallint", "bigint"),
        ("smallint", "double"),
        ("smallint", "decimal(5,2)"),
        ("integer", "bigint"),
        ("integer", "double"),
        ("integer", "decimal(5,2)"),
        ("bigint", "double"),
        ("bigint", "decimal(18,2)"),
        ("real", "double"),
        ("decimal(10,2)", "decimal(18,2)"),
        ("date", "timestamp(0)"),
        ("date", "timestamp(9)"),
        ("timestamp(3)", "timestamp(6)"),
        ("time(0)", "time(9)"),
    ] {
        let mapped = [value("a", narrower), value("b", wider)];
        assert_eq!(
            read_type_of(&mapped),
            Some(named(wider)),
            "{narrower} to {wider}"
        );
    }
}

#[test]
fn types_that_no_step_widens_give_the_type_of_the_first_by_name() {
    for (first, second) in [
        ("varchar(10)", "string"),
        ("binary(16)", "binary"),
        ("interval_day_second(3)", "interval_day_second(9)"),
        ("real", "bigint"),
        ("double", "decimal(38,9)"),
        ("tinyint", "smallint"),
        ("time(6)", "timestamp(6)"),
    ] {
        let mapped = [value("a", first), value("b", second)];
        assert_eq!(
            read_type_of(&mapped),
            Some(named(first)),
            "{first} and {second}"
        );
    }
}
