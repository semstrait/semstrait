use semstrait_sql::functions::{registry, Signature};

fn signature(name: &str) -> &'static Signature {
    registry().get(name).expect("a function").canonical().1
}

#[test]
fn a_signature_counts_its_required_optional_and_variadic_arguments() {
    for (function, count) in [
        ("row_number", (0, Some(0))),
        ("upper", (1, Some(1))),
        ("lpad", (2, Some(3))),
        ("greatest", (1, None)),
        ("concat_ws", (2, None)),
    ] {
        assert_eq!(signature(function).argument_count(), count, "{function}");
    }
}

#[test]
fn a_signature_finds_a_parameter_by_its_name_and_never_its_variadic_tail() {
    let found = |function, name| {
        (signature(function).parameter(name)).map(|(index, parameter)| (index, parameter.name))
    };
    assert_eq!(found("nullif", "right"), Some((1, "right")));
    assert_eq!(found("nullif", "rigth"), None);
    assert_eq!(found("concat_ws", "value"), None);
}
