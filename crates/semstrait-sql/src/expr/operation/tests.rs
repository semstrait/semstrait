use crate::datatype::{DataType, TypeFamily};
use crate::expr::{BinaryOpKind, LikeKind, OperationKind, OperationOption, UnaryOpKind};
use crate::functions::{
    ArgumentForm, Constraint, ExactType, Nullability, OptionDecl, ParameterDecl, ParameterType,
    Presence, ReturnRule, Signature,
};
use semstrait_common::Dialect;

const fn operand(name: &'static str, ty: ParameterType) -> ParameterDecl {
    ParameterDecl {
        name,
        ty,
        form: ArgumentForm::Value,
        presence: Presence::Required,
    }
}

const fn operands(parameters: &'static [ParameterDecl]) -> Signature {
    Signature {
        parameters,
        variadic: None,
        constraints: &[],
    }
}

const NUMBER: ParameterType = ParameterType::Family(&TypeFamily::Number);
const BOOLEAN: ParameterType = ParameterType::Family(&TypeFamily::Boolean);
const STRING: ParameterType = ParameterType::Family(&TypeFamily::String);
const BOOLEAN_RESULT: ReturnRule = ReturnRule::Fixed(ExactType::Type(&DataType::Boolean));

const PLUS: Signature = Signature {
    constraints: &[Constraint::Families(PLUS_ROWS)],
    ..operands(&[
        operand("left", ParameterType::Any),
        operand("right", ParameterType::Any),
    ])
};

const PLUS_ROWS: &[&[TypeFamily]] = &[
    &[TypeFamily::Number, TypeFamily::Number],
    &[TypeFamily::DateAndTimestamp, TypeFamily::Interval],
    &[TypeFamily::DateAndTimestamp, TypeFamily::YearMonthInterval],
    &[TypeFamily::DateAndTimestamp, TypeFamily::DayTimeInterval],
    &[TypeFamily::Time, TypeFamily::Interval],
    &[TypeFamily::Time, TypeFamily::YearMonthInterval],
    &[TypeFamily::Time, TypeFamily::DayTimeInterval],
    &[TypeFamily::Interval, TypeFamily::DateAndTimestamp],
    &[TypeFamily::YearMonthInterval, TypeFamily::DateAndTimestamp],
    &[TypeFamily::DayTimeInterval, TypeFamily::DateAndTimestamp],
    &[TypeFamily::Interval, TypeFamily::Time],
    &[TypeFamily::YearMonthInterval, TypeFamily::Time],
    &[TypeFamily::DayTimeInterval, TypeFamily::Time],
];

const MINUS: Signature = Signature {
    constraints: &[Constraint::Families(PLUS_ROWS.split_at(7).0)],
    ..PLUS
};

const TWO_NUMBERS: Signature = operands(&[operand("left", NUMBER), operand("right", NUMBER)]);

const COMPARED: Signature = Signature {
    constraints: &[Constraint::OneFamily],
    ..operands(&[
        operand("left", ParameterType::Any),
        operand("right", ParameterType::Any),
    ])
};

const OVERFLOW: OptionDecl<OperationOption> = OptionDecl {
    name: OperationOption::Overflow,
    values: &["SILENT", "SATURATE", "ERROR"],
    canonical: "ERROR",
};

const fn on_division_by_zero(canonical: &'static str) -> OptionDecl<OperationOption> {
    OptionDecl {
        name: OperationOption::OnDivisionByZero,
        values: &["NULL", "ERROR"],
        canonical,
    }
}

struct Row {
    operations: &'static [OperationKind],
    operands: Signature,
    result: ReturnRule,
    nullability: Nullability,
    options: &'static [OptionDecl<OperationOption>],
}

static RULE_5: &[Row] = &[
    Row {
        operations: &[OperationKind::Binary(BinaryOpKind::Add)],
        operands: PLUS,
        result: ReturnRule::Sum,
        nullability: Nullability::Mirror,
        options: &[OVERFLOW],
    },
    Row {
        operations: &[OperationKind::Binary(BinaryOpKind::Subtract)],
        operands: MINUS,
        result: ReturnRule::Sum,
        nullability: Nullability::Mirror,
        options: &[OVERFLOW],
    },
    Row {
        operations: &[OperationKind::Binary(BinaryOpKind::Multiply)],
        operands: TWO_NUMBERS,
        result: ReturnRule::Product,
        nullability: Nullability::Mirror,
        options: &[OVERFLOW],
    },
    Row {
        operations: &[OperationKind::Binary(BinaryOpKind::Divide)],
        operands: TWO_NUMBERS,
        result: ReturnRule::Quotient,
        nullability: Nullability::Mirror,
        options: &[on_division_by_zero("ERROR")],
    },
    Row {
        operations: &[OperationKind::Binary(BinaryOpKind::Mod)],
        operands: TWO_NUMBERS,
        result: ReturnRule::Mod,
        nullability: Nullability::Mirror,
        options: &[on_division_by_zero("ERROR")],
    },
    Row {
        operations: &[OperationKind::Binary(BinaryOpKind::SafeDivide)],
        operands: TWO_NUMBERS,
        result: ReturnRule::Quotient,
        nullability: Nullability::ByOption(&[
            ("NULL", Nullability::Always),
            ("ERROR", Nullability::Mirror),
        ]),
        options: &[on_division_by_zero("NULL")],
    },
    Row {
        operations: &[OperationKind::Unary(UnaryOpKind::Negate)],
        operands: operands(&[operand("operand", NUMBER)]),
        result: ReturnRule::SameAs(0),
        nullability: Nullability::Mirror,
        options: &[OVERFLOW],
    },
    Row {
        operations: &[
            OperationKind::Binary(BinaryOpKind::Eq),
            OperationKind::Binary(BinaryOpKind::NotEq),
            OperationKind::Binary(BinaryOpKind::Lt),
            OperationKind::Binary(BinaryOpKind::LtEq),
            OperationKind::Binary(BinaryOpKind::Gt),
            OperationKind::Binary(BinaryOpKind::GtEq),
        ],
        operands: COMPARED,
        result: BOOLEAN_RESULT,
        nullability: Nullability::Mirror,
        options: &[],
    },
    Row {
        operations: &[OperationKind::Binary(BinaryOpKind::IsNotDistinctFrom)],
        operands: COMPARED,
        result: BOOLEAN_RESULT,
        nullability: Nullability::Never,
        options: &[],
    },
    Row {
        operations: &[
            OperationKind::Binary(BinaryOpKind::And),
            OperationKind::Binary(BinaryOpKind::Or),
        ],
        operands: operands(&[operand("left", BOOLEAN), operand("right", BOOLEAN)]),
        result: BOOLEAN_RESULT,
        nullability: Nullability::Kleene,
        options: &[],
    },
    Row {
        operations: &[OperationKind::Unary(UnaryOpKind::Not)],
        operands: operands(&[operand("operand", BOOLEAN)]),
        result: BOOLEAN_RESULT,
        nullability: Nullability::Mirror,
        options: &[],
    },
    Row {
        operations: &[
            OperationKind::Like(LikeKind::Like),
            OperationKind::Like(LikeKind::NotLike),
            OperationKind::Like(LikeKind::ILike),
            OperationKind::Like(LikeKind::NotILike),
        ],
        operands: operands(&[operand("value", STRING), operand("pattern", STRING)]),
        result: BOOLEAN_RESULT,
        nullability: Nullability::Mirror,
        options: &[],
    },
    Row {
        operations: &[OperationKind::Between],
        operands: Signature {
            constraints: &[Constraint::OneFamily],
            ..operands(&[
                operand("value", ParameterType::Any),
                operand("low", ParameterType::Any),
                operand("high", ParameterType::Any),
            ])
        },
        result: BOOLEAN_RESULT,
        nullability: Nullability::Kleene,
        options: &[],
    },
    Row {
        operations: &[OperationKind::InList],
        operands: Signature {
            parameters: &[operand("value", ParameterType::Any)],
            variadic: Some(operand("item", ParameterType::Any)),
            constraints: &[Constraint::OneFamily],
        },
        result: BOOLEAN_RESULT,
        nullability: Nullability::Kleene,
        options: &[],
    },
    Row {
        operations: &[OperationKind::IsNull],
        operands: operands(&[operand("operand", ParameterType::Any)]),
        result: BOOLEAN_RESULT,
        nullability: Nullability::Never,
        options: &[],
    },
];

#[test]
fn the_operations_are_rule_5s_each_once() {
    let operations: Vec<OperationKind> = OperationKind::all().collect();
    let rule_5 = RULE_5.iter().flat_map(|row| row.operations);
    assert_eq!(operations.len(), rule_5.clone().count());
    for kind in rule_5 {
        let found = operations.iter().filter(|operation| *operation == kind);
        assert_eq!(found.count(), 1, "{kind:?}");
    }
}

#[test]
fn each_canonical_declaration_is_rule_5s_row() {
    for kind in OperationKind::all() {
        let rows: Vec<&Row> = RULE_5
            .iter()
            .filter(|row| row.operations.contains(&kind))
            .collect();
        let [row] = rows[..] else {
            panic!("{kind:?} is in {} rows of rule 5", rows.len())
        };
        let canonical = &kind.operation().canonical;
        assert_eq!(canonical.signature, row.operands, "{kind:?}'s operands");
        assert_eq!(canonical.returns, row.result, "{kind:?}'s result");
        assert_eq!(
            canonical.nullability, row.nullability,
            "{kind:?}'s NULL rule"
        );
        assert_eq!(canonical.options, row.options, "{kind:?}'s options");
    }
}

#[test]
fn each_kind_reaches_its_operation() {
    let names = [
        BinaryOpKind::Add.operation(),
        UnaryOpKind::Not.operation(),
        LikeKind::ILike.operation(),
    ]
    .map(|operation| operation.canonical.name);
    assert_eq!(names, ["add", "not", "ilike"]);
}

#[test]
fn the_lookup_finds_each_name_and_alias_as_its_operation() {
    for kind in OperationKind::all() {
        let canonical = &kind.operation().canonical;
        for name in std::iter::once(canonical.name).chain(canonical.aliases.iter().copied()) {
            assert_eq!(OperationKind::from_name(name), Some(kind), "{name}");
        }
    }
}

#[test]
fn the_lookup_finds_nothing_for_another_spelling() {
    for name in ["ADD", "plus", "in_list", "i_like", "upper", ""] {
        assert_eq!(OperationKind::from_name(name), None, "{name:?}");
    }
}

enum DataFusion {
    Name(&'static str),
    Options(&'static [OptionDecl<OperationOption>]),
}

static RULE_8: &[(&[OperationKind], DataFusion)] = &[
    (
        &[
            OperationKind::Like(LikeKind::ILike),
            OperationKind::Like(LikeKind::NotILike),
        ],
        DataFusion::Name("ilike"),
    ),
    (
        &[OperationKind::Binary(BinaryOpKind::SafeDivide)],
        DataFusion::Options(&[OptionDecl {
            name: OperationOption::OnDivisionByZero,
            values: &["ERROR"],
            canonical: "ERROR",
        }]),
    ),
    (
        &[
            OperationKind::Binary(BinaryOpKind::Add),
            OperationKind::Binary(BinaryOpKind::Subtract),
            OperationKind::Binary(BinaryOpKind::Multiply),
            OperationKind::Unary(UnaryOpKind::Negate),
        ],
        DataFusion::Options(&[OptionDecl {
            name: OperationOption::Overflow,
            values: &["SILENT"],
            canonical: "SILENT",
        }]),
    ),
];

#[test]
fn datafusion_declares_rule_8s_operations_and_no_other() {
    for kind in OperationKind::all() {
        let operation = kind.operation();
        let canonical = &operation.canonical;
        let datafusion = operation
            .on(Dialect::DataFusion)
            .unwrap_or_else(|| panic!("DataFusion has {kind:?}"));
        let change = RULE_8.iter().find(|(kinds, _)| kinds.contains(&kind));
        assert_eq!(
            std::ptr::eq(datafusion, canonical),
            change.is_none(),
            "{kind:?}'s entry"
        );
        let (name, aliases, options): (_, &[&str], _) = match change {
            Some((_, DataFusion::Name(name))) => (*name, &[], canonical.options),
            Some((_, DataFusion::Options(options))) => (canonical.name, &[], *options),
            None => (canonical.name, canonical.aliases, canonical.options),
        };
        assert_eq!(datafusion.name, name, "{kind:?}'s name");
        assert_eq!(datafusion.aliases, aliases, "{kind:?}'s aliases");
        assert_eq!(datafusion.options, options, "{kind:?}'s options");
        assert_eq!(
            datafusion.signature, canonical.signature,
            "{kind:?}'s operands"
        );
        assert_eq!(datafusion.returns, canonical.returns, "{kind:?}'s result");
        assert_eq!(
            datafusion.nullability, canonical.nullability,
            "{kind:?}'s NULL rule"
        );
    }
}

#[test]
fn on_substrait_gives_every_canonical_declaration() {
    for kind in OperationKind::all() {
        let operation = kind.operation();
        assert!(
            operation
                .on(Dialect::Substrait)
                .is_some_and(|declaration| std::ptr::eq(declaration, &operation.canonical)),
            "{kind:?}"
        );
    }
}
