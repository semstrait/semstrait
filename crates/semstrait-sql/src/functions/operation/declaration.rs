use semstrait_common::Dialect;

use super::{
    BinaryOpKind, LikeKind, Operation, OperationDeclaration, OperationKind, OperationOption,
    UnaryOpKind,
};
use crate::datatype::{DataType, TypeFamily};
use crate::functions::{
    value, Constraint, ExactType, Nullability, OptionDecl, ParameterDecl, ParameterType,
    ReturnRule, Signature, BOOLEAN, NUMBER, STRING,
};

impl OperationKind {
    pub fn operation(self) -> &'static Operation {
        match self {
            Self::Binary(op) => op.operation(),
            Self::Unary(op) => op.operation(),
            Self::Like(kind) => kind.operation(),
            Self::Between => &BETWEEN,
            Self::InList => &IN_LIST,
            Self::IsNull => &IS_NULL,
        }
    }
}

impl BinaryOpKind {
    pub fn operation(self) -> &'static Operation {
        match self {
            Self::Add => &ADD,
            Self::Subtract => &SUBTRACT,
            Self::Multiply => &MULTIPLY,
            Self::Divide => &DIVIDE,
            Self::SafeDivide => &SAFE_DIVIDE,
            Self::Mod => &MOD,
            Self::Eq => &EQ,
            Self::NotEq => &NOT_EQ,
            Self::Lt => &LT,
            Self::LtEq => &LT_EQ,
            Self::Gt => &GT,
            Self::GtEq => &GT_EQ,
            Self::IsNotDistinctFrom => &IS_NOT_DISTINCT_FROM,
            Self::And => &AND,
            Self::Or => &OR,
        }
    }
}

impl UnaryOpKind {
    pub fn operation(self) -> &'static Operation {
        match self {
            Self::Negate => &NEGATE,
            Self::Not => &NOT,
        }
    }
}

impl LikeKind {
    pub fn operation(self) -> &'static Operation {
        match self {
            Self::Like => &LIKE,
            Self::NotLike => &NOT_LIKE,
            Self::ILike => &ILIKE,
            Self::NotILike => &NOT_ILIKE,
        }
    }
}

const fn operands(
    parameters: &'static [ParameterDecl],
    constraints: &'static [Constraint],
) -> Signature {
    Signature {
        parameters,
        variadic: None,
        constraints,
    }
}

const TWO_NUMBERS: Signature = operands(&[value("left", NUMBER), value("right", NUMBER)], &[]);

const LEFT_AND_RIGHT: &[ParameterDecl] = &[
    value("left", ParameterType::Any),
    value("right", ParameterType::Any),
];

const PLUS: Signature = operands(LEFT_AND_RIGHT, &[Constraint::Families(PLUS_ROWS)]);

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

const COMPARED: Signature = operands(LEFT_AND_RIGHT, &[Constraint::OneFamily]);

const fn on_division_by_zero(canonical: &'static str) -> OptionDecl<OperationOption> {
    OptionDecl {
        name: OperationOption::OnDivisionByZero,
        values: &["NULL", "ERROR"],
        canonical,
    }
}

const fn declaration(
    name: &'static str,
    signature: Signature,
    returns: ReturnRule,
) -> OperationDeclaration {
    OperationDeclaration {
        name,
        aliases: &[],
        signature,
        returns,
        nullability: Nullability::Mirror,
        options: &[],
    }
}

const fn overflowing(
    name: &'static str,
    signature: Signature,
    returns: ReturnRule,
) -> OperationDeclaration {
    OperationDeclaration {
        options: &[OptionDecl {
            name: OperationOption::Overflow,
            values: &["SILENT", "SATURATE", "ERROR"],
            canonical: "ERROR",
        }],
        ..declaration(name, signature, returns)
    }
}

const fn boolean(name: &'static str, signature: Signature) -> OperationDeclaration {
    declaration(
        name,
        signature,
        ReturnRule::Fixed(ExactType::Type(&DataType::Boolean)),
    )
}

const fn kleene(name: &'static str, signature: Signature) -> OperationDeclaration {
    OperationDeclaration {
        nullability: Nullability::Kleene,
        ..boolean(name, signature)
    }
}

const fn canonical(canonical: OperationDeclaration) -> Operation {
    Operation {
        canonical,
        dialects: &[],
    }
}

const fn datafusion(
    declaration: OperationDeclaration,
) -> [(Dialect, Option<OperationDeclaration>); 1] {
    [(
        Dialect::DataFusion,
        Some(OperationDeclaration {
            aliases: &[],
            ..declaration
        }),
    )]
}

const fn silent_overflow(canonical: OperationDeclaration) -> OperationDeclaration {
    OperationDeclaration {
        options: &[OptionDecl {
            name: OperationOption::Overflow,
            values: &["SILENT"],
            canonical: "SILENT",
        }],
        ..canonical
    }
}

const CANONICAL_ADD: OperationDeclaration = overflowing("add", PLUS, ReturnRule::Sum);
const CANONICAL_SUBTRACT: OperationDeclaration = overflowing("subtract", MINUS, ReturnRule::Sum);
const CANONICAL_MULTIPLY: OperationDeclaration =
    overflowing("multiply", TWO_NUMBERS, ReturnRule::Product);
const CANONICAL_NEGATE: OperationDeclaration = overflowing(
    "negate",
    operands(&[value("operand", NUMBER)], &[]),
    ReturnRule::SameAs(0),
);

static ADD: Operation = Operation {
    canonical: CANONICAL_ADD,
    dialects: &datafusion(silent_overflow(CANONICAL_ADD)),
};
static SUBTRACT: Operation = Operation {
    canonical: CANONICAL_SUBTRACT,
    dialects: &datafusion(silent_overflow(CANONICAL_SUBTRACT)),
};
static MULTIPLY: Operation = Operation {
    canonical: CANONICAL_MULTIPLY,
    dialects: &datafusion(silent_overflow(CANONICAL_MULTIPLY)),
};
static NEGATE: Operation = Operation {
    canonical: CANONICAL_NEGATE,
    dialects: &datafusion(silent_overflow(CANONICAL_NEGATE)),
};
static DIVIDE: Operation = canonical(OperationDeclaration {
    options: &[on_division_by_zero("ERROR")],
    ..declaration("divide", TWO_NUMBERS, ReturnRule::Quotient)
});
static MOD: Operation = canonical(OperationDeclaration {
    options: &[on_division_by_zero("ERROR")],
    ..declaration("mod", TWO_NUMBERS, ReturnRule::Mod)
});

const CANONICAL_SAFE_DIVIDE: OperationDeclaration = OperationDeclaration {
    nullability: Nullability::ByOption(&[
        ("NULL", Nullability::Always),
        ("ERROR", Nullability::Mirror),
    ]),
    options: &[on_division_by_zero("NULL")],
    ..declaration("safe_divide", TWO_NUMBERS, ReturnRule::Quotient)
};

static SAFE_DIVIDE: Operation = Operation {
    canonical: CANONICAL_SAFE_DIVIDE,
    dialects: &datafusion(OperationDeclaration {
        options: &[OptionDecl {
            values: &["ERROR"],
            ..on_division_by_zero("ERROR")
        }],
        ..CANONICAL_SAFE_DIVIDE
    }),
};

static EQ: Operation = canonical(boolean("eq", COMPARED));
static NOT_EQ: Operation = canonical(boolean("not_eq", COMPARED));
static LT: Operation = canonical(boolean("lt", COMPARED));
static LT_EQ: Operation = canonical(boolean("lt_eq", COMPARED));
static GT: Operation = canonical(boolean("gt", COMPARED));
static GT_EQ: Operation = canonical(boolean("gt_eq", COMPARED));
static IS_NOT_DISTINCT_FROM: Operation = canonical(OperationDeclaration {
    nullability: Nullability::Never,
    ..boolean("is_not_distinct_from", COMPARED)
});

const TWO_BOOLEANS: Signature = operands(&[value("left", BOOLEAN), value("right", BOOLEAN)], &[]);

static AND: Operation = canonical(kleene("and", TWO_BOOLEANS));
static OR: Operation = canonical(kleene("or", TWO_BOOLEANS));
static NOT: Operation = canonical(boolean("not", operands(&[value("operand", BOOLEAN)], &[])));

const PATTERN: Signature = operands(&[value("value", STRING), value("pattern", STRING)], &[]);

static LIKE: Operation = canonical(boolean("like", PATTERN));
static NOT_LIKE: Operation = canonical(boolean("not_like", PATTERN));

const CANONICAL_ILIKE: OperationDeclaration = boolean("ilike", PATTERN);
const CANONICAL_NOT_ILIKE: OperationDeclaration = boolean("not_ilike", PATTERN);

static ILIKE: Operation = Operation {
    canonical: CANONICAL_ILIKE,
    dialects: &datafusion(OperationDeclaration {
        name: "ilike",
        ..CANONICAL_ILIKE
    }),
};
static NOT_ILIKE: Operation = Operation {
    canonical: CANONICAL_NOT_ILIKE,
    dialects: &datafusion(OperationDeclaration {
        name: "ilike",
        ..CANONICAL_NOT_ILIKE
    }),
};

static BETWEEN: Operation = canonical(kleene(
    "between",
    operands(
        &[
            value("value", ParameterType::Any),
            value("low", ParameterType::Any),
            value("high", ParameterType::Any),
        ],
        &[Constraint::OneFamily],
    ),
));
static IN_LIST: Operation = canonical(kleene(
    "in",
    Signature {
        variadic: Some(value("item", ParameterType::Any)),
        ..operands(
            &[value("value", ParameterType::Any)],
            &[Constraint::OneFamily],
        )
    },
));
static IS_NULL: Operation = canonical(OperationDeclaration {
    nullability: Nullability::Never,
    ..boolean(
        "is_null",
        operands(&[value("operand", ParameterType::Any)], &[]),
    )
});
