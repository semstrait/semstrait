use semstrait_sql::datatype::{
    promote, DataType, Decimal, SubstraitTypeSystem, TypeFamily, TypeSystem,
};
use semstrait_sql::expr::{self, BinaryOpKind, LikeKind, UnaryOpKind};
use semstrait_sql::functions::{Constraint, ReturnRule, Signature};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FunctionName {
    pub urn: &'static str,
    pub name: &'static str,
}

pub(crate) const AGGREGATE_APPROX: &str = "extension:io.substrait:functions_aggregate_approx";
pub(crate) const AGGREGATE_GENERIC: &str = "extension:io.substrait:functions_aggregate_generic";
pub(crate) const ARITHMETIC: &str = "extension:io.substrait:functions_arithmetic";
pub(crate) const ARITHMETIC_DECIMAL: &str = "extension:io.substrait:functions_arithmetic_decimal";
pub(crate) const BOOLEAN: &str = "extension:io.substrait:functions_boolean";
pub(crate) const COMPARISON: &str = "extension:io.substrait:functions_comparison";
pub(crate) const DATETIME: &str = "extension:io.substrait:functions_datetime";
pub(crate) const LOGARITHMIC: &str = "extension:io.substrait:functions_logarithmic";
pub(crate) const ROUNDING: &str = "extension:io.substrait:functions_rounding";
pub(crate) const ROUNDING_DECIMAL: &str = "extension:io.substrait:functions_rounding_decimal";
pub(crate) const STRING: &str = "extension:io.substrait:functions_string";
pub(crate) const SEMSTRAIT: &str = "extension:semstrait:semstrait";

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Argument {
    Call(usize),
    CallOr(usize, Constant),
    Literal(Constant),
    Enum(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Constant {
    Integer(i32),
    BigInt(i64),
    String(&'static str),
}

impl Constant {
    pub(crate) fn literal(self) -> expr::Literal {
        match self {
            Self::Integer(value) => expr::Literal::Integer(value),
            Self::BigInt(value) => expr::Literal::BigInt(value),
            Self::String(value) => expr::Literal::String(value.to_owned()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Form {
    pub name: FunctionName,
    pub declared: &'static [&'static str],
    pub options: &'static [(&'static str, &'static str)],
    pub arguments: &'static [Argument],
}

const fn form(urn: &'static str, name: &'static str) -> Form {
    Form {
        name: FunctionName { urn, name },
        declared: &[],
        options: &[],
        arguments: &[],
    }
}

pub(crate) const CANONICAL_OPTIONS: &[(&str, &str)] = &[
    ("overflow", "ERROR"),
    ("case_sensitivity", "CASE_SENSITIVE"),
];

fn declared(name: FunctionName, operands: Operands) -> &'static [&'static str] {
    let floats = operands == Operands::Floats;
    match (name.urn, name.name) {
        (ARITHMETIC, "add" | "subtract" | "multiply") if floats => &["rounding"],
        (ARITHMETIC, "divide") if floats => &["rounding", "on_domain_error", "on_division_by_zero"],
        (ARITHMETIC, "divide") => &["overflow", "on_domain_error", "on_division_by_zero"],
        (ARITHMETIC, "modulus") => &["division_type", "overflow", "on_domain_error"],
        (ARITHMETIC, "negate" | "abs") if floats => &[],
        (ARITHMETIC, "add" | "subtract" | "multiply" | "negate" | "abs" | "sum" | "avg")
        | (ARITHMETIC_DECIMAL, "add" | "subtract" | "multiply" | "divide" | "modulus")
        | (ARITHMETIC_DECIMAL, "sum" | "avg")
        | (AGGREGATE_GENERIC, "count") => &["overflow"],
        (ARITHMETIC, "asin" | "acos" | "atan2") => &["rounding", "on_domain_error"],
        (ARITHMETIC, "sqrt" | "exp" | "sin" | "cos" | "tan" | "atan" | "sinh" | "cosh")
        | (ARITHMETIC, "tanh" | "degrees" | "radians" | "corr" | "median")
        | (ARITHMETIC, "std_dev" | "variance") => &["rounding"],
        (LOGARITHMIC, _) => &["rounding", "on_domain_error", "on_log_zero"],
        (STRING, "like" | "strpos" | "replace" | "starts_with" | "ends_with" | "contains") => {
            &["case_sensitivity"]
        }
        (STRING, "upper" | "lower" | "initcap") => &["char_set"],
        (STRING, "concat") => &["null_handling"],
        _ => &[],
    }
}

impl Form {
    const fn with_options(self, options: &'static [(&'static str, &'static str)]) -> Self {
        Self { options, ..self }
    }

    const fn with_arguments(self, arguments: &'static [Argument]) -> Self {
        Self { arguments, ..self }
    }

    fn on(self, operands: Operands) -> Self {
        Self {
            declared: declared(self.name, operands),
            ..self
        }
    }

    fn places(&self, arity: usize) -> bool {
        let placed =
            (self.arguments.iter()).filter(|argument| matches!(argument, Call(_) | CallOr(..)));
        self.arguments.is_empty() || arity <= placed.count()
    }

    pub(crate) fn overload_types(
        &self,
        signature: &Signature,
        returns: &ReturnRule,
        result: &DataType,
        types: &[DataType],
    ) -> Vec<Option<DataType>> {
        use ReturnRule::{AvgOf, Fixed, Mod, Product, Promote, Quotient, Sum};
        if self.name.urn == SEMSTRAIT {
            return vec![None; types.len()];
        }
        let promoted = promote(&SubstraitTypeSystem, types);
        let decimal = self.name.urn == ARITHMETIC_DECIMAL;
        let compared = signature.constraints.contains(&Constraint::OneFamily);
        let widened = matches!(result, DataType::Double | DataType::String);
        let overload = |data_type: &DataType| match returns {
            Sum | Product | Quotient | Mod if decimal => Decimal::of(data_type).map(DataType::from),
            Sum | Product | Quotient | Mod | Promote => Some(promoted.clone()),
            Fixed(_) if compared => Some(promoted.clone()),
            Fixed(_) if widened && data_type.family() == result.family() => Some(result.clone()),
            AvgOf if SubstraitTypeSystem.avg_of(data_type) != *result => Some(result.clone()),
            _ => None,
        };
        (types.iter())
            .map(|data_type| {
                overload(data_type)
                    .filter(|overload| overload != data_type && *overload != DataType::Unknown)
            })
            .collect()
    }
}

use Argument::{Call, CallOr, Enum, Literal};
use Constant::{BigInt, Integer};

const REGEXP: &[(&str, &str)] = &[
    ("case_sensitivity", "CASE_SENSITIVE"),
    ("multiline", "MULTILINE_DISABLED"),
    ("dotall", "DOTALL_DISABLED"),
];
const SWAPPED: &[Argument] = &[Call(1), Call(0)];
const ROUND: Form = form(ROUNDING, "round")
    .with_arguments(&[Call(0), CallOr(1, Integer(0))])
    .with_options(&[("rounding", "TIE_AWAY_FROM_ZERO")]);
const TRIM: &[Argument] = &[Call(0), CallOr(1, Constant::String(" "))];
const PAD: &[Argument] = &[Call(0), Call(1), CallOr(2, Constant::String(" "))];

static SAME_NAME: &[(&str, &[&str])] = &[
    (
        STRING,
        &[
            "upper",
            "lower",
            "concat_ws",
            "replace",
            "reverse",
            "repeat",
            "left",
            "right",
            "starts_with",
            "ends_with",
            "contains",
            "initcap",
            "octet_length",
            "bit_length",
            "concat",
            "string_agg",
        ],
    ),
    (
        ARITHMETIC,
        &[
            "exp",
            "power",
            "sign",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "atan2",
            "degrees",
            "radians",
            "rank",
            "row_number",
        ],
    ),
    (ROUNDING, &["ceil", "floor"]),
    (LOGARITHMIC, &["ln", "log10", "log2"]),
    (COMPARISON, &["coalesce", "nullif", "greatest", "least"]),
    (AGGREGATE_GENERIC, &["count"]),
    (BOOLEAN, &["bool_and", "bool_or"]),
    (AGGREGATE_APPROX, &["approx_count_distinct"]),
];

static FORMS: &[(&str, Form)] = &[
    (
        "substring",
        form(STRING, "substring").with_options(&[("negative_start", "LEFT_OF_BEGINNING")]),
    ),
    ("length", form(STRING, "char_length")),
    ("position", form(STRING, "strpos").with_arguments(SWAPPED)),
    ("trim", form(STRING, "trim").with_arguments(TRIM)),
    ("ltrim", form(STRING, "ltrim").with_arguments(TRIM)),
    ("rtrim", form(STRING, "rtrim").with_arguments(TRIM)),
    ("lpad", form(STRING, "lpad").with_arguments(PAD)),
    ("rpad", form(STRING, "rpad").with_arguments(PAD)),
    (
        "regexp_extract",
        form(STRING, "regexp_match_substring")
            .with_arguments(&[
                Call(0),
                Call(1),
                Literal(BigInt(1)),
                Literal(BigInt(1)),
                Call(2),
            ])
            .with_options(REGEXP),
    ),
    (
        "regexp_replace",
        form(STRING, "regexp_replace")
            .with_arguments(&[
                Call(0),
                Call(1),
                Call(2),
                Literal(BigInt(1)),
                Literal(BigInt(0)),
            ])
            .with_options(REGEXP),
    ),
    (
        "regexp_count",
        form(STRING, "regexp_count_substring")
            .with_arguments(&[Call(0), Call(1), Literal(BigInt(1))])
            .with_options(REGEXP),
    ),
    (
        "regexp_instr",
        form(STRING, "regexp_strpos")
            .with_arguments(&[Call(0), Call(1), Literal(BigInt(1)), Literal(BigInt(1))])
            .with_options(REGEXP),
    ),
    ("round", ROUND),
    (
        "sqrt",
        form(ARITHMETIC, "sqrt").with_options(&[("on_domain_error", "ERROR")]),
    ),
    ("log", form(LOGARITHMIC, "logb").with_arguments(SWAPPED)),
    (
        "any_value",
        form(AGGREGATE_GENERIC, "any_value").with_options(&[("ignore_nulls", "TRUE")]),
    ),
    (
        "stddev_samp",
        form(ARITHMETIC, "std_dev").with_options(&[("distribution", "SAMPLE")]),
    ),
    (
        "stddev_pop",
        form(ARITHMETIC, "std_dev").with_options(&[("distribution", "POPULATION")]),
    ),
    (
        "var_samp",
        form(ARITHMETIC, "variance").with_options(&[("distribution", "SAMPLE")]),
    ),
    (
        "var_pop",
        form(ARITHMETIC, "variance").with_options(&[("distribution", "POPULATION")]),
    ),
    ("corr", form(ARITHMETIC, "corr").with_arguments(SWAPPED)),
    (
        "median",
        form(ARITHMETIC, "median").with_arguments(&[Enum("EXACT"), Call(0)]),
    ),
];

pub(crate) fn function(canonical: &'static str, operands: Operands, arity: usize) -> Form {
    let same_name = || {
        (SAME_NAME.iter())
            .find(|(_, names)| names.contains(&canonical))
            .map(|&(urn, _)| form(urn, canonical))
    };
    let row = match (canonical, operands) {
        ("sum" | "avg" | "min" | "max" | "abs", _) => Some(arithmetic(canonical, operands)),
        ("ceil" | "floor", Operands::Decimals) => Some(form(ROUNDING_DECIMAL, canonical)),
        ("round", Operands::Decimals) => Some(Form {
            name: FunctionName {
                urn: ROUNDING_DECIMAL,
                ..ROUND.name
            },
            ..ROUND
        }),
        ("median", Operands::Decimals) => None,
        _ => (FORMS.iter())
            .find(|(name, _)| *name == canonical)
            .map(|&(_, form)| form)
            .or_else(same_name),
    };
    let form = (row.filter(|row| row.places(arity))).unwrap_or(form(SEMSTRAIT, canonical));
    form.on(operands)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Operands {
    Integers,
    Floats,
    Decimals,
    Datetimes,
    NoOverload,
    Others,
}

impl Operands {
    pub(crate) fn of(types: &[DataType]) -> Self {
        let any = |test: fn(&DataType) -> bool| types.iter().any(test);
        let interval = |data_type: &DataType| {
            matches!(
                data_type,
                DataType::Interval
                    | DataType::IntervalYearMonth
                    | DataType::IntervalDaySecond { .. }
            )
        };
        let datetime =
            |data_type: &DataType| matches!(data_type, DataType::Date | DataType::Timestamp { .. });
        let no_overload =
            |data_type: &DataType| matches!(data_type, DataType::Time { .. } | DataType::Interval);
        let float = |data_type: &DataType| matches!(data_type, DataType::Real | DataType::Double);
        if any(|data_type| matches!(data_type, DataType::Decimal { .. })) {
            Self::Decimals
        } else if any(no_overload) {
            Self::NoOverload
        } else if any(datetime) || any(interval) {
            Self::Datetimes
        } else if any(|data_type| data_type.family() != Some(TypeFamily::Number)) {
            Self::Others
        } else if any(float) {
            Self::Floats
        } else {
            Self::Integers
        }
    }
}

pub(crate) fn binary(op: BinaryOpKind, operands: Operands) -> Form {
    let form = match op {
        BinaryOpKind::Add => arithmetic("add", operands),
        BinaryOpKind::Subtract => arithmetic("subtract", operands),
        BinaryOpKind::Multiply => arithmetic("multiply", operands),
        BinaryOpKind::Divide | BinaryOpKind::SafeDivide => arithmetic("divide", operands),
        BinaryOpKind::Mod if operands == Operands::Integers => {
            arithmetic("modulus", operands).with_options(&[("on_domain_error", "ERROR")])
        }
        BinaryOpKind::Mod if operands == Operands::Decimals => arithmetic("modulus", operands),
        BinaryOpKind::Eq => form(COMPARISON, "equal"),
        BinaryOpKind::NotEq => form(COMPARISON, "not_equal"),
        BinaryOpKind::Lt => form(COMPARISON, "lt"),
        BinaryOpKind::LtEq => form(COMPARISON, "lte"),
        BinaryOpKind::Gt => form(COMPARISON, "gt"),
        BinaryOpKind::GtEq => form(COMPARISON, "gte"),
        BinaryOpKind::IsNotDistinctFrom => form(COMPARISON, "is_not_distinct_from"),
        BinaryOpKind::And => form(BOOLEAN, "and"),
        BinaryOpKind::Or => form(BOOLEAN, "or"),
        _ => form(SEMSTRAIT, op.operation().canonical.name),
    };
    form.on(operands)
}

fn arithmetic(name: &'static str, operands: Operands) -> Form {
    match operands {
        Operands::Integers | Operands::Floats => form(ARITHMETIC, name),
        Operands::Decimals => form(ARITHMETIC_DECIMAL, name),
        Operands::Datetimes => form(DATETIME, name),
        Operands::NoOverload | Operands::Others => form(SEMSTRAIT, name),
    }
}

pub(crate) fn unary(op: UnaryOpKind, operands: Operands) -> Form {
    let form = match (op, operands) {
        (UnaryOpKind::Negate, Operands::Integers | Operands::Floats) => form(ARITHMETIC, "negate"),
        (UnaryOpKind::Not, _) => NOT,
        _ => form(SEMSTRAIT, op.operation().canonical.name),
    };
    form.on(operands)
}

pub(crate) fn like(kind: LikeKind) -> Form {
    let like = form(STRING, "like").on(Operands::Others);
    match kind {
        LikeKind::ILike | LikeKind::NotILike => {
            like.with_options(&[("case_sensitivity", "CASE_INSENSITIVE")])
        }
        _ => like,
    }
}

pub(crate) const NOT: Form = form(BOOLEAN, "not");

pub(crate) const IS_NULL: Form = form(COMPARISON, "is_null");

pub(crate) const EXTRACT: Form = form(DATETIME, "extract");

pub(crate) fn component(component: &str) -> Option<(&'static str, bool)> {
    Some(match component {
        "year" => ("YEAR", false),
        "quarter" => ("QUARTER", true),
        "month" => ("MONTH", true),
        "week" => ("ISO_WEEK", true),
        "day" => ("DAY", true),
        "day_of_week" => ("MONDAY_DAY_OF_WEEK", true),
        "day_of_year" => ("DAY_OF_YEAR", true),
        "hour" => ("HOUR", false),
        "minute" => ("MINUTE", false),
        "second" => ("SECOND", false),
        _ => return None,
    })
}

pub(crate) const ASSUME_TIMEZONE: Form = form(DATETIME, "assume_timezone");
pub(crate) const ROUND_CALENDAR: Form = form(DATETIME, "round_calendar");
pub(crate) const LOCAL_TIMESTAMP: Form = form(DATETIME, "local_timestamp");

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Calendar {
    pub unit: &'static str,
    pub multiple: i64,
    pub origin: &'static str,
    pub anchor: &'static str,
}

pub(crate) fn calendar(grain: &str, week_start: Option<&str>) -> Option<Calendar> {
    const EPOCH: &str = "1970-01-01";
    let (unit, multiple, origin, anchor) = match (grain, week_start) {
        ("minute", _) => ("MINUTE", 1, "HOUR", EPOCH),
        ("hour", _) => ("HOUR", 1, "DAY", EPOCH),
        ("day", _) => ("DAY", 1, "MONTH", EPOCH),
        ("week", Some("monday")) => ("WEEK", 1, "ISO_WEEK", "1970-01-05"),
        ("week", Some("sunday")) => ("WEEK", 1, "US_WEEK", "1970-01-04"),
        ("month", _) => ("MONTH", 1, "YEAR", EPOCH),
        ("quarter", _) => ("MONTH", 3, "YEAR", EPOCH),
        ("year", _) => ("YEAR", 1, "YEAR", EPOCH),
        _ => return None,
    };
    Some(Calendar {
        unit,
        multiple,
        origin,
        anchor,
    })
}
