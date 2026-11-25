use super::data_type::resolved;
use super::error::ParseError;
use super::leaf::{authored_reference, LeafResolver};
use super::literal::{lit, untyped_literal, LIT};
use super::token::{is_compound_expr_string, tokenize_leaf, LeafToken};
use super::value::{deny_unknown_fields, get_optional_value, lookup};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    AggregateAllowed, BinaryOpKind, Call, CallError, CastFailure, Expr, FunctionRef, Leaf,
    LikeKind, Literal, NoAggregate, OperationKind, RowExpr, UnaryOpKind,
};
use semstrait_sql::functions::{registry, ArgumentForm, ParameterDecl};
use serde_yaml::{Mapping, Value};
use std::collections::BTreeMap;
use std::slice;

pub fn parse_expr<L: LeafResolver>(value: &Value) -> Result<RowExpr<L>, ParseError> {
    match value {
        Value::String(s) if is_compound_expr_string(s) => {
            Err(ParseError::InlineExprUnsupported { expr: s.clone() })
        }
        Value::String(_) | Value::Number(_) | Value::Bool(_) | Value::Null | Value::Mapping(_) => {
            parse_value::<L>(value)
        }
        other => Err(ParseError::UnexpectedShape(format!(
            "top-level expects string, scalar, or mapping; got {other:?}"
        ))),
    }
}

fn parse_value<L: LeafResolver>(value: &Value) -> Result<RowExpr<L>, ParseError> {
    match value {
        Value::String(s) => match tokenize_leaf(s)? {
            LeafToken::Literal(literal) => Ok(untyped(literal)),
            LeafToken::Name(n) => authored_reference(n, "expression").map(L::bare_name),
            LeafToken::NameWithAccessor { .. } => Err(ParseError::InvalidValue {
                field: "expression",
                reason: "accessor only valid in dim:/measure:/metric: bodies".into(),
            }),
        },
        Value::Number(_) | Value::Bool(_) | Value::Null => Ok(untyped(untyped_literal(value)?)),
        Value::Mapping(m) => parse_tagged_mapping::<L>(m),
        other => Err(ParseError::UnexpectedShape(format!(
            "expected scalar, mapping, or single-key tag inside an expression; got {other:?}"
        ))),
    }
}

fn untyped<L: LeafResolver>(literal: Literal) -> RowExpr<L> {
    Expr::Leaf(L::from_untyped(literal))
}

fn parse_tagged_mapping<L: LeafResolver>(map: &Mapping) -> Result<RowExpr<L>, ParseError> {
    let mut entries = map.iter();
    let (Some((k, v)), None) = (entries.next(), entries.next()) else {
        return Err(ParseError::AmbiguousTag(map.len()));
    };
    let tag = k
        .as_str()
        .ok_or_else(|| ParseError::UnexpectedShape("tag key must be a string".into()))?;

    if let Some(operation) = OperationKind::from_name(tag) {
        return match operation {
            OperationKind::Binary(op) => parse_binary_op::<L>(op, v),
            OperationKind::Unary(UnaryOpKind::Not) => parse_not_fold::<L>(v),
            OperationKind::Unary(op) => Ok(Expr::UnaryOp {
                op,
                operand: Box::new(parse_value::<L>(v)?),
            }),
            OperationKind::Like(kind) => parse_like_kind::<L>(v, kind),
            OperationKind::Between => parse_between::<L>(v, false),
            OperationKind::InList => parse_in::<L>(v),
            OperationKind::IsNull => Ok(Expr::IsNull(Box::new(parse_value::<L>(v)?))),
            _ => Err(ParseError::UnknownTag(tag.to_owned())),
        };
    }

    let Some(tree_tag) = TreeTag::of(tag) else {
        return match registry().get(tag) {
            Some(function) => parse_call::<L>(tag, function, v),
            None => Err(ParseError::UnknownTag(tag.to_owned())),
        };
    };
    match tree_tag {
        TreeTag::Lit => lit(v, Expr::Literal, untyped),
        TreeTag::Col => {
            let name = v.as_str().ok_or_else(|| ParseError::InvalidValue {
                field: "col",
                reason: "expected string name".into(),
            })?;
            authored_reference(name, "col").map(Expr::Column)
        }
        TreeTag::Semantic => match L::from_semantic_tag(tag, v)? {
            Some(leaf) => Ok(Expr::Leaf(leaf)),
            None => Err(ParseError::TagNotAllowedAtSite {
                tag: tag.to_owned(),
                site: "physical-mapping",
            }),
        },
        TreeTag::Case => parse_case::<L>(v),
        TreeTag::If => parse_if::<L>(v),
        TreeTag::Cast => parse_cast::<L>(v),
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum TreeTag {
    Lit,
    Col,
    Semantic,
    Case,
    If,
    Cast,
}

impl TreeTag {
    pub(crate) const ALL: [(&'static str, Self); 9] = [
        (LIT, Self::Lit),
        ("col", Self::Col),
        ("field", Self::Semantic),
        ("dim", Self::Semantic),
        ("measure", Self::Semantic),
        ("metric", Self::Semantic),
        ("case", Self::Case),
        ("if", Self::If),
        ("cast", Self::Cast),
    ];

    fn of(name: &str) -> Option<Self> {
        Self::ALL
            .iter()
            .find_map(|&(tag, tree_tag)| (tag == name).then_some(tree_tag))
    }
}

fn parse_call<L: LeafResolver>(
    tag: &str,
    function: FunctionRef<AggregateAllowed>,
    body: &Value,
) -> Result<RowExpr<L>, ParseError> {
    let function = FunctionRef::<NoAggregate>::try_from(function).map_err(|_| {
        ParseError::AggregateInRowExpr {
            function: tag.to_owned(),
        }
    })?;
    let refused = |reason| ParseError::CallRefused {
        function: tag.to_owned(),
        reason,
    };
    let args = match body {
        Value::Sequence(args) => read_args(function, args)?,
        Value::Mapping(named)
            if named.len() > 1
                || (named.keys().next().and_then(Value::as_str)).is_some_and(|key| {
                    function.parameter(key).is_ok()
                        && TreeTag::of(key).is_none()
                        && registry().get(key).is_none()
                }) =>
        {
            read_named_args(tag, function, named)?
        }
        one => read_args(function, slice::from_ref(one))?,
    };
    let call = Call::new(function, args).map_err(|reason| refused(reason.to_string()))?;
    check_group(&call).map_err(refused)?;
    Ok(Expr::Function(call))
}

fn check_group<L: Leaf>(call: &Call<L, NoAggregate>) -> Result<(), String> {
    let ("regexp_extract", [_, pattern, group]) = (call.name(), call.args()) else {
        return Ok(());
    };
    let Some(Ok(Literal::BigInt(group))) =
        (group.literal()).map(|group| group.read_as(&DataType::BigInt))
    else {
        return Ok(());
    };
    if group < 0 {
        return Err(format!(
            "`regexp_extract` has no group {group}: a group is not below 0"
        ));
    }
    let Some(
        Literal::String(pattern)
        | Literal::VarChar { value: pattern, .. }
        | Literal::FixedChar { value: pattern, .. },
    ) = pattern.literal()
    else {
        return Ok(());
    };
    match regex_syntax::parse(pattern).map(|pattern| pattern.properties().explicit_captures_len()) {
        Ok(last) if usize::try_from(group).is_ok_and(|group| group > last) => Err(format!(
            "`regexp_extract`'s `pattern` has no group {group}: its last group is {last}"
        )),
        _ => Ok(()),
    }
}

fn read_args<L: LeafResolver>(
    function: FunctionRef<NoAggregate>,
    values: &[Value],
) -> Result<Vec<RowExpr<L>>, ParseError> {
    let mut args = values
        .iter()
        .map(parse_value::<L>)
        .collect::<Result<Vec<_>, _>>()?;
    let mut words: Vec<_> = values.iter().map(word).collect();
    while let Ok(parameters) = function.parameters(&args) {
        let Some((arg, word)) = (args.iter_mut().zip(&mut words).zip(parameters))
            .filter(|(_, parameter)| takes_a_literal(parameter))
            .find_map(|((arg, word), _)| Some((arg, word.take()?)))
        else {
            break;
        };
        *arg = untyped(Literal::String(word));
    }
    Ok(function.arguments_by_position(args))
}

fn read_named_args<L: LeafResolver>(
    tag: &str,
    function: FunctionRef<NoAggregate>,
    named: &Mapping,
) -> Result<Vec<RowExpr<L>>, ParseError> {
    let refused = |refusal| mapping_refusal(tag, function, refusal);
    let parameters = (named.keys())
        .map(|key| {
            let key = match key.as_str() {
                Some(key) => key.to_owned(),
                None => serde_yaml::to_string(key)
                    .map_err(|error| ParseError::UnexpectedShape(error.to_string()))?
                    .trim_end()
                    .to_owned(),
            };
            function.parameter(&key).map_err(refused)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let args = (parameters.into_iter().zip(named.values()))
        .map(|(parameter, value)| {
            let arg = match word(value) {
                Some(word) if takes_a_literal(parameter) => untyped(Literal::String(word)),
                _ => parse_value::<L>(value)?,
            };
            Ok((parameter.name, arg))
        })
        .collect::<Result<BTreeMap<_, _>, ParseError>>()?;
    function.arguments_by_name(args).map_err(refused)
}

fn mapping_refusal(
    tag: &str,
    function: FunctionRef<NoAggregate>,
    refusal: CallError,
) -> ParseError {
    let function_name = tag.to_owned();
    match refusal {
        CallError::UnknownParameter { name, .. } => ParseError::UnknownParameter {
            function: function_name,
            key: name,
            parameters: (function.canonical().1.parameters.iter())
                .map(|parameter| parameter.name)
                .collect(),
        },
        CallError::MissingParameter { parameter, .. } => ParseError::MissingParameter {
            function: function_name,
            parameter,
        },
        CallError::UnwrittenParameter { parameter, .. } => ParseError::UnwrittenParameter {
            function: function_name,
            parameter,
        },
        other => ParseError::CallRefused {
            function: function_name,
            reason: other.to_string(),
        },
    }
}

fn word(value: &Value) -> Option<String> {
    match tokenize_leaf(value.as_str()?) {
        Ok(LeafToken::Name(word)) => Some(word),
        _ => None,
    }
}

fn takes_a_literal(parameter: &ParameterDecl) -> bool {
    matches!(
        parameter.form,
        ArgumentForm::Enum(_) | ArgumentForm::Constant
    )
}

fn parse_binary_op<L: LeafResolver>(
    op: BinaryOpKind,
    body: &Value,
) -> Result<RowExpr<L>, ParseError> {
    let (left, right) = parse_pair_body(body, "binary_op")?;
    Ok(Expr::BinaryOp {
        op,
        left: Box::new(parse_value::<L>(left)?),
        right: Box::new(parse_value::<L>(right)?),
    })
}

fn parse_pair_body<'v>(
    body: &'v Value,
    tag: &'static str,
) -> Result<(&'v Value, &'v Value), ParseError> {
    if let Some(seq) = body.as_sequence() {
        if seq.len() != 2 {
            return Err(ParseError::InvalidValue {
                field: tag,
                reason: format!("expected 2-element sequence, got {}", seq.len()),
            });
        }
        return Ok((&seq[0], &seq[1]));
    }
    let m = body.as_mapping().ok_or_else(|| ParseError::InvalidValue {
        field: tag,
        reason: "expected `[left, right]` sequence or `{ left, right }` mapping".into(),
    })?;
    let left = get_required_value(m, "left", "left")?;
    let right = get_required_value(m, "right", "right")?;
    deny_unknown_fields(m, tag, &["left", "right"])?;
    Ok((left, right))
}

const LIKE_NEGATIONS: [(LikeKind, LikeKind); 2] = [
    (LikeKind::Like, LikeKind::NotLike),
    (LikeKind::ILike, LikeKind::NotILike),
];

fn parse_not_fold<L: LeafResolver>(body: &Value) -> Result<RowExpr<L>, ParseError> {
    Ok(match parse_value::<L>(body)? {
        Expr::InList {
            value,
            list,
            negated,
        } => Expr::InList {
            value,
            list,
            negated: !negated,
        },
        Expr::Like {
            value,
            pattern,
            kind,
        } => {
            let negation = LIKE_NEGATIONS.iter().find_map(|&(like, negation)| {
                (kind == like)
                    .then_some(negation)
                    .or((kind == negation).then_some(like))
            });
            let like = Expr::Like {
                value,
                pattern,
                kind: negation.unwrap_or(kind),
            };
            if negation.is_some() {
                like
            } else {
                !like
            }
        }
        Expr::Between {
            value,
            low,
            high,
            negated,
        } => Expr::Between {
            value,
            low,
            high,
            negated: !negated,
        },
        other => !other,
    })
}

fn parse_case<L: LeafResolver>(value: &Value) -> Result<RowExpr<L>, ParseError> {
    let m = value.as_mapping().ok_or_else(|| ParseError::InvalidValue {
        field: "case",
        reason: "expected mapping with `whens`, optional `else`".into(),
    })?;
    let whens_node = get_required_value(m, "whens", "case.whens")?;
    let whens_seq = whens_node
        .as_sequence()
        .ok_or_else(|| ParseError::InvalidValue {
            field: "case.whens",
            reason: "expected sequence of `{ when, then }` pairs".into(),
        })?;
    let mut whens: Vec<(RowExpr<L>, RowExpr<L>)> = Vec::with_capacity(whens_seq.len());
    for entry in whens_seq {
        let pair = entry.as_mapping().ok_or_else(|| ParseError::InvalidValue {
            field: "case.whens[]",
            reason: "expected `{ when, then }` mapping".into(),
        })?;
        let when_node = get_required_value(pair, "when", "case.whens[].when")?;
        let then_node = get_required_value(pair, "then", "case.whens[].then")?;
        let when = parse_value::<L>(when_node)?;
        let then = parse_value::<L>(then_node)?;
        deny_unknown_fields(pair, "case.whens[]", &["when", "then"])?;
        whens.push((when, then));
    }
    let else_ = match get_optional_value(m, "else") {
        Some(v) => Some(Box::new(parse_value::<L>(v)?)),
        None => None,
    };
    deny_unknown_fields(m, "case", &["whens", "else"])?;
    (Expr::Case { whens, else_ })
        .checked()
        .map_err(ParseError::InvalidExpression)
}

fn parse_if<L: LeafResolver>(value: &Value) -> Result<RowExpr<L>, ParseError> {
    let m = value.as_mapping().ok_or_else(|| ParseError::InvalidValue {
        field: "if",
        reason: "expected mapping with `cond`, `then`, optional `else`".into(),
    })?;
    let cond = parse_value::<L>(get_required_value(m, "cond", "if.cond")?)?;
    let then = parse_value::<L>(get_required_value(m, "then", "if.then")?)?;
    let else_ = match get_optional_value(m, "else") {
        Some(v) => Some(Box::new(parse_value::<L>(v)?)),
        None => None,
    };
    deny_unknown_fields(m, "if", &["cond", "then", "else"])?;
    Ok(Expr::Case {
        whens: vec![(cond, then)],
        else_,
    })
}

pub(crate) const LIST: &str = "list";

fn parse_in<L: LeafResolver>(value: &Value) -> Result<RowExpr<L>, ParseError> {
    let m = value.as_mapping().ok_or_else(|| ParseError::InvalidValue {
        field: "in",
        reason: "expected mapping with `value`, `list`".into(),
    })?;
    let value_expr = parse_value::<L>(get_required_value(m, "value", "in.value")?)?;
    let list = get_required_value(m, LIST, "in.list")?
        .as_sequence()
        .ok_or_else(|| ParseError::InvalidValue {
            field: "in.list",
            reason: "expected sequence".into(),
        })?
        .iter()
        .map(parse_value::<L>)
        .collect::<Result<_, _>>()?;
    deny_unknown_fields(m, "in", &["value", LIST])?;
    let in_list = Expr::InList {
        value: Box::new(value_expr),
        list,
        negated: false,
    };
    in_list.checked().map_err(ParseError::InvalidExpression)
}

fn parse_between<L: LeafResolver>(value: &Value, negated: bool) -> Result<RowExpr<L>, ParseError> {
    let m = value.as_mapping().ok_or_else(|| ParseError::InvalidValue {
        field: "between",
        reason: "expected mapping with `value`, `low`, `high`".into(),
    })?;
    let v_expr = parse_value::<L>(get_required_value(m, "value", "between.value")?)?;
    let low = parse_value::<L>(get_required_value(m, "low", "between.low")?)?;
    let high = parse_value::<L>(get_required_value(m, "high", "between.high")?)?;
    deny_unknown_fields(m, "between", &["value", "low", "high"])?;
    Ok(Expr::Between {
        value: Box::new(v_expr),
        low: Box::new(low),
        high: Box::new(high),
        negated,
    })
}

fn parse_like_kind<L: LeafResolver>(
    value: &Value,
    kind: LikeKind,
) -> Result<RowExpr<L>, ParseError> {
    let m = value.as_mapping().ok_or_else(|| ParseError::InvalidValue {
        field: "like",
        reason: "expected mapping with `value`, `pattern`".into(),
    })?;
    let v_expr = parse_value::<L>(get_required_value(m, "value", "like.value")?)?;
    let pattern = parse_value::<L>(get_required_value(m, "pattern", "like.pattern")?)?;
    deny_unknown_fields(m, "like", &["value", "pattern"])?;
    Ok(Expr::Like {
        value: Box::new(v_expr),
        pattern: Box::new(pattern),
        kind,
    })
}

fn parse_cast<L: LeafResolver>(value: &Value) -> Result<RowExpr<L>, ParseError> {
    let m = value.as_mapping().ok_or_else(|| ParseError::InvalidValue {
        field: "cast",
        reason: "expected mapping with `input`, `target`, optional `on_failure`".into(),
    })?;
    let input = parse_value::<L>(get_required_value(m, "input", "cast.input")?)?;
    let target = resolved(get_required_value(m, "target", "cast.target")?)?;
    let on_failure = match get_optional_str(m, "on_failure")? {
        Some(s) => parse_cast_failure(s)?,
        None => CastFailure::Error,
    };
    deny_unknown_fields(m, "cast", &["input", "target", "on_failure"])?;
    Ok(Expr::Cast {
        input: Box::new(input),
        target,
        on_failure,
    })
}

pub(crate) const CAST_FAILURES: [(&str, CastFailure); 2] =
    [("error", CastFailure::Error), ("null", CastFailure::Null)];

fn parse_cast_failure(s: &str) -> Result<CastFailure, ParseError> {
    CAST_FAILURES
        .iter()
        .find_map(|&(word, failure)| (word == s).then_some(failure))
        .ok_or_else(|| ParseError::InvalidValue {
            field: "on_failure",
            reason: format!("unknown CastFailure `{s}`"),
        })
}

fn get_required_value<'m>(
    map: &'m Mapping,
    key: &'static str,
    field: &'static str,
) -> Result<&'m Value, ParseError> {
    lookup(map, key).ok_or(ParseError::MissingField(field))
}

fn get_optional_str<'m>(map: &'m Mapping, key: &str) -> Result<Option<&'m str>, ParseError> {
    match lookup(map, key) {
        Some(v) => v
            .as_str()
            .map(Some)
            .ok_or_else(|| ParseError::InvalidValue {
                field: "value",
                reason: format!("expected string for `{key}`"),
            }),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{col, field, physical_col};
    use semstrait_sql::datatype::DataType;
    use semstrait_sql::expr::{
        DimensionAccessor, MeasureAccessor, NamedReference, PhysicalLeaf, SemanticLeaf,
        UntypedLiteral,
    };

    fn parse_yaml<L: LeafResolver>(yaml: &str) -> RowExpr<L> {
        let value: Value = serde_yaml::from_str(yaml).expect("valid YAML");
        parse_expr::<L>(&value).expect("parse_expr succeeds")
    }

    fn try_parse_yaml<L: LeafResolver>(yaml: &str) -> Result<RowExpr<L>, ParseError> {
        let value: Value = serde_yaml::from_str(yaml).expect("valid YAML");
        parse_expr::<L>(&value)
    }

    fn unwrap_block<L: LeafResolver>(expr: RowExpr<L>) -> RowExpr<L> {
        expr
    }

    #[test]
    fn top_level_bare_string_becomes_field_at_semantic_site() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("revenue"));
        assert_eq!(e, field("revenue").expect("a name"));
    }

    #[test]
    fn top_level_bare_string_becomes_column_at_physical_site() {
        let e = unwrap_block(parse_yaml::<PhysicalLeaf>("amount"));
        assert_eq!(e, physical_col("amount").expect("a name"));
    }

    #[test]
    fn top_level_compound_string_is_rejected() {
        for yaml in [
            r#""a / b""#,
            r#""x != 'y'""#,
            r#""(a) + b""#,
            r#""a * 2""#,
            r#""a b""#,
        ] {
            let err = try_parse_yaml::<SemanticLeaf>(yaml).unwrap_err();
            assert!(
                matches!(err, ParseError::InlineExprUnsupported { .. }),
                "{yaml:?} → {err:?}"
            );
        }
    }

    #[test]
    fn top_level_dash_name_is_leaf_not_rejected() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("shopify-shipping_country"));
        assert_eq!(e, field("shopify-shipping_country").expect("a name"));
    }

    #[test]
    fn top_level_dotted_accessor_is_rejected_as_misplaced_accessor() {
        let err = try_parse_yaml::<SemanticLeaf>("country.previous").unwrap_err();
        assert!(matches!(
            err,
            ParseError::InvalidValue {
                field: "expression",
                ..
            }
        ));
    }

    #[test]
    fn top_level_quoted_string_literal_is_leaf() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("\"'Germany'\""));
        assert_eq!(
            e,
            Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::String(
                "Germany".into()
            ))))
        );
    }

    #[test]
    fn top_level_quoted_dotted_name_is_leaf() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(r#"'"a.b"'"#));
        assert_eq!(e, field("a.b").expect("a name"));
    }

    #[test]
    fn bare_int_top_level_becomes_literal_block() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("42"));
        assert_eq!(
            e,
            Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::BigInt(42))))
        );
    }

    #[test]
    fn bare_float_top_level_becomes_literal_block() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("1.5"));
        assert_eq!(
            e,
            Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::Double(1.5))))
        );
    }

    #[test]
    fn bare_bool_top_level_becomes_literal_block() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("true"));
        assert_eq!(
            e,
            Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::Boolean(
                true
            ))))
        );
    }

    #[test]
    fn bare_null_top_level_becomes_literal_block() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("null"));
        assert_eq!(
            e,
            Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::Null(
                DataType::String
            ))))
        );
    }

    #[test]
    fn bare_identifier_inside_expr_becomes_field_at_semantic_site() {
        let yaml = "{ add: [revenue, cost] }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::BinaryOp { op, left, right } => {
                assert_eq!(op, BinaryOpKind::Add);
                assert_eq!(*left, field("revenue").expect("a name"));
                assert_eq!(*right, field("cost").expect("a name"));
            }
            other => panic!("expected BinaryOp, got {other:?}"),
        }
    }

    #[test]
    fn bare_identifier_inside_expr_becomes_column_at_physical_site() {
        let yaml = "{ add: [amount, tax] }";
        let e = unwrap_block(parse_yaml::<PhysicalLeaf>(yaml));
        match e {
            Expr::BinaryOp { left, right, .. } => {
                assert_eq!(*left, physical_col("amount").expect("a name"));
                assert_eq!(*right, physical_col("tax").expect("a name"));
            }
            other => panic!("expected BinaryOp, got {other:?}"),
        }
    }

    #[test]
    fn quoted_string_inside_expr_becomes_string_literal() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("{ eq: [country, \"'DE'\"] }"));
        match e {
            Expr::BinaryOp { right, .. } => {
                assert_eq!(
                    *right,
                    Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::String(
                        "DE".into()
                    ))))
                );
            }
            other => panic!("expected BinaryOp, got {other:?}"),
        }
    }

    #[test]
    fn accessor_in_plain_expr_is_rejected() {
        let err = try_parse_yaml::<SemanticLeaf>("{ add: [revenue.previous, 1] }").unwrap_err();
        assert!(matches!(
            err,
            ParseError::InvalidValue {
                field: "expression",
                ..
            }
        ));
    }

    #[test]
    fn sugar_binary_seq_form() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("{ add: [1, 2] }"));
        match e {
            Expr::BinaryOp { op, .. } => assert_eq!(op, BinaryOpKind::Add),
            other => panic!("expected BinaryOp, got {other:?}"),
        }
    }

    #[test]
    fn sugar_binary_pair_form() {
        let yaml = "{ subtract: { left: 10, right: 3 } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::BinaryOp { op, left, right } => {
                assert_eq!(op, BinaryOpKind::Subtract);
                assert_eq!(
                    *left,
                    Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::BigInt(10))))
                );
                assert_eq!(
                    *right,
                    Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::BigInt(3))))
                );
            }
            other => panic!("expected BinaryOp, got {other:?}"),
        }
    }

    #[test]
    fn sugar_binary_eq_lt_gteq_and_or() {
        for (tag, want) in [
            ("eq", BinaryOpKind::Eq),
            ("not_eq", BinaryOpKind::NotEq),
            ("lt", BinaryOpKind::Lt),
            ("lt_eq", BinaryOpKind::LtEq),
            ("gt", BinaryOpKind::Gt),
            ("gt_eq", BinaryOpKind::GtEq),
            ("and", BinaryOpKind::And),
            ("or", BinaryOpKind::Or),
            ("multiply", BinaryOpKind::Multiply),
            ("divide", BinaryOpKind::Divide),
            ("safe_divide", BinaryOpKind::SafeDivide),
            ("mod", BinaryOpKind::Mod),
        ] {
            let yaml = format!("{{ {tag}: [1, 2] }}");
            let e = unwrap_block(parse_yaml::<SemanticLeaf>(&yaml));
            match e {
                Expr::BinaryOp { op, .. } => {
                    assert_eq!(op, want, "tag {tag}");
                }
                other => panic!("expected BinaryOp for {tag}, got {other:?}"),
            }
        }
    }

    #[test]
    fn sugar_binary_seq_wrong_arity_rejected() {
        let err = try_parse_yaml::<SemanticLeaf>("{ add: [1, 2, 3] }").unwrap_err();
        assert!(matches!(err, ParseError::InvalidValue { .. }));
    }

    #[test]
    fn sugar_unary_negate() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("{ negate: 5 }"));
        match e {
            Expr::UnaryOp { op, operand } => {
                assert_eq!(op, UnaryOpKind::Negate);
                assert_eq!(
                    *operand,
                    Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::BigInt(5))))
                );
            }
            other => panic!("expected UnaryOp, got {other:?}"),
        }
    }

    #[test]
    fn col_at_semantic_site() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("{ col: revenue }"));
        assert_eq!(e, col("revenue").expect("a name"));
    }

    #[test]
    fn col_at_physical_site() {
        let e = unwrap_block(parse_yaml::<PhysicalLeaf>("{ col: revenue }"));
        assert_eq!(e, physical_col("revenue").expect("a name"));
    }

    #[test]
    fn field_tag_short_form() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("{ field: revenue }"));
        assert_eq!(e, field("revenue").expect("a name"));
    }

    #[test]
    fn dim_tag_with_lag_accessor() {
        let yaml = "{ dim: { name: dt, accessor: { lag: 2 } } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        assert_eq!(
            e,
            Expr::Leaf(SemanticLeaf::Dimension(
                NamedReference::unqualified("dt").expect("a name"),
                Some(DimensionAccessor::Lag(2))
            ))
        );
    }

    #[test]
    fn measure_tag_with_previous_accessor() {
        let yaml = "{ measure: { name: revenue, accessor: previous } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        assert_eq!(
            e,
            Expr::Leaf(SemanticLeaf::Measure(
                NamedReference::unqualified("revenue").expect("a name"),
                Some(MeasureAccessor::Previous)
            ))
        );
    }

    #[test]
    fn semantic_tag_at_physical_site_is_rejected() {
        let err = try_parse_yaml::<PhysicalLeaf>("{ field: revenue }").unwrap_err();
        assert!(matches!(
            err,
            ParseError::TagNotAllowedAtSite {
                site: "physical-mapping",
                ..
            }
        ));
    }

    #[test]
    fn dim_at_physical_site_is_rejected() {
        let err = try_parse_yaml::<PhysicalLeaf>("{ dim: x }").unwrap_err();
        assert!(matches!(
            err,
            ParseError::TagNotAllowedAtSite {
                site: "physical-mapping",
                ..
            }
        ));
    }

    #[test]
    fn not_in_flips_in_list_negation() {
        let yaml = "{ not: { in: { value: country, list: [\"'DE'\", \"'FR'\"] } } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::InList { negated, .. } => assert!(negated),
            other => panic!("expected InList(negated), got {other:?}"),
        }
    }

    #[test]
    fn not_like_flips_to_not_like_kind() {
        let yaml = "{ not: { like: { value: name, pattern: \"'%a%'\" } } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::Like { kind, .. } => assert_eq!(kind, LikeKind::NotLike),
            other => panic!("expected Like(NotLike), got {other:?}"),
        }
    }

    #[test]
    fn not_ilike_flips_to_not_ilike_kind() {
        let yaml = "{ not: { ilike: { value: name, pattern: \"'%a%'\" } } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::Like { kind, .. } => assert_eq!(kind, LikeKind::NotILike),
            other => panic!("expected Like(NotILike), got {other:?}"),
        }
    }

    #[test]
    fn not_between_flips_negation() {
        let yaml = "{ not: { between: { value: x, low: 1, high: 10 } } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::Between { negated, .. } => assert!(negated),
            other => panic!("expected Between(negated), got {other:?}"),
        }
    }

    #[test]
    fn not_on_eq_wraps_in_unary_op() {
        let yaml = "{ not: { eq: [a, b] } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::UnaryOp { op, .. } => assert_eq!(op, UnaryOpKind::Not),
            other => panic!("expected UnaryOp(Not), got {other:?}"),
        }
    }

    #[test]
    fn case_with_else() {
        let yaml = r#"
case:
  whens:
    - when: { gt: [revenue, 0] }
      then: { lit: 1 }
  else: { lit: 0 }
"#;
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::Case { whens, else_ } => {
                assert_eq!(whens.len(), 1);
                assert!(else_.is_some());
            }
            other => panic!("expected Case, got {other:?}"),
        }
    }

    #[test]
    fn empty_case_is_rejected() {
        let yaml = "{ case: { whens: [] } }";
        let err = try_parse_yaml::<SemanticLeaf>(yaml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Empty") || msg.contains("invalid expression"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn if_lowers_to_case_with_one_when() {
        let yaml = r#"
if:
  cond: { gt: [revenue, 0] }
  then: { lit: 1 }
  else: { lit: 0 }
"#;
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::Case { whens, else_ } => {
                assert_eq!(whens.len(), 1);
                assert!(else_.is_some());
            }
            other => panic!("expected Case, got {other:?}"),
        }
    }

    #[test]
    fn if_without_else_is_legal() {
        let yaml = r#"
if:
  cond: { eq: [a, b] }
  then: { lit: 1 }
"#;
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::Case { whens, else_ } => {
                assert_eq!(whens.len(), 1);
                assert!(else_.is_none());
            }
            other => panic!("expected Case, got {other:?}"),
        }
    }

    #[test]
    fn in_emits_non_negated() {
        let yaml = "{ in: { value: country, list: [\"'DE'\", \"'FR'\"] } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::InList { negated, list, .. } => {
                assert!(!negated);
                assert_eq!(list.len(), 2);
            }
            other => panic!("expected InList, got {other:?}"),
        }
    }

    #[test]
    fn empty_in_list_is_rejected() {
        let yaml = "{ in: { value: country, list: [] } }";
        let err = try_parse_yaml::<SemanticLeaf>(yaml).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("EmptyInList")
                || msg.contains("Empty")
                || msg.contains("invalid expression"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn between_emits_non_negated() {
        let yaml = "{ between: { value: x, low: 1, high: 10 } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::Between { negated, .. } => assert!(!negated),
            other => panic!("expected Between, got {other:?}"),
        }
    }

    #[test]
    fn like_emits_like_kind() {
        let yaml = "{ like: { value: name, pattern: \"'%a%'\" } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::Like { kind, .. } => assert_eq!(kind, LikeKind::Like),
            other => panic!("expected Like, got {other:?}"),
        }
    }

    #[test]
    fn ilike_emits_ilike_kind() {
        let yaml = "{ ilike: { value: name, pattern: \"'%a%'\" } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::Like { kind, .. } => assert_eq!(kind, LikeKind::ILike),
            other => panic!("expected Like(ILike), got {other:?}"),
        }
    }

    #[test]
    fn is_null_with_bare_arg() {
        let e = unwrap_block(parse_yaml::<SemanticLeaf>("{ is_null: revenue }"));
        assert!(matches!(e, Expr::IsNull(_)));
    }

    #[test]
    fn cast_to_string_default_failure() {
        let yaml = "{ cast: { input: x, target: string } }";
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::Cast {
                target, on_failure, ..
            } => {
                assert!(matches!(target, DataType::String));
                assert!(matches!(on_failure, CastFailure::Error));
            }
            other => panic!("expected Cast, got {other:?}"),
        }
    }

    #[test]
    fn cast_with_on_failure_null() {
        let yaml = r#"{ cast: { input: x, target: integer, on_failure: "null" } }"#;
        let e = unwrap_block(parse_yaml::<SemanticLeaf>(yaml));
        match e {
            Expr::Cast { on_failure, .. } => assert!(matches!(on_failure, CastFailure::Null)),
            other => panic!("expected Cast, got {other:?}"),
        }
    }

    #[test]
    fn unknown_tag_is_rejected() {
        let err = try_parse_yaml::<SemanticLeaf>("{ unknown_tag: x }").unwrap_err();
        assert!(matches!(err, ParseError::UnknownTag(t) if t == "unknown_tag"));
    }

    #[test]
    fn ambiguous_two_key_top_level_is_rejected() {
        let err = try_parse_yaml::<SemanticLeaf>("{ lit: 1, col: x }").unwrap_err();
        assert!(matches!(err, ParseError::AmbiguousTag(2)));
    }

    #[test]
    fn case_unknown_field_is_rejected() {
        let yaml = r#"
case:
  whens: [{ when: a, then: 1 }]
  bogus: 1
"#;
        let err = try_parse_yaml::<SemanticLeaf>(yaml).unwrap_err();
        assert!(matches!(
            err,
            ParseError::InvalidValue { field: "case", .. }
        ));
    }

    #[test]
    fn binary_pair_form_unknown_field_is_rejected() {
        let yaml = "{ add: { left: 1, right: 2, bogus: 3 } }";
        let err = try_parse_yaml::<SemanticLeaf>(yaml).unwrap_err();
        assert!(matches!(err, ParseError::InvalidValue { .. }));
    }

    #[test]
    fn no_function_s_name_or_alias_is_a_tag_of_the_tree() {
        for (tag, _) in TreeTag::ALL {
            assert_eq!(registry().get(tag), None, "`{tag}` names a function");
            let read = try_parse_yaml::<SemanticLeaf>(&format!("{{ {tag}: x }}"));
            assert!(
                !matches!(read, Err(ParseError::UnknownTag(_))),
                "`{tag}` is read as the tree's own: {read:?}"
            );
        }
    }
}
