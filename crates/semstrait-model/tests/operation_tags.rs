mod common;

use common::{built, expression_model, parsed_expression, refused_expression};
use semstrait_model::{ParseErrorKind, SemanticModel, UnresolvedSemanticExpr};
use semstrait_sql::datatype::{CanonicalTypeSystem, DataType};
use semstrait_sql::expr::{
    type_tree, BinaryOpKind, Expr, InputTypes, LikeKind, NamedReference, OperationKind,
    SemanticLeaf, TypeError, Typed, UnaryOpKind,
};
use semstrait_sql::DocumentDialect;

type Tree = UnresolvedSemanticExpr;

fn field(name: &str) -> Box<Tree> {
    Box::new(semstrait_model::expr::field(name).expect("a name"))
}

fn binary(op: BinaryOpKind) -> Tree {
    Expr::BinaryOp {
        op,
        left: field("a"),
        right: field("b"),
    }
}

fn unary(op: UnaryOpKind) -> Tree {
    Expr::UnaryOp {
        op,
        operand: field("a"),
    }
}

fn like(kind: LikeKind) -> Tree {
    Expr::Like {
        value: field("a"),
        pattern: field("b"),
        kind,
    }
}

fn between(negated: bool) -> Tree {
    Expr::Between {
        value: field("a"),
        low: field("b"),
        high: field("c"),
        negated,
    }
}

fn in_list(negated: bool) -> Tree {
    Expr::InList {
        value: field("a"),
        list: vec![*field("b"), *field("c")],
        negated,
    }
}

const PAIR: &str = "[a, b]";
const PATTERN: &str = "{value: a, pattern: b}";
const RANGE: &str = "{value: a, low: b, high: c}";
const LIST: &str = "{value: a, list: [b, c]}";

fn todays_tags() -> Vec<(&'static str, String, Tree)> {
    let pair = |tag, op| (tag, PAIR.to_owned(), binary(op));
    vec![
        pair("add", BinaryOpKind::Add),
        pair("subtract", BinaryOpKind::Subtract),
        pair("multiply", BinaryOpKind::Multiply),
        pair("divide", BinaryOpKind::Divide),
        pair("safe_divide", BinaryOpKind::SafeDivide),
        pair("mod", BinaryOpKind::Mod),
        pair("eq", BinaryOpKind::Eq),
        pair("not_eq", BinaryOpKind::NotEq),
        pair("lt", BinaryOpKind::Lt),
        pair("lt_eq", BinaryOpKind::LtEq),
        pair("gt", BinaryOpKind::Gt),
        pair("gt_eq", BinaryOpKind::GtEq),
        pair("and", BinaryOpKind::And),
        pair("or", BinaryOpKind::Or),
        (
            "add",
            "{left: a, right: b}".to_owned(),
            binary(BinaryOpKind::Add),
        ),
        ("negate", "a".to_owned(), unary(UnaryOpKind::Negate)),
        ("not", "a".to_owned(), unary(UnaryOpKind::Not)),
        ("like", PATTERN.to_owned(), like(LikeKind::Like)),
        ("ilike", PATTERN.to_owned(), like(LikeKind::ILike)),
        (
            "not",
            format!("{{like: {PATTERN}}}"),
            like(LikeKind::NotLike),
        ),
        (
            "not",
            format!("{{ilike: {PATTERN}}}"),
            like(LikeKind::NotILike),
        ),
        ("between", RANGE.to_owned(), between(false)),
        ("not", format!("{{between: {RANGE}}}"), between(true)),
        ("in", LIST.to_owned(), in_list(false)),
        ("not", format!("{{in: {LIST}}}"), in_list(true)),
        ("is_null", "a".to_owned(), Expr::IsNull(field("a"))),
    ]
}

#[test]
fn each_operation_tag_parses_to_its_node() {
    for (tag, body, tree) in todays_tags() {
        let declared = OperationKind::from_name(tag).expect("today's tag is a declared name");
        for spelling in std::iter::once(&tag).chain(declared.operation().canonical.aliases) {
            assert_eq!(
                parsed_expression(&format!("{{{spelling}: {body}}}")),
                tree,
                "`{spelling}: {body}`"
            );
        }
    }
}

#[test]
fn a_negated_like_parses_by_its_declared_name() {
    for (tag, kind) in [
        ("not_like", LikeKind::NotLike),
        ("not_ilike", LikeKind::NotILike),
    ] {
        assert_eq!(
            parsed_expression(&format!("{{{tag}: {PATTERN}}}")),
            like(kind),
            "`{tag}`"
        );
    }
}

#[test]
fn an_undeclared_tag_is_refused_as_unknown() {
    for tag in ["plus", "equals", "AND", "not_in", "not_between", "isnull"] {
        let kinds = refused_expression(&format!("{{{tag}: {PAIR}}}"));
        assert_eq!(
            kinds,
            [ParseErrorKind::UnknownTag {
                tag: tag.to_owned()
            }],
            "`{tag}`"
        );
    }
}

struct Columns;

impl InputTypes<SemanticLeaf> for Columns {
    fn column(&self, column: &NamedReference) -> Typed {
        let data_type = match column.name() {
            "n" => DataType::BigInt,
            _ => DataType::String,
        };
        Typed::new(data_type, true)
    }

    fn leaf(&self, leaf: &SemanticLeaf) -> Typed {
        unreachable!("the test types columns only: {leaf:?}")
    }
}

fn typed(tree: Tree) -> Result<Typed, TypeError> {
    type_tree(
        tree,
        &Columns,
        DocumentDialect::Canonical,
        &CanonicalTypeSystem,
        None,
    )
    .map(|(_, typed)| typed)
}

#[test]
fn is_not_distinct_from_types_a_boolean_that_is_never_null_over_one_family() {
    let column = |name| Box::new(semstrait_model::expr::col(name).expect("a name"));
    let strings = parsed_expression("{is_not_distinct_from: [{col: a}, {col: b}]}");
    assert_eq!(
        strings,
        Expr::BinaryOp {
            op: BinaryOpKind::IsNotDistinctFrom,
            left: column("a"),
            right: column("b"),
        }
    );
    assert_eq!(typed(strings), Ok(Typed::new(DataType::Boolean, false)));
    assert_eq!(
        typed(parsed_expression(
            "{is_not_distinct_from: [{col: a}, {col: n}]}"
        )),
        Err(TypeError::Arguments {
            name: "is_not_distinct_from",
            types: vec![DataType::String.into(), DataType::BigInt.into()],
        })
    );
}

#[test]
fn the_writer_writes_a_negated_like_under_its_declared_name() {
    for tag in ["not_like", "not_ilike"] {
        let model = built(&expression_model(&format!(
            "{{{tag}: {{value: name, pattern: {{lit: 'A%'}}}}}}"
        )));
        let written = serde_json::to_value(&model).expect("the writer writes the model");
        assert_eq!(
            written.pointer("/dimensions/0/expr"),
            Some(&serde_json::json!({tag: {"value": {"field": "name"}, "pattern": {"lit": "A%"}}})),
            "`{tag}`"
        );
        let back: SemanticModel = serde_json::from_value(written).expect("the JSON reads back");
        assert_eq!(back, model, "`{tag}`");
    }
}
