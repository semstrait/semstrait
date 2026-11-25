use std::fmt::Display;

use serde_json::{json, Value};

use semstrait_common::io::Persistable;
use semstrait_common::{Dialect, Tree};
use semstrait_ir::{Parameter, ParameterKey};
use semstrait_manifest::{CompileError, PhysicalExprId, ReferenceSite, SemanticManifest};
use semstrait_model::expr::{dim, lit};
use semstrait_model::{Dataset, Dimension, DimensionSite, Measure, MeasureSite, SemanticModel};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    Call, CallClause, CallError, Expr, FunctionRef, JsonForm, JsonFormError, Literal, OverError,
    WindowBound, WindowFrame, WindowFrameKind,
};
use semstrait_sql::functions::{Additivity, ClauseRule, Nullability};
use semstrait_sql::DocumentDialect;

mod common;
use common::*;

fn reading_model() -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "sales".to_owned();
    for (name, data_type) in [
        ("name", DataType::String),
        ("region", DataType::String),
        ("ordered_on", DataType::Date),
        ("cost", DataType::Double),
    ] {
        insert_dimension(
            &mut model,
            Dimension::builder(name)
                .data_type(data_type)
                .categorical()
                .build(),
        );
    }
    for (name, data_type, expr) in [
        (
            "shout",
            DataType::String,
            call("upper", vec![dim("name").expect("a name")]),
        ),
        (
            "label",
            DataType::String,
            call(
                "concat",
                vec![dim("name").expect("a name"), dim("region").expect("a name")],
            ),
        ),
        (
            "ordered_in",
            DataType::BigInt,
            call(
                "extract",
                vec![lit("year"), dim("ordered_on").expect("a name")],
            ),
        ),
        (
            "cost_rounded",
            DataType::Double,
            call(
                "round",
                vec![dim("cost").expect("a name"), lit(Literal::Integer(2))],
            ),
        ),
    ] {
        insert_dimension(
            &mut model,
            Dimension::builder(name)
                .data_type(data_type)
                .categorical()
                .expr(expr)
                .build(),
        );
    }
    insert_measure(
        &mut model,
        Measure::builder("revenue")
            .data_type(DataType::BigInt)
            .agg(agg("sum"))
            .build(),
    );
    insert_measure(
        &mut model,
        Measure::builder("price")
            .data_type(DataType::BigInt)
            .agg(agg("max"))
            .build(),
    );
    insert_dataset(
        &mut model,
        Dataset::builder("orders")
            .table("warehouse.orders")
            .dimensions(
                [
                    "name",
                    "region",
                    "ordered_on",
                    "cost",
                    "shout",
                    "label",
                    "ordered_in",
                    "cost_rounded",
                ]
                .map(DimensionSite::r#ref),
            )
            .measures(["revenue", "price"].map(MeasureSite::r#ref))
            .build(),
    );
    model
}

async fn compiled() -> (SemanticManifest, String) {
    let manifest = compile_model(reading_model()).await;
    let json = serde_json::to_string(&manifest).expect("a manifest writes");
    (manifest, json)
}

async fn compiled_json() -> String {
    compiled().await.1
}

async fn compiled_json_and_argument(name: &str) -> (String, String) {
    let (manifest, json) = compiled().await;
    let root = signature(&manifest, "orders", name).expression_id;
    let call = manifest
        .expressions
        .get(&root)
        .expect("the entry")
        .expr
        .expr();
    let argument = call.children()[0];
    let argument = serde_json::to_string(&JsonForm::new(argument, DocumentDialect::Canonical))
        .expect("an argument writes");
    (json, argument)
}

fn edited(json: &str, from: &str, to: &str) -> String {
    assert_eq!(
        json.matches(from).count(),
        1,
        "the manifest holds `{from}` once"
    );
    json.replacen(from, to, 1)
}

fn refused(json: &str, refusal: impl Display) {
    let error = SemanticManifest::from_bytes(json.as_bytes()).expect_err("reading refuses it");
    assert!(error.contains(&refusal.to_string()), "{error}");
}

const LEAF_IS_A_LINK: &str =
    "a stored expression's only leaf is a link, its entry's expression id in UUID text";

fn pool_refusal(rule: &str) -> String {
    let site = ReferenceSite::ExpressionPool;
    format!("structural violation at {site:?}: {rule}")
}

#[tokio::test]
async fn a_compiled_manifest_names_its_dialect_canonical() {
    let json: Value = serde_json::from_str(&compiled_json().await).expect("a JSON object");
    assert_eq!(json["dialect"], "canonical");
}

#[tokio::test]
async fn a_scalar_call_is_stored_with_its_name_and_arguments_only() {
    let (manifest, json) = compiled().await;
    let name = signature(&manifest, "orders", "name").expression_id;
    let upper =
        format!(r#""expr":{{"type":"scalar","content":{{"name":"upper","args":["{name}"]}}}}"#);
    assert!(json.contains(&upper), "{json}");
}

#[tokio::test]
async fn a_call_read_back_holds_its_function_s_handle() {
    let json = compiled_json().await;
    let read = SemanticManifest::from_bytes(json.as_bytes()).expect("the manifest reads");
    let root = signature(&read, "orders", "revenue").expression_id;
    let Expr::Function(Call::Aggregate(sum, _)) =
        read.expressions.get(&root).expect("the entry").expr.expr()
    else {
        panic!("`revenue` is stored as an aggregate call");
    };
    let FunctionRef::Aggregate(catalog_sum, _) = function("sum") else {
        panic!("`sum` is an aggregate function");
    };
    let function = sum.function();
    assert!(std::ptr::eq(function, catalog_sum), "the catalog's `sum`");
    let declaration = &function.declarations.canonical;
    assert_eq!(declaration.nullability, Nullability::EmptyGroupNull);
    assert_eq!(declaration.specific.clauses.distinct, ClauseRule::Allowed);
    assert_eq!(function.additivity, Additivity::Additive);
}

#[tokio::test]
async fn a_stored_call_with_an_enum_or_a_constant_argument_reads_back() {
    let (manifest, json) = compiled().await;
    for name in ["extract", "round"] {
        assert!(
            json.contains(&format!(r#""name":"{name}""#)),
            "`{name}` is stored"
        );
    }
    let read = SemanticManifest::from_bytes(json.as_bytes()).expect("the manifest reads");
    assert_eq!(read, manifest);
}

#[tokio::test]
async fn reading_refuses_a_dialect_other_than_canonical() {
    let json = compiled_json().await;
    for (written, dialect) in [
        (r#""datafusion""#, "datafusion"),
        (r#""nope""#, "nope"),
        (r#""Canonical""#, "Canonical"),
        ("5", "5"),
    ] {
        let json = edited(
            &json,
            r#""dialect":"canonical""#,
            &format!(r#""dialect":{written}"#),
        );
        let dialect = dialect.to_owned();
        refused(&json, CompileError::ManifestDialect { dialect });
    }
}

#[tokio::test]
async fn reading_refuses_a_data_type_that_is_not_its_canonical_name() {
    let json = compiled_json().await;
    for spelling in ["long", "BIGINT"] {
        let json = json.replace(
            r#""data_type":"bigint""#,
            &format!(r#""data_type":"{spelling}""#),
        );
        refused(&json, format!("`{spelling}`"));
    }
}

#[tokio::test]
async fn reading_refuses_a_call_name_that_no_function_has_in_the_manifest_s_dialect() {
    let json = edited(
        &compiled_json().await,
        r#""name":"upper""#,
        r#""name":"yell""#,
    );
    refused(
        &json,
        JsonFormError::UnknownFunction {
            name: "yell".to_owned(),
            dialect: DocumentDialect::Canonical,
        },
    );
}

#[tokio::test]
async fn reading_refuses_a_tag_that_is_not_the_function_s_type() {
    let json = edited(
        &compiled_json().await,
        r#"{"type":"scalar","content":{"name":"upper""#,
        r#"{"type":"aggregate","content":{"name":"upper""#,
    );
    refused(
        &json,
        JsonFormError::NotItsFunctionType {
            name: "upper".to_owned(),
            tag: "aggregate",
        },
    );
}

#[tokio::test]
async fn reading_refuses_a_wrong_number_of_arguments() {
    let (json, argument) = compiled_json_and_argument("shout").await;
    let json = edited(
        &json,
        r#"{"name":"upper","args":["#,
        &format!(r#"{{"name":"upper","args":[{argument},"#),
    );
    refused(
        &json,
        CallError::TooManyArguments {
            function: "upper",
            got: 2,
        },
    );
}

#[tokio::test]
async fn reading_refuses_an_argument_that_does_not_fit_its_signature() {
    let json = compiled_json().await;
    let cost = r#"{"type":"column","content":["ps0","cost"]}"#;
    refused(
        &edited(&json, r#"{"type":"literal","content":{"integer":2}}"#, cost),
        CallError::NotALiteral {
            function: "round",
            parameter: "scale",
        },
    );
    refused(
        &edited(&json, r#"{"string":"year"}"#, r#"{"string":"yeer"}"#),
        CallError::EnumValue {
            function: "extract",
            parameter: "component",
            value: Literal::String("yeer".to_owned()),
        },
    );
}

#[tokio::test]
async fn reading_refuses_a_refused_or_no_effect_clause() {
    let json = compiled_json().await;
    refused(
        &edited(
            &json,
            r#"{"name":"sum","#,
            r#"{"name":"sum","nulls":"ignore_nulls","#,
        ),
        CallError::ClauseRefused {
            function: "sum",
            clause: CallClause::Nulls,
        },
    );
    refused(
        &edited(
            &json,
            r#"{"name":"max","#,
            r#"{"name":"max","distinct":true,"#,
        ),
        CallError::ClauseRefused {
            function: "max",
            clause: CallClause::Distinct,
        },
    );
}

#[tokio::test]
async fn reading_refuses_a_call_with_no_required_clause() {
    let (json, argument) = compiled_json_and_argument("revenue").await;
    let json = edited(
        &json,
        &format!(r#"{{"name":"sum","args":[{argument}]"#),
        r#"{"name":"percentile_cont","args":[{"type":"literal","content":{"double":"0.5"}}]"#,
    );
    refused(
        &json,
        CallError::ClauseRequired {
            function: "percentile_cont",
            clause: CallClause::WithinGroup,
        },
    );
}

#[tokio::test]
async fn reading_refuses_a_leaf_that_is_not_a_link() {
    let (json, argument) = compiled_json_and_argument("shout").await;
    let json = edited(&json, &format!(r#""args":[{argument}]"#), r#""args":[7]"#);
    refused(&json, pool_refusal(LEAF_IS_A_LINK));
}

#[tokio::test]
async fn reading_refuses_a_stored_column_with_an_empty_part() {
    let (json, column) = (compiled_json().await, r#"["ps0","region"]"#);
    assert!(json.contains(column), "the manifest stores `{column}`");
    refused(
        &json.replace(column, r#"["ps0",""]"#),
        r#"invalid value: string "", expected a name path of one part or more, none of them empty"#,
    );
}

#[tokio::test]
async fn reading_refuses_a_parameter_in_a_stored_expression() {
    let (json, argument) = compiled_json_and_argument("shout").await;
    assert!(SemanticManifest::from_bytes(json.as_bytes()).is_ok());
    let parameter = serde_json::to_string(&Parameter {
        key: ParameterKey::RequestTemporalAxis,
        data_type: DataType::Date,
    })
    .expect("a parameter writes");
    let json = edited(
        &json,
        &format!(r#""args":[{argument}]"#),
        &format!(r#""args":[{parameter}]"#),
    );
    refused(&json, pool_refusal(LEAF_IS_A_LINK));
}

#[tokio::test]
async fn reading_refuses_a_link_that_is_not_uuid_text() {
    let (json, argument) = compiled_json_and_argument("shout").await;
    for link in ["7", r#""26C8A8C0-1F01-5EF4-99A3-87AB96316377""#] {
        let json = edited(
            &json,
            &format!(r#""args":[{argument}]"#),
            &format!(r#""args":[{link}]"#),
        );
        refused(&json, pool_refusal(LEAF_IS_A_LINK));
    }
}

#[tokio::test]
async fn reading_refuses_a_window_frame_that_sql_refuses() {
    use WindowBound::{CurrentRow, Following, Preceding, UnboundedFollowing, UnboundedPreceding};
    use WindowFrameKind::{Groups, Range, Rows};
    let (json, argument) = compiled_json_and_argument("shout").await;
    let upper = format!(r#"{{"type":"scalar","content":{{"name":"upper","args":[{argument}]}}}}"#);
    let argument: Value = serde_json::from_str(&argument).expect("an argument reads");
    let first_value = |kind, start, end, keys| {
        let key = json!({"expr": argument, "dir": {"asc": {"nulls": "unspecified"}}});
        let window_frame = serde_json::to_value(WindowFrame::new(kind, start, end));
        let node = json!({"type": "window", "content": {
            "name": "first_value",
            "args": [argument],
            "over": {"order_by": vec![key; keys], "window_frame": window_frame.expect("writes")},
        }});
        edited(&json, &upper, &node.to_string())
    };
    let allowed = first_value(Rows, UnboundedPreceding, CurrentRow, 1);
    assert!(SemanticManifest::from_bytes(allowed.as_bytes()).is_ok());
    for (kind, start, end, keys, error) in [
        (
            Rows,
            UnboundedFollowing,
            UnboundedFollowing,
            1,
            OverError::StartsUnboundedFollowing,
        ),
        (
            Rows,
            UnboundedPreceding,
            UnboundedPreceding,
            1,
            OverError::EndsUnboundedPreceding,
        ),
        (
            Rows,
            CurrentRow,
            Preceding(1),
            1,
            OverError::EndsBeforeStart,
        ),
        (
            Rows,
            Following(1),
            Preceding(1),
            1,
            OverError::EndsBeforeStart,
        ),
        (
            Rows,
            Following(1),
            CurrentRow,
            1,
            OverError::EndsBeforeStart,
        ),
        (
            Range,
            Preceding(5),
            CurrentRow,
            0,
            OverError::RangeOffsetKeys { keys: 0 },
        ),
        (
            Range,
            CurrentRow,
            Following(5),
            2,
            OverError::RangeOffsetKeys { keys: 2 },
        ),
        (
            Groups,
            UnboundedPreceding,
            UnboundedFollowing,
            0,
            OverError::GroupsWithoutOrderBy,
        ),
        (
            Groups,
            Preceding(1),
            Following(1),
            0,
            OverError::GroupsWithoutOrderBy,
        ),
    ] {
        let function = "first_value";
        refused(
            &first_value(kind, start, end, keys),
            CallError::Over { function, error },
        );
    }
}

#[tokio::test]
async fn validate_refuses_a_dialect_other_than_canonical() {
    let mut manifest = compile_model(reading_model()).await;
    manifest.dialect = DocumentDialect::Dialect(Dialect::Substrait);
    let faults = manifest.validate().unwrap_or_else(|faults| faults);
    let faults: Vec<_> = faults.into_iter().map(|fault| fault.kind).collect();
    assert_eq!(
        faults,
        [CompileError::ManifestDialect {
            dialect: "substrait".to_owned(),
        }]
    );
}

async fn faults_of_concat_with(options: &str) -> (Vec<CompileError>, PhysicalExprId) {
    let (manifest, json) = compiled().await;
    let concat = signature(&manifest, "orders", "label").expression_id;
    let json = edited(
        &json,
        r#"{"name":"concat","#,
        &format!(r#"{{"name":"concat","options":{options},"#),
    );
    let read = SemanticManifest::from_bytes(json.as_bytes()).expect("reading reads any option");
    let faults = read.validate().unwrap_or_else(|faults| faults);
    (faults.into_iter().map(|fault| fault.kind).collect(), concat)
}

#[tokio::test]
async fn validate_takes_an_option_that_the_declaration_has() {
    let (faults, _) = faults_of_concat_with(r#"{"null_handling":"IGNORE_NULLS"}"#).await;
    assert_eq!(faults, []);
}

#[tokio::test]
async fn validate_refuses_an_option_name_that_the_declaration_does_not_have() {
    let (faults, concat) = faults_of_concat_with(r#"{"null_policy":"IGNORE_NULLS"}"#).await;
    assert_eq!(
        faults,
        [CompileError::UndeclaredOption {
            expr_id: concat,
            error: CallError::UnknownOption {
                function: "concat",
                option: "null_policy".to_owned(),
            },
        }]
    );
}

#[tokio::test]
async fn validate_refuses_an_option_value_that_the_declaration_does_not_have() {
    let (faults, concat) = faults_of_concat_with(r#"{"null_handling":"SKIP_NULLS"}"#).await;
    assert_eq!(
        faults,
        [CompileError::UndeclaredOption {
            expr_id: concat,
            error: CallError::OptionValue {
                function: "concat",
                option: "null_handling",
                value: "SKIP_NULLS".to_owned(),
            },
        }]
    );
}
