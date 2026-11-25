#![cfg(feature = "rest")]

mod common;

use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::http::{header, Method, Request, StatusCode};
use common::{compiled, semstrait, SALES, SPEND_BOOKINGS};
use http_body_util::BodyExt;
use prost::Message;
use semstrait_api::router;
use semstrait_common::Dialect;
use semstrait_manifest::SemanticManifest;
use serde_json::{json, Value};
use substrait::proto::Plan;
use tower::ServiceExt;

async fn answered(
    model: &str,
    method: Method,
    uri: &str,
    body: &'static str,
) -> (StatusCode, Option<String>, Bytes) {
    answered_in(Dialect::Substrait, model, method, uri, body).await
}

async fn answered_in(
    dialect: Dialect,
    model: &str,
    method: Method,
    uri: &str,
    body: impl Into<Body>,
) -> (StatusCode, Option<String>, Bytes) {
    let manifest = compiled(model).await;
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .body(body.into())
        .expect("the request builds");
    let response = router(Arc::new(semstrait()), Arc::new(manifest), dialect)
        .oneshot(request)
        .await
        .expect("the router answers");
    let (parts, body) = response.into_parts();
    let bytes = body.collect().await.expect("the body reads").to_bytes();
    let content_type = (parts.headers.get(header::CONTENT_TYPE))
        .map(|value| value.to_str().expect("the content type is text").to_owned());
    (parts.status, content_type, bytes)
}

const JSON: Option<&str> = Some("application/json");

const BY_REGION: &str = r#"{"select": ["region"]}"#;

const BY_REVENUE: &str = r#"{"select": ["revenue"]}"#;

fn unknown_revenue() -> Value {
    json!({ "message": "`revenue` is not in the semantic definitions", "severity": "error" })
}

fn ambiguous_candidates() -> Value {
    json!({
        "message": "at `orders`: `orders` is chosen by name order over `returns`: \
                    name one in `from`",
        "severity": "warning",
    })
}

fn parsed(body: &[u8]) -> Value {
    serde_json::from_slice(body).expect("the body is JSON")
}

#[tokio::test]
async fn health_answers_status_ok() {
    let (status, content_type, body) = answered(SALES, Method::GET, "/health", "").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type.as_deref(), JSON);
    assert_eq!(parsed(&body), json!({ "status": "ok" }));
}

#[tokio::test]
async fn schema_lists_only_the_public_semantic_sources_each_with_its_names_by_role() {
    let (status, content_type, body) = answered(SPEND_BOOKINGS, Method::GET, "/schema", "").await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type.as_deref(), JSON);
    assert_eq!(
        parsed(&body),
        json!([{
            "name": "spend_bookings",
            "dimensions": ["campaign_id"],
            "measures": ["cost", "revenue"],
            "metrics": ["return_on_spend"],
        }])
    );
}

#[tokio::test]
async fn compile_answers_the_manifest_json_of_the_model_in_the_body() {
    let (status, content_type, body) =
        answered(SPEND_BOOKINGS, Method::POST, "/compile", SALES).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type.as_deref(), JSON);
    let manifest: SemanticManifest = serde_json::from_slice(&body).expect("the body is a manifest");
    assert_eq!(manifest.metadata.source_model_uri.as_deref(), Some("sales"));
}

#[tokio::test]
async fn compile_answers_400_with_the_errors_of_a_model_that_parse_refuses() {
    let (status, content_type, body) =
        answered(SALES, Method::POST, "/compile", "semantic_model: [").await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(content_type.as_deref(), JSON);
    assert_eq!(
        parsed(&body),
        json!({
            "errors": [{
                "message": "YAML syntax error: semantic_model: invalid type: sequence, \
                            expected struct YamlSemanticModel at line 1 column 17",
                "severity": "error",
            }],
            "advisories": [],
        })
    );
}

#[tokio::test]
async fn compile_answers_400_without_the_value_to_a_model_that_uses_an_environment_variable() {
    let home = std::env::var("HOME").expect("HOME is set");
    let model = SALES.replacen("name: sales", "name: ${HOME}", 1);
    let (status, _, body) =
        answered_in(Dialect::Substrait, SALES, Method::POST, "/compile", model).await;

    let leaked = String::from_utf8_lossy(&body).contains(&home);
    assert_eq!((status, leaked), (StatusCode::BAD_REQUEST, false));
    assert_eq!(
        parsed(&body)["errors"],
        json!([{
            "message": "a posted model may not use `${...}`, because the server expands \
                        no environment variable in HTTP input",
            "severity": "error",
        }])
    );
}

#[tokio::test]
async fn validate_answers_valid_with_the_advisories_of_an_accepted_request() {
    let (status, content_type, body) = answered(SALES, Method::POST, "/validate", BY_REGION).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type.as_deref(), JSON);
    assert_eq!(
        parsed(&body),
        json!({ "valid": true, "errors": [], "advisories": [ambiguous_candidates()] })
    );
}

#[tokio::test]
async fn validate_answers_400_not_valid_with_the_errors_of_a_refused_request() {
    let (status, content_type, body) = answered(SALES, Method::POST, "/validate", BY_REVENUE).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(content_type.as_deref(), JSON);
    assert_eq!(
        parsed(&body),
        json!({ "valid": false, "errors": [unknown_revenue()], "advisories": [] })
    );
}

#[tokio::test]
async fn explain_answers_the_text_and_the_advisories_of_an_accepted_request() {
    let (status, content_type, body) = answered(SALES, Method::POST, "/explain", BY_REGION).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type.as_deref(), JSON);
    let body = parsed(&body);
    let text = body["text"].as_str().expect("the text is a string");
    assert!(text.contains("Read[\"\".warehouse.fact_rows => region:string?]"));
    assert_eq!(body["advisories"], json!([ambiguous_candidates()]));
}

#[tokio::test]
async fn plan_answers_protobuf_bytes_that_decode_into_the_substrait_plan_of_an_accepted_request() {
    let (status, content_type, body) = answered(SALES, Method::POST, "/plan", BY_REGION).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type.as_deref(), Some("application/x-protobuf"));
    let plan = Plan::decode(body).expect("the body decodes into a Substrait plan");
    assert_eq!(plan.relations.len(), 1);
}

#[tokio::test]
async fn explain_and_plan_answer_400_with_the_errors_of_a_refused_request() {
    for uri in ["/explain", "/plan"] {
        let (status, content_type, body) = answered(SALES, Method::POST, uri, BY_REVENUE).await;

        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}");
        assert_eq!(content_type.as_deref(), JSON, "{uri}");
        assert_eq!(
            parsed(&body),
            json!({ "errors": [unknown_revenue()], "advisories": [] }),
            "{uri}"
        );
    }
}

#[tokio::test]
async fn a_body_with_an_unknown_key_or_a_refused_where_is_refused_with_400_on_each_request_route() {
    for (document, message) in [
        (
            r#"{"select": ["region"], "colour": 1}"#,
            "the Request document is refused: unknown field `colour`, expected one of `from`, \
             `select`, `where`, `filters`, `order_by`, `limit`, `offset`, `context` at line 1 \
             column 31",
        ),
        (
            r#"{"select": ["region"], "where": {"nope": ["channel"]}}"#,
            "the Request document is refused: unknown reserved tag: `nope` at line 1 column 53",
        ),
    ] {
        for uri in ["/validate", "/explain", "/plan"] {
            let (status, content_type, body) = answered(SALES, Method::POST, uri, document).await;

            assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}");
            assert_eq!(content_type.as_deref(), JSON, "{uri}");
            let refused = json!({ "message": message, "severity": "error" });
            assert_eq!(parsed(&body)["errors"], json!([refused]), "{uri}");
        }
    }
}

#[tokio::test]
async fn a_body_that_is_not_utf_8_is_refused_with_400_and_its_error_on_each_route_that_reads_one() {
    for uri in ["/compile", "/validate", "/explain", "/plan"] {
        let (status, content_type, body) =
            answered_in(Dialect::Substrait, SALES, Method::POST, uri, &b"\xff"[..]).await;

        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}");
        assert_eq!(content_type.as_deref(), JSON, "{uri}");
        let not_text = json!({
            "message": "malformed payload from the request body: \
                        invalid utf-8 sequence of 1 bytes from index 0",
            "severity": "error",
        });
        assert_eq!(parsed(&body)["errors"], json!([not_text]), "{uri}");
    }
}

#[tokio::test]
async fn explain_and_plan_answer_500_when_the_dialect_of_the_server_has_no_engine_adapter() {
    for uri in ["/explain", "/plan"] {
        let (status, content_type, body) =
            answered_in(Dialect::Iceberg, SALES, Method::POST, uri, BY_REGION).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{uri}");
        assert_eq!(content_type.as_deref(), JSON, "{uri}");
        let no_adapter = json!({
            "message": "the dialect Iceberg has no engine adapter",
            "severity": "error",
        });
        assert_eq!(
            parsed(&body),
            json!({ "errors": [no_adapter], "advisories": [] }),
            "{uri}"
        );
    }
}
