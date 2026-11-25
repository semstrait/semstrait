use std::iter::once;
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use prost::Message;
use semstrait_common::{split_by_severity, Diagnose, Diagnostic, DialectRelease};
use semstrait_ir::SemanticRole;
use semstrait_manifest::SemanticManifest;
use semstrait_planner::{Request, RequestContext};
use serde_json::{json, Value};

use crate::error::{refusal, wired, InputErrorKind, Refusal, SemstraitError};
use crate::orchestrator::Semstrait;
use crate::reader::malformed;
use crate::wire::RequestPart;

#[derive(Clone)]
struct Served {
    semstrait: Arc<Semstrait>,
    manifest: Arc<SemanticManifest>,
    dialect: DialectRelease,
}

pub fn router(
    semstrait: Arc<Semstrait>,
    manifest: Arc<SemanticManifest>,
    dialect: impl Into<DialectRelease>,
) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
        .route("/schema", get(schema))
        .route("/compile", post(compile))
        .route("/validate", post(validate))
        .route("/explain", post(explain))
        .route("/plan", post(plan))
        .with_state(Served {
            semstrait,
            manifest,
            dialect: dialect.into(),
        })
}

async fn schema(State(served): State<Served>) -> Json<Value> {
    let definitions = &served.manifest.semantic_definitions.entries;
    let public = (served.manifest.semantic_sources.values()).filter(|source| source.public);
    let sources = public.map(|source| {
        let named = |role| {
            (definitions.values())
                .filter(|definition| definition.role() == role)
                .filter(|definition| source.semantic_bitmask.contains(definition.bit_position))
                .map(|definition| &definition.name)
                .collect::<Vec<_>>()
        };
        json!({
            "name": source.name,
            "dimensions": named(SemanticRole::Dimension),
            "measures": named(SemanticRole::Measure),
            "metrics": named(SemanticRole::Metric),
        })
    });
    Json(sources.collect())
}

async fn compile(
    State(served): State<Served>,
    body: Bytes,
) -> Result<Json<SemanticManifest>, Response> {
    let yaml = text(&body).map_err(refused)?;
    if yaml.contains("${") {
        let kind = SemstraitError::Input(InputErrorKind::EnvVarInPostedModel);
        return Err(refused(refusal(kind)));
    }
    let compiled = served.semstrait.compile_from_yaml(yaml).await;
    let (manifest, _) = compiled.map_err(refused)?;
    Ok(Json(manifest))
}

async fn validate(State(served): State<Served>, body: Bytes) -> Response {
    let validated = requested(&body, |request, context| {
        (served.semstrait).validate(&served.manifest, request, context)
    });
    let valid = validated.is_ok();
    let (status, errors, advisories) = match validated {
        Ok(advisories) => (StatusCode::OK, json!([]), described(&advisories)),
        Err(refusal) => reported(refusal),
    };
    let body = json!({ "valid": valid, "errors": errors, "advisories": advisories });
    (status, Json(body)).into_response()
}

async fn explain(State(served): State<Served>, body: Bytes) -> Result<Json<Value>, Response> {
    let explained = requested(&body, |request, context| {
        (served.semstrait).explain(&served.manifest, request, context, served.dialect)
    });
    let (text, advisories) = explained.map_err(refused)?;
    Ok(Json(explanation(&text, &advisories)))
}

pub(crate) fn explanation(text: &str, advisories: &[Diagnostic<SemstraitError>]) -> Value {
    json!({ "text": text, "advisories": described(advisories) })
}

async fn plan(State(served): State<Served>, body: Bytes) -> Result<impl IntoResponse, Response> {
    let planned = requested(&body, |request, context| {
        (served.semstrait).plan(&served.manifest, request, context, served.dialect)
    });
    let (plan, _) = planned.map_err(refused)?;
    let content_type = [(header::CONTENT_TYPE, "application/x-protobuf")];
    Ok((content_type, plan.encode_to_vec()))
}

fn requested<T>(
    body: &[u8],
    verb: impl FnOnce(&Request, &RequestContext) -> Result<T, Refusal>,
) -> Result<T, Refusal> {
    let (request, context) = RequestPart::read_json(text(body)?).map_err(wired)?;
    verb(&request, &context)
}

fn text(body: &[u8]) -> Result<&str, Refusal> {
    std::str::from_utf8(body).map_err(|error| malformed("the request body", error))
}

fn refused(refusal: Refusal) -> Response {
    let (status, errors, advisories) = reported(refusal);
    let body = json!({ "errors": errors, "advisories": advisories });
    (status, Json(body)).into_response()
}

fn reported((head, rest): Refusal) -> (StatusCode, Value, Value) {
    let status = match head.kind {
        SemstraitError::Adapt(_) | SemstraitError::NoEngineAdapter(_) => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
        _ => StatusCode::BAD_REQUEST,
    };
    let (errors, advisories) = split_by_severity(rest);
    let errors = described(once(&head).chain(&errors));
    (status, errors, described(&advisories))
}

fn described<'a>(diagnostics: impl IntoIterator<Item = &'a Diagnostic<SemstraitError>>) -> Value {
    let described = |diagnostic: &Diagnostic<SemstraitError>| {
        let severity = diagnostic.severity.to_string();
        json!({ "message": diagnostic.kind.message(), "severity": severity })
    };
    diagnostics.into_iter().map(described).collect()
}
