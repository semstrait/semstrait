#![cfg(feature = "api-rest")]

#[path = "../../semstrait-api/tests/common/mod.rs"]
mod common;

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::sales;
use semstrait::Dialect;
use semstrait_api::router;
use tower::ServiceExt;

#[tokio::test]
async fn the_api_rest_feature_forwards_to_the_router_of_semstrait_api() {
    let health = Request::get("/health")
        .body(Body::empty())
        .expect("the request builds");

    let response = router(
        Arc::new(common::semstrait()),
        Arc::new(sales().await),
        Dialect::Substrait,
    )
    .oneshot(health)
    .await
    .expect("the router answers");

    assert_eq!(response.status(), StatusCode::OK);
}
