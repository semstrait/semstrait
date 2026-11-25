#[path = "../../semstrait-api/tests/common/mod.rs"]
mod common;

use semstrait::{
    run, Dialect, Input, Request, RequestContext, RequestPart, Selection, SemstraitError,
};
use semstrait_common::io::Persistable;
use semstrait_model::{ModelBuildErrorKind, ParseErrorKind};
use semstrait_planner::{Advisory, LookedIn, PlanError};
use substrait::proto::Plan;
use tempfile::tempdir;

use common::{kinds, only_retired_keys, sales, written, SALES};

async fn planned(input: Input, dialect: Dialect) -> (Plan, Vec<SemstraitError>) {
    let request = Request::new(vec![Selection::Field("region".into())]);
    let planned = run(input, &request, &RequestContext::default(), dialect).await;
    let (plan, warnings) = planned.expect("the input plans");
    (plan, kinds(warnings))
}

#[tokio::test]
async fn run_gives_the_substrait_plan_of_each_input_of_one_semantic_model() {
    let dir = tempdir().expect("a temporary directory");
    let manifest = sales().await;
    let model_path = written(&dir, "sales.yaml", SALES);
    let manifest_path = written(&dir, "sales.json", manifest.to_bytes().expect("it encodes"));

    for input in [
        Input::ModelYaml(SALES.into()),
        Input::ModelPath(model_path),
        Input::Manifest(manifest),
        Input::ManifestPath(manifest_path),
    ] {
        let (plan, _) = planned(input, Dialect::Substrait).await;

        assert_eq!(plan.relations.len(), 1);
    }
}

#[tokio::test]
async fn run_gives_the_substrait_plan_of_model_yaml_with_its_warnings_and_advisories() {
    let (plan, warnings) = planned(Input::ModelYaml(SALES.into()), Dialect::Substrait).await;

    assert_eq!(plan.relations.len(), 1);
    assert!(matches!(
        warnings.as_slice(),
        [
            SemstraitError::Build(ModelBuildErrorKind::Parse(
                ParseErrorKind::DataKindKeysBlockRetired { .. }
            )),
            SemstraitError::Advisory(Advisory::AmbiguousCandidates { .. }),
        ]
    ));
}

#[tokio::test]
async fn run_refuses_a_request_with_the_warnings_of_load_beside_the_plan_error() {
    let request = Request::new(vec![Selection::Field("revenue".into())]);

    let refused = run(
        Input::ModelYaml(SALES.into()),
        &request,
        &RequestContext::default(),
        Dialect::Substrait,
    )
    .await;

    let (refusal, warnings) = refused.expect_err("the request is refused");
    let unknown = PlanError::UnknownName {
        name: "revenue".into(),
        looked_in: LookedIn::SemanticDefinitions,
    };
    assert!(matches!(refusal.kind, SemstraitError::Plan(kind) if kind == unknown));
    assert!(only_retired_keys(warnings));
}

#[tokio::test]
async fn run_plans_the_request_and_the_context_that_a_wire_document_gives() {
    let (request, context) =
        RequestPart::read_yaml("select: [region]\ncontext: {timezone: Europe/Stockholm}")
            .expect("the document reads");

    let (plan, _) = run(
        Input::ModelYaml(SALES.into()),
        &request,
        &context,
        Dialect::Substrait,
    )
    .await
    .expect("the model plans");

    assert_eq!(plan.relations.len(), 1);
}

#[cfg(feature = "datafusion")]
#[tokio::test]
async fn run_gives_a_plan_for_datafusion_with_the_datafusion_feature() {
    let (plan, _) = planned(Input::ModelYaml(SALES.into()), Dialect::DataFusion).await;

    assert_eq!(plan.relations.len(), 1);
}
