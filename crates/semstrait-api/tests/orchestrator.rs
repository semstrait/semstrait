mod common;

use std::sync::Arc;
use std::time::SystemTime;

use bytes::Bytes;
use semstrait_api::{NoEngineAdapter, SemstraitBuilder, SemstraitError};
use semstrait_catalog::NoopCatalogProvider;
use semstrait_common::io::{FileSystem, InMemoryFileSystem, Path};
use semstrait_common::DataKindName;
use semstrait_common::Dialect;
use semstrait_manifest::CompileError;
use semstrait_model::ParseErrorKind;
use semstrait_planner::{Advisory, LookedIn, PlanError, Request, RequestContext, Selection};

use common::{kinds, only_retired_keys, sales, sales_model, semstrait, SALES};

const LAKE: &str = r#"
semantic_model:
  name: lake
  dimensions:
    - name: region
      data_type: string
      type: categorical
  datasets:
    - name: orders
      extras:
        storage:
          format: parquet
          paths:
            - s3://lake/orders/
        semantic_mapping: auto
      dimensions:
        - ref: region
          is_key: true
"#;

fn by_region() -> Request {
    Request::new(vec![Selection::Field("region".into())])
}

#[tokio::test]
async fn compile_from_yaml_gives_the_manifest_stamped_at_the_compile_with_its_warnings() {
    let before = SystemTime::now();
    let (manifest, warnings) = semstrait()
        .compile_from_yaml(SALES)
        .await
        .expect("the model compiles");

    assert_eq!(manifest.metadata.source_model_uri.as_deref(), Some("sales"));
    assert_eq!(manifest.created_at, manifest.updated_at);
    assert!((before..=SystemTime::now()).contains(&manifest.created_at));
    assert!(only_retired_keys(warnings));
}

#[tokio::test]
async fn compile_reads_the_catalog_the_builder_holds_with_the_warnings_beside_its_refusal() {
    let semstrait = SemstraitBuilder::default()
        .with_catalog(Arc::new(NoopCatalogProvider::default()))
        .with_filesystem(Arc::new(InMemoryFileSystem::new()))
        .build();

    let Err((refusal, warnings)) = semstrait.compile_from_yaml(SALES).await else {
        panic!("a catalog that holds no table refuses the compile");
    };

    assert!(matches!(
        refusal.kind,
        SemstraitError::Compile(CompileError::CatalogSourceMissing { ref locator, .. })
            if locator == "warehouse.fact_rows"
    ));
    assert!(only_retired_keys(warnings));
}

#[tokio::test]
async fn compile_lists_a_path_source_through_the_filesystem_the_builder_holds() {
    let object = "s3://lake/orders/part-0.parquet";
    let filesystem = InMemoryFileSystem::new();
    filesystem
        .write(&Path::new(object), Bytes::from_static(b"PAR1"))
        .await
        .expect("the object is written");
    let semstrait = SemstraitBuilder::default()
        .with_filesystem(Arc::new(filesystem))
        .build();

    let (manifest, _) = semstrait
        .compile_from_yaml(LAKE)
        .await
        .expect("the model compiles");

    let stored = serde_json::to_string(&manifest).expect("the manifest serializes");
    assert!(stored.contains(object));
}

fn lake_at(path: &str) -> String {
    LAKE.replace("s3://lake/orders/", path)
}

#[tokio::test]
async fn the_default_filesystem_lists_a_local_object_and_its_folder_at_their_absolute_paths() {
    let dir = tempfile::tempdir().expect("a temporary folder");
    let folder = format!("{}/orders/", dir.path().display());
    let object = format!("{folder}part-0.parquet");
    std::fs::create_dir(&folder).expect("the folder is created");
    std::fs::write(&object, b"PAR1").expect("the object is written");

    for path in [&folder, &object] {
        let (manifest, _) = SemstraitBuilder::default()
            .build()
            .compile_from_yaml(&lake_at(path))
            .await
            .expect("the model compiles");

        let stored = serde_json::to_string(&manifest).expect("the manifest serializes");
        assert!(stored.contains(&format!("file://{object}")), "{path}");
    }
}

#[tokio::test]
async fn the_default_filesystem_refuses_a_missing_local_object() {
    let dir = tempfile::tempdir().expect("a temporary folder");
    let object = format!("{}/orders/part-0.parquet", dir.path().display());

    let refusal = SemstraitBuilder::default()
        .build()
        .compile_from_yaml(&lake_at(&object))
        .await;

    assert!(matches!(
        refusal,
        Err((diagnostic, _)) if matches!(
            diagnostic.kind,
            SemstraitError::Compile(CompileError::StorageObjectMissing { .. })
        )
    ));
}

#[tokio::test]
async fn compile_from_model_on_the_built_model_gives_the_same_model_id() {
    let (from_model, warnings) = semstrait()
        .compile_from_model(sales_model())
        .await
        .expect("the model compiles");

    assert_eq!(from_model.model_id, sales().await.model_id);
    assert!(warnings.is_empty());
}

#[tokio::test]
async fn a_model_that_parse_refuses_gives_the_parse_diagnostic_on_the_error_arm() {
    let refusal = semstrait().compile_from_yaml("semantic_model: [").await;

    assert!(matches!(
        refusal,
        Err((diagnostic, warnings))
            if matches!(diagnostic.kind, SemstraitError::Parse(ParseErrorKind::YamlSyntax { .. }))
                && warnings.is_empty()
    ));
}

#[tokio::test]
async fn an_accepted_request_is_valid_and_gives_the_explain_text_and_the_plan_with_its_advisories()
{
    let (manifest, request, context) = (sales().await, by_region(), RequestContext::default());

    let validated = semstrait().validate(&manifest, &request, &context);
    let explained = semstrait().explain(&manifest, &request, &context, Dialect::Substrait);
    let planned = semstrait().plan(&manifest, &request, &context, Dialect::Substrait);

    let (text, explain_advisories) = explained.expect("the request explains");
    let (plan, plan_advisories) = planned.expect("the request plans");
    assert!(text.contains("warehouse.fact_rows"));
    assert_eq!(plan.relations.len(), 1);
    for advisories in [
        validated.expect("the request is valid"),
        explain_advisories,
        plan_advisories,
    ] {
        assert!(matches!(
            kinds(advisories).as_slice(),
            [SemstraitError::Advisory(Advisory::AmbiguousCandidates { chosen, tied, .. })]
                if chosen.as_ref() == "orders" && tied == &[DataKindName::from("returns")]
        ));
    }
}

#[tokio::test]
async fn a_refused_request_gives_the_plan_error_wrapped_in_semstrait_error_on_each_verb() {
    let manifest = sales().await;
    let request = Request::new(vec![Selection::Field("revenue".into())]);
    let context = RequestContext::default();

    let refusals = [
        semstrait().validate(&manifest, &request, &context).err(),
        semstrait()
            .explain(&manifest, &request, &context, Dialect::Substrait)
            .err(),
        semstrait()
            .plan(&manifest, &request, &context, Dialect::Substrait)
            .err(),
    ];

    for refusal in refusals {
        let (diagnostic, advisories) = refusal.expect("the request is refused");
        let unknown = PlanError::UnknownName {
            name: "revenue".into(),
            looked_in: LookedIn::SemanticDefinitions,
        };
        assert!(matches!(diagnostic.kind, SemstraitError::Plan(kind) if kind == unknown));
        assert!(advisories.is_empty());
    }
}

async fn refusals(dialect: Dialect) -> [Option<(SemstraitError, Vec<SemstraitError>)>; 2] {
    let (manifest, request, context) = (sales().await, by_region(), RequestContext::default());
    let planned = semstrait().plan(&manifest, &request, &context, dialect);
    let explained = semstrait().explain(&manifest, &request, &context, dialect);
    [planned.err(), explained.err()]
        .map(|refused| refused.map(|(diagnostic, warnings)| (diagnostic.kind, kinds(warnings))))
}

async fn has_no_engine_adapter(dialect: Dialect, expected: NoEngineAdapter) -> bool {
    refusals(dialect).await.iter().all(|refused| {
        matches!(
            refused,
            Some((SemstraitError::NoEngineAdapter(kind), warnings))
                if *kind == expected && warnings.is_empty()
        )
    })
}

#[tokio::test]
async fn substrait_gives_a_plan_and_iceberg_gives_the_typed_error() {
    assert!(matches!(refusals(Dialect::Substrait).await, [None, None]));
    let no_rules = NoEngineAdapter::NoRules(Dialect::Iceberg);
    assert!(has_no_engine_adapter(Dialect::Iceberg, no_rules).await);
}

#[cfg(not(feature = "datafusion"))]
#[tokio::test]
async fn datafusion_gives_the_typed_error_without_the_datafusion_feature() {
    let feature_off = NoEngineAdapter::DataFusionFeatureOff;
    assert!(has_no_engine_adapter(Dialect::DataFusion, feature_off).await);
}

#[cfg(feature = "datafusion")]
mod datafusion {
    use std::iter;

    use semstrait_common::Dialect;
    use semstrait_planner::{Request, RequestContext, Selection};
    use substrait::proto::rel::RelType;
    use substrait::proto::{plan_rel, Plan};

    use crate::common::{compiled, semstrait, SPEND_BOOKINGS};

    async fn produced(dialect: Dialect) -> (Plan, String) {
        let manifest = compiled(SPEND_BOOKINGS).await;
        let names = ["campaign_id", "cost", "revenue"];
        let request = Request::new(names.map(|name| Selection::Field(name.into())).to_vec());
        let context = RequestContext::default();
        let planned = semstrait().plan(&manifest, &request, &context, dialect);
        let explained = semstrait().explain(&manifest, &request, &context, dialect);
        (
            planned.expect("the request plans").0,
            explained.expect("the request explains").0,
        )
    }

    fn below_root(plan: &Plan) -> Vec<&RelType> {
        let root = plan
            .relations
            .iter()
            .find_map(|relation| match &relation.rel_type {
                Some(plan_rel::RelType::Root(root)) => root.input.as_ref(),
                _ => None,
            });
        iter::successors(root, |rel| match rel.rel_type.as_ref()? {
            RelType::Project(project) => project.input.as_deref(),
            RelType::Aggregate(aggregate) => aggregate.input.as_deref(),
            RelType::Filter(filter) => filter.input.as_deref(),
            _ => None,
        })
        .filter_map(|rel| rel.rel_type.as_ref())
        .collect()
    }

    fn a_filter_above_a_join(text: &str) -> bool {
        let relations: Vec<&str> = text.lines().map(str::trim_start).collect();
        relations.windows(2).any(|pair| {
            matches!(pair, [above, below] if above.starts_with("Filter[") && below.starts_with("Join["))
        })
    }

    #[tokio::test]
    async fn a_relationship_filter_is_the_post_join_filter_under_substrait_and_a_filter_above_the_join_under_datafusion(
    ) {
        let ((substrait, substrait_text), (datafusion, datafusion_text)) = (
            produced(Dialect::Substrait).await,
            produced(Dialect::DataFusion).await,
        );

        assert!(!a_filter_above_a_join(&substrait_text), "{substrait_text}");
        assert!(a_filter_above_a_join(&datafusion_text), "{datafusion_text}");
        let (substrait, datafusion) = (below_root(&substrait), below_root(&datafusion));
        let [.., RelType::Join(faithful)] = substrait.as_slice() else {
            panic!("under Substrait the join is the lowest relation below the root");
        };
        let [.., RelType::Filter(filter), RelType::Join(join)] = datafusion.as_slice() else {
            panic!("under DataFusion a filter is above the join");
        };
        assert!(faithful.post_join_filter.is_some());
        assert_eq!(filter.condition, faithful.post_join_filter);
        assert_eq!(join.post_join_filter, None);
    }
}
