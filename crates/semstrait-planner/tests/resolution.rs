mod common;

use common::golden::{self, Golden};
use common::{
    compiled, orders_catalog, planned, read_table, refusal, select, skeleton, with_datasets,
    Catalog, ORDERS,
};
use semstrait_catalog::FileFormat;
use semstrait_common::io::InMemoryFileSystem;
use semstrait_common::DataKindName;
use semstrait_ir::SemAnnotation;
use semstrait_manifest::SemanticManifest;
use semstrait_model::expr::{field, lit};
use semstrait_planner::{LookedIn, PlanError, Request};
use semstrait_sql::expr::ExprOperationExt;

async fn a_from_that_names_no_public_source() -> Golden {
    let request = select(&["region", "revenue"]).with_from("nope");
    let (error, advisories) = refusal(&skeleton().await, &request);

    assert_eq!(
        error.kind,
        PlanError::UnknownName {
            name: "nope".into(),
            looked_in: LookedIn::PublicSources,
        }
    );
    assert_eq!(error.kind.path(), None);
    assert!(advisories.is_empty());
    Golden::message("from_unknown", &error)
}

#[tokio::test]
async fn a_from_that_names_no_public_source_is_unknown_name() {
    a_from_that_names_no_public_source()
        .await
        .assert_committed();
}

const TARGETS: &str = r#"
    - name: targets
      extras:
        storage:
          tables: [warehouse.targets]
        semantic_mapping:
          region: region
          target: target
      dimensions:
        - ref: region
      measures:
        - name: target
          data_type: double
          agg: sum
"#;

async fn skeleton_and_targets() -> SemanticManifest {
    with_datasets(&format!("{ORDERS}{TARGETS}"), &targets_catalog()).await
}

fn targets_catalog() -> Catalog {
    orders_catalog().table(
        "warehouse.targets",
        FileFormat::Iceberg,
        &[("region", "string"), ("target", "double")],
    )
}

async fn unserved_by_from(case: &str, request: &Request, name: &str) -> Golden {
    let (error, advisories) = refusal(&skeleton_and_targets().await, request);
    let from = request.from.clone().expect("a request with `from`");

    assert_eq!(
        error.kind,
        PlanError::UnknownName {
            name: name.into(),
            looked_in: LookedIn::Source(from),
        }
    );
    assert_eq!(error.kind.path(), None);
    assert!(advisories.is_empty());
    Golden::message(case, &error)
}

async fn a_name_that_from_does_not_serve() -> Golden {
    let request = select(&["region", "target"]).with_from("orders");
    unserved_by_from("from_unserved", &request, "target").await
}

#[tokio::test]
async fn a_name_that_from_does_not_serve_is_unknown_name_and_never_widened() {
    a_name_that_from_does_not_serve().await.assert_committed();
}

async fn a_leaf_that_from_does_not_serve() -> Golden {
    let predicate = field("channel").expect("a name").eq(lit("web"));
    let request = (select(&["target"]).with_from("targets")).with_predicate(predicate);
    unserved_by_from("from_unserved_leaf", &request, "channel").await
}

#[tokio::test]
async fn a_predicate_leaf_that_from_does_not_serve_is_unknown_name() {
    a_leaf_that_from_does_not_serve().await.assert_committed();
}

async fn a_filter_that_from_does_not_serve() -> Golden {
    let request = (select(&["target"]).with_from("targets")).with_filters(vec!["is_eu".into()]);
    unserved_by_from("from_unserved_filter", &request, "is_eu").await
}

#[tokio::test]
async fn an_activated_filter_that_from_does_not_serve_is_unknown_name() {
    a_filter_that_from_does_not_serve().await.assert_committed();
}

async fn a_demand_that_no_public_source_serves() -> Golden {
    let request = select(&["revenue", "target"]);
    let (error, advisories) = refusal(&skeleton_and_targets().await, &request);

    assert_eq!(
        error.kind,
        PlanError::NoCandidate {
            nearest: vec![
                ("targets".into(), vec!["revenue".into()]),
                ("orders".into(), vec!["target".into()]),
            ],
        }
    );
    assert_eq!(error.kind.path(), None);
    assert!(advisories.is_empty());
    Golden::message("no_candidate", &error)
}

#[tokio::test]
async fn a_demand_that_no_public_source_serves_is_no_candidate_naming_the_nearest() {
    a_demand_that_no_public_source_serves()
        .await
        .assert_committed();
}

const FORECASTS: &str = r#"
    - name: forecasts
      extras:
        storage:
          tables: [warehouse.forecasts]
        semantic_mapping:
          region: region
          revenue: forecast
      dimensions:
        - ref: region
      measures:
        - ref: revenue
"#;

#[tokio::test]
async fn a_tie_on_every_term_goes_to_name_order() {
    let catalog = targets_catalog().table(
        "warehouse.forecasts",
        FileFormat::Iceberg,
        &[("region", "string"), ("forecast", "double")],
    );
    let manifest = with_datasets(&format!("{ORDERS}{TARGETS}{FORECASTS}"), &catalog).await;

    let (plan, _) = planned(&manifest, &select(&["region"]));

    assert_eq!(read_table(&plan), "forecasts");
}

const REGIONS: &str = r#"
    - name: regions
      mode: all
      dimensions:
        - ref: region
      datasets:
        - name: target_regions
          extras:
            storage:
              tables: [warehouse.targets]
        - name: forecast_regions
          extras:
            storage:
              tables: [warehouse.forecasts]
"#;

#[tokio::test]
async fn a_dataset_outranks_a_composition_that_exposes_fewer_semantics() {
    let catalog = targets_catalog().table(
        "warehouse.forecasts",
        FileFormat::Iceberg,
        &[("region", "string"), ("forecast", "double")],
    );
    let manifest = with_datasets(
        &format!("{ORDERS}{TARGETS}  unionsets:\n{REGIONS}"),
        &catalog,
    )
    .await;

    let (plan, advisories) = planned(&manifest, &select(&["region"]));

    assert_eq!(read_table(&plan), "targets");
    assert!(advisories.is_empty());
}

const JOINS: &str = r#"
semantic_model:
  name: joins
  dimensions:
    - name: customer_id
      data_type: string
      type: categorical
    - name: region
      data_type: string
      type: categorical
  joinsets:
    - name: chain
      dimensions:
        - ref: customer_id
        - ref: region
      datasets:
        - name: chain_a
          extras:
            storage:
              tables: [warehouse.a]
        - name: chain_b
          extras:
            storage:
              tables: [warehouse.b]
        - name: chain_c
          extras:
            storage:
              tables: [warehouse.c]
      relationships:
        - name: chain_a_to_b
          from: chain_a
          to: chain_b
          on: [customer_id]
          cardinality: many_to_one
        - name: chain_b_to_c
          from: chain_b
          to: chain_c
          on: [customer_id]
          cardinality: many_to_one
    - name: pair
      dimensions:
        - ref: customer_id
        - ref: region
      datasets:
        - name: pair_a
          extras:
            storage:
              tables: [warehouse.a]
        - name: pair_b
          extras:
            storage:
              tables: [warehouse.b]
      relationships:
        - name: pair_a_to_b
          from: pair_a
          to: pair_b
          on: [customer_id]
          cardinality: many_to_one
"#;

#[tokio::test]
async fn compositions_count_the_compositions_beneath_a_source() {
    let columns = [("customer_id", "string"), ("region", "string")];
    let catalog = ["warehouse.a", "warehouse.b", "warehouse.c"]
        .into_iter()
        .fold(Catalog::default(), |catalog, table| {
            catalog.table(table, FileFormat::Iceberg, &columns)
        });
    let manifest = compiled(JOINS, &catalog, &InMemoryFileSystem::new()).await;

    let (plan, advisories) = planned(&manifest, &select(&["region"]));

    let pair = DataKindName::new("pair");
    assert!(
        (plan.root.common().semantic_meta)
            .iter()
            .any(|subject| matches!(subject, SemAnnotation::Rank { scope_root, .. } if *scope_root == pair)),
        "{plan:?}"
    );
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        vec![
            a_from_that_names_no_public_source().await,
            a_name_that_from_does_not_serve().await,
            a_leaf_that_from_does_not_serve().await,
            a_filter_that_from_does_not_serve().await,
            a_demand_that_no_public_source_serves().await,
        ]
    })
    .await;
}
