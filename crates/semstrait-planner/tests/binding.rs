mod common;

use common::golden::{self, Golden};
use common::{at_granularity, harness, refusal, select, skeleton};
use semstrait_common::Granularity;
use semstrait_ir::SemanticRole;
use semstrait_model::expr::{col, field, lit, measure};
use semstrait_model::UnresolvedSemanticExpr;
use semstrait_planner::{Direction, LookedIn, OrderKey, PlanError, Request, Selection};
use semstrait_sql::datatype::{DataType, TypeOrFamily};
use semstrait_sql::expr::{
    Expr, ExprOperationExt, MeasureAccessor, NamedReference, SemanticLeaf, TypeError,
};

async fn binding_refuses(case: &str, request: &Request, expected: PlanError) -> Golden {
    let (error, advisories) = refusal(&skeleton().await, request);

    assert_eq!(error.kind, expected);
    assert_eq!(error.kind.path(), None);
    assert!(advisories.is_empty());
    Golden::message(case, &error)
}

async fn an_empty_select() -> Golden {
    binding_refuses("empty_select", &select(&[]), PlanError::EmptySelect).await
}

#[tokio::test]
async fn a_request_that_selects_nothing_is_empty_select() {
    an_empty_select().await.assert_committed();
}

async fn a_repeated_selection() -> Golden {
    let expected = PlanError::RepeatedSelection {
        name: "region".into(),
    };
    binding_refuses("repeated_select", &select(&["region", "region"]), expected).await
}

#[tokio::test]
async fn a_name_selected_twice_is_repeated_selection() {
    a_repeated_selection().await.assert_committed();
}

async fn a_selection_of_another_role() -> Golden {
    let request = Request::new(vec![
        Selection::Measure {
            name: "region".into(),
        },
        Selection::Field("revenue".into()),
    ]);
    let expected = PlanError::SelectionRoleMismatch {
        name: "region".into(),
        selected: SemanticRole::Measure,
        role: SemanticRole::Dimension,
    };
    binding_refuses("role_mismatch", &request, expected).await
}

#[tokio::test]
async fn a_typed_selection_whose_role_is_not_the_manifests_is_selection_role_mismatch() {
    a_selection_of_another_role().await.assert_committed();
}

async fn a_selected_filter() -> Golden {
    let expected = PlanError::NotSelectable {
        name: "is_eu".into(),
        role: SemanticRole::Filter,
    };
    binding_refuses("not_selectable", &select(&["is_eu"]), expected).await
}

#[tokio::test]
async fn a_selected_named_filter_is_not_selectable() {
    a_selected_filter().await.assert_committed();
}

async fn an_order_key_not_selected() -> Golden {
    let request = select(&["region"]).with_order_by(vec![OrderKey::new("revenue", Direction::Asc)]);
    let expected = PlanError::OrderByNotSelected {
        name: "revenue".into(),
    };
    binding_refuses("order_by_not_selected", &request, expected).await
}

#[tokio::test]
async fn an_order_by_name_outside_select_is_order_by_not_selected() {
    an_order_key_not_selected().await.assert_committed();
}

fn read(name: &str) -> UnresolvedSemanticExpr {
    field(name).expect("a name")
}

fn region_and_revenue_where(predicate: UnresolvedSemanticExpr) -> Request {
    select(&["region", "revenue"]).with_predicate(predicate)
}

async fn a_predicate_of_mixed_placement() -> Golden {
    let predicate = read("channel").eq(lit("web")) & read("revenue").gt(lit(160));
    let expected = PlanError::MixedPlacement {
        dimension: "channel".into(),
        aggregated: "revenue".into(),
    };
    let request = region_and_revenue_where(predicate);
    binding_refuses("mixed_placement", &request, expected).await
}

#[tokio::test]
async fn a_predicate_that_reads_a_dimension_beside_a_measure_is_mixed_placement() {
    a_predicate_of_mixed_placement().await.assert_committed();
}

async fn a_filter_read_by_the_predicate() -> Golden {
    let expected = PlanError::FilterInRequestPredicate {
        name: "eu_web".into(),
    };
    let request = region_and_revenue_where(read("eu_web"));
    binding_refuses("filter_leaf", &request, expected).await
}

#[tokio::test]
async fn a_filter_leaf_in_the_request_predicate_is_filter_in_request_predicate() {
    a_filter_read_by_the_predicate().await.assert_committed();
}

async fn a_column_read_by_the_predicate() -> Golden {
    let amount = NamedReference::unqualified("amount").expect("a name");
    let predicate = col("amount").expect("a name").gt(lit(10));
    let expected = PlanError::ColumnInRequestPredicate { column: amount };
    let request = region_and_revenue_where(predicate);
    binding_refuses("column_leaf", &request, expected).await
}

#[tokio::test]
async fn a_column_leaf_in_the_request_predicate_is_column_in_request_predicate() {
    a_column_read_by_the_predicate().await.assert_committed();
}

async fn an_accessor_read_by_the_predicate() -> Golden {
    let revenue = NamedReference::unqualified("revenue").expect("a name");
    let previous = Expr::Leaf(SemanticLeaf::Measure(
        revenue,
        Some(MeasureAccessor::Previous),
    ));
    let expected = PlanError::AccessorInRequestPredicate {
        name: "revenue".into(),
    };
    let request = region_and_revenue_where(previous.gt(lit(10)));
    binding_refuses("accessor_leaf", &request, expected).await
}

#[tokio::test]
async fn a_leaf_with_a_sugar_accessor_in_the_request_predicate_is_refused() {
    an_accessor_read_by_the_predicate().await.assert_committed();
}

async fn a_leaf_of_another_role() -> Golden {
    let predicate = measure("region").expect("a name").eq(lit("eu"));
    let expected = PlanError::PredicateRoleMismatch {
        name: "region".into(),
        stated: SemanticRole::Measure,
        role: SemanticRole::Dimension,
    };
    let request = region_and_revenue_where(predicate);
    binding_refuses("leaf_role_mismatch", &request, expected).await
}

#[tokio::test]
async fn a_typed_leaf_whose_role_is_not_the_manifests_is_predicate_role_mismatch() {
    a_leaf_of_another_role().await.assert_committed();
}

async fn a_predicate_that_does_not_type() -> Golden {
    let predicate = read("region").eq(lit(5));
    let expected = PlanError::PredicateTypeError {
        error: TypeError::Arguments {
            name: "eq",
            types: vec![
                TypeOrFamily::Type(DataType::String),
                TypeOrFamily::Type(DataType::Integer),
            ],
        },
    };
    let request = region_and_revenue_where(predicate);
    binding_refuses("predicate_type", &request, expected).await
}

#[tokio::test]
async fn a_predicate_that_does_not_type_against_the_manifest_is_predicate_type_error() {
    a_predicate_that_does_not_type().await.assert_committed();
}

async fn a_predicate_that_is_no_condition() -> Golden {
    let expected = PlanError::PredicateNotACondition {
        got: TypeOrFamily::Type(DataType::String),
    };
    let request = region_and_revenue_where(read("region"));
    binding_refuses("predicate_not_a_condition", &request, expected).await
}

#[tokio::test]
async fn a_predicate_whose_root_is_not_a_boolean_is_predicate_not_a_condition() {
    a_predicate_that_is_no_condition().await.assert_committed();
}

async fn a_request_with_two_faults() -> Golden {
    let expected = PlanError::RepeatedSelection {
        name: "region".into(),
    };
    let request = select(&["region", "region"]).with_predicate(read("eu_web"));
    binding_refuses("two_faults", &request, expected).await
}

#[tokio::test]
async fn a_request_with_two_faults_gets_the_first_in_bindings_order() {
    a_request_with_two_faults().await.assert_committed();
}

async fn a_granularity_on_a_categorical_dimension() -> Golden {
    let expected = PlanError::GranularityOnNonTemporalDimension {
        name: "region".into(),
        requested: Granularity::Month,
    };
    let request = at_granularity("region", Granularity::Month, &["revenue"]);
    binding_refuses("granularity_on_non_temporal", &request, expected).await
}

#[tokio::test]
async fn a_granularity_on_a_dimension_that_is_not_temporal_is_refused() {
    a_granularity_on_a_categorical_dimension()
        .await
        .assert_committed();
}

async fn an_hour_on_a_date() -> Golden {
    let expected = PlanError::GranularityFinerThanType {
        name: "ordered_at".into(),
        requested: Granularity::Hour,
        data_type: DataType::Date,
    };
    let request = at_granularity("ordered_at", Granularity::Hour, &["revenue"]);
    binding_refuses("hour_on_date", &request, expected).await
}

#[tokio::test]
async fn an_hour_on_a_date_dimension_is_granularity_finer_than_type() {
    an_hour_on_a_date().await.assert_committed();
}

async fn a_name_no_definition_holds() -> Golden {
    let expected = PlanError::UnknownName {
        name: "nope".into(),
        looked_in: LookedIn::SemanticDefinitions,
    };
    binding_refuses("unknown_name", &select(&["nope"]), expected).await
}

#[tokio::test]
async fn a_selected_name_that_no_semantic_definition_holds_is_unknown_name() {
    a_name_no_definition_holds().await.assert_committed();
}

async fn a_leaf_no_definition_holds() -> Golden {
    let expected = PlanError::UnknownName {
        name: "nope".into(),
        looked_in: LookedIn::SemanticDefinitions,
    };
    let request = region_and_revenue_where(read("nope").eq(lit("x")));
    binding_refuses("unknown_leaf", &request, expected).await
}

#[tokio::test]
async fn a_predicate_leaf_that_no_semantic_definition_holds_is_unknown_name() {
    a_leaf_no_definition_holds().await.assert_committed();
}

async fn a_filter_no_definition_holds() -> Golden {
    let expected = PlanError::UnknownName {
        name: "nope".into(),
        looked_in: LookedIn::SemanticDefinitions,
    };
    let request = select(&["region", "revenue"]).with_filters(vec!["nope".into()]);
    binding_refuses("unknown_filter", &request, expected).await
}

#[tokio::test]
async fn an_activated_filter_that_no_semantic_definition_holds_is_unknown_name() {
    a_filter_no_definition_holds().await.assert_committed();
}

async fn a_filter_that_names_a_dimension() -> Golden {
    let expected = PlanError::UnknownName {
        name: "region".into(),
        looked_in: LookedIn::NamedFilters,
    };
    let request = select(&["region", "revenue"]).with_filters(vec!["region".into()]);
    binding_refuses("filter_not_named_filter", &request, expected).await
}

#[tokio::test]
async fn an_activated_filter_whose_definition_is_not_a_filter_is_unknown_name() {
    a_filter_that_names_a_dimension().await.assert_committed();
}

async fn the_harness_error_case() -> Golden {
    let manifest = harness().await;

    let request = select(&["nope"]);
    let (first, _) = refusal(&manifest, &request);
    let (second, _) = refusal(&manifest, &request);

    assert_eq!(first.kind, second.kind);
    assert_eq!(
        first.kind,
        PlanError::UnknownName {
            name: "nope".into(),
            looked_in: LookedIn::SemanticDefinitions,
        }
    );
    assert_eq!(first.kind.path(), None);
    Golden::message("harness_unknown_name", &first)
}

#[tokio::test]
async fn the_harness_request_for_an_unknown_name_is_refused_alike_when_planned_twice() {
    the_harness_error_case().await.assert_committed();
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        vec![
            a_name_no_definition_holds().await,
            the_harness_error_case().await,
            an_empty_select().await,
            a_repeated_selection().await,
            a_selection_of_another_role().await,
            a_selected_filter().await,
            an_order_key_not_selected().await,
            a_granularity_on_a_categorical_dimension().await,
            an_hour_on_a_date().await,
            a_predicate_of_mixed_placement().await,
            a_filter_read_by_the_predicate().await,
            a_request_with_two_faults().await,
            a_column_read_by_the_predicate().await,
            an_accessor_read_by_the_predicate().await,
            a_leaf_of_another_role().await,
            a_predicate_that_does_not_type().await,
            a_predicate_that_is_no_condition().await,
            a_leaf_no_definition_holds().await,
            a_filter_no_definition_holds().await,
            a_filter_that_names_a_dimension().await,
        ]
    })
    .await;
}
