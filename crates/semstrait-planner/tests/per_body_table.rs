mod common;

use common::golden::{self, Golden};
use common::{
    advisory_kinds, alias, orders_catalog, output_names, planned, qualified, refusal_by, scan_of,
    select, skeleton, spend_bookings, txn_by_grain, with_datasets, Replaced, ORDERS,
    SPEND_BOOKINGS, TRANSACTIONS, TWINS, TXN_MONTHLY_AGG, TXN_WEEKLY_AGG,
};
use semstrait_common::{Diagnose, Diagnostic};
use semstrait_ir::{
    Cardinality, IrErrorKind, JoinType, Parameter, ParameterKey, PhysicalExpr, Schema,
    SchemaColumn, VirtualTable,
};
use semstrait_ir::{Integrity, JoinsetJoinType};
use semstrait_manifest::{SemanticManifest, SourceAlias};
use semstrait_model::expr::{dim, lit};
use semstrait_planner::{
    Combine, Decision, DecisionValue, Operand, PhysicalRead, PlanError, Planner, ReadModel,
    Relationship, Request, Strategy, StrategyInput,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, ExprOperationExt, Literal};

fn operand(manifest: &SemanticManifest, name: &str, names: &[&str]) -> Operand {
    Operand {
        source: alias(manifest, name).into(),
        names: names.iter().map(|&name| name.into()).collect(),
        null_filled: Vec::new(),
    }
}

#[derive(Debug)]
struct ReplacedJoinset {
    decision: Decision,
    filter: PhysicalExpr,
}

impl Strategy<Relationship> for ReplacedJoinset {
    fn decide(
        &self,
        _: &dyn ReadModel,
        relationship: &Relationship,
        _: &StrategyInput,
    ) -> Result<Decision, PlanError> {
        assert_eq!(
            relationship.join_type,
            JoinsetJoinType::Join(JoinType::Inner)
        );
        assert_eq!(relationship.cardinality, Cardinality::OneToOne);
        assert_eq!(relationship.integrity, Integrity::Assumed);
        assert_eq!(relationship.filter.as_ref(), Some(&self.filter));
        Ok(self.decision.clone())
    }
}

fn joined(manifest: &SemanticManifest) -> Decision {
    let operands =
        ["spend", "bookings"].map(|end| operand(manifest, &format!("spend_bookings.{end}"), &[]));
    Decision {
        stand_in: None,
        operands: operands.into(),
        combine: Combine::Join,
        physical_read: None,
    }
}

fn deciding(manifest: &SemanticManifest, decision: Decision) -> Planner {
    let day = Expr::Literal(Literal::Date("2026-03-15".into()));
    let filter = qualified(&alias(manifest, "spend_bookings.spend"), "date").gt_eq(day);
    Planner::new().with_joinset_strategy(Box::new(ReplacedJoinset { decision, filter }))
}

fn leaf() -> Decision {
    Decision {
        stand_in: None,
        operands: Vec::new(),
        combine: Combine::Leaf,
        physical_read: None,
    }
}

fn not_allowed(
    planner: &Planner,
    manifest: &SemanticManifest,
    request: &Request,
    source: &str,
    value: &DecisionValue,
) -> Diagnostic<PlanError> {
    not_allowed_at(planner, manifest, request, source, source, value)
}

fn not_allowed_at(
    planner: &Planner,
    manifest: &SemanticManifest,
    request: &Request,
    path: &str,
    source: &str,
    value: &DecisionValue,
) -> Diagnostic<PlanError> {
    let (error, advisories) = refusal_by(planner, manifest, request);

    let PlanError::DecisionNotAllowed {
        semantic_source,
        value: refused,
        ..
    } = &error.kind
    else {
        panic!("the table check's error, got {error:?}");
    };
    assert_eq!(semantic_source.as_str(), source);
    assert_eq!(refused, value);
    let at = error.kind.path().map(ToString::to_string);
    assert_eq!(at.as_deref(), Some(path));
    assert!(advisories.is_empty());
    error
}

async fn a_value_the_dataset_row_refuses(
    case: &str,
    decision: Decision,
    value: DecisionValue,
) -> Golden {
    let planner = Planner::new().with_dataset_strategy(Box::new(Replaced(decision)));
    let request = select(&["revenue"]);

    let error = not_allowed(&planner, &skeleton().await, &request, "orders", &value);

    Golden::message(case, &error)
}

async fn a_stand_in_for_a_dataset() -> Golden {
    let decision = Decision {
        stand_in: Some("targets".into()),
        ..leaf()
    };
    a_value_the_dataset_row_refuses(
        "dataset_stand_in",
        decision,
        DecisionValue::StandIn("targets".into()),
    )
    .await
}

async fn a_join_for_a_dataset() -> Golden {
    let decision = Decision {
        combine: Combine::Join,
        ..leaf()
    };
    a_value_the_dataset_row_refuses(
        "dataset_join",
        decision,
        DecisionValue::Combine(Combine::Join),
    )
    .await
}

async fn an_operand_for_a_dataset() -> Golden {
    let manifest = skeleton().await;
    let decision = Decision {
        operands: vec![operand(&manifest, "orders", &[])],
        ..leaf()
    };
    let value = DecisionValue::NotAnOperand("orders".into());

    a_value_the_dataset_row_refuses("dataset_operand", decision, value).await
}

async fn a_combine_the_joinset_row_refuses(case: &str, combine: Combine) -> Golden {
    let manifest = spend_bookings(SPEND_BOOKINGS).await;
    let planner = deciding(
        &manifest,
        Decision {
            combine,
            ..joined(&manifest)
        },
    );
    let (request, value) = (select(&["campaign_id"]), DecisionValue::Combine(combine));

    let error = not_allowed(&planner, &manifest, &request, "spend_bookings", &value);

    Golden::message(case, &error)
}

async fn a_stack_for_a_joinset() -> Golden {
    a_combine_the_joinset_row_refuses("joinset_stack", Combine::Stack).await
}

async fn a_drill_across_for_a_joinset() -> Golden {
    a_combine_the_joinset_row_refuses("joinset_drill_across", Combine::DrillAcross).await
}

async fn an_operand_the_joinset_row_refuses(
    operands: &[&str],
    combine: Combine,
) -> Diagnostic<PlanError> {
    let manifest = spend_bookings(SPEND_BOOKINGS).await;
    let decision = Decision {
        operands: (operands.iter())
            .map(|&source| operand(&manifest, source, &[]))
            .collect(),
        combine,
        ..joined(&manifest)
    };
    let value = DecisionValue::NotAnOperand("spend_bookings".into());
    let planner = deciding(&manifest, decision);

    not_allowed(
        &planner,
        &manifest,
        &select(&["revenue"]),
        "spend_bookings",
        &value,
    )
}

async fn an_operand_that_is_no_end_of_a_joinset() -> Golden {
    let operands = [
        "spend_bookings.spend",
        "spend_bookings.bookings",
        "spend_bookings",
    ];

    let error = an_operand_the_joinset_row_refuses(&operands, Combine::Join).await;

    Golden::message("joinset_not_an_operand", &error)
}

const ALL_ORDERS: &str = r#"
    - name: all_orders
      mode: all
      dimensions:
        - ref: region
      measures:
        - ref: revenue
      datasets:
        - name: eu_orders
          extras:
            storage:
              tables: [warehouse.orders]
            semantic_mapping: {region: region, revenue: amount}
        - name: us_orders
          extras:
            storage:
              tables: [warehouse.orders]
            semantic_mapping: {region: region, revenue: amount}
"#;

async fn with_all_orders() -> SemanticManifest {
    with_datasets(
        &format!("{ORDERS}  unionsets:\n{ALL_ORDERS}"),
        &orders_catalog(),
    )
    .await
}

fn stacked(manifest: &SemanticManifest) -> Decision {
    let branches = ["eu_orders", "us_orders"]
        .map(|branch| operand(manifest, &format!("all_orders.{branch}"), &["revenue"]));
    Decision {
        stand_in: None,
        operands: branches.into(),
        combine: Combine::Stack,
        physical_read: None,
    }
}

async fn a_value_the_unionset_row_refuses(
    decision: impl FnOnce(&SemanticManifest) -> Decision,
    value: DecisionValue,
) -> Diagnostic<PlanError> {
    let manifest = with_all_orders().await;
    let replaced = Replaced(decision(&manifest));
    let planner = Planner::new().with_unionset_strategy(Box::new(replaced));
    let request = select(&["revenue"]).with_from("all_orders");

    not_allowed(&planner, &manifest, &request, "all_orders", &value)
}

async fn a_stand_in_for_a_unionset() -> Golden {
    let decision = |manifest: &SemanticManifest| Decision {
        stand_in: Some("eu_orders".into()),
        ..stacked(manifest)
    };
    let value = DecisionValue::StandIn("eu_orders".into());

    let error = a_value_the_unionset_row_refuses(decision, value).await;

    Golden::message("unionset_stand_in", &error)
}

async fn a_join_for_a_unionset() -> Golden {
    let decision = |manifest: &SemanticManifest| Decision {
        combine: Combine::Join,
        ..stacked(manifest)
    };
    let value = DecisionValue::Combine(Combine::Join);

    let error = a_value_the_unionset_row_refuses(decision, value).await;

    Golden::message("unionset_join", &error)
}

async fn a_branch_of_a_unionset_not_read() -> Golden {
    let decision = |manifest: &SemanticManifest| Decision {
        operands: vec![operand(manifest, "all_orders.eu_orders", &["revenue"])],
        ..stacked(manifest)
    };
    let value = DecisionValue::NotRead("all_orders.us_orders".into());

    let error = a_value_the_unionset_row_refuses(decision, value).await;

    Golden::message("unionset_not_read", &error)
}

fn drilled(manifest: &SemanticManifest) -> Decision {
    let children = [
        ("txn_monthly_agg", "revenue_target"),
        ("transactions", "quantity"),
    ]
    .map(|(child, name)| operand(manifest, &format!("txn_by_grain.{child}"), &[name]));
    Decision {
        stand_in: None,
        operands: children.into(),
        combine: Combine::DrillAcross,
        physical_read: None,
    }
}

async fn a_value_the_grainset_row_refuses(
    decision: impl FnOnce(&SemanticManifest) -> Decision,
    value: DecisionValue,
) -> Diagnostic<PlanError> {
    let manifest = txn_by_grain(&[TRANSACTIONS, TXN_MONTHLY_AGG, TXN_WEEKLY_AGG]).await;
    let replaced = Replaced(decision(&manifest));
    let planner = Planner::new().with_grainset_strategy(Box::new(replaced));
    let request = select(&["quantity", "revenue_target"]).with_from("txn_by_grain");

    not_allowed(&planner, &manifest, &request, "txn_by_grain", &value)
}

async fn a_combine_the_grainset_row_refuses(case: &str, combine: Combine) -> Golden {
    let decision = |manifest: &SemanticManifest| Decision {
        combine,
        ..drilled(manifest)
    };

    let error = a_value_the_grainset_row_refuses(decision, DecisionValue::Combine(combine)).await;

    Golden::message(case, &error)
}

async fn a_stack_for_a_grainset() -> Golden {
    a_combine_the_grainset_row_refuses("grainset_stack", Combine::Stack).await
}

async fn a_join_for_a_grainset() -> Golden {
    a_combine_the_grainset_row_refuses("grainset_join", Combine::Join).await
}

#[tokio::test]
async fn the_built_in_dataset_decision_passes_its_row_and_is_planned() {
    let (plan, advisories) = planned(&skeleton().await, &select(&["region", "revenue"]));

    assert_eq!(output_names(&plan), ["region", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_leaf_with_no_physical_read_is_decision_not_allowed() {
    let planner = Planner::new().with_dataset_strategy(Box::new(Replaced(leaf())));
    let request = select(&["revenue"]);

    not_allowed(
        &planner,
        &skeleton().await,
        &request,
        "orders",
        &DecisionValue::NoPhysicalRead,
    );
}

#[tokio::test]
async fn a_stand_in_for_a_dataset_is_decision_not_allowed() {
    a_stand_in_for_a_dataset().await.assert_committed();
}

#[tokio::test]
async fn a_combine_other_than_leaf_for_a_dataset_is_decision_not_allowed() {
    a_join_for_a_dataset().await.assert_committed();
}

#[tokio::test]
async fn an_operand_for_a_dataset_is_decision_not_allowed() {
    an_operand_for_a_dataset().await.assert_committed();
}

#[tokio::test]
async fn an_operand_whose_alias_names_no_source_is_decision_not_allowed_naming_the_alias() {
    let manifest = skeleton().await;
    let unknown = SourceAlias::new("ssd99");
    assert!(
        (manifest.semantic_sources.values()).all(|source| source.alias != unknown),
        "no source has the alias {unknown}"
    );
    let decision = Decision {
        operands: vec![Operand {
            source: unknown.clone(),
            names: Vec::new(),
            null_filled: Vec::new(),
        }],
        ..leaf()
    };
    let planner = Planner::new().with_dataset_strategy(Box::new(Replaced(decision)));
    let value = DecisionValue::UnknownOperand(unknown);

    let error = not_allowed(&planner, &manifest, &select(&["revenue"]), "orders", &value);

    assert!(error.to_string().contains("`ssd99`"), "{error}");
}

#[tokio::test]
async fn an_operand_for_a_unionset_branch_is_decision_not_allowed() {
    let manifest = with_all_orders().await;
    let eu_orders = "all_orders.eu_orders";
    let decision = Decision {
        operands: vec![operand(&manifest, eu_orders, &[])],
        ..leaf()
    };
    let planner = Planner::new().with_dataset_strategy(Box::new(Replaced(decision)));
    let request = select(&["revenue"]).with_from("all_orders");
    let value = DecisionValue::NotAnOperand(eu_orders.into());

    not_allowed_at(
        &planner,
        &manifest,
        &request,
        "all_orders / all_orders.eu_orders",
        eu_orders,
        &value,
    );
}

#[tokio::test]
async fn a_stack_for_a_joinset_is_decision_not_allowed() {
    a_stack_for_a_joinset().await.assert_committed();
}

#[tokio::test]
async fn a_drill_across_for_a_joinset_is_decision_not_allowed() {
    a_drill_across_for_a_joinset().await.assert_committed();
}

#[tokio::test]
async fn a_cross_for_a_keyed_joinset_is_decision_not_allowed_before_its_operands() {
    let manifest = spend_bookings(SPEND_BOOKINGS).await;
    let (spend, bookings) = ("spend_bookings.spend", "spend_bookings.bookings");
    let decision = Decision {
        operands: vec![
            operand(&manifest, spend, &["revenue"]),
            operand(&manifest, bookings, &[]),
        ],
        combine: Combine::Cross,
        ..joined(&manifest)
    };
    let planner = deciding(&manifest, decision);
    let (request, value) = (
        select(&["campaign_id"]),
        DecisionValue::Combine(Combine::Cross),
    );

    not_allowed(&planner, &manifest, &request, "spend_bookings", &value);
}

#[tokio::test]
async fn a_join_that_leaves_out_an_end_is_decision_not_allowed_before_its_operands() {
    let manifest = spend_bookings(SPEND_BOOKINGS).await;
    let decision = Decision {
        operands: vec![operand(&manifest, "spend_bookings.spend", &["revenue"])],
        ..joined(&manifest)
    };
    let planner = deciding(&manifest, decision);
    let (request, value) = (
        select(&["revenue"]),
        DecisionValue::NotRead("spend_bookings.bookings".into()),
    );

    not_allowed(&planner, &manifest, &request, "spend_bookings", &value);
}

#[tokio::test]
async fn a_pass_through_that_leaves_out_an_end_it_may_not_prune_is_decision_not_allowed() {
    let manifest = spend_bookings(SPEND_BOOKINGS).await;
    let decision = Decision {
        operands: vec![operand(&manifest, "spend_bookings.bookings", &["revenue"])],
        combine: Combine::PassThrough,
        ..joined(&manifest)
    };
    let planner = deciding(&manifest, decision);
    let (request, value) = (
        select(&["revenue"]),
        DecisionValue::NotRead("spend_bookings.spend".into()),
    );

    not_allowed(&planner, &manifest, &request, "spend_bookings", &value);
}

#[tokio::test]
async fn a_stand_in_that_is_no_alternative_is_decision_not_allowed() {
    let manifest = spend_bookings(SPEND_BOOKINGS).await;
    let decision = Decision {
        stand_in: Some("spend_bookings.spend".into()),
        operands: Vec::new(),
        combine: Combine::PassThrough,
        ..joined(&manifest)
    };
    let planner = deciding(&manifest, decision);
    let (request, value) = (
        select(&["campaign_id"]),
        DecisionValue::StandIn("spend_bookings.spend".into()),
    );

    not_allowed(&planner, &manifest, &request, "spend_bookings", &value);
}

#[tokio::test]
async fn an_operand_that_is_no_end_is_decision_not_allowed() {
    an_operand_that_is_no_end_of_a_joinset()
        .await
        .assert_committed();
}

#[tokio::test]
async fn a_pass_through_of_an_operand_that_is_no_end_is_decision_not_allowed() {
    an_operand_the_joinset_row_refuses(&["spend_bookings"], Combine::PassThrough).await;
}

#[tokio::test]
async fn the_built_in_unionset_decision_passes_its_row_and_is_planned() {
    let request = select(&["region", "revenue"]).with_from("all_orders");

    let (plan, advisories) = planned(&with_all_orders().await, &request);

    assert_eq!(output_names(&plan), ["region", "revenue"]);
    assert!(advisories.is_empty());
}

#[tokio::test]
async fn a_stand_in_for_a_unionset_is_decision_not_allowed() {
    a_stand_in_for_a_unionset().await.assert_committed();
}

#[tokio::test]
async fn a_combine_other_than_stack_for_a_unionset_is_decision_not_allowed() {
    a_join_for_a_unionset().await.assert_committed();
}

#[tokio::test]
async fn a_branch_not_read_with_no_predicate_is_decision_not_allowed() {
    a_branch_of_a_unionset_not_read().await.assert_committed();
}

#[tokio::test]
async fn a_branch_not_read_that_the_predicate_does_not_exclude_is_decision_not_allowed() {
    let manifest = with_all_orders().await;
    let decision = Decision {
        operands: vec![operand(&manifest, "all_orders.eu_orders", &["revenue"])],
        ..stacked(&manifest)
    };
    let planner = Planner::new().with_unionset_strategy(Box::new(Replaced(decision)));
    let eu = dim("region").expect("a name").eq(lit("eu"));
    let request = (select(&["revenue"]).with_from("all_orders")).with_predicate(eu);
    let value = DecisionValue::NotRead("all_orders.us_orders".into());

    not_allowed(&planner, &manifest, &request, "all_orders", &value);
}

#[tokio::test]
async fn an_operand_that_is_no_branch_is_decision_not_allowed() {
    let itself = |manifest: &SemanticManifest| {
        let mut decision = stacked(manifest);
        decision
            .operands
            .push(operand(manifest, "all_orders", &["revenue"]));
        decision
    };
    let value = DecisionValue::NotAnOperand("all_orders".into());

    a_value_the_unionset_row_refuses(itself, value).await;
}

#[tokio::test]
async fn a_stack_for_a_grainset_is_decision_not_allowed() {
    a_stack_for_a_grainset().await.assert_committed();
}

#[tokio::test]
async fn a_join_for_a_grainset_is_decision_not_allowed() {
    a_join_for_a_grainset().await.assert_committed();
}

#[tokio::test]
async fn a_stand_in_that_is_no_child_is_decision_not_allowed() {
    let itself = |manifest: &SemanticManifest| Decision {
        stand_in: Some("txn_by_grain".into()),
        operands: Vec::new(),
        combine: Combine::PassThrough,
        ..drilled(manifest)
    };
    let value = DecisionValue::StandIn("txn_by_grain".into());

    a_value_the_grainset_row_refuses(itself, value).await;
}

#[tokio::test]
async fn an_operand_that_is_no_child_is_decision_not_allowed() {
    let itself = |manifest: &SemanticManifest| {
        let mut decision = drilled(manifest);
        decision
            .operands
            .push(operand(manifest, "txn_by_grain", &["quantity"]));
        decision
    };
    let value = DecisionValue::NotAnOperand("txn_by_grain".into());

    a_value_the_grainset_row_refuses(itself, value).await;
}

#[tokio::test]
async fn a_child_not_read_that_serves_a_name_no_operand_serves_is_decision_not_allowed() {
    let without_target = |manifest: &SemanticManifest| {
        let children = [("transactions", "quantity"), ("txn_weekly_agg", "quantity")]
            .map(|(child, name)| operand(manifest, &format!("txn_by_grain.{child}"), &[name]));
        Decision {
            operands: children.into(),
            ..drilled(manifest)
        }
    };
    let value = DecisionValue::NotRead("txn_by_grain.txn_monthly_agg".into());

    a_value_the_grainset_row_refuses(without_target, value).await;
}

#[tokio::test]
async fn a_grainset_over_fewer_than_two_children_is_decision_not_allowed() {
    for combine in [Combine::DrillAcross, Combine::PassThrough] {
        let one = |manifest: &SemanticManifest| {
            let transactions = operand(manifest, "txn_by_grain.transactions", &["quantity"]);
            Decision {
                operands: vec![transactions],
                combine,
                ..drilled(manifest)
            }
        };

        a_value_the_grainset_row_refuses(one, DecisionValue::Combine(combine)).await;
    }
}

async fn a_plan_that_validate_refuses() -> Golden {
    let manifest = with_datasets(&format!("{ORDERS}{TWINS}"), &orders_catalog()).await;
    let request = select(&["region"]);
    let key = ParameterKey::RequestDimensionsMinusTemporal;
    let unbound = Expr::Leaf(Parameter {
        key,
        data_type: DataType::String,
    });
    let region = SchemaColumn {
        name: "region".into(),
        data_type: DataType::String,
        nullable: true,
    };
    let read = PhysicalRead {
        reader: Some(VirtualTable::new(vec![vec![unbound]]).into()),
        schema: Schema {
            columns: vec![region],
        },
    };
    let decision = Decision {
        physical_read: Some(read),
        ..leaf()
    };
    let planner = Planner::new().with_dataset_strategy(Box::new(Replaced(decision)));

    let (error, advisories) = refusal_by(&planner, &manifest, &request);

    let refused = IrErrorKind::UnboundParameter { key };
    assert!(error.kind.message().contains(&refused.message()));
    let cause = error.kind.cause().map(ToString::to_string);
    assert_eq!(cause, Some(refused.to_string()));
    assert_eq!(error.kind, PlanError::PlanRefused { error: refused });
    let (plan, raised) = planned(&manifest, &request);
    let anchor = scan_of(&plan).common.anchor.0;
    assert_eq!(error.notes, [format!("node {anchor} (ScanNode)")]);
    assert!(!raised.is_empty());
    assert_eq!(advisory_kinds(advisories), advisory_kinds(raised));
    Golden::message("plan_refused", &error)
}

#[tokio::test]
async fn a_decision_whose_plan_validate_refuses_is_plan_refused() {
    a_plan_that_validate_refuses().await.assert_committed();
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        vec![
            a_stand_in_for_a_dataset().await,
            a_join_for_a_dataset().await,
            an_operand_for_a_dataset().await,
            a_stack_for_a_joinset().await,
            a_drill_across_for_a_joinset().await,
            an_operand_that_is_no_end_of_a_joinset().await,
            a_stand_in_for_a_unionset().await,
            a_join_for_a_unionset().await,
            a_branch_of_a_unionset_not_read().await,
            a_stack_for_a_grainset().await,
            a_join_for_a_grainset().await,
            a_plan_that_validate_refuses().await,
        ]
    })
    .await;
}
