mod common;

use common::golden::{self, Golden};
use common::{
    agg, by_name, column, compile_model_on, constituent, dated, dated_on, dimension, fixture_fs,
    has_signature, identity_mapping, lake_folder, listed, measure, refusal_golden_on,
    semantic_alias, signature, stored_tree,
};

use semstrait_common::Granularity;
use semstrait_ir::PhysicalExpr;
use semstrait_ir::{SemanticRole, TemporalType};
use semstrait_manifest::{
    mint_keyset_id, mint_semantic_id, mint_semantic_source_id as id, CompileError, RoleDefinition,
    SemanticManifest, SemanticSource,
};
use semstrait_model::expr::{dim, lit, measure as measure_leaf, metric as metric_leaf};
use semstrait_model::{
    CrossFilter, DimensionSite, Grainset, GrainsetBuilder, Integrity, Joinset, Measure,
    MeasureSite, Metric, MetricSite, NestedDataset, NestedGrainset, NestedJoinset, NestedUnionset,
    Optional, Relationship, SemanticModel, SemanticModelBuilder, StorageFormat, TemporalShape,
    UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{BinaryOpKind, Expr, ExprOperationExt, NamedReference};

fn many_to_one(from: &str, to: &str, key: &str) -> Relationship {
    Relationship::builder()
        .name(format!("{from}_to_{to}"))
        .from(from)
        .to(to)
        .on([key])
        .many_to_one()
        .build()
        .expect("the fixture relationship must build")
}

fn events_on(axis: &str) -> TemporalType {
    TemporalType::events(mint_semantic_id(SemanticRole::Dimension, axis))
}

fn events(granularity: Granularity) -> TemporalShape {
    TemporalShape::events("date", Some(granularity))
}

fn pools(dimensions: &[&str], measures: &[&str]) -> SemanticModelBuilder {
    let mut built = SemanticModel::builder().name("sales");
    for name in dimensions {
        let data_type = if name.ends_with("date") {
            DataType::Date
        } else {
            DataType::String
        };
        built = built.dimension(dimension(name, data_type));
    }
    for name in measures {
        built = built.measure(measure(name, DataType::Double, "sum"));
    }
    built
}

async fn compiled(model: SemanticModelBuilder, datasets: &[&str]) -> SemanticManifest {
    let model = model.build().expect("the fixture model must build").0;
    compile_model_on(model, &fixture_fs(datasets).await).await
}

fn condition(manifest: &SemanticManifest, name: &str) -> PhysicalExpr {
    let RoleDefinition::Joinset {
        on_expr_id: Some(on),
        ..
    } = &by_name(manifest, name).role_definition
    else {
        panic!("`{name}` is a Joinset with a condition");
    };
    stored_tree(manifest, *on)
}

fn equates(manifest: &SemanticManifest, left: &str, right: &str, key: &str) -> PhysicalExpr {
    Expr::BinaryOp {
        op: BinaryOpKind::Eq,
        left: Box::new(column(semantic_alias(manifest, left), key)),
        right: Box::new(column(semantic_alias(manifest, right), key)),
    }
}

fn nested_under(manifest: &SemanticManifest, name: &str, parent: &str, letter: &str) {
    let entry = by_name(manifest, name);
    assert!(!entry.public, "`{name}` is private");
    assert_eq!(entry.originators, [by_name(manifest, parent).id.clone()]);
    assert!(
        entry.alias.as_str().starts_with(&format!("ss{letter}")),
        "`{name}`'s alias is `ss{letter}<n>`: {}",
        entry.alias.as_str()
    );
}

#[tokio::test]
async fn a_joinset_over_a_unionset_whose_branch_is_a_joinset_compiles() {
    let web = NestedJoinset::builder("web")
        .dataset(constituent(
            "web_orders",
            &["order_id", "customer_id", "session_id", "revenue"],
        ))
        .dataset(constituent("sessions", &["session_id", "device_id"]))
        .dataset(constituent("devices", &["device_id"]))
        .relationship(many_to_one("web_orders", "sessions", "session_id"))
        .relationship(many_to_one("sessions", "devices", "device_id"))
        .build();
    let orders = NestedUnionset::builder("orders")
        .union_all()
        .joinset(web)
        .dataset(constituent(
            "store_orders",
            &["order_id", "customer_id", "revenue"],
        ))
        .build();
    let sales = Joinset::builder("sales")
        .dataset(constituent("customers", &["customer_id", "region"]))
        .unionset(orders)
        .dimensions(
            [
                "customer_id",
                "region",
                "order_id",
                "session_id",
                "device_id",
            ]
            .map(DimensionSite::r#ref),
        )
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(
            Relationship::builder()
                .name("orders_to_customers")
                .from("orders")
                .to("customers")
                .on(["customer_id"])
                .many_to_one()
                .integrity(Integrity::Enforced)
                .cross_filter(CrossFilter::Left)
                .build()
                .expect("the fixture relationship must build"),
        )
        .build();
    let manifest = compiled(
        pools(
            &[
                "customer_id",
                "region",
                "order_id",
                "session_id",
                "device_id",
            ],
            &["revenue"],
        )
        .joinset(sales),
        &[
            "customers",
            "web_orders",
            "sessions",
            "devices",
            "store_orders",
        ],
    )
    .await;

    nested_under(&manifest, "sales.orders", "sales", "u");
    nested_under(&manifest, "sales.orders.web", "sales.orders", "j");
    let web = &by_name(&manifest, "sales.orders.web").id;
    let mut units: Vec<&str> = (manifest.semantic_sources.values())
        .filter(|entry: &&SemanticSource| entry.originators.contains(web))
        .filter(|entry| matches!(entry.role_definition, RoleDefinition::Joinset { .. }))
        .map(|entry| entry.name.as_str())
        .collect();
    units.sort_unstable();
    assert_eq!(
        units,
        [
            "sales.orders.web.sessions_to_devices",
            "sales.orders.web.web_orders_to_sessions",
        ]
    );
    assert_eq!(
        condition(&manifest, "sales"),
        equates(&manifest, "sales.orders", "sales.customers", "customer_id")
    );

    let stored = serde_json::to_value(&by_name(&manifest, "sales").role_definition)
        .expect("serialize Joinset");
    assert_eq!(
        stored["content"]["integrity"], "enforced",
        "the manifest's own `Integrity` still serializes `enforced`: {stored}"
    );
    assert_eq!(
        stored["content"]["cross_filter"], "left",
        "the manifest's own `CrossFilter` still serializes `left`: {stored}"
    );
}

#[tokio::test]
async fn four_nested_compositions_compile() {
    let names = ["customer_id", "date", "revenue"];
    let l4 = NestedGrainset::builder("l4")
        .temporal(events(Granularity::Month))
        .dataset(dated("l4a", &names, Granularity::Day))
        .dataset(dated("l4b", &names, Granularity::Month))
        .build();
    let l3 = NestedUnionset::builder("l3")
        .union_all()
        .temporal(events(Granularity::Month))
        .dataset(dated("l3a", &names, Granularity::Month))
        .grainset(l4)
        .build();
    let l2 = NestedJoinset::builder("l2")
        .temporal(events(Granularity::Month))
        .dataset(constituent("l2a", &["customer_id", "region"]))
        .unionset(l3)
        .relationship(many_to_one("l3", "l2a", "customer_id"))
        .build();
    let l1 = Grainset::builder("l1")
        .dataset(dated(
            "l1a",
            &["customer_id", "region", "date", "revenue"],
            Granularity::Day,
        ))
        .joinset(l2)
        .dimensions(["customer_id", "region", "date"].map(DimensionSite::r#ref))
        .measure(MeasureSite::r#ref("revenue"))
        .build();
    let manifest = compiled(
        pools(&["customer_id", "region", "date"], &["revenue"]).grainset(l1),
        &["l1a", "l2a", "l3a", "l4a", "l4b"],
    )
    .await;

    nested_under(&manifest, "l1.l2", "l1", "j");
    nested_under(&manifest, "l1.l2.l3", "l1.l2", "u");
    nested_under(&manifest, "l1.l2.l3.l4", "l1.l2.l3", "g");
    assert_eq!(
        condition(&manifest, "l1.l2"),
        equates(&manifest, "l1.l2.l3", "l1.l2.l2a", "customer_id")
    );
}

#[tokio::test]
async fn a_cross_grain_condition_names_a_joinset_child_by_its_own_alias() {
    let daily = NestedJoinset::builder("daily")
        .temporal(events(Granularity::Day))
        .dataset(dated(
            "clicks_daily",
            &["campaign_id", "date", "clicks"],
            Granularity::Day,
        ))
        .dataset(dated(
            "cost_daily",
            &["campaign_id", "date", "cost"],
            Granularity::Day,
        ))
        .relationship(
            Relationship::builder()
                .name("clicks_to_cost")
                .from("clicks_daily")
                .to("cost_daily")
                .on(["campaign_id"])
                .many_to_many()
                .optional(Optional::Both)
                .cross_filter(CrossFilter::None)
                .build()
                .expect("the fixture relationship must build"),
        )
        .build();
    let spend = Grainset::builder("spend")
        .joinset(daily)
        .dataset(dated(
            "spend_monthly",
            &["campaign_id", "date", "clicks", "cost"],
            Granularity::Month,
        ))
        .dimension(DimensionSite::r#ref("campaign_id").with_key())
        .dimension(DimensionSite::r#ref("date"))
        .measures(["clicks", "cost"].map(MeasureSite::r#ref))
        .build();
    let manifest = compiled(
        pools(&["campaign_id", "date"], &["clicks", "cost"]).grainset(spend),
        &["clicks_daily", "cost_daily", "spend_monthly"],
    )
    .await;

    nested_under(&manifest, "spend.daily", "spend", "j");
    assert_eq!(
        condition(&manifest, "spend.daily_spend_monthly"),
        equates(
            &manifest,
            "spend.daily",
            "spend.spend_monthly",
            "campaign_id"
        )
    );
}

#[tokio::test]
async fn a_composed_grainset_childs_signature_also_reads_the_names_its_filter_reads() {
    let web_budget = Metric::builder("web_budget")
        .data_type(DataType::Double)
        .expr(measure_leaf("budget").expect("a name"))
        .filter(dim("channel").expect("a name").eq(lit("web")))
        .build();
    let web_budget_copy = Metric::builder("web_budget_copy")
        .data_type(DataType::Double)
        .expr(metric_leaf("web_budget").expect("a name"))
        .build();
    let j = NestedJoinset::builder("j")
        .dataset(dated(
            "budget_daily",
            &["campaign_id", "date", "budget"],
            Granularity::Day,
        ))
        .dataset(constituent("campaigns", &["campaign_id"]))
        .relationship(many_to_one("budget_daily", "campaigns", "campaign_id"))
        .build();
    let dimensions = ["campaign_id", "channel", "date"];
    let spend = Grainset::builder("spend")
        .joinset(j)
        .dataset(dated(
            "budget_monthly",
            &["campaign_id", "channel", "date", "budget"],
            Granularity::Month,
        ))
        .dimensions(dimensions.map(DimensionSite::r#ref))
        .measure(MeasureSite::r#ref("budget"))
        .metrics(["web_budget", "web_budget_copy"].map(MetricSite::r#ref))
        .build();
    let model = pools(&dimensions, &["budget"])
        .metric(web_budget)
        .metric(web_budget_copy)
        .grainset(spend);
    let manifest = compiled(model, &["budget_daily", "campaigns", "budget_monthly"]).await;
    for metric in ["web_budget", "web_budget_copy"] {
        assert!(has_signature(&manifest, "spend.budget_monthly", metric));
        assert!(
            !has_signature(&manifest, "spend.j", metric),
            "`j` does not map `channel`, so `{metric}` has no signature there"
        );
    }
}

const SPEND: [&str; 3] = ["campaign_id", "date", "spend"];

fn spend_family(children: impl FnOnce(GrainsetBuilder) -> GrainsetBuilder) -> SemanticModelBuilder {
    spend_over(&["campaign_id", "date"], children)
}

fn spend_over(
    dimensions: &[&str],
    children: impl FnOnce(GrainsetBuilder) -> GrainsetBuilder,
) -> SemanticModelBuilder {
    let spend = children(Grainset::builder("spend"))
        .dimensions(dimensions.iter().map(|name| DimensionSite::r#ref(*name)))
        .measure(MeasureSite::r#ref("spend"))
        .build();
    pools(dimensions, &["spend"]).grainset(spend)
}

fn spend_at(name: &str, granularity: Granularity) -> NestedDataset {
    dated(name, &SPEND, granularity)
}

async fn refusal(
    case: &str,
    model: SemanticModelBuilder,
    datasets: &[&str],
    expected: CompileError,
) -> Golden {
    let model = model.build().expect("the fixture model must build").0;
    refusal_golden_on(case, model, &fixture_fs(datasets).await, expected).await
}

fn stored_grain(manifest: &SemanticManifest, name: &str) -> Option<String> {
    let json = serde_json::to_value(manifest).expect("the manifest serializes");
    json["semantic_sources"][by_name(manifest, name).id.as_str()]["temporal"]["granularity"]
        .as_str()
        .map(str::to_owned)
}

fn joinset_child_shape<'m>(
    manifest: &'m SemanticManifest,
    expectation: &str,
) -> &'m semstrait_ir::TemporalShape {
    by_name(manifest, "spend.j")
        .temporal
        .as_ref()
        .expect(expectation)
}

fn faults(manifest: &SemanticManifest) -> Vec<CompileError> {
    (manifest
        .validate()
        .expect_err("the changed artifact is refused"))
    .into_iter()
    .map(|fault| fault.kind)
    .collect()
}

fn shape_of<'m>(
    manifest: &'m mut SemanticManifest,
    name: &str,
) -> &'m mut semstrait_ir::TemporalShape {
    let id = by_name(manifest, name).id.clone();
    (manifest.semantic_sources.get_mut(&id))
        .and_then(|entry| entry.temporal.as_mut())
        .expect("the entry stores a shape")
}

macro_rules! declaring {
    ($built:expr, $granularity:expr) => {
        match $granularity {
            Some(granularity) => $built.temporal(events(granularity)),
            None => $built,
        }
        .build()
    };
}

fn stack(
    name: &str,
    branches: [(&str, Granularity); 2],
    granularity: Option<Granularity>,
) -> NestedUnionset {
    let [(first, first_granularity), (second, second_granularity)] = branches;
    let built = NestedUnionset::builder(name)
        .union_all()
        .dataset(spend_at(first, first_granularity))
        .dataset(spend_at(second, second_granularity));
    declaring!(built, granularity)
}

const U_WM: [(&str, Granularity); 2] = [("wk", Granularity::Week), ("mo", Granularity::Month)];

const U_DM: [(&str, Granularity); 2] = [
    ("web_daily", Granularity::Day),
    ("store_monthly", Granularity::Month),
];

#[tokio::test]
async fn a_grainset_child_with_no_time_axis_compiles() {
    let stack = NestedUnionset::builder("stack")
        .union_all()
        .dataset(constituent("east", &SPEND))
        .dataset(constituent("west", &SPEND))
        .build();
    let manifest = compiled(
        spend_family(|spend| {
            spend
                .unionset(stack)
                .dataset(spend_at("spend_monthly", Granularity::Month))
        }),
        &["east", "west", "spend_monthly"],
    )
    .await;
    assert_eq!(stored_grain(&manifest, "spend.stack"), None);
}

#[tokio::test]
async fn a_unionset_over_week_and_month_serves_no_declared_granularity() {
    for golden in week_and_month_refusals().await {
        golden.assert_committed();
    }
}

async fn week_and_month_refusals() -> Vec<Golden> {
    let folders = ["wk", "mo", "spend_year"];
    let family = |granularity| {
        spend_family(|spend| {
            spend
                .unionset(stack("u_wm", U_WM, granularity))
                .dataset(spend_at("spend_year", Granularity::Year))
        })
    };
    let not_served =
        |stated, constituent: &str, granularity| CompileError::StatedGranularityNotServed {
            composite: id("spend.u_wm"),
            stated,
            constituents: vec![(id(&format!("spend.u_wm.{constituent}")), granularity)],
        };
    vec![
        refusal(
            "a_unionset_over_week_and_month_is_refused_undeclared",
            family(None),
            &folders,
            CompileError::ComposedConstituentMissingGranularity {
                composite: id("spend"),
                constituent: id("spend.u_wm"),
            },
        )
        .await,
        refusal(
            "a_unionset_over_week_and_month_declaring_week_is_refused",
            family(Some(Granularity::Week)),
            &folders,
            not_served(Granularity::Week, "mo", Granularity::Month),
        )
        .await,
        refusal(
            "a_unionset_over_week_and_month_declaring_month_is_refused",
            family(Some(Granularity::Month)),
            &folders,
            not_served(Granularity::Month, "wk", Granularity::Week),
        )
        .await,
    ]
}

fn u_dm_family(granularity: Option<Granularity>) -> SemanticModelBuilder {
    spend_family(|spend| {
        spend
            .unionset(stack("u_dm", U_DM, granularity))
            .dataset(spend_at("spend_year", Granularity::Year))
    })
}

const U_DM_FOLDERS: [&str; 3] = ["web_daily", "store_monthly", "spend_year"];

#[tokio::test]
async fn a_unionset_over_day_and_month_is_refused_undeclared() {
    day_and_month_refusal().await.assert_committed();
}

async fn day_and_month_refusal() -> Golden {
    refusal(
        "a_unionset_over_day_and_month_is_refused_undeclared",
        u_dm_family(None),
        &U_DM_FOLDERS,
        CompileError::ComposedConstituentMissingGranularity {
            composite: id("spend"),
            constituent: id("spend.u_dm"),
        },
    )
    .await
}

#[tokio::test]
async fn a_unionset_over_day_and_month_declaring_month_compiles() {
    let manifest = compiled(u_dm_family(Some(Granularity::Month)), &U_DM_FOLDERS).await;
    assert_eq!(
        stored_grain(&manifest, "spend.u_dm").as_deref(),
        Some("month")
    );
}

fn j_family(joined: Option<Granularity>, targets: Option<Granularity>) -> SemanticModelBuilder {
    let target = ["campaign_id", "date"];
    let targets_gs = NestedGrainset::builder("targets_gs")
        .dataset(dated("targets_monthly", &target, Granularity::Month))
        .dataset(dated("targets_yearly", &target, Granularity::Year));
    let j = NestedJoinset::builder("j")
        .dataset(spend_at("spend_daily", Granularity::Day))
        .grainset(declaring!(targets_gs, targets))
        .relationship(many_to_one("spend_daily", "targets_gs", "campaign_id"));
    let j = declaring!(j, joined);
    spend_family(|spend| {
        spend
            .joinset(j)
            .dataset(spend_at("spend_monthly", Granularity::Month))
    })
}

const J_FOLDERS: [&str; 4] = [
    "spend_daily",
    "targets_monthly",
    "targets_yearly",
    "spend_monthly",
];

#[tokio::test]
async fn a_joinset_over_a_grainset_is_refused_absent_a_declaration_on_each() {
    for golden in joinset_over_grainset_refusals().await {
        golden.assert_committed();
    }
}

async fn joinset_over_grainset_refusals() -> Vec<Golden> {
    let targets = CompileError::ComposedConstituentMissingGranularity {
        composite: id("spend.j"),
        constituent: id("spend.j.targets_gs"),
    };
    vec![
        refusal(
            "a_joinset_over_a_grainset_is_refused_undeclared",
            j_family(None, None),
            &J_FOLDERS,
            targets.clone(),
        )
        .await,
        refusal(
            "a_joinset_over_a_grainset_is_refused_declaring_the_joinset_only",
            j_family(Some(Granularity::Year), None),
            &J_FOLDERS,
            targets,
        )
        .await,
        refusal(
            "a_joinset_over_a_grainset_is_refused_declaring_the_grainset_only",
            j_family(None, Some(Granularity::Year)),
            &J_FOLDERS,
            CompileError::ComposedConstituentMissingGranularity {
                composite: id("spend"),
                constituent: id("spend.j"),
            },
        )
        .await,
    ]
}

#[tokio::test]
async fn a_joinset_over_a_grainset_declaring_year_on_each_compiles() {
    let manifest = compiled(
        j_family(Some(Granularity::Year), Some(Granularity::Year)),
        &J_FOLDERS,
    )
    .await;
    for name in ["spend.j", "spend.j.targets_gs"] {
        assert_eq!(
            stored_grain(&manifest, name).as_deref(),
            Some("year"),
            "{name}"
        );
    }
}

fn u_g_family(granularity: Option<Granularity>) -> SemanticModelBuilder {
    let g_inner = NestedGrainset::builder("g_inner")
        .dataset(spend_at("gi_day", Granularity::Day))
        .dataset(spend_at("gi_month", Granularity::Month));
    let u_g = NestedUnionset::builder("u_g")
        .union_all()
        .grainset(declaring!(g_inner, granularity))
        .dataset(spend_at("d_day", Granularity::Day));
    let u_g = declaring!(u_g, granularity);
    spend_family(|spend| {
        spend
            .unionset(u_g)
            .dataset(spend_at("spend_year", Granularity::Year))
    })
}

const U_G_FOLDERS: [&str; 4] = ["gi_day", "gi_month", "d_day", "spend_year"];

#[tokio::test]
async fn a_unionset_over_a_grainset_branch_is_refused_undeclared() {
    grainset_branch_refusal().await.assert_committed();
}

async fn grainset_branch_refusal() -> Golden {
    refusal(
        "a_unionset_over_a_grainset_branch_is_refused_undeclared",
        u_g_family(None),
        &U_G_FOLDERS,
        CompileError::ComposedConstituentMissingGranularity {
            composite: id("spend.u_g"),
            constituent: id("spend.u_g.g_inner"),
        },
    )
    .await
}

#[tokio::test]
async fn a_unionset_over_a_grainset_branch_declaring_month_compiles() {
    let manifest = compiled(u_g_family(Some(Granularity::Month)), &U_G_FOLDERS).await;
    for name in ["spend.u_g", "spend.u_g.g_inner"] {
        assert_eq!(
            stored_grain(&manifest, name).as_deref(),
            Some("month"),
            "{name}"
        );
    }
}

#[tokio::test]
async fn a_joinset_stating_a_shape_that_is_neither_tied_shape_is_refused() {
    neither_tied_shape_refusal().await.assert_committed();
}

async fn neither_tied_shape_refusal() -> Golden {
    refusal(
        "a_joinset_stating_a_shape_that_is_neither_tied_shape_is_refused",
        shapeless_family(Some(Granularity::Month)),
        &SHAPELESS_FOLDERS,
        CompileError::ComposedTemporalDisagreement {
            composite: id("spend.j"),
            shapes: vec![
                (id("spend.j.orders_daily"), events_on("order_date")),
                (id("spend.j.shipments_daily"), events_on("ship_date")),
                (id("spend.j"), events_on("date")),
            ],
        },
    )
    .await
}

#[tokio::test]
async fn a_shapeless_joinset_child_is_refused() {
    shapeless_child_refusal().await.assert_committed();
}

async fn shapeless_child_refusal() -> Golden {
    refusal(
        "a_shapeless_joinset_child_is_refused",
        shapeless_family(None),
        &SHAPELESS_FOLDERS,
        CompileError::ComposedConstituentMissingTemporalShape {
            composite: id("spend"),
            constituent: id("spend.j"),
        },
    )
    .await
}

#[tokio::test]
async fn a_shapeless_joinset_child_carries_the_shape_it_states() {
    let j = NestedJoinset::builder("j")
        .dataset(constituent(
            "orders_daily",
            &["campaign_id", "date", "spend"],
        ))
        .dataset(constituent("shipments_daily", &["campaign_id"]))
        .relationship(many_to_one(
            "orders_daily",
            "shipments_daily",
            "campaign_id",
        ))
        .temporal(TemporalShape::timeseries("date", Some(Granularity::Day)))
        .build();
    let sibling = NestedDataset::builder("spend_year")
        .path(lake_folder("spend_year"))
        .format(StorageFormat::Parquet)
        .semantic_mapping(identity_mapping(&SPEND))
        .temporal(TemporalShape::timeseries("date", Some(Granularity::Year)))
        .build();
    let manifest = compiled(
        spend_family(|spend| spend.joinset(j).dataset(sibling)),
        &SHAPELESS_FOLDERS,
    )
    .await;

    let stored = joinset_child_shape(&manifest, "the Joinset child stores the shape it states");
    assert_eq!(
        stored.temporal_type,
        TemporalType::timeseries(mint_semantic_id(SemanticRole::Dimension, "date"))
    );
    assert_eq!(stored.granularity, Some(Granularity::Day));
}

#[tokio::test]
async fn a_tied_joinset_child_with_no_granularity_carries_the_tied_shape_it_states() {
    let manifest = compiled(
        tie_family(
            Some(TemporalShape::events("order_date", None)),
            "order_date",
        ),
        &SHAPELESS_FOLDERS,
    )
    .await;

    let stored = joinset_child_shape(
        &manifest,
        "the Joinset child stores the tied shape it states",
    );
    assert_eq!(stored.temporal_type, events_on("order_date"));
    assert_eq!(stored.granularity, Some(Granularity::Day));
}

const SHAPELESS_FOLDERS: [&str; 3] = ["orders_daily", "shipments_daily", "spend_year"];

fn shapeless_family(granularity: Option<Granularity>) -> SemanticModelBuilder {
    tie_family(granularity.map(events), "date")
}

fn tie_family(stated: Option<TemporalShape>, sibling_axis: &str) -> SemanticModelBuilder {
    let j = NestedJoinset::builder("j")
        .dataset(dated_on(
            "orders_daily",
            &["campaign_id", "order_date", "spend"],
            "order_date",
            Granularity::Day,
        ))
        .dataset(dated_on(
            "shipments_daily",
            &["campaign_id", "ship_date"],
            "ship_date",
            Granularity::Day,
        ))
        .relationship(many_to_one(
            "orders_daily",
            "shipments_daily",
            "campaign_id",
        ));
    let j = match stated {
        Some(shape) => j.temporal(shape),
        None => j,
    }
    .build();
    let mut dimensions = vec!["campaign_id"];
    if !["order_date", "ship_date"].contains(&sibling_axis) {
        dimensions.push(sibling_axis);
    }
    dimensions.extend(["order_date", "ship_date"]);
    spend_over(&dimensions, |spend| {
        spend.joinset(j).dataset(dated_on(
            "spend_year",
            &["campaign_id", sibling_axis, "spend"],
            sibling_axis,
            Granularity::Year,
        ))
    })
}

#[tokio::test]
async fn validate_restates_a_nested_compositions_temporal_refusals() {
    let u_dm = compiled(u_dm_family(Some(Granularity::Month)), &U_DM_FOLDERS).await;
    let mut blanked = u_dm.clone();
    shape_of(&mut blanked, "spend.u_dm").granularity = None;
    assert_eq!(
        faults(&blanked),
        [CompileError::ComposedConstituentMissingGranularity {
            composite: id("spend"),
            constituent: id("spend.u_dm"),
        }]
    );

    let mut finer = u_dm;
    shape_of(&mut finer, "spend.u_dm").granularity = Some(Granularity::Day);
    assert_eq!(
        faults(&finer),
        [CompileError::StatedGranularityNotServed {
            composite: id("spend.u_dm"),
            stated: Granularity::Day,
            constituents: vec![(id("spend.u_dm.store_monthly"), Granularity::Month)],
        }]
    );

    let mut moved = compiled(
        j_family(Some(Granularity::Year), Some(Granularity::Year)),
        &J_FOLDERS,
    )
    .await;
    let date = mint_semantic_id(SemanticRole::Dimension, "date");
    shape_of(&mut moved, "spend.j").temporal_type = TemporalType::snapshot(date.clone());
    let events = TemporalType::events(date.clone());
    let snapshot = TemporalType::snapshot(date);
    assert_eq!(
        faults(&moved),
        [
            CompileError::ComposedTemporalDisagreement {
                composite: id("spend.j"),
                shapes: vec![
                    (id("spend.j.targets_gs.targets_yearly"), events.clone()),
                    (id("spend.j.spend_daily"), events.clone()),
                    (id("spend.j.targets_gs.targets_monthly"), events.clone()),
                    (id("spend.j.targets_gs"), events.clone()),
                    (id("spend.j"), snapshot.clone()),
                ],
            },
            CompileError::ComposedTemporalDisagreement {
                composite: id("spend"),
                shapes: vec![
                    (id("spend.spend_monthly"), events.clone()),
                    (id("spend.j"), snapshot),
                    (id("spend"), events),
                ],
            },
        ]
    );
}

const COSTS: [&str; 3] = ["campaign_id", "date", "cost"];

const CAMPAIGN_SPEND_FOLDERS: [&str; 3] = ["campaigns", "spend_daily", "spend_monthly"];

fn campaign_spend(granularity: Option<Granularity>) -> SemanticModelBuilder {
    campaign_spend_over(granularity, CAMPAIGN_SPEND.map(DimensionSite::r#ref))
}

const CAMPAIGN_SPEND: [&str; 3] = ["campaign_id", "campaign_name", "date"];

fn campaign_spend_over(
    granularity: Option<Granularity>,
    sites: [DimensionSite; 3],
) -> SemanticModelBuilder {
    let spend_gs = NestedGrainset::builder("spend_gs")
        .dataset(dated("spend_daily", &COSTS, Granularity::Day))
        .dataset(dated("spend_monthly", &COSTS, Granularity::Month));
    let dimensions = CAMPAIGN_SPEND;
    let joined = Joinset::builder("campaign_spend")
        .dataset(constituent("campaigns", &["campaign_id", "campaign_name"]))
        .grainset(declaring!(spend_gs, granularity))
        .dimensions(sites)
        .measure(MeasureSite::r#ref("cost"))
        .relationship(
            Relationship::builder()
                .name("campaigns_to_spend_gs")
                .from("campaigns")
                .to("spend_gs")
                .on(["campaign_id"])
                .one_to_many()
                .build()
                .expect("the fixture relationship must build"),
        )
        .build();
    pools(&dimensions, &["cost"]).joinset(joined)
}

#[tokio::test]
async fn a_joinset_over_a_grainset_declaring_month_compiles() {
    let manifest = compiled(
        campaign_spend(Some(Granularity::Month)),
        &CAMPAIGN_SPEND_FOLDERS,
    )
    .await;
    let spend_gs = semantic_alias(&manifest, "campaign_spend.spend_gs");
    assert_eq!(
        condition(&manifest, "campaign_spend"),
        equates(
            &manifest,
            "campaign_spend.campaigns",
            "campaign_spend.spend_gs",
            "campaign_id"
        )
    );
    let cost = signature(&manifest, "campaign_spend", "cost").expression_id;
    let read: Vec<&NamedReference> = (manifest.expressions.columns(&cost).into_iter())
        .map(|(_, column)| column)
        .collect();
    assert_eq!(
        read,
        [&NamedReference::qualified(spend_gs, "cost").expect("an alias and a name")]
    );
    assert_eq!(
        stored_grain(&manifest, "campaign_spend.spend_gs").as_deref(),
        Some("month")
    );
}

#[tokio::test]
async fn a_joinset_over_a_grainset_with_no_granularity_is_refused() {
    undeclared_grainset_refusal().await.assert_committed();
}

async fn undeclared_grainset_refusal() -> Golden {
    refusal(
        "a_joinset_over_a_grainset_with_no_granularity_is_refused",
        campaign_spend(None),
        &CAMPAIGN_SPEND_FOLDERS,
        CompileError::ComposedConstituentMissingGranularity {
            composite: id("campaign_spend"),
            constituent: id("campaign_spend.spend_gs"),
        },
    )
    .await
}

fn keyed(name: &str, keysets: &[&str]) -> DimensionSite {
    (DimensionSite::r#ref(name).with_keysets(keysets.iter().copied())).expect("named keysets")
}

fn keysets_of(manifest: &SemanticManifest, source_name: &str, semantic: &str) -> Vec<String> {
    (signature(manifest, source_name, semantic).keysets.iter())
        .map(|keyset| manifest.keysets[keyset].name.as_str().to_owned())
        .collect()
}

#[tokio::test]
async fn a_nested_grainset_holds_only_the_root_keysets_it_serves() {
    let sites = [
        keyed("campaign_id", &["campaign", "spend"]),
        keyed("campaign_name", &["campaign"]),
        keyed("date", &["spend"]),
    ];
    let model = campaign_spend_over(Some(Granularity::Month), sites);
    let manifest = compiled(model, &CAMPAIGN_SPEND_FOLDERS).await;
    let spend_gs = by_name(&manifest, "campaign_spend.spend_gs");
    let RoleDefinition::Grainset { keysets, .. } = &spend_gs.role_definition else {
        panic!("`spend_gs` is a Grainset");
    };
    assert_eq!(keysets, &[mint_keyset_id(&spend_gs.id, "spend")]);
    let daily = "campaign_spend.spend_gs.spend_daily";
    assert_eq!(keysets_of(&manifest, daily, "campaign_id"), ["spend"]);
    assert_eq!(
        keysets_of(&manifest, "campaign_spend.campaigns", "campaign_id"),
        ["campaign"]
    );
}

#[tokio::test]
async fn a_nested_unionset_guarantees_only_the_root_keysets_it_serves() {
    let branch = |name| constituent(name, &["order_id", "customer_id", "revenue"]);
    let orders = NestedUnionset::builder("orders")
        .union_all()
        .dataset(branch("web_orders"))
        .dataset(branch("store_orders"))
        .build();
    let sales = Joinset::builder("sales")
        .dataset(constituent("customers", &["customer_id", "region"]))
        .unionset(orders)
        .dimension(keyed("customer_id", &["customer"]))
        .dimension(keyed("region", &["customer"]))
        .dimension(keyed("order_id", &["order"]))
        .measure(MeasureSite::r#ref("revenue"))
        .relationship(many_to_one("orders", "customers", "customer_id"))
        .build();
    let model = pools(&["customer_id", "region", "order_id"], &["revenue"]).joinset(sales);
    let manifest = compiled(model, &["customers", "web_orders", "store_orders"]).await;
    assert_eq!(keysets_of(&manifest, "sales.orders", "order_id"), ["order"]);
    assert!(keysets_of(&manifest, "sales.orders", "customer_id").is_empty());
    assert_eq!(
        keysets_of(&manifest, "sales.customers", "customer_id"),
        ["customer"]
    );
}

fn half_grains(name: &str) -> NestedGrainset {
    NestedGrainset::builder(name)
        .temporal(events(Granularity::Month))
        .dataset(dated(
            &format!("{name}_clicks"),
            &["campaign_id", "date", "click_id"],
            Granularity::Day,
        ))
        .dataset(dated(
            &format!("{name}_channels"),
            &["channel", "date"],
            Granularity::Month,
        ))
        .build()
}

#[derive(Clone, Copy)]
enum Reach {
    Operand,
    Branch,
}

impl Reach {
    const BOTH: [Self; 2] = [Self::Operand, Self::Branch];

    const fn operand(self) -> &'static str {
        match self {
            Self::Operand => "spend_eu",
            Self::Branch => "spend_u",
        }
    }
}

const CAMPAIGN_CLICKS_FOLDERS: [&str; 5] = [
    "campaigns",
    "spend_eu_clicks",
    "spend_eu_channels",
    "spend_us_clicks",
    "spend_us_channels",
];

fn campaign_clicks(reach: Reach, basis: &[&str], measure: Measure) -> SemanticModelBuilder {
    clicks_exposing(reach, basis, vec![measure], Vec::new())
}

fn clicks_exposing(
    reach: Reach,
    basis: &[&str],
    measures: Vec<Measure>,
    metrics: Vec<Metric>,
) -> SemanticModelBuilder {
    let joined = Joinset::builder("campaign_clicks");
    let joined = match reach {
        Reach::Operand => joined.grainset(half_grains("spend_eu")),
        Reach::Branch => joined.unionset(
            NestedUnionset::builder("spend_u")
                .union_all()
                .grainset(half_grains("spend_eu"))
                .grainset(half_grains("spend_us"))
                .build(),
        ),
    };
    let campaigns: Vec<&str> = (basis.iter().copied()).chain(["campaign_name"]).collect();
    let dimensions = [
        "campaign_id",
        "campaign_name",
        "channel",
        "click_id",
        "date",
    ];
    let mut joined = joined
        .dataset(constituent("campaigns", &campaigns))
        .dimensions(dimensions.map(DimensionSite::r#ref));
    let mut pooled = pools(&dimensions, &[]);
    for measure in measures {
        joined = joined.measure(MeasureSite::r#ref(measure.body.name.as_str()));
        pooled = pooled.measure(measure);
    }
    for metric in metrics {
        joined = joined.metric(MetricSite::r#ref(metric.body.name.as_str()));
        pooled = pooled.metric(metric);
    }
    let joined = joined
        .relationship(
            Relationship::builder()
                .name("spend_to_campaigns")
                .from(reach.operand())
                .to("campaigns")
                .on(basis.iter().copied())
                .many_to_one()
                .build()
                .expect("the fixture relationship must build"),
        )
        .build();
    pooled.joinset(joined)
}

fn click_where(dimension: &str, value: &str) -> UnresolvedSemanticExpr {
    Expr::Case {
        whens: vec![(
            dim(dimension).expect("a name").eq(lit(value)),
            dim("click_id").expect("a name"),
        )],
        else_: None,
    }
}

fn clicks_where(name: &str, dimension: &str, value: &str) -> Measure {
    Measure::builder(name)
        .data_type(DataType::BigInt)
        .agg(agg("count"))
        .expr(click_where(dimension, value))
        .build()
}

fn brand_clicks() -> Measure {
    clicks_where("brand_clicks", "campaign_name", "brand")
}

#[tokio::test]
async fn a_join_basis_no_one_child_maps_whole_is_refused_at_each_depth() {
    for golden in split_basis_refusals().await {
        golden.assert_committed();
    }
}

async fn at_each_reach(
    cases: [&str; 2],
    refused: impl Fn(Reach) -> (SemanticModelBuilder, CompileError),
) -> Vec<Golden> {
    let mut goldens = Vec::new();
    for (reach, case) in Reach::BOTH.into_iter().zip(cases) {
        let (model, expected) = refused(reach);
        goldens.push(refusal(case, model, &CAMPAIGN_CLICKS_FOLDERS, expected).await);
    }
    goldens
}

async fn split_basis_refusals() -> Vec<Golden> {
    let cases = [
        "a_join_basis_no_one_child_maps_whole_is_refused",
        "a_join_basis_no_one_child_of_a_branch_maps_whole_is_refused",
    ];
    at_each_reach(cases, |reach| {
        let model = campaign_clicks(reach, &["campaign_id", "channel"], brand_clicks());
        let expected = CompileError::JoinKeyNotMapped {
            relationship: "spend_to_campaigns".to_owned(),
            key: "channel".to_owned(),
            semantic_source: format!("campaign_clicks.{}", reach.operand()),
        };
        (model, expected)
    })
    .await
}

#[tokio::test]
async fn a_measure_no_one_child_maps_whole_is_refused_at_each_depth() {
    for golden in split_value_refusals().await {
        golden.assert_committed();
    }
}

async fn split_value_refusals() -> Vec<Golden> {
    let cases = [
        "a_measure_no_one_child_maps_whole_is_refused",
        "a_measure_no_one_child_of_a_branch_maps_whole_is_refused",
    ];
    at_each_reach(cases, |reach| {
        let web_clicks = clicks_where("web_clicks", "channel", "web");
        let model = campaign_clicks(reach, &["campaign_id"], web_clicks);
        let expected = CompileError::UnservedComposedSemantics {
            semantics: "web_clicks".to_owned(),
            semantic_source: "campaign_clicks".to_owned(),
        };
        (model, expected)
    })
    .await
}

#[tokio::test]
async fn a_row_aggregating_metric_no_one_child_maps_whole_is_refused_at_each_depth() {
    for golden in split_metric_refusals().await {
        golden.assert_committed();
    }
}

async fn split_metric_refusals() -> Vec<Golden> {
    let cases = [
        "a_row_aggregating_metric_no_one_child_maps_whole_is_refused",
        "a_row_aggregating_metric_no_one_child_of_a_branch_maps_whole_is_refused",
    ];
    at_each_reach(cases, |reach| {
        let web_clicks = Metric::builder("web_clicks")
            .data_type(DataType::BigInt)
            .agg(agg("count"))
            .expr(click_where("channel", "web"))
            .build();
        let model = clicks_exposing(reach, &["campaign_id"], Vec::new(), vec![web_clicks]);
        let expected = CompileError::UnservedComposedSemantics {
            semantics: "web_clicks".to_owned(),
            semantic_source: "campaign_clicks".to_owned(),
        };
        (model, expected)
    })
    .await
}

#[tokio::test]
async fn a_metric_over_a_post_aggregate_formula_compiles_at_each_depth() {
    for reach in Reach::BOTH {
        let clicks = Measure::builder("clicks")
            .data_type(DataType::BigInt)
            .agg(agg("count"))
            .expr(dim("click_id").expect("a name"))
            .build();
        let web_click_days = Metric::builder("web_click_days")
            .data_type(DataType::BigInt)
            .agg(agg("count"))
            .expr(Expr::Case {
                whens: vec![(
                    dim("channel").expect("a name").eq(lit("web")),
                    measure_leaf("clicks").expect("a name"),
                )],
                else_: None,
            })
            .build();
        let model = clicks_exposing(reach, &["campaign_id"], vec![clicks], vec![web_click_days]);
        let manifest = compiled(model, &CAMPAIGN_CLICKS_FOLDERS).await;
        assert!(has_signature(
            &manifest,
            "campaign_clicks",
            "web_click_days"
        ));
        let grainsets = (manifest.semantic_sources.values())
            .filter(|source| matches!(source.role_definition, RoleDefinition::Grainset { .. }));
        for grainset in grainsets {
            assert_eq!(
                listed(&manifest, grainset).count(),
                0,
                "`{}` is nested, exposes no Metric and so computes none; the composition above keeps `web_click_days`",
                grainset.name
            );
        }
    }
}

#[tokio::test]
async fn a_measure_one_child_maps_whole_compiles_at_each_depth() {
    for reach in Reach::BOTH {
        let model = campaign_clicks(reach, &["campaign_id"], brand_clicks());
        let manifest = compiled(model, &CAMPAIGN_CLICKS_FOLDERS).await;
        assert!(has_signature(&manifest, "campaign_clicks", "brand_clicks"));
        let operand = format!("campaign_clicks.{}", reach.operand());
        assert!(
            !has_signature(&manifest, &operand, "brand_clicks"),
            "`{operand}` does not map `campaign_name`"
        );
    }
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        let mut goldens = vec![
            day_and_month_refusal().await,
            grainset_branch_refusal().await,
            neither_tied_shape_refusal().await,
            shapeless_child_refusal().await,
            undeclared_grainset_refusal().await,
        ];
        goldens.extend(week_and_month_refusals().await);
        goldens.extend(joinset_over_grainset_refusals().await);
        goldens.extend(split_basis_refusals().await);
        goldens.extend(split_value_refusals().await);
        goldens.extend(split_metric_refusals().await);
        goldens
    })
    .await;
}

#[tokio::test]
async fn a_constituent_grainset_whose_children_share_a_granularity_derives_it() {
    let columns = ["campaign_id", "campaign_name", "date", "cost"];
    let spend_gs = NestedGrainset::builder("spend_gs")
        .dataset(dated("spend_daily", &COSTS, Granularity::Day))
        .dataset(dated("spend_daily_named", &columns, Granularity::Day));
    let dimensions = CAMPAIGN_SPEND;
    let joined = Joinset::builder("campaign_spend")
        .dataset(constituent("campaigns", &["campaign_id", "campaign_name"]))
        .grainset(declaring!(spend_gs, None::<Granularity>))
        .dimensions(dimensions.map(DimensionSite::r#ref))
        .measure(MeasureSite::r#ref("cost"))
        .relationship(
            Relationship::builder()
                .name("campaigns_to_spend_gs")
                .from("campaigns")
                .to("spend_gs")
                .on(["campaign_id"])
                .one_to_many()
                .build()
                .expect("the fixture relationship must build"),
        )
        .build();
    let manifest = compiled(
        pools(&dimensions, &["cost"]).joinset(joined),
        &["campaigns", "spend_daily", "spend_daily_named"],
    )
    .await;
    assert_eq!(
        stored_grain(&manifest, "campaign_spend.spend_gs").as_deref(),
        Some("day")
    );
}
