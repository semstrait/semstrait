mod common;

use std::error::Error;

use common::golden::{self, Golden};
use common::runner::{self, gate};
use common::{
    aggregate, aggregate_function, leaf, literal, name, plan, project, rules, scalar, schema,
};
use semstrait_adapter::SubstraitRules;
use semstrait_ir::{
    AggregateNode, Cardinality, CrossNode, Delivery, EmitForm, FilterNode, IrErrorKind, JoinNode,
    JoinType, MultisetNode, MultisetOp, Name, PhysicalExpr, PlanNode, ScanNode, SemAnnotation,
    SourceRole, VirtualTable,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{AggregateCall, ExprOperationExt, IntoLiteral, Literal};

const LISTED: &[&str] = &[
    "predicate/request_predicate",
    "predicate/relationship_filter",
    "join_type/inner",
    "join_type/left",
    "join_type/right",
    "join_type/full",
    "cross/non_equi_filter",
    "drill_across/one_row_children",
    "drill_across/by_campaign",
    "drill_across/null_campaign_group",
    "drill_across/left_deep_chain",
    "distinct_from/not_distinct_from",
    "distinct_from/equal_control",
    "union/typed_null_fill",
    "union/first_branch_folded",
    "unique_union/keyed_by_customer",
    "unique_union/no_key",
    "unique_union/keyed_by_sale",
    "re_aggregated_count/empty_union_branch",
    "re_aggregated_count/empty_union_branch_control",
    "re_aggregated_count/no_matching_customer",
    "re_aggregated_count/no_matching_customer_control",
];

#[test]
fn the_listing_holds_each_case_and_no_other() -> Result<(), Box<dyn Error>> {
    runner::listing(LISTED)?;
    Ok(())
}

fn table(
    alias: &str,
    columns: &[(&str, &str, DataType)],
    rows: Vec<Vec<PhysicalExpr>>,
) -> Result<PlanNode, IrErrorKind> {
    let physical: Vec<_> = (columns.iter())
        .map(|(_, column, data_type)| (*column, data_type.clone()))
        .collect();
    let scan = ScanNode::new(VirtualTable::new(rows), schema(&physical))?;
    let fields = (columns.iter())
        .map(|(field, column, _)| (name(&format!("{alias}.{field}")), leaf(&["ps0", column])))
        .collect();
    in_name_order(scan, fields)
}

fn in_name_order(
    input: impl Into<PlanNode>,
    mut fields: Vec<(Name, PhysicalExpr)>,
) -> Result<PlanNode, IrErrorKind> {
    fields.sort_by(|(left, _), (right, _)| left.cmp(right));
    project(input, fields)
}

fn value(value: Option<impl IntoLiteral>, data_type: DataType) -> PhysicalExpr {
    value.map_or_else(|| literal(Literal::Null(data_type)), literal)
}

fn orders() -> Result<PlanNode, IrErrorKind> {
    let columns = [
        ("order_id", "order_id", DataType::BigInt),
        ("customer_id", "cust_key", DataType::BigInt),
        ("region", "region_code", DataType::String),
        ("amount", "amount", DataType::BigInt),
    ];
    let rows: [(i64, Option<i64>, Option<&str>, i64); 8] = [
        (1, Some(1), Some("EU"), 100),
        (2, Some(1), Some("EU"), 50),
        (3, Some(2), Some("EU"), 70),
        (4, Some(3), Some("US"), 30),
        (5, Some(9), Some("US"), 20),
        (6, Some(2), None, 40),
        (7, None, Some("APAC"), 10),
        (8, Some(4), Some("US"), 25),
    ];
    let rows = rows.map(|(id, customer, region, amount)| {
        vec![
            literal(id),
            value(customer, DataType::BigInt),
            value(region, DataType::String),
            literal(amount),
        ]
    });
    table("ssd0", &columns, rows.into())
}

fn customers() -> Result<PlanNode, IrErrorKind> {
    let columns = [
        ("customer_id", "customer_id", DataType::BigInt),
        ("segment", "segment", DataType::String),
        ("region", "region", DataType::String),
    ];
    let rows: [(i64, &str, Option<&str>); 5] = [
        (1, "SMB", Some("EU")),
        (2, "SMB", Some("US")),
        (3, "ENT", Some("US")),
        (4, "SMB", None),
        (5, "ENT", Some("APAC")),
    ];
    let rows = rows.map(|(id, segment, region)| {
        vec![
            literal(id),
            literal(segment),
            value(region, DataType::String),
        ]
    });
    table("ssd1", &columns, rows.into())
}

fn regions() -> Result<PlanNode, IrErrorKind> {
    let columns = [
        ("region", "reg", DataType::String),
        ("region_name", "region_name", DataType::String),
    ];
    let rows = [
        ("EU", "Europe"),
        ("US", "United States"),
        ("APAC", "Asia Pacific"),
        ("LATAM", "Latin America"),
    ];
    let rows = rows.map(|(region, region_name)| vec![literal(region), literal(region_name)]);
    table("ssd2", &columns, rows.into())
}

fn joinset(semantic_source: &str, cardinality: Cardinality) -> SemAnnotation {
    SemAnnotation::SemanticSource {
        semantic_source: semantic_source.into(),
        source_role: SourceRole::Joinset,
        originators: Vec::new(),
        physical_source: None,
        cardinality: Some(cardinality),
        delivery: None,
        answered_by: None,
        pruned: Vec::new(),
    }
}

fn grainset(semantic_source: &str, cardinality: Cardinality) -> SemAnnotation {
    SemAnnotation::SemanticSource {
        semantic_source: semantic_source.into(),
        source_role: SourceRole::Grainset,
        originators: Vec::new(),
        physical_source: None,
        cardinality: Some(cardinality),
        delivery: None,
        answered_by: None,
        pruned: Vec::new(),
    }
}

fn unionset(semantic_source: &str, delivery: Delivery) -> SemAnnotation {
    SemAnnotation::SemanticSource {
        semantic_source: semantic_source.into(),
        source_role: SourceRole::Unionset,
        originators: Vec::new(),
        physical_source: None,
        cardinality: None,
        delivery: Some(delivery),
        answered_by: None,
        pruned: Vec::new(),
    }
}

fn orders_to_customers(
    orders: PlanNode,
    join_type: JoinType,
    post_join_filter: Option<PhysicalExpr>,
) -> Result<PlanNode, IrErrorKind> {
    let on = leaf(&["ssd0", "customer_id"]).eq(leaf(&["ssd1", "customer_id"]));
    let join = JoinNode::new(orders, customers()?, join_type, on, post_join_filter)?;
    Ok(PlanNode::from(join).with_trace([joinset("orders_customers", Cardinality::ManyToOne)]))
}

enum Smb {
    RequestPredicate,
    RelationshipFilter,
}

fn sales(smb: Smb) -> Result<PlanNode, IrErrorKind> {
    let segment = leaf(&["ssd1", "segment"]).eq(literal("SMB"));
    let (post_join_filter, predicate) = match smb {
        Smb::RequestPredicate => (None, Some(segment)),
        Smb::RelationshipFilter => (Some(segment), None),
    };
    let r1 = orders_to_customers(orders()?, JoinType::Left, post_join_filter)?;
    let smb = match predicate {
        Some(predicate) => FilterNode::new(r1, predicate)?.into(),
        None => r1,
    };
    let region = scalar(
        "coalesce",
        vec![leaf(&["ssd0", "region"]), leaf(&["ssd1", "region"])],
    );
    let r1 = project(
        smb,
        vec![
            (name("ssj3.amount"), leaf(&["ssd0", "amount"])),
            (name("ssd0.region"), leaf(&["ssd0", "region"])),
            (name("ssj3.region"), region),
            (name("ssj3.segment"), leaf(&["ssd1", "segment"])),
        ],
    )?;
    let on = leaf(&["ssd0", "region"]).eq(leaf(&["ssd2", "region"]));
    let sales = JoinNode::new(r1, regions()?, JoinType::Inner, on, None)?;
    let region = scalar(
        "coalesce",
        vec![leaf(&["ssj3", "region"]), leaf(&["ssd2", "region"])],
    );
    let sales = project(
        PlanNode::from(sales).with_trace([joinset("sales", Cardinality::ManyToOne)]),
        vec![
            (name("ssj4.amount"), leaf(&["ssj3", "amount"])),
            (name("ssj4.region"), region),
            (name("ssj4.region_name"), leaf(&["ssd2", "region_name"])),
            (name("ssj4.segment"), leaf(&["ssj3", "segment"])),
        ],
    )?;
    let groupings = ["region_name", "segment"].map(|field| (name(field), leaf(&["ssj4", field])));
    let revenue = aggregate("sum", vec![leaf(&["ssj4", "amount"])], None);
    Ok(AggregateNode::new(sales, groupings.into(), vec![(name("revenue"), revenue)])?.into())
}

async fn request_predicate() -> Result<Golden, Box<dyn Error>> {
    Ok(gate(
        "predicate/request_predicate",
        &plan(sales(Smb::RequestPredicate)?),
    )
    .await?
    .1)
}

#[tokio::test]
async fn a_left_join_under_an_inner_join_gives_the_revenue_by_region_and_segment(
) -> Result<(), Box<dyn Error>> {
    request_predicate().await?.assert_committed();
    Ok(())
}

async fn relationship_filter() -> Result<Golden, Box<dyn Error>> {
    Ok(gate(
        "predicate/relationship_filter",
        &plan(sales(Smb::RelationshipFilter)?),
    )
    .await?
    .1)
}

#[tokio::test]
async fn a_relationship_filter_in_the_left_join_gives_its_rows() -> Result<(), Box<dyn Error>> {
    relationship_filter().await?.assert_committed();
    Ok(())
}

async fn join_type_cases() -> Result<Vec<Golden>, Box<dyn Error>> {
    let cases = [
        (JoinType::Inner, "join_type/inner"),
        (JoinType::Left, "join_type/left"),
        (JoinType::Right, "join_type/right"),
        (JoinType::Full, "join_type/full"),
    ];
    let mut goldens = Vec::new();
    for (join_type, case) in cases {
        let (from, to) = (leaf(&["ssd0", "region"]), leaf(&["ssd1", "region"]));
        let region = match join_type {
            JoinType::Right => vec![to, from],
            _ => vec![from, to],
        };
        let joinset = project(
            orders_to_customers(orders()?, join_type, None)?,
            vec![
                (name("ssj2.order_id"), leaf(&["ssd0", "order_id"])),
                (name("ssj2.region"), scalar("coalesce", region)),
                (name("ssj2.segment"), leaf(&["ssd1", "segment"])),
            ],
        )?;
        let root =
            ["order_id", "region", "segment"].map(|field| (name(field), leaf(&["ssj2", field])));
        let semantic = plan(project(joinset, root.into())?);
        goldens.push(gate(case, &semantic).await?.1);
    }
    Ok(goldens)
}

#[tokio::test]
async fn each_join_type_gives_its_rows_and_the_coalesce_of_a_name_both_map(
) -> Result<(), Box<dyn Error>> {
    join_type_cases()
        .await?
        .iter()
        .for_each(Golden::assert_committed);
    Ok(())
}

#[test]
fn the_post_join_filter_rule_moves_the_filter_and_the_emit_form_above_the_join_once(
) -> Result<(), Box<dyn Error>> {
    let smb = leaf(&["ssd1", "segment"]).eq(literal("SMB"));
    let emit = EmitForm::new([0, 3]);
    let offered = orders_to_customers(orders()?, JoinType::Left, Some(smb.clone()))?
        .with_emit_form(emit.clone())?;

    let replacement = rules().rewrite_node(&offered);

    let inner = orders_to_customers(orders()?, JoinType::Left, None)?;
    let expected = PlanNode::from(FilterNode::new(inner.clone(), smb)?).with_emit_form(emit)?;
    assert_eq!(replacement, Some(expected));
    let replacement = replacement.ok_or("a replacement")?;
    assert_eq!(
        replacement.common().output_schema,
        offered.common().output_schema
    );
    assert_eq!(rules().rewrite_node(&inner), None);
    Ok(())
}

const DAILY: [(&str, Option<&str>, i64); 4] = [
    ("2026-03-01", Some("A"), 10),
    ("2026-03-02", Some("A"), 20),
    ("2026-04-01", Some("A"), 30),
    ("2026-03-15", Some("B"), 40),
];

const MONTHLY: [(&str, Option<&str>, i64); 3] = [
    ("2026-03-01", Some("A"), 100),
    ("2026-04-01", Some("A"), 100),
    ("2026-03-01", Some("B"), 50),
];

const YEARLY: [(&str, Option<&str>, i64); 2] = [
    ("2026-01-01", Some("A"), 200),
    ("2026-01-01", Some("B"), 50),
];

fn spend(
    alias: &str,
    measure: &str,
    groupings: &[&str],
    rows: &[(&str, Option<&str>, i64)],
) -> Result<PlanNode, IrErrorKind> {
    let columns = [
        ("date", "date", DataType::Date),
        ("campaign", "campaign", DataType::String),
        (measure, measure, DataType::BigInt),
    ];
    let rows = (rows.iter())
        .map(|(date, campaign, total)| {
            let date = literal(Literal::Date((*date).to_owned()));
            vec![date, value(*campaign, DataType::String), literal(*total)]
        })
        .collect();
    let field = |field: &str| name(&format!("{alias}.{field}"));
    let groupings = (groupings.iter())
        .map(|grouping| (field(grouping), leaf(&[alias, grouping])))
        .collect();
    let total = aggregate("sum", vec![leaf(&[alias, measure])], None);
    let spend = table(alias, &columns, rows)?;
    Ok(AggregateNode::new(spend, groupings, vec![(field(measure), total)])?.into())
}

fn drill_across(
    grainset_alias: &str,
    (first_alias, first): (&str, PlanNode),
    later: Vec<(&str, PlanNode)>,
) -> Result<PlanNode, IrErrorKind> {
    let mut earlier = vec![first_alias];
    (later.into_iter()).try_fold(first, |joined, (alias, child)| {
        let on = coalesced(&earlier, "campaign").is_not_distinct_from(leaf(&[alias, "campaign"]));
        earlier.push(alias);
        let join = JoinNode::new(joined, child, JoinType::Full, on, None)?;
        Ok(PlanNode::from(join).with_trace([grainset(grainset_alias, Cardinality::OneToOne)]))
    })
}

fn coalesced(children: &[&str], field: &str) -> PhysicalExpr {
    match children {
        [child] => leaf(&[child, field]),
        _ => scalar(
            "coalesce",
            (children.iter())
                .map(|child| leaf(&[child, field]))
                .collect(),
        ),
    }
}

type Requested<'r> = [(&'r str, &'r [&'r str])];

fn frame(alias: &str, joined: PlanNode, request: &Requested) -> Result<PlanNode, IrErrorKind> {
    let fields = (request.iter())
        .map(|(field, children)| {
            (
                name(&format!("{alias}.{field}")),
                coalesced(children, field),
            )
        })
        .collect();
    let root = (request.iter())
        .map(|(field, _)| (name(field), leaf(&[alias, field])))
        .collect();
    project(in_name_order(joined, fields)?, root)
}

async fn one_row_children() -> Result<Golden, Box<dyn Error>> {
    let crossed = CrossNode::new(
        spend("ssd1", "budget", &[], &YEARLY)?,
        spend("ssd0", "clicks", &[], &DAILY)?,
    );
    let crossed = PlanNode::from(crossed).with_trace([grainset("ssg2", Cardinality::OneToOne)]);
    let root = frame(
        "ssg2",
        crossed,
        &[("clicks", &["ssd0"]), ("budget", &["ssd1"])],
    )?;

    Ok(gate("drill_across/one_row_children", &plan(root)).await?.1)
}

#[tokio::test]
async fn two_one_row_children_crossed_give_one_row() -> Result<(), Box<dyn Error>> {
    one_row_children().await?.assert_committed();
    Ok(())
}

async fn drill_across_by_campaign_cases() -> Result<Vec<Golden>, Box<dyn Error>> {
    let null_campaign = |total| [("2026-03-01", None, total)];
    let cases = [
        ("drill_across/by_campaign", MONTHLY.to_vec(), DAILY.to_vec()),
        (
            "drill_across/null_campaign_group",
            [&MONTHLY[..], &null_campaign(20)[..]].concat(),
            [&DAILY[..], &null_campaign(10)[..]].concat(),
        ),
    ];
    let mut goldens = Vec::new();
    for (case, monthly, daily) in cases {
        let monthly = ("ssd1", spend("ssd1", "budget", &["campaign"], &monthly)?);
        let daily = ("ssd0", spend("ssd0", "clicks", &["campaign"], &daily)?);
        let request: [(&str, &[&str]); 3] = [
            ("campaign", &["ssd1", "ssd0"]),
            ("clicks", &["ssd0"]),
            ("budget", &["ssd1"]),
        ];
        let root = frame(
            "ssg2",
            drill_across("ssg2", monthly, vec![daily])?,
            &request,
        )?;
        goldens.push(gate(case, &plan(root)).await?.1);
    }
    Ok(goldens)
}

#[tokio::test]
async fn a_drill_across_by_campaign_joins_the_null_group_once_with_each_childs_value(
) -> Result<(), Box<dyn Error>> {
    drill_across_by_campaign_cases()
        .await?
        .iter()
        .for_each(Golden::assert_committed);
    Ok(())
}

fn spend_chain() -> Result<PlanNode, IrErrorKind> {
    let budget = [&YEARLY[..], &[("2026-01-01", Some("C"), 30)]].concat();
    let impressions = [
        ("2026-03-01", Some("A"), 1000),
        ("2026-03-01", Some("C"), 300),
        ("2026-03-01", Some("D"), 70),
    ];
    let campaign = ["campaign"];
    let joined = drill_across(
        "ssg3",
        ("ssd2", spend("ssd2", "budget", &campaign, &budget)?),
        vec![
            (
                "ssd1",
                spend("ssd1", "impressions", &campaign, &impressions)?,
            ),
            ("ssd0", spend("ssd0", "clicks", &campaign, &DAILY)?),
        ],
    )?;
    let request: [(&str, &[&str]); 4] = [
        ("campaign", &["ssd2", "ssd1", "ssd0"]),
        ("budget", &["ssd2"]),
        ("impressions", &["ssd1"]),
        ("clicks", &["ssd0"]),
    ];
    frame("ssg3", joined, &request)
}

async fn left_deep_chain() -> Result<Golden, Box<dyn Error>> {
    Ok(gate("drill_across/left_deep_chain", &plan(spend_chain()?))
        .await?
        .1)
}

#[tokio::test]
async fn a_left_deep_chain_keeps_every_group_with_each_childs_value_or_null(
) -> Result<(), Box<dyn Error>> {
    left_deep_chain().await?.assert_committed();
    Ok(())
}

fn tiered_orders() -> Result<PlanNode, IrErrorKind> {
    let orders: [(i64, i64); 3] = [(1, 100), (2, 50), (3, 10)];
    let orders = orders.map(|(id, amount)| vec![literal(id), literal(amount)]);
    let columns = [
        ("order_id", "order_id", DataType::BigInt),
        ("amount", "amount", DataType::BigInt),
    ];
    let orders = table("ssd0", &columns, orders.into())?;
    let tiers = [("silver", 50_i64), ("gold", 100)];
    let tiers = tiers.map(|(tier, floor)| vec![literal(tier), literal(floor)]);
    let columns = [
        ("tier", "tier", DataType::String),
        ("floor", "floor", DataType::BigInt),
    ];
    let tiers = table("ssd1", &columns, tiers.into())?;
    let crossed = PlanNode::from(CrossNode::new(orders, tiers))
        .with_trace([joinset("ssj2", Cardinality::ManyToMany)]);
    let reached = leaf(&["ssd0", "amount"]).gt_eq(leaf(&["ssd1", "floor"]));
    let joinset = project(
        FilterNode::new(crossed, reached)?,
        vec![
            (name("ssj2.order_id"), leaf(&["ssd0", "order_id"])),
            (name("ssj2.tier"), leaf(&["ssd1", "tier"])),
        ],
    )?;
    let root = ["order_id", "tier"].map(|field| (name(field), leaf(&["ssj2", field])));
    project(joinset, root.into())
}

async fn non_equi_filter() -> Result<Golden, Box<dyn Error>> {
    Ok(gate("cross/non_equi_filter", &plan(tiered_orders()?))
        .await?
        .1)
}

#[tokio::test]
async fn a_cross_under_a_non_equi_filter_gives_the_pairs_that_pass_it() -> Result<(), Box<dyn Error>>
{
    non_equi_filter().await?.assert_committed();
    Ok(())
}

fn shop_orders() -> Result<PlanNode, IrErrorKind> {
    let columns = [
        ("customer_id", "customer_id", DataType::String),
        ("region", "region", DataType::String),
        ("channel", "channel", DataType::String),
        ("amount", "amount", DataType::BigInt),
    ];
    let rows: [(&str, &str, Option<&str>, i64); 4] = [
        ("c1", "eu", Some("web"), 100),
        ("c1", "eu", Some("app"), 50),
        ("c1", "us", Some("web"), 200),
        ("c2", "us", None, 10),
    ];
    let rows = rows.map(|(customer_id, region, channel, amount)| {
        vec![
            literal(customer_id),
            literal(region),
            value(channel, DataType::String),
            literal(amount),
        ]
    });
    table("ssd0", &columns, rows.into())
}

fn channels(
    condition: fn(PhysicalExpr, PhysicalExpr) -> PhysicalExpr,
) -> Result<PlanNode, Box<dyn Error>> {
    let field = |field: &str| (name(&format!("ssd0.{field}")), leaf(&["ssd0", field]));
    let customers = vec![leaf(&["ssd0", "customer_id"])];
    let buyers = AggregateCall::builder(aggregate_function("count"), customers)
        .distinct(true)
        .build()?;
    let frame = AggregateNode::new(
        shop_orders()?,
        vec![field("channel")],
        vec![(name("ssd0.buyers"), buyers)],
    )?;
    let revenue = aggregate("sum", vec![leaf(&["ssd0", "amount"])], None);
    let step_1 = AggregateNode::new(
        shop_orders()?,
        vec![field("channel"), field("region")],
        vec![(name("ssd0.revenue"), revenue)],
    )?;
    let regions = aggregate("count", vec![leaf(&["ssd0", "revenue"])], None);
    let step_2 = AggregateNode::new(
        step_1,
        vec![(name("ssd0.__plan_join_channel"), leaf(&["ssd0", "channel"]))],
        vec![(name("ssd0.regions_with_revenue"), regions)],
    )?;
    let on = condition(
        leaf(&["ssd0", "channel"]),
        leaf(&["ssd0", "__plan_join_channel"]),
    );
    let join = JoinNode::new(frame, step_2, JoinType::Inner, on, None)?;
    let root = ["channel", "regions_with_revenue", "buyers"]
        .map(|field| (name(field), leaf(&["ssd0", field])));
    Ok(project(join, root.into())?)
}

async fn not_distinct_from() -> Result<Golden, Box<dyn Error>> {
    Ok(gate(
        "distinct_from/not_distinct_from",
        &plan(channels(ExprOperationExt::is_not_distinct_from)?),
    )
    .await?
    .1)
}

async fn equal_control() -> Result<Golden, Box<dyn Error>> {
    Ok(gate(
        "distinct_from/equal_control",
        &plan(channels(ExprOperationExt::eq)?),
    )
    .await?
    .1)
}

#[tokio::test]
async fn a_join_of_two_subtrees_on_is_not_distinct_from_keeps_the_null_group(
) -> Result<(), Box<dyn Error>> {
    not_distinct_from().await?.assert_committed();
    equal_control().await?.assert_committed();
    Ok(())
}

fn union(op: MultisetOp, branches: Vec<PlanNode>) -> Result<PlanNode, IrErrorKind> {
    let union = MultisetNode::new(branches, op)?;
    Ok(PlanNode::from(union).with_trace([unionset("sales", Delivery::Rows)]))
}

type Sales<'a> = &'a [(&'a str, i64)];

fn sales_branch(key: Option<&str>, rows: Sales<'_>) -> Result<PlanNode, IrErrorKind> {
    let key_column = key.map(|key| (key, key, DataType::String));
    let columns: Vec<_> = (key_column.into_iter())
        .chain([("amount", "amount", DataType::BigInt)])
        .collect();
    let rows = (rows.iter())
        .map(|(id, amount)| {
            let id = key.map(|_| literal(*id));
            id.into_iter().chain([literal(*amount)]).collect()
        })
        .collect();
    table("ssu2", &columns, rows)
}

fn null_filled_sales() -> Result<PlanNode, IrErrorKind> {
    let web = sales_branch(Some("customer_id"), &[("c1", 10)])?;
    let store = ScanNode::new(
        VirtualTable::new(vec![vec![literal(20_i64)]]),
        schema(&[("amount", DataType::BigInt)]),
    )?;
    let store = project(
        store,
        vec![
            (name("ssu2.amount"), leaf(&["ps0", "amount"])),
            (
                name("ssu2.customer_id"),
                value(None::<&str>, DataType::String),
            ),
        ],
    )?;
    let sales = union(MultisetOp::UnionAll, vec![web, store])?;
    let root = ["amount", "customer_id"].map(|field| (name(field), leaf(&["ssu2", field])));
    project(sales, root.into())
}

async fn typed_null_fill() -> Result<Golden, Box<dyn Error>> {
    let root = null_filled_sales()?;

    let [sales] = root.children()[..] else {
        panic!("the root projection's one input");
    };
    let fields = [
        ("ssu2.amount", DataType::BigInt),
        ("ssu2.customer_id", DataType::String),
    ];
    assert_eq!(sales.common().output_schema, schema(&fields));
    Ok(gate("union/typed_null_fill", &plan(root)).await?.1)
}

#[tokio::test]
async fn a_branch_s_typed_null_fill_keeps_the_union_s_field_type() -> Result<(), Box<dyn Error>> {
    typed_null_fill().await?.assert_committed();
    Ok(())
}

#[rustfmt::skip]
const UNIQUE_UNIONS: [(&str, Option<&str>, Sales, Sales); 3] = [
    ("unique_union/keyed_by_customer", Some("customer_id"), &[("c1", 10)], &[("c2", 10)]),
    ("unique_union/no_key", None, &[("c1", 10)], &[("c2", 10)]),
    ("unique_union/keyed_by_sale", Some("sale_id"), &[("s1", 10)], &[("s2", 10), ("s1", 10)]),
];

fn unique_revenue(
    key: Option<&str>,
    web: Sales<'_>,
    store: Sales<'_>,
) -> Result<PlanNode, IrErrorKind> {
    let branches = vec![sales_branch(key, web)?, sales_branch(key, store)?];
    let sales = union(MultisetOp::UnionDistinct, branches)?;
    let revenue = aggregate("sum", vec![leaf(&["ssu2", "amount"])], None);
    Ok(AggregateNode::new(sales, Vec::new(), vec![(name("revenue"), revenue)])?.into())
}

async fn unique_union_cases() -> Result<Vec<Golden>, Box<dyn Error>> {
    let mut goldens = Vec::new();
    for (case, key, web, store) in UNIQUE_UNIONS {
        let semantic = plan(unique_revenue(key, web, store)?);
        goldens.push(gate(case, &semantic).await?.1);
    }
    Ok(goldens)
}

#[tokio::test]
async fn a_unique_union_compares_every_output_field() -> Result<(), Box<dyn Error>> {
    unique_union_cases()
        .await?
        .iter()
        .for_each(Golden::assert_committed);
    Ok(())
}

fn totals(rows: Sales<'_>, key: PhysicalExpr, kept: PhysicalExpr) -> Result<PlanNode, IrErrorKind> {
    let kept = FilterNode::new(sales_branch(Some("customer_id"), rows)?, kept)?;
    let total = aggregate("sum", vec![leaf(&["ssu2", "amount"])], None);
    let groupings = vec![(name("ssu2.customer_id"), key)];
    Ok(AggregateNode::new(kept, groupings, vec![(name("ssu2.amount"), total)])?.into())
}

async fn first_branch_folded() -> Result<Golden, Box<dyn Error>> {
    let customer = leaf(&["ssu2", "customer_id"]);
    let web_key = scalar("coalesce", vec![customer.clone(), literal("c0")]);
    let web = totals(&[("c1", 10)], web_key, literal("web").eq(literal("store")))?;
    let store = totals(
        &[("c3", 7)],
        customer,
        literal("store").eq(literal("store")),
    )?;
    let sales = union(MultisetOp::UnionAll, vec![web, store])?;
    Ok(gate("union/first_branch_folded", &plan(sales)).await?.1)
}

#[tokio::test]
async fn a_union_whose_first_branch_folds_to_no_row_gives_its_other_branch_s_rows(
) -> Result<(), Box<dyn Error>> {
    first_branch_folded().await?.assert_committed();
    Ok(())
}

async fn re_aggregated_count_cases() -> Result<Vec<Golden>, Box<dyn Error>> {
    let empty = ScanNode::new(
        VirtualTable::new(Vec::new()),
        schema(&[("ssu2.order_count", DataType::BigInt)]),
    )?;
    let count = aggregate("count", vec![leaf(&["ssd0", "order_id"])], None);
    let partials = AggregateNode::new(
        orders()?,
        vec![(name("ssd0.customer_id"), leaf(&["ssd0", "customer_id"]))],
        vec![(name("ssd0.order_count"), count)],
    )?;
    let joined = orders_to_customers(partials.into(), JoinType::Inner, None)?;
    let vip = FilterNode::new(joined, leaf(&["ssd1", "segment"]).eq(literal("vip")))?;
    let cases = [
        (
            "re_aggregated_count/empty_union_branch",
            "ssu2",
            PlanNode::from(empty),
        ),
        (
            "re_aggregated_count/no_matching_customer",
            "ssd0",
            vip.into(),
        ),
    ];
    let mut goldens = Vec::new();
    for (case, alias, partials) in cases {
        let re_aggregated = aggregate("sum", vec![leaf(&[alias, "order_count"])], None);
        let control = AggregateNode::new(
            partials,
            Vec::new(),
            vec![(name("order_count"), re_aggregated)],
        )?;
        let order_count = scalar("coalesce", vec![leaf(&["order_count"]), literal(0_i64)]);
        let coalesced = project(control.clone(), vec![(name("order_count"), order_count)])?;
        goldens.push(gate(case, &plan(coalesced)).await?.1);
        goldens.push(gate(&format!("{case}_control"), &plan(control)).await?.1);
    }
    Ok(goldens)
}

#[tokio::test]
async fn a_re_aggregated_count_over_no_partial_aggregate_is_zero() -> Result<(), Box<dyn Error>> {
    re_aggregated_count_cases()
        .await?
        .iter()
        .for_each(Golden::assert_committed);
    Ok(())
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        let mut goldens = vec![
            request_predicate()
                .await
                .expect("predicate/request_predicate"),
            relationship_filter()
                .await
                .expect("predicate/relationship_filter"),
        ];
        goldens.extend(join_type_cases().await.expect("join_type"));
        goldens.push(non_equi_filter().await.expect("cross/non_equi_filter"));
        goldens.push(
            one_row_children()
                .await
                .expect("drill_across/one_row_children"),
        );
        goldens.extend(
            drill_across_by_campaign_cases()
                .await
                .expect("drill_across/by_campaign"),
        );
        goldens.push(
            left_deep_chain()
                .await
                .expect("drill_across/left_deep_chain"),
        );
        goldens.push(
            not_distinct_from()
                .await
                .expect("distinct_from/not_distinct_from"),
        );
        goldens.push(equal_control().await.expect("distinct_from/equal_control"));
        goldens.push(typed_null_fill().await.expect("union/typed_null_fill"));
        goldens.push(
            first_branch_folded()
                .await
                .expect("union/first_branch_folded"),
        );
        goldens.extend(unique_union_cases().await.expect("unique_union"));
        goldens.extend(
            re_aggregated_count_cases()
                .await
                .expect("re_aggregated_count"),
        );
        goldens
    })
    .await;
}
