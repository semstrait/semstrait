#![allow(
    clippy::result_large_err,
    reason = "a function in `plan`'s place returns what `plan` returns, its PlanError unboxed"
)]

mod common;

use std::error::Error;

use common::golden::{self, Golden, GENERATOR};
use common::runner::{self, Planned};
use common::{
    adapted, aggregate, files_read, leaf, literal, name, plan, produced, project, relations, rules,
    scalar, scanned, schema, Faithful,
};
use prost::Message;
use semstrait_adapter::explain;
use semstrait_adapter_datafusion::DataFusionRules;
use semstrait_catalog::FileFormat as CatalogFileFormat;
use semstrait_common::Release;
use semstrait_ir::{
    AggregateNode, EmitForm, FetchNode, FilesScan, FilterNode, GranularityGroup, IrErrorKind,
    PlanNode, ProjectNode, ScanNode, SemAnnotation, SemanticPlan, SortNode, SourceRole, TableScan,
    VirtualTable,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    Call, Expr, ExprOperationExt, FunctionRef, Literal, NullOrdering, OptionSetting, ScalarCall,
    SortDir, SortKey,
};
use semstrait_sql::functions::{registry, OptionName, ScalarOption};
use substrait::proto::extensions::simple_extension_declaration::MappingType;
use substrait::proto::read_rel::local_files::file_or_files::{FileFormat, PathType};
use substrait::proto::read_rel::ReadType;
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::{Emit, EmitKind};
use substrait::proto::set_rel::SetOp;
use substrait::proto::{Plan, ReadRel};

const ORDERS_FOLDER: &str = "s3://semstrait-e2e/orders/";

const ORDERS: [&str; 2] = [
    "s3://semstrait-e2e/orders/part-0.parquet",
    "s3://semstrait-e2e/orders/part-1.parquet",
];

const ORDER_COLUMNS: [(&str, DataType); 4] = [
    ("id", DataType::BigInt),
    ("region", DataType::String),
    ("status", DataType::String),
    ("amount", DataType::BigInt),
];

const EVENTS: &str = "s3://semstrait-e2e/events/";

const BUCKETED: [(&str, &str); 3] = [
    ("2026-03-31T23:30:00", "2026-10-17"),
    ("2026-11-01T05:30:00", "2026-10-18"),
    ("2026-11-01T06:30:00", "2026-10-19"),
];

const LISTED: &[&str] = &[
    "unionset/implicit/two_paths_union",
    "read/files/one_path_no_union",
    "read/catalog/no_catalog_table",
    "read/catalog/catalog_named_table",
    "read/partitions/default_partition_null",
    "read/files/empty_read_ungrouped",
    "read/files/empty_read_grouped",
    "function/behavior/regexp_extract_null_group",
    "function/behavior/concat_null_argument",
    "temporal/shape/timezone_and_week_start",
    "temporal/shape/null_timezone_bucket",
    "function/behavior/extract_bigint_components",
];

#[test]
fn the_listing_holds_each_case_and_no_other() -> Result<(), Box<dyn Error>> {
    runner::listing(LISTED)?;
    Ok(())
}

fn by_hand(built: Result<SemanticPlan, IrErrorKind>) -> Planned {
    Ok((built.expect("a hand-built plan"), Vec::new()))
}

fn two_paths_union_plan() -> Result<SemanticPlan, IrErrorKind> {
    let files = FilesScan::new(
        Some(CatalogFileFormat::Parquet),
        None,
        ORDERS_FOLDER,
        ORDERS.map(String::from).to_vec(),
    );
    let scan = PlanNode::from(ScanNode::new(files, schema(&ORDER_COLUMNS))?).with_trace([
        SemAnnotation::SemanticSource {
            semantic_source: "orders".into(),
            source_role: SourceRole::Dataset,
            originators: Vec::new(),
            physical_source: Some(name("orders")),
            cardinality: None,
            delivery: None,
            answered_by: None,
            pruned: Vec::new(),
        },
    ]);
    let paid_flag = leaf(&["ps0", "status"]).eq(literal("paid"));
    let flagged = ProjectNode::new(scan, vec![(name("__plan_filter_paid"), paid_flag)])?;
    let flagged = PlanNode::from(flagged).with_emit_form(EmitForm::new([1, 2, 3, 4]))?;
    let not_cancelled = FilterNode::new(flagged, leaf(&["status"]).neq(literal("cancelled")))?;
    let amount = || vec![leaf(&["amount"])];
    let aggregates = vec![
        (name("total"), aggregate("sum", amount(), None)),
        (
            name("paid"),
            aggregate("sum", amount(), Some(leaf(&["__plan_filter_paid"]))),
        ),
    ];
    let regions = vec![(name("region"), leaf(&["region"]))];
    let totals = PlanNode::from(AggregateNode::new(not_cancelled, regions, aggregates)?)
        .with_trace([SemAnnotation::Rank {
            scope_root: "orders".into(),
            compositions: 0,
            granularity_fit: GranularityGroup::NotSelected,
            granularity: None,
        }]);
    let unpaid = leaf(&["total"]) - leaf(&["paid"]);
    let unpaid = ProjectNode::new(totals, vec![(name("unpaid"), unpaid)])?;
    let over_ten = FilterNode::new(unpaid, leaf(&["total"]).gt(literal(10_i64)))?;
    let outputs = ["region", "total", "paid", "unpaid"].map(|field| (name(field), leaf(&[field])));
    let root = project(over_ten, outputs.into())?;
    let by_paid = SortKey {
        expr: leaf(&["paid"]),
        dir: SortDir::Asc {
            nulls: NullOrdering::Last,
        },
    };
    let fetched = FetchNode::new(SortNode::new(root, vec![by_paid])?, 1, Some(2))?;
    Ok(plan(fetched))
}

async fn two_paths_union() -> Result<Golden, Box<dyn Error>> {
    let (plan, golden) = runner::hand_built("unionset/implicit/two_paths_union", |_, _, _| {
        by_hand(two_paths_union_plan())
    })
    .await?;

    let version = plan.version.as_ref().expect("a version");
    let release = [
        version.major_number,
        version.minor_number,
        version.patch_number,
    ];
    assert_eq!(release, [0, 85, 0]);
    let mut functions: Vec<_> = (plan.extensions.iter())
        .filter_map(|declaration| match &declaration.mapping_type {
            Some(MappingType::ExtensionFunction(function)) => Some(function.name.as_str()),
            _ => None,
        })
        .collect();
    functions.sort_unstable();
    assert_eq!(functions, ["equal", "gt", "not_equal", "subtract", "sum"]);
    let relations = relations(&plan);
    let [.., totals, _, flagged, union, part_0, part_1] = relations.as_slice() else {
        panic!("the plan's relations");
    };
    let Some(RelType::Aggregate(totals)) = &totals.rel_type else {
        panic!("an AggregateRel");
    };
    assert_eq!(
        totals.advanced_extension, None,
        "trace in the explain copy only"
    );
    let Some(RelType::Project(flagged)) = &flagged.rel_type else {
        panic!("a ProjectRel");
    };
    let emit = (flagged.common.as_ref()).and_then(|common| common.emit_kind.as_ref());
    let output_mapping = vec![1, 2, 3, 4];
    assert_eq!(emit, Some(&EmitKind::Emit(Emit { output_mapping })));
    let Some(RelType::Set(union)) = &union.rel_type else {
        panic!("a SetRel");
    };
    assert_eq!(union.op(), SetOp::UnionAll);
    assert_eq!(union.inputs.iter().collect::<Vec<_>>(), [*part_0, *part_1]);
    let names = ORDER_COLUMNS.map(|(column, _)| column);
    for (read, path) in [part_0, part_1].into_iter().zip(ORDERS) {
        let Some(RelType::Read(read)) = &read.rel_type else {
            panic!("a ReadRel");
        };
        assert_eq!(
            read.advanced_extension, None,
            "trace in the explain copy only"
        );
        let base_schema = read.base_schema.as_ref().expect("a base schema");
        assert_eq!(base_schema.names, names);
        let Some(ReadType::LocalFiles(files)) = &read.read_type else {
            panic!("a LocalFiles read");
        };
        let [item] = files.items.as_slice() else {
            panic!("one item");
        };
        assert_eq!(item.path_type, Some(PathType::UriFile(path.to_owned())));
        assert!(matches!(item.file_format, Some(FileFormat::Parquet(_))));
    }
    Ok(golden)
}

#[tokio::test]
async fn unionset_implicit_two_paths_union() -> Result<(), Box<dyn Error>> {
    two_paths_union().await?.assert_committed();
    Ok(())
}

fn one_path_no_union_plan() -> Result<SemanticPlan, IrErrorKind> {
    let [path, _] = ORDERS;
    let files = FilesScan::new(
        Some(CatalogFileFormat::Parquet),
        None,
        ORDERS_FOLDER,
        vec![path.to_owned()],
    );
    let regions = scanned(files, &[("region", DataType::String)], None)?;
    let groupings = vec![(name("region"), leaf(&["region"]))];
    Ok(plan(AggregateNode::new(regions, groupings, Vec::new())?))
}

async fn one_path_no_union() -> Result<Golden, Box<dyn Error>> {
    let [path, _] = ORDERS;

    let (plan, golden) = runner::hand_built("read/files/one_path_no_union", |_, _, _| {
        by_hand(one_path_no_union_plan())
    })
    .await?;

    assert!(
        !(relations(&plan).iter())
            .any(|relation| matches!(relation.rel_type, Some(RelType::Set(_)))),
        "one path has no SetRel"
    );
    assert_eq!(
        files_read(scan(&plan)),
        [PathType::UriFile(path.to_owned())]
    );
    Ok(golden)
}

#[tokio::test]
async fn read_files_one_path_no_union() -> Result<(), Box<dyn Error>> {
    one_path_no_union().await?.assert_committed();
    Ok(())
}

fn customer_counts(names: [&str; 3]) -> Result<SemanticPlan, IrErrorKind> {
    let table = TableScan::named_table(names.map(String::from), None);
    let columns = [("id", DataType::BigInt), ("segment", DataType::String)];
    let customers = scanned(table, &columns, None)?;
    let segments = vec![(name("segment"), leaf(&["segment"]))];
    let n = aggregate("count", vec![leaf(&["id"])], None);
    Ok(plan(AggregateNode::new(
        customers,
        segments,
        vec![(name("n"), n)],
    )?))
}

async fn no_catalog_table() -> Result<Golden, Box<dyn Error>> {
    let case = "read/catalog/no_catalog_table";
    let built: runner::PlanFn = |_, _, _| by_hand(customer_counts(["", "sales", "customers"]));
    let written: &[&str] = &["sales", "customers"];
    let (plan, golden) = runner::hand_built(case, built).await?;

    let Some(ReadType::NamedTable(table)) = &scan(&plan).read_type else {
        panic!("a NamedTable read");
    };
    assert_eq!(table.names, written, "{case}: the names written");
    Ok(golden)
}

async fn catalog_named_table() -> Result<Golden, Box<dyn Error>> {
    let case = "read/catalog/catalog_named_table";
    let built: runner::PlanFn = |_, _, _| by_hand(customer_counts(["lake", "sales", "customers"]));
    let written: &[&str] = &["lake", "sales", "customers"];
    let (plan, golden) = runner::hand_built(case, built).await?;

    let Some(ReadType::NamedTable(table)) = &scan(&plan).read_type else {
        panic!("a NamedTable read");
    };
    assert_eq!(table.names, written, "{case}: the names written");
    Ok(golden)
}

#[tokio::test]
async fn read_catalog_no_catalog_table() -> Result<(), Box<dyn Error>> {
    no_catalog_table().await?.assert_committed();
    Ok(())
}

#[tokio::test]
async fn read_catalog_catalog_named_table() -> Result<(), Box<dyn Error>> {
    catalog_named_table().await?.assert_committed();
    Ok(())
}

fn default_partition_null_plan() -> Result<SemanticPlan, IrErrorKind> {
    let folder = FilesScan::new(Some(CatalogFileFormat::Parquet), None, EVENTS, Vec::new());
    let columns = [
        ("year", DataType::Integer),
        ("region", DataType::String),
        ("amount", DataType::BigInt),
    ];
    let events = scanned(folder, &columns, None)?;
    let groupings = ["year", "region"].map(|column| (name(column), leaf(&[column])));
    let total = aggregate("sum", vec![leaf(&["amount"])], None);
    Ok(plan(AggregateNode::new(
        events,
        groupings.into(),
        vec![(name("total"), total)],
    )?))
}

async fn default_partition_null() -> Result<Golden, Box<dyn Error>> {
    let (plan, golden) = runner::hand_built("read/partitions/default_partition_null", |_, _, _| {
        by_hand(default_partition_null_plan())
    })
    .await?;

    assert_eq!(
        files_read(scan(&plan)),
        [PathType::UriFolder(EVENTS.to_owned())]
    );
    Ok(golden)
}

#[tokio::test]
async fn read_partitions_default_partition_null() -> Result<(), Box<dyn Error>> {
    default_partition_null().await?.assert_committed();
    Ok(())
}

fn empty_totals(groupings: &[&str]) -> Result<SemanticPlan, IrErrorKind> {
    let columns: Vec<_> = (groupings.iter())
        .map(|column| (*column, DataType::String))
        .chain([("amount", DataType::BigInt)])
        .collect();
    let empty = scanned(VirtualTable::new(Vec::new()), &columns, None)?;
    let groupings = (groupings.iter())
        .map(|column| (name(column), leaf(&[*column])))
        .collect();
    let amount = || vec![leaf(&["amount"])];
    let aggregates = vec![
        (name("n"), aggregate("count", amount(), None)),
        (name("total"), aggregate("sum", amount(), None)),
    ];
    Ok(plan(AggregateNode::new(empty, groupings, aggregates)?))
}

async fn empty_read_ungrouped() -> Result<Golden, Box<dyn Error>> {
    let case = "read/files/empty_read_ungrouped";
    let (plan, golden) = runner::hand_built(case, |_, _, _| by_hand(empty_totals(&[]))).await?;

    let Some(ReadType::VirtualTable(table)) = &scan(&plan).read_type else {
        panic!("a VirtualTable read");
    };
    assert!(
        table.expressions.is_empty(),
        "{case}: the empty read has no row"
    );
    Ok(golden)
}

async fn empty_read_grouped() -> Result<Golden, Box<dyn Error>> {
    let case = "read/files/empty_read_grouped";
    let (plan, golden) =
        runner::hand_built(case, |_, _, _| by_hand(empty_totals(&["region"]))).await?;

    let Some(ReadType::VirtualTable(table)) = &scan(&plan).read_type else {
        panic!("a VirtualTable read");
    };
    assert!(
        table.expressions.is_empty(),
        "{case}: the empty read has no row"
    );
    Ok(golden)
}

#[tokio::test]
async fn read_files_empty_read_ungrouped() -> Result<(), Box<dyn Error>> {
    empty_read_ungrouped().await?.assert_committed();
    Ok(())
}

#[tokio::test]
async fn read_files_empty_read_grouped() -> Result<(), Box<dyn Error>> {
    empty_read_grouped().await?.assert_committed();
    Ok(())
}

fn regexp_extract_null_group_plan() -> Result<SemanticPlan, IrErrorKind> {
    let rows = ["bc", "abc", "ac"].map(|v| vec![literal(v)]);
    let values = scanned(
        VirtualTable::new(rows.into()),
        &[("v", DataType::String)],
        None,
    )?;
    let group = |n: i32| {
        let args = vec![leaf(&["v"]), literal("(a)?(b)(c)"), literal(n)];
        scalar("regexp_extract", args)
    };
    let fields = vec![
        (name("v"), leaf(&["v"])),
        (name("g1"), group(1)),
        (name("g2"), group(2)),
    ];
    Ok(plan(project(values, fields)?))
}

async fn regexp_extract_null_group() -> Result<Golden, Box<dyn Error>> {
    let (_, golden) =
        runner::hand_built("function/behavior/regexp_extract_null_group", |_, _, _| {
            by_hand(regexp_extract_null_group_plan())
        })
        .await?;
    Ok(golden)
}

#[tokio::test]
async fn function_behavior_regexp_extract_null_group() -> Result<(), Box<dyn Error>> {
    regexp_extract_null_group().await?.assert_committed();
    Ok(())
}

fn concat_null_argument_plan() -> Result<SemanticPlan, IrErrorKind> {
    let rows = [("US", Some("purchase")), ("x", None)].map(|(a, b)| {
        let b = b.map_or_else(|| literal(Literal::Null(DataType::String)), literal);
        vec![literal(a), b]
    });
    let columns = [("a", DataType::String), ("b", DataType::String)];
    let pairs = scanned(VirtualTable::new(rows.into()), &columns, None)?;
    let c = scalar("concat", vec![leaf(&["a"]), literal("-"), leaf(&["b"])]);
    Ok(plan(project(pairs, vec![(name("c"), c)])?))
}

async fn concat_null_argument() -> Result<Golden, Box<dyn Error>> {
    let (_, golden) = runner::hand_built("function/behavior/concat_null_argument", |_, _, _| {
        by_hand(concat_null_argument_plan())
    })
    .await?;
    Ok(golden)
}

#[tokio::test]
async fn function_behavior_concat_null_argument() -> Result<(), Box<dyn Error>> {
    concat_null_argument().await?.assert_committed();
    Ok(())
}

fn timezone_and_week_start_plan() -> Result<SemanticPlan, IrErrorKind> {
    let values = timestamps_and_dates(BUCKETED)?;
    let truncated = |grain, column, setting| {
        let args = vec![literal(grain), leaf(&[column]), literal(setting)];
        scalar("date_trunc", args)
    };
    let fields = vec![
        (name("m"), truncated("month", "ts", "Europe/Stockholm")),
        (name("h"), truncated("hour", "ts", "America/New_York")),
        (name("ws"), truncated("week", "d", "sunday")),
        (name("wm"), truncated("week", "d", "monday")),
    ];
    Ok(plan(project(values, fields)?))
}

async fn timezone_and_week_start() -> Result<Golden, Box<dyn Error>> {
    let (_, golden) = runner::hand_built("temporal/shape/timezone_and_week_start", |_, _, _| {
        by_hand(timezone_and_week_start_plan())
    })
    .await?;
    Ok(golden)
}

#[tokio::test]
async fn temporal_shape_timezone_and_week_start() -> Result<(), Box<dyn Error>> {
    timezone_and_week_start().await?.assert_committed();
    Ok(())
}

fn null_timezone_bucket_plan() -> Result<SemanticPlan, IrErrorKind> {
    let values = timestamps_and_dates(BUCKETED)?;
    let zone = literal(Literal::Null(DataType::String));
    let m = scalar("date_trunc", vec![literal("month"), leaf(&["ts"]), zone]);
    Ok(plan(project(values, vec![(name("m"), m)])?))
}

async fn null_timezone_bucket() -> Result<Golden, Box<dyn Error>> {
    let case = "temporal/shape/null_timezone_bucket";
    let (_, golden) =
        runner::hand_built(case, |_, _, _| by_hand(null_timezone_bucket_plan())).await?;
    Ok(golden)
}

#[tokio::test]
async fn temporal_shape_null_timezone_bucket() -> Result<(), Box<dyn Error>> {
    null_timezone_bucket().await?.assert_committed();
    Ok(())
}

fn extract_bigint_components_plan() -> Result<SemanticPlan, IrErrorKind> {
    let values = timestamps_and_dates([
        ("2026-03-31T23:30:15", "2026-03-31"),
        ("2026-04-05T00:00:00", "2026-04-05"),
        ("2027-01-01T00:00:00", "2027-01-01"),
    ])?;
    let on_d = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "day_of_week",
        "day_of_year",
    ]
    .map(|component| (component, "d"));
    let on_ts = ["hour", "minute", "second"].map(|component| (component, "ts"));
    let fields = (on_d.into_iter().chain(on_ts))
        .map(|(component, column)| {
            let args = vec![literal(component), leaf(&[column])];
            (name(component), scalar("extract", args))
        })
        .collect();
    let components = project(values, fields)?;
    let compared = FilterNode::new(components, leaf(&["year"]).gt_eq(literal(2026_i64)))?;
    Ok(plan(compared))
}

async fn extract_bigint_components() -> Result<Golden, Box<dyn Error>> {
    let (_, golden) =
        runner::hand_built("function/behavior/extract_bigint_components", |_, _, _| {
            by_hand(extract_bigint_components_plan())
        })
        .await?;
    Ok(golden)
}

#[tokio::test]
async fn function_behavior_extract_bigint_components() -> Result<(), Box<dyn Error>> {
    extract_bigint_components().await?.assert_committed();
    Ok(())
}

#[test]
fn extract_day_of_week_is_written_from_dow_on_datafusion_53() -> Result<(), Box<dyn Error>> {
    let rows = vec![vec![literal(Literal::Date("2026-03-31".to_owned()))]];
    let values = scanned(VirtualTable::new(rows), &[("d", DataType::Date)], None)?;
    let extracted = scalar("extract", vec![literal("day_of_week"), leaf(&["d"])]);
    let semantic = plan(project(values, vec![(name("day_of_week"), extracted)])?);
    semantic.validate()?;

    let fifty_three = adapted(explain(&semantic, &DataFusionRules::new(Release(53))))?;
    assert!(
        fifty_three.contains("add(modulus(add(date_part('dow'"),
        "the explain text at DataFusion 53: {fifty_three}"
    );
    let default = adapted(explain(&semantic, &DataFusionRules::default()))?;
    assert!(
        default.contains("date_part('isodow'"),
        "the explain text at the default release: {default}"
    );
    Ok(())
}

#[test]
fn vocab_other_functions_keep_their_faithful_form() -> Result<(), Box<dyn Error>> {
    let values = scanned(
        VirtualTable::new(vec![vec![literal("abc")]]),
        &[("v", DataType::String)],
        None,
    )?;
    let v = || leaf(&["v"]);
    let Some(FunctionRef::Scalar(concat)) = registry().get("concat") else {
        panic!("the catalog declares the scalar `concat`");
    };
    let ignore = OptionSetting {
        name: ScalarOption::NullHandling.name().to_owned(),
        value: "IGNORE_NULLS".to_owned(),
    };
    let skipping = ScalarCall::builder(concat, vec![v(), literal("-")]).options(vec![ignore]);
    let skipping = Expr::Function(Call::Scalar(skipping.build()?));
    let others = [
        ("upper", vec![v()]),
        ("concat_ws", vec![literal("-"), v(), v()]),
        ("regexp_count", vec![v(), literal("b")]),
        ("regexp_match", vec![v(), literal("(b)")]),
    ];
    let fields = (others.into_iter())
        .map(|(function, args)| (name(function), scalar(function, args)))
        .chain([(name("concat"), skipping)])
        .collect();
    let calls = plan(project(values, fields)?);
    calls.validate()?;

    assert_eq!(
        produced(&calls, &rules())?.encode_to_vec(),
        produced(&calls, &Faithful)?.encode_to_vec(),
        "the productions under DataFusionRules and the faithful rules are byte-equal"
    );
    Ok(())
}

fn timestamps_and_dates(rows: [(&str, &str); 3]) -> Result<PlanNode, IrErrorKind> {
    let rows = rows.map(|(ts, d)| {
        let ts = Literal::Timestamp {
            value: ts.to_owned(),
            precision: 6,
        };
        vec![literal(ts), literal(Literal::Date(d.to_owned()))]
    });
    let columns = [("ts", DataType::timestamp()), ("d", DataType::Date)];
    scanned(VirtualTable::new(rows.into()), &columns, None)
}

fn scan(plan: &Plan) -> &ReadRel {
    let Some(RelType::Read(read)) = relations(plan)
        .pop()
        .and_then(|scan| scan.rel_type.as_ref())
    else {
        panic!("the scan is a ReadRel");
    };
    read
}

#[test]
fn the_generator_is_asked_only_by_its_name_with_its_switch_set() {
    let runs: [(&[&str], bool, bool); 8] = [
        (&[], false, false),
        (&["--ignored"], false, false),
        (&["--include-ignored"], false, false),
        (&[GENERATOR], false, false),
        (&[], true, false),
        (&["--include-ignored"], true, false),
        (&[GENERATOR], true, true),
        (&["--exact", GENERATOR], true, true),
    ];
    for (args, switch, asked) in runs {
        let run = args.iter().map(|arg| (*arg).to_owned());
        assert_eq!(
            golden::asked(run, switch),
            asked,
            "arguments {args:?}, switch set: {switch}"
        );
    }
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async {
        let mut goldens = vec![
            two_paths_union()
                .await
                .expect("unionset/implicit/two_paths_union"),
            one_path_no_union()
                .await
                .expect("read/files/one_path_no_union"),
        ];
        goldens.push(
            no_catalog_table()
                .await
                .expect("read/catalog/no_catalog_table"),
        );
        goldens.push(
            catalog_named_table()
                .await
                .expect("read/catalog/catalog_named_table"),
        );
        goldens.push(
            default_partition_null()
                .await
                .expect("read/partitions/default_partition_null"),
        );
        goldens.push(
            empty_read_ungrouped()
                .await
                .expect("read/files/empty_read_ungrouped"),
        );
        goldens.push(
            empty_read_grouped()
                .await
                .expect("read/files/empty_read_grouped"),
        );
        goldens.push(
            regexp_extract_null_group()
                .await
                .expect("function/behavior/regexp_extract_null_group"),
        );
        goldens.push(
            concat_null_argument()
                .await
                .expect("function/behavior/concat_null_argument"),
        );
        goldens.push(
            timezone_and_week_start()
                .await
                .expect("temporal/shape/timezone_and_week_start"),
        );
        goldens.push(
            null_timezone_bucket()
                .await
                .expect("temporal/shape/null_timezone_bucket"),
        );
        goldens.push(
            extract_bigint_components()
                .await
                .expect("function/behavior/extract_bigint_components"),
        );
        goldens
    })
    .await;
}
