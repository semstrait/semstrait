mod common;

use std::collections::BTreeMap;

use common::{agg, call};
use semstrait_common::Granularity;
use semstrait_common::SemanticsName;
use semstrait_model::expr::field;
use semstrait_model::{
    parse, Agg, DimensionSite, DimensionType, MeasureSite, MetricSite, SemanticMapping,
    SemanticMappingValue, StorageFormat, TemporalShape, TemporalShapeKind, UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{
    BinaryOpKind, Expr, ExprOperationExt, Literal, SemanticLeaf, UntypedLiteral,
};

const FIXTURE: &str = include_str!("../../../test_data/acme_eu_ad_platform_v3.yaml");

fn site_names<I, R>(
    sites: &semstrait_model::EntityBTreeMap<semstrait_model::SemanticSite<I, R>>,
) -> Vec<String>
where
    I: semstrait_model::AuthoringBody,
    R: semstrait_model::AuthoringBody,
{
    let mut names: Vec<String> = sites
        .values()
        .map(|s| s.name().as_ref().to_owned())
        .collect();
    names.sort();
    names
}

fn lit_str(s: &str) -> UnresolvedSemanticExpr {
    Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::String(
        s.into(),
    ))))
}
fn lit_int(i: i64) -> UnresolvedSemanticExpr {
    Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::BigInt(i))))
}
fn lit_null() -> UnresolvedSemanticExpr {
    Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::Null(
        DataType::String,
    ))))
}

fn base_path(platform: &str) -> String {
    let account = "5f3a9c2e-8b41-4d6a-9e07-2c1b4a7d8e90";
    format!("s3://semstrait-e2e/example/exports/acme/eu/{platform}/{account}/")
}

fn block_expr(expr: &UnresolvedSemanticExpr) -> &UnresolvedSemanticExpr {
    expr
}

#[test]
fn model_parsing_produces_expected_ir() {
    let builder = parse(FIXTURE).unwrap_or_else(|diags| {
        for d in &diags {
            eprintln!("PARSE: {:?}", d);
        }
        panic!("model fixture failed to parse");
    });
    let (model, build_diags) = builder.build().unwrap_or_else(|diags| {
        for d in &diags {
            eprintln!("BUILD: {:?}", d);
        }
        panic!("model fixture failed to build");
    });
    assert!(
        build_diags.is_empty(),
        "expected zero build diagnostics; got: {build_diags:?}"
    );

    assert_eq!(model.name, "acme-eu-ad-platforms");
    assert_eq!(
        model.labels,
        vec![
            "mmm".to_string(),
            "acme".into(),
            "eu".into(),
            "paid-media".into(),
            "independent-variables".into()
        ]
    );
    assert!(
        model
            .unionsets
            .values()
            .any(|u| u.body.base.name == "paid_media_campaign_performance"),
        "unionset present"
    );
    assert!(
        model
            .datasets
            .values()
            .any(|d| d.body.base.name == "shopify"),
        "shopify dataset present"
    );
    assert_eq!(model.datasets.len(), 1, "exactly one top-level dataset");
    assert_eq!(model.unionsets.len(), 1, "exactly one top-level unionset");
    assert_eq!(model.grainsets.len(), 0);
    assert_eq!(model.joinsets.len(), 0);

    let dims: BTreeMap<&str, &_> = model
        .dimensions
        .values()
        .map(|d| (d.name().as_str(), d))
        .collect();
    assert_eq!(
        dims.keys().map(|k| (*k).to_owned()).collect::<Vec<_>>(),
        vec![
            "campaign",
            "country",
            "currency",
            "dataset_name",
            "date",
            "export_account_id",
            "measurement_channel",
            "measurement_source",
            "traffic_source",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>(),
    );
    let dt = |name: &str| dims.get(name).expect(name).dim_type.clone();
    assert!(matches!(dt("campaign"), DimensionType::Categorical));
    assert!(matches!(dt("country"), DimensionType::Categorical));
    assert!(matches!(dt("currency"), DimensionType::Categorical));
    assert!(matches!(
        dt("measurement_channel"),
        DimensionType::Categorical
    ));
    assert!(matches!(
        dt("measurement_source"),
        DimensionType::Categorical
    ));
    assert!(matches!(dt("traffic_source"), DimensionType::Categorical));
    match dt("dataset_name") {
        DimensionType::Metadata(b) => match b.source {
            semstrait_model::MetadataSource::Path(p) => assert_eq!(p.token, 5),
            other => panic!("dataset_name should be Metadata(Path), got {other:?}"),
        },
        other => panic!("dataset_name should be Metadata, got {other:?}"),
    }
    match dt("export_account_id") {
        DimensionType::Metadata(b) => match b.source {
            semstrait_model::MetadataSource::Path(p) => assert_eq!(p.token, 6),
            other => panic!("export_account_id should be Metadata(Path), got {other:?}"),
        },
        other => panic!("export_account_id should be Metadata, got {other:?}"),
    }
    match dt("date") {
        DimensionType::Temporal(b) => assert_eq!(
            b.granularities,
            vec![
                Granularity::Day,
                Granularity::Week,
                Granularity::Month,
                Granularity::Quarter,
                Granularity::Year
            ]
        ),
        other => panic!("date should be Temporal, got {other:?}"),
    }
    for name in [
        "campaign",
        "country",
        "currency",
        "dataset_name",
        "export_account_id",
        "measurement_channel",
        "measurement_source",
        "traffic_source",
    ] {
        assert_eq!(
            *dims.get(name).unwrap().data_type(),
            DataType::String,
            "{name}"
        );
    }
    assert_eq!(*dims.get("date").unwrap().data_type(), DataType::Date);

    let ms_expr = block_expr(
        dims.get("measurement_source")
            .unwrap()
            .body
            .body
            .expr
            .as_ref()
            .unwrap(),
    );
    let expected_ms = call(
        "concat",
        vec![
            field("measurement_channel").expect("a name"),
            lit_str(" - "),
            field("traffic_source").expect("a name"),
        ],
    );
    assert_eq!(
        ms_expr, &expected_ms,
        "measurement_source CONCAT must equal the constructed expected tree"
    );

    let measures: BTreeMap<&str, &_> = model
        .measures
        .values()
        .map(|m| (m.name().as_str(), m))
        .collect();
    let expected_measures: Vec<&str> = vec![
        "actions",
        "clicks",
        "cost",
        "impressions",
        "platform_conversions",
        "platform_revenue",
    ];
    assert_eq!(
        measures.keys().map(|k| (*k).to_owned()).collect::<Vec<_>>(),
        expected_measures
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
    );
    for n in expected_measures {
        let m = measures.get(n).unwrap();
        assert_eq!(m.agg, agg("sum"), "{n} agg = sum");
        assert_eq!(
            *m.data_type(),
            DataType::Decimal {
                precision: 18,
                scale: 2
            },
            "{n} data_type = decimal(18,2)"
        );
    }

    let metrics: BTreeMap<&str, &_> = model
        .metrics
        .values()
        .map(|m| (m.name().as_str(), m))
        .collect();
    let expected_blocks: BTreeMap<&str, UnresolvedSemanticExpr> = [
        (
            "cpc",
            (field("cost").expect("a name") / field("clicks").expect("a name")),
        ),
        (
            "cpm",
            ((field("cost").expect("a name") / field("impressions").expect("a name"))
                * lit_int(1000)),
        ),
        (
            "ctr",
            (field("clicks").expect("a name") / field("impressions").expect("a name")),
        ),
        (
            "cpa",
            (field("cost").expect("a name") / field("platform_conversions").expect("a name")),
        ),
        (
            "roas",
            (field("platform_revenue").expect("a name") / field("cost").expect("a name")),
        ),
        (
            "conversion_rate",
            (field("platform_conversions").expect("a name") / field("clicks").expect("a name")),
        ),
    ]
    .into_iter()
    .collect();
    assert_eq!(metrics.len(), expected_blocks.len());
    for (name, expected_tree) in &expected_blocks {
        let metric = metrics
            .get(*name)
            .unwrap_or_else(|| panic!("metric {name}"));
        assert_eq!(
            *metric.data_type(),
            DataType::Decimal {
                precision: 18,
                scale: 4
            }
        );
        assert_eq!(
            block_expr(metric.expr().unwrap()),
            expected_tree,
            "metric {name}: Block expr structural equality"
        );
    }

    let union = model
        .unionsets
        .values()
        .find(|u| u.body.base.name == "paid_media_campaign_performance")
        .unwrap();
    assert_eq!(union.body.mode, semstrait_model::UnionMode::All);
    let temporal: &TemporalShape = union.body.base.extras.temporal.as_ref().unwrap();
    assert!(temporal.granularity.is_none(), "no granularity on complex");
    match &temporal.kind {
        TemporalShapeKind::Events(b) => assert_eq!(b.event_time.as_ref(), "date"),
        other => panic!("expected Events, got {other:?}"),
    }
    let keysets: Vec<(String, Vec<String>)> = union
        .semantic_interface
        .keysets()
        .into_iter()
        .map(|(k, members)| (k, members.iter().map(|n| n.as_ref().to_owned()).collect()))
        .collect();
    assert_eq!(
        keysets,
        vec![
            (
                "natural".to_owned(),
                vec!["campaign".to_owned(), "date".to_owned()]
            ),
            (
                "platform".to_owned(),
                vec!["date".to_owned(), "export_account_id".to_owned()]
            ),
        ]
    );
    let union_dims = &union.semantic_interface.dimensions;
    let names = site_names(union_dims);
    assert_eq!(
        names,
        vec![
            "campaign",
            "country",
            "currency",
            "dataset_name",
            "date",
            "export_account_id",
            "market",
            "measurement_channel",
            "measurement_source",
            "traffic_source",
        ]
    );
    let ref_count = union_dims
        .values()
        .filter(|e| matches!(e, DimensionSite::Reference(_)))
        .count();
    assert_eq!(ref_count, 9, "9 ref entries");
    let market_expr = match union_dims
        .values()
        .find(|e| e.name().as_ref() == "market")
        .unwrap()
    {
        DimensionSite::Inline(d) => block_expr(d.body.expr().unwrap()),
        _ => panic!("market should be Inline"),
    };
    let counts = count_op_kinds(market_expr);
    assert_eq!(counts.case, 4, "4 nested case roots");
    assert_eq!(counts.in_list, 1, "one in-list (platform names)");
    assert_eq!(counts.binop_eq, 2, "two eq sugars (facebookads, impact)");
    assert_eq!(counts.regexp_like, 3, "three regexp_like calls");
    assert_eq!(counts.regexp_extract, 3, "three regexp_extract calls");
    assert_eq!(counts.upper, 3, "three upper calls");
    match market_expr {
        Expr::Case { whens, else_ } => {
            assert_eq!(whens.len(), 2, "two outer when arms");
            match &whens[0].0 {
                Expr::InList {
                    value,
                    list,
                    negated,
                } => {
                    assert!(!negated);
                    assert_eq!(value.as_ref(), &field("dataset_name").expect("a name"));
                    let expected_platforms: Vec<UnresolvedSemanticExpr> = vec![
                        lit_str("adwords"),
                        lit_str("facebookads"),
                        lit_str("bing"),
                        lit_str("tiktok"),
                        lit_str("klaviyo"),
                    ];
                    assert_eq!(list, &expected_platforms);
                }
                other => panic!("first when[0] = InList expected, got {other:?}"),
            }
            assert_eq!(
                &whens[1].0,
                &field("dataset_name").expect("a name").eq(lit_str("impact"))
            );
            let else_box = else_.as_ref().expect("else present");
            assert_eq!(else_box.as_ref(), &lit_str(""));
        }
        other => panic!("market root must be Case, got {other:?}"),
    }

    let m_refs = site_names(&union.semantic_interface.measures);
    assert_eq!(
        m_refs,
        vec![
            "actions",
            "clicks",
            "cost",
            "impressions",
            "platform_conversions",
            "platform_revenue",
        ]
    );
    for e in union.semantic_interface.measures.values() {
        assert!(matches!(e, MeasureSite::Reference(_)));
    }
    let met_refs = site_names(&union.semantic_interface.metrics);
    assert_eq!(
        met_refs,
        vec!["conversion_rate", "cpa", "cpc", "cpm", "ctr", "roas"]
    );
    for e in union.semantic_interface.metrics.values() {
        assert!(matches!(e, MetricSite::Reference(_)));
    }

    let nested = &union.body.datasets;
    assert_eq!(nested.len(), 6);
    let nested_names: Vec<String> = nested.iter().map(|d| d.body.base.name.clone()).collect();
    assert_eq!(
        nested_names,
        vec![
            "adwords_campaign_data",
            "bing_campaign_data",
            "facebook_adset_data",
            "tiktok_ad_data",
            "impact_adset_data",
            "klaviyo_adset_data",
        ]
    );
    let platforms = [
        "adwords",
        "bing",
        "facebookads",
        "tiktok",
        "impact",
        "klaviyo",
    ];
    for (d, platform) in nested.iter().zip(platforms) {
        let extras = &d.body.base.extras;
        let cat = extras.catalog.as_ref().expect(&d.body.base.name);
        assert_eq!(cat.alias, "polaris");
        let storage = extras.storage.as_ref().expect("storage");
        assert_eq!(storage.format, Some(StorageFormat::Parquet));
        assert_eq!(storage.paths, vec![base_path(platform)]);
        let t = extras.temporal.as_ref().expect("temporal");
        assert_eq!(t.granularity, Some(Granularity::Day));
        match &t.kind {
            TemporalShapeKind::Events(b) => assert_eq!(b.event_time.as_ref(), "date"),
            other => panic!(
                "nested.{}: expected Events, got {other:?}",
                d.body.base.name
            ),
        }
    }

    let adwords = nested
        .iter()
        .find(|d| d.body.base.name == "adwords_campaign_data")
        .unwrap();
    let amap = match &adwords.body.base.extras.semantic_mapping {
        SemanticMapping::Explicit(m) => m,
        other => panic!("expected Explicit mapping, got {other:?}"),
    };
    let mc = SemanticsName("measurement_channel".into());
    assert_eq!(
        amap.get(&mc).unwrap(),
        &SemanticMappingValue::Literal(Literal::String("Paid Search".into())),
        "adwords measurement_channel = lit Paid Search"
    );
    let date = SemanticsName("date".into());
    assert_eq!(
        amap.get(&date).unwrap(),
        &SemanticMappingValue::Column("date".into()),
        "adwords date = bare column 'date'"
    );
    let actions = SemanticsName("actions".into());
    assert_eq!(
        amap.get(&actions).unwrap(),
        &SemanticMappingValue::Untyped(UntypedLiteral(Literal::BigInt(0))),
        "adwords actions = untyped lit 0"
    );
    let pr = SemanticsName("platform_revenue".into());
    assert_eq!(
        amap.get(&pr).unwrap(),
        &SemanticMappingValue::Column("adwords_allConvValue".into()),
        "adwords platform_revenue = bare column 'adwords_allConvValue'"
    );

    let klaviyo = nested
        .iter()
        .find(|d| d.body.base.name == "klaviyo_adset_data")
        .unwrap();
    let kmap = match &klaviyo.body.base.extras.semantic_mapping {
        SemanticMapping::Explicit(m) => m,
        other => panic!("expected Explicit mapping, got {other:?}"),
    };
    let clicks = SemanticsName("clicks".into());
    assert_eq!(
        kmap.get(&clicks).unwrap(),
        &SemanticMappingValue::Column("klaviyo-clicks2".into()),
        "klaviyo clicks: dashed identifier survives as one Column atom"
    );

    let shopify = model
        .datasets
        .values()
        .find(|d| d.body.base.name == "shopify")
        .unwrap();
    let s_extras = &shopify.body.base.extras;
    assert_eq!(s_extras.catalog.as_ref().unwrap().alias, "polaris");
    let s_storage = s_extras.storage.as_ref().unwrap();
    assert_eq!(s_storage.format, Some(StorageFormat::Parquet));
    assert_eq!(s_storage.paths, vec![base_path("shopify")]);
    let s_t = s_extras.temporal.as_ref().unwrap();
    assert_eq!(s_t.granularity, Some(Granularity::Day));
    assert!(matches!(&s_t.kind, TemporalShapeKind::Events(_)));

    let s_dims = &shopify.semantic_interface.dimensions;
    let s_dim_names = site_names(s_dims);
    assert_eq!(
        s_dim_names,
        vec![
            "brand",
            "conversion_event",
            "country",
            "currency",
            "dataset_name",
            "date",
            "export_account_id",
            "market",
            "order_action_type",
            "sale_action_type",
            "sale_line_type",
            "sales_channel",
        ]
    );
    let s_refs: Vec<&str> = s_dims
        .values()
        .filter(|e| matches!(e, DimensionSite::Reference(_)))
        .map(|e| e.name().as_ref())
        .collect();
    assert_eq!(s_refs.len(), 5, "5 ref entries");
    assert!(s_refs.contains(&"country"), "shopify country is a ref");

    let shopify_market_expr = match shopify
        .semantic_interface
        .dimensions
        .values()
        .find(|e| e.name().as_ref() == "market")
        .unwrap()
    {
        DimensionSite::Inline(d) => block_expr(d.body.expr().unwrap()),
        _ => panic!("shopify.market should be Inline"),
    };
    let expected_shopify_market = Expr::Case {
        whens: vec![
            (
                field("country").expect("a name").eq(lit_str("Germany")),
                lit_str("DE"),
            ),
            (
                field("country").expect("a name").eq(lit_str("Spain")),
                lit_str("ES"),
            ),
            (
                field("country").expect("a name").eq(lit_str("France")),
                lit_str("FR"),
            ),
            (
                field("country")
                    .expect("a name")
                    .eq(lit_str("United Kingdom")),
                lit_str("GB"),
            ),
            (
                field("country").expect("a name").eq(lit_str("Italy")),
                lit_str("IT"),
            ),
        ],
        else_: Some(Box::new(lit_null())),
    };
    assert_eq!(
        shopify_market_expr, &expected_shopify_market,
        "shopify.market full case cascade equality"
    );

    let mut s_meas: Vec<(String, Agg, DataType)> = shopify
        .semantic_interface
        .measures
        .values()
        .map(|e| match e {
            MeasureSite::Inline(m) => (
                m.name().as_ref().to_owned(),
                m.agg,
                m.data_type().resolve().expect("a valid type"),
            ),
            MeasureSite::Reference(_) => panic!("shopify measures are inline"),
            other => panic!("unknown MeasureSite variant: {other:?}"),
        })
        .collect();
    s_meas.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        s_meas,
        vec![
            (
                "orders".to_string(),
                agg("sum"),
                DataType::Decimal {
                    precision: 18,
                    scale: 2
                }
            ),
            (
                "revenue".into(),
                agg("sum"),
                DataType::Decimal {
                    precision: 18,
                    scale: 2
                }
            ),
            (
                "tax".into(),
                agg("sum"),
                DataType::Decimal {
                    precision: 18,
                    scale: 2
                }
            ),
        ]
    );

    let smap = match &shopify.body.base.extras.semantic_mapping {
        SemanticMapping::Explicit(m) => m,
        other => panic!("expected Explicit mapping, got {other:?}"),
    };
    let s_country_key = SemanticsName("country".into());
    assert_eq!(
        smap.get(&s_country_key).unwrap(),
        &SemanticMappingValue::Column("shopify-shipping_country".into()),
        "shopify country mapping: dashed column atom"
    );
    let brand = SemanticsName("brand".into());
    assert_eq!(
        smap.get(&brand).unwrap(),
        &SemanticMappingValue::Literal(Literal::String("Acme".into()))
    );
}

#[derive(Default)]
struct OpCounts {
    case: usize,
    in_list: usize,
    binop_eq: usize,
    regexp_like: usize,
    regexp_extract: usize,
    upper: usize,
}

fn count_op_kinds(expr: &UnresolvedSemanticExpr) -> OpCounts {
    use semstrait_common::{Tree, Visitor};
    use std::ops::ControlFlow;
    struct C(OpCounts);
    impl Visitor<UnresolvedSemanticExpr> for C {
        type Output = ();
        fn f_down(&mut self, n: &UnresolvedSemanticExpr) -> ControlFlow<()> {
            match n {
                Expr::Case { .. } => self.0.case += 1,
                Expr::InList { .. } => self.0.in_list += 1,
                Expr::BinaryOp {
                    op: BinaryOpKind::Eq,
                    ..
                } => self.0.binop_eq += 1,
                Expr::Function(call) => match call.name() {
                    "regexp_like" => self.0.regexp_like += 1,
                    "regexp_extract" => self.0.regexp_extract += 1,
                    "upper" => self.0.upper += 1,
                    _ => {}
                },
                _ => {}
            }
            ControlFlow::Continue(())
        }
        fn f_up(&mut self, _n: &UnresolvedSemanticExpr) -> ControlFlow<()> {
            ControlFlow::Continue(())
        }
    }
    let mut c = C(OpCounts::default());
    let _ = expr.apply(&mut c);
    c.0
}
