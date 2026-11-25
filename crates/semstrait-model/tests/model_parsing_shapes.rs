use semstrait_model::expr::field;
use semstrait_model::{parse, DimensionSite, SemanticMappingValue, UnresolvedSemanticExpr};
use semstrait_sql::expr::{Expr, Literal, SemanticLeaf, UntypedLiteral};

const FIXTURE: &str = include_str!("../../../test_data/acme_eu_ad_platform_v3.yaml");

fn shape_tags(expr: &UnresolvedSemanticExpr) -> Vec<String> {
    use semstrait_common::Visitor;
    use std::ops::ControlFlow;
    struct Tags(Vec<String>);
    impl Visitor<UnresolvedSemanticExpr> for Tags {
        type Output = ();
        fn f_down(&mut self, n: &UnresolvedSemanticExpr) -> ControlFlow<()> {
            self.0.push(match n {
                Expr::Literal(_) | Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(_))) => {
                    "Literal".into()
                }
                Expr::Leaf(SemanticLeaf::Field(_)) => "Field".into(),
                Expr::Column(_) => "Column".into(),
                Expr::Leaf(SemanticLeaf::Dimension { .. }) => "Dimension".into(),
                Expr::Leaf(SemanticLeaf::Measure { .. }) => "Measure".into(),
                Expr::Leaf(SemanticLeaf::Metric { .. }) => "Metric".into(),
                Expr::Leaf(_) => "Leaf?".into(),
                Expr::BinaryOp { op, .. } => format!("BinaryOp({op:?})"),
                Expr::UnaryOp { op, .. } => format!("UnaryOp({op:?})"),
                Expr::Function(call) => format!("Function({})", call.name()),
                Expr::Cast { .. } => "Cast".into(),
                Expr::Case { .. } => "Case".into(),
                Expr::InList { negated, .. } => format!("InList(neg={negated})"),
                Expr::Between { .. } => "Between".into(),
                Expr::Like { kind, .. } => format!("Like({kind:?})"),
                Expr::IsNull(_) => "IsNull".into(),
                _ => "?".into(),
            });
            ControlFlow::Continue(())
        }
        fn f_up(&mut self, _n: &UnresolvedSemanticExpr) -> ControlFlow<()> {
            ControlFlow::Continue(())
        }
    }
    let mut tags = Tags(Vec::new());
    use semstrait_common::Tree;
    let _ = expr.apply(&mut tags);
    tags.0
}

#[test]
fn model_parsing_produces_expected_shapes() {
    let builder = match parse(FIXTURE) {
        Ok(b) => b,
        Err(diags) => {
            for d in &diags {
                eprintln!("PARSE: {:?}", d);
            }
            panic!("model fixture failed to parse");
        }
    };
    let (model, build_diags) = match builder.build() {
        Ok(pair) => pair,
        Err(diags) => {
            for d in &diags {
                eprintln!("BUILD: {:?}", d);
            }
            panic!("model fixture failed to build");
        }
    };
    assert!(
        build_diags.is_empty(),
        "expected zero build diagnostics; got: {build_diags:?}"
    );

    assert_eq!(model.name, "acme-eu-ad-platforms");
    assert!(model
        .unionsets
        .values()
        .any(|u| u.body.base.name == "paid_media_campaign_performance"));
    assert!(model
        .datasets
        .values()
        .any(|d| d.body.base.name == "shopify"));
    assert_eq!(model.dimensions.len(), 9);
    assert_eq!(model.measures.len(), 6);
    assert_eq!(model.metrics.len(), 6);

    let measurement_source = model
        .dimensions
        .values()
        .find(|d| d.name().as_str() == "measurement_source")
        .expect("measurement_source dimension present");
    let inner = measurement_source
        .body
        .body
        .expr
        .as_ref()
        .expect("has expr");
    let Expr::Function(call) = inner else {
        panic!("expected a call of `concat`, got {inner:?}");
    };
    assert_eq!(call.name(), "concat", "concat: → concat");
    assert_eq!(
        call.args(),
        [
            field("measurement_channel").expect("a name"),
            Expr::Leaf(SemanticLeaf::Untyped(UntypedLiteral(Literal::String(
                " - ".into()
            )))),
            field("traffic_source").expect("a name"),
        ]
    );

    let cpc = model
        .metrics
        .values()
        .find(|m| m.name().as_str() == "cpc")
        .expect("cpc metric present");
    let cpc_expr = cpc.expr().expect("cpc has expr");
    assert_eq!(
        cpc_expr,
        &(field("cost").expect("a name") / field("clicks").expect("a name")),
        "expected divide(cost, clicks) for cpc"
    );

    let unionset = model
        .unionsets
        .values()
        .find(|u| u.body.base.name == "paid_media_campaign_performance")
        .expect("unionset present");
    let market_entry = unionset
        .semantic_interface
        .dimensions
        .values()
        .find(|d| d.name().as_str() == "market")
        .expect("market dim attached to unionset");
    let market_expr = match market_entry {
        DimensionSite::Inline(d) => d.body.expr().expect("has expr"),
        other => panic!("computed `market` should be inline, got {other:?}"),
    };
    assert!(
        matches!(market_expr, Expr::Case { .. }),
        "market root = Case"
    );
    let tags = shape_tags(market_expr);
    assert!(tags.iter().any(|t| t.starts_with("Case")), "Case present");
    assert!(
        tags.iter().any(|t| t == "BinaryOp(Eq)"),
        "eq: sugar → BinaryOp(Eq); tags={tags:?}"
    );
    assert!(
        tags.iter().any(|t| t == "InList(neg=false)"),
        "in: sugar → InList(negated=false); tags={tags:?}"
    );
    for function in ["upper", "regexp_like", "regexp_extract"] {
        assert!(
            tags.iter().any(|t| *t == format!("Function({function})")),
            "{function}: → a call of {function}; tags={tags:?}"
        );
    }
    assert!(
        tags.iter().any(|t| t == "Field") && !tags.iter().any(|t| t == "Column"),
        "market reads semantics names; tags={tags:?}"
    );

    let shopify = model
        .datasets
        .values()
        .find(|d| d.body.base.name == "shopify")
        .expect("shopify present");
    let shopify_market = shopify
        .semantic_interface
        .dimensions
        .values()
        .find(|d| d.name().as_str() == "market")
        .expect("shopify.market attached");
    let shopify_market_expr = match shopify_market {
        DimensionSite::Inline(d) => d.body.expr().expect("has expr"),
        other => panic!("shopify.market should be inline, got {other:?}"),
    };
    let stags = shape_tags(shopify_market_expr);
    assert!(
        stags
            .iter()
            .filter(|t| t.as_str() == "BinaryOp(Eq)")
            .count()
            == 5,
        "shopify.market has 5 country branches; tags={stags:?}"
    );

    let adwords = unionset
        .body
        .datasets
        .iter()
        .find(|d| d.body.base.name == "adwords_campaign_data")
        .expect("adwords nested dataset present");
    let mapping = match &adwords.body.base.extras.semantic_mapping {
        semstrait_model::SemanticMapping::Explicit(m) => m,
        other => panic!("expected Explicit mapping, got {other:?}"),
    };
    let date_v = mapping
        .get(&semstrait_common::SemanticsName("date".into()))
        .expect("date mapping present");
    assert!(matches!(date_v, SemanticMappingValue::Column(s) if s == "date"));
    let mc_v = mapping
        .get(&semstrait_common::SemanticsName(
            "measurement_channel".into(),
        ))
        .expect("measurement_channel mapping present");
    assert!(
        matches!(
            mc_v,
            SemanticMappingValue::Literal(Literal::String(s)) if s == "Paid Search"
        ),
        "expected Literal(String('Paid Search')), got {mc_v:?}"
    );
    let actions_v = mapping
        .get(&semstrait_common::SemanticsName("actions".into()))
        .expect("actions mapping present");
    assert!(
        matches!(
            actions_v,
            SemanticMappingValue::Untyped(UntypedLiteral(Literal::BigInt(0)))
        ),
        "expected Untyped(BigInt(0)), got {actions_v:?}"
    );

    let cost = model
        .measures
        .values()
        .find(|m| m.name().as_str() == "cost")
        .expect("cost measure present");
    assert_eq!(
        cost.agg,
        semstrait_model::Agg::of_name("sum").expect("a catalog aggregate")
    );

    assert_eq!(unionset.body.datasets.len(), 6);
}
