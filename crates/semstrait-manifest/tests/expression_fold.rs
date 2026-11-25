use semstrait_ir::PhysicalExpr;
use semstrait_manifest::{Link, SemanticManifest};
use semstrait_model::expr::{dim, lit};
use semstrait_model::{
    Dataset, Dimension, DimensionSite, NestedDataset, SemanticMapping, SemanticModel,
    StorageFormat, Unionset, UnionsetBuilder, UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{Expr, ExprOperationExt, Literal};

mod common;
use common::*;

fn location(name: &str) -> String {
    format!("s3://bucket/{name}/")
}

fn vocabulary(bodied: Vec<Dimension>) -> impl Iterator<Item = Dimension> {
    let dataset_name = Dimension::builder("dataset_name")
        .data_type(DataType::String)
        .metadata(path_token(1))
        .build();
    std::iter::once(dataset_name).chain(bodied)
}

fn google_ads(bodied: Vec<Dimension>) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "ads".to_owned();
    let mut dataset = Dataset::builder("google_ads")
        .path(location("google_ads"))
        .format(StorageFormat::Parquet);
    for dimension in vocabulary(bodied) {
        dataset = dataset.dimension(DimensionSite::r#ref(dimension.name().as_str()));
        insert_dimension(&mut model, dimension);
    }
    insert_dataset(&mut model, dataset.build());
    model
}

fn unionset_model(
    mut stack: UnionsetBuilder,
    dimensions: impl IntoIterator<Item = Dimension>,
) -> SemanticModel {
    let mut model = SemanticModel::default();
    model.name = "ads".to_owned();
    for dimension in dimensions {
        stack = stack.dimension(DimensionSite::r#ref(dimension.name().as_str()));
        insert_dimension(&mut model, dimension);
    }
    let stack = stack.union_all().build();
    let id = stack.body.base.id.clone().expect("the mint is eager");
    model.unionsets.insert(id, stack);
    model
}

fn model(name: &str, declared: DataType, expr: UnresolvedSemanticExpr) -> SemanticModel {
    google_ads(vec![bodied(name, declared, expr)])
}

async fn stored(name: &str, declared: DataType, expr: UnresolvedSemanticExpr) -> PhysicalExpr {
    let manifest = compile_model(model(name, declared, expr)).await;
    tree(&manifest, signature(&manifest, "google_ads", name))
}

fn string(value: &str) -> Literal {
    Literal::String(value.to_owned())
}

fn is_google() -> Dimension {
    bodied(
        "is_google",
        DataType::Boolean,
        dim("dataset_name")
            .expect("a name")
            .eq(untyped(string("google"))),
    )
}

fn is_google2(declared: DataType) -> [Dimension; 2] {
    [
        bodied("ds2", declared, dim("dataset_name").expect("a name")),
        bodied(
            "is_google2",
            DataType::Boolean,
            dim("ds2").expect("a name").eq(untyped(string("google"))),
        ),
    ]
}

fn entry<'m>(manifest: &'m SemanticManifest, source: &str, semantic: &str) -> &'m Expr<Link> {
    stored_expr(
        manifest,
        signature(manifest, source, semantic).expression_id,
    )
}

fn link_to(manifest: &SemanticManifest, source: &str, semantic: &str) -> Expr<Link> {
    link(signature(manifest, source, semantic).expression_id)
}

#[tokio::test]
async fn compile_stores_each_tree_folded() {
    let cases = [
        (
            "is_google",
            DataType::Boolean,
            dim("dataset_name")
                .expect("a name")
                .eq(untyped(string("google"))),
            Literal::Boolean(false),
        ),
        (
            "total",
            DataType::Integer,
            untyped(Literal::BigInt(1)) + untyped(Literal::BigInt(2)),
            Literal::Integer(3),
        ),
    ];
    for (name, declared, expr, folded) in cases {
        assert_eq!(
            stored(name, declared, expr).await,
            Expr::Literal(folded),
            "{name}"
        );
    }
}

#[tokio::test]
async fn compile_stores_a_function_call_and_double_arithmetic_unfolded() {
    let double = |value| Expr::Literal(Literal::Double(value));
    let cases = [
        (
            "is_shouted",
            DataType::Boolean,
            call("upper", vec![dim("dataset_name").expect("a name")])
                .eq(untyped(string("GOOGLE_ADS"))),
            physical_call("upper", vec![Expr::Literal(string("google_ads"))])
                .eq(Expr::Literal(string("GOOGLE_ADS"))),
        ),
        (
            "ratio",
            DataType::Double,
            lit(Literal::Double(1.5)) + lit(Literal::Double(2.5)),
            double(1.5) + double(2.5),
        ),
    ];
    for (name, declared, expr, unfolded) in cases {
        assert_eq!(stored(name, declared, expr).await, unfolded, "{name}");
    }
}

#[tokio::test]
async fn a_link_to_a_literal_entry_folds_as_that_literal() {
    let dataset_upper = bodied(
        "dataset_upper",
        DataType::String,
        call("upper", vec![dim("dataset_name").expect("a name")]),
    );
    let [ds2, is_google2] = is_google2(DataType::String);
    let manifest = compile_model(google_ads(vec![
        is_google(),
        dataset_upper,
        ds2,
        is_google2,
    ]))
    .await;
    let entry = |semantic| entry(&manifest, "google_ads", semantic);
    let dataset_name = link_to(&manifest, "google_ads", "dataset_name");
    for is_google in ["is_google", "is_google2"] {
        assert_eq!(
            entry(is_google),
            &Expr::Literal(Literal::Boolean(false)),
            "{is_google}"
        );
    }
    assert_eq!(
        entry("dataset_upper"),
        &physical_call("upper", vec![dataset_name.clone()])
    );
    assert_eq!(entry("dataset_name"), &Expr::Literal(string("google_ads")));
    assert_eq!(entry("ds2"), &dataset_name);
}

#[tokio::test]
async fn a_linked_literal_folds_on_each_branch_of_a_unionset() {
    let bodied = || {
        let [ds2, is_google2] = is_google2(DataType::String);
        vec![is_google(), ds2, is_google2]
    };
    let branch = |name: &str| {
        NestedDataset::builder(name)
            .path(location(name))
            .format(StorageFormat::Parquet)
            .semantic_mapping(SemanticMapping::explicit(Default::default()))
            .build()
    };
    let stack = Unionset::builder("stacked")
        .dataset(branch("google_ads"))
        .dataset(branch("google"));
    let manifest = compile_model(unionset_model(stack, vocabulary(bodied()))).await;
    for (branch, is) in [("stacked.google_ads", false), ("stacked.google", true)] {
        for is_google in ["is_google", "is_google2"] {
            assert_eq!(
                entry(&manifest, branch, is_google),
                &Expr::Literal(Literal::Boolean(is)),
                "{is_google} on {branch}"
            );
        }
    }
}

#[tokio::test]
async fn a_unionset_branch_s_literal_under_a_cast_is_stored_as_the_cast_s_value() {
    let branch = |name: &str, mapping: SemanticMapping| {
        NestedDataset::builder(name)
            .table(format!("warehouse.{name}"))
            .semantic_mapping(mapping)
            .build()
    };
    let may_19 = || "2026-05-19".to_owned();
    let day = dimension("day", DataType::Date);
    let is_may_19 = bodied(
        "is_may_19",
        DataType::Boolean,
        dim("day")
            .expect("a name")
            .eq(untyped(Literal::String(may_19()))),
    );
    let campaigns = Unionset::builder("campaigns")
        .dataset(branch(
            "adwords",
            SemanticMapping::builder()
                .literal("day", Literal::String(may_19()))
                .build(),
        ))
        .dataset(branch(
            "facebook",
            SemanticMapping::builder().column("day", "day").build(),
        ));
    let manifest = compile_model(unionset_model(campaigns, [day, is_may_19])).await;
    let entry = |branch, semantic| entry(&manifest, branch, semantic);
    let date = || Expr::Literal(Literal::Date(may_19()));
    assert_eq!(entry("campaigns.adwords", "day"), &date());
    assert_eq!(
        entry("campaigns.adwords", "is_may_19"),
        &Expr::Literal(Literal::Boolean(true))
    );
    assert_eq!(
        entry("campaigns.facebook", "is_may_19"),
        &link_to(&manifest, "campaigns.facebook", "day").eq(date())
    );
}

#[tokio::test]
async fn a_cast_of_a_literal_folds_in_the_chain() {
    let varchar = DataType::VarChar { length: 20 };
    let manifest = compile_model(google_ads(is_google2(varchar).into())).await;
    let google_ads = Literal::VarChar {
        value: "google_ads".to_owned(),
        length: 20,
    };
    assert_eq!(
        entry(&manifest, "google_ads", "ds2"),
        &Expr::Literal(google_ads)
    );
    assert_eq!(
        entry(&manifest, "google_ads", "is_google2"),
        &Expr::Literal(Literal::Boolean(false))
    );
}

#[tokio::test]
async fn a_cast_that_the_canonical_cast_gives_no_value_stops_the_chain() {
    let ds2 = bodied("ds2", DataType::Date, dim("dataset_name").expect("a name"));
    let is_late = bodied(
        "is_late",
        DataType::Boolean,
        dim("ds2")
            .expect("a name")
            .gt(untyped(string("2026-01-01"))),
    );
    let manifest = compile_model(google_ads(vec![ds2, is_late])).await;
    assert_eq!(
        entry(&manifest, "google_ads", "ds2"),
        &cast(
            link_to(&manifest, "google_ads", "dataset_name"),
            DataType::Date
        )
    );
    assert_eq!(
        entry(&manifest, "google_ads", "is_late"),
        &link_to(&manifest, "google_ads", "ds2")
            .gt(Expr::Literal(Literal::Date("2026-01-01".to_owned())))
    );
}
