mod common;

use common::{
    agg, by_name, column, constituent, fixture_fs, semantic_alias, stored_tree, test_provenance,
};

use semstrait_ir::PhysicalExpr;
use semstrait_manifest::{compile, CompileError, PhysicalExprId, RoleDefinition, SemanticManifest};
use semstrait_model::{
    Dimension, DimensionSite, Integrity, Joinset, Measure, MeasureSite, Relationship, SemanticModel,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{BinaryOpKind, Expr};

const CONSTITUENTS: [&str; 3] = ["orders", "regions", "customers"];

fn joinset(name: &str, members: &[&str], relationships: Vec<Relationship>) -> Joinset {
    let mut built = Joinset::builder(name)
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"));
    for member in members {
        built = built.dataset(constituent(member, &["customer_id", "region", "revenue"]));
    }
    for relationship in relationships {
        built = built.relationship(relationship);
    }
    built.build()
}

fn model(joinsets: Vec<Joinset>, root: Vec<Relationship>) -> SemanticModel {
    let mut built = SemanticModel::builder()
        .name("sales")
        .dimension(
            Dimension::builder("customer_id")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .dimension(
            Dimension::builder("region")
                .data_type(DataType::String)
                .categorical()
                .build(),
        )
        .measure(
            Measure::builder("revenue")
                .data_type(DataType::Double)
                .agg(agg("sum"))
                .build(),
        );
    for one in joinsets {
        built = built.joinset(one);
    }
    for one in root {
        built = built.relationship(one);
    }
    built.build().expect("the fixture model must build").0
}

fn related(name: &str, from: &str, to: &str, key: &str) -> Relationship {
    Relationship::builder()
        .name(name)
        .from(from)
        .to(to)
        .on([key])
        .many_to_one()
        .build()
        .expect("the fixture relationship must build")
}

fn related_with(name: &str, from: &str, to: &str, key: &str, integrity: Integrity) -> Relationship {
    Relationship::builder()
        .name(name)
        .from(from)
        .to(to)
        .on([key])
        .many_to_one()
        .integrity(integrity)
        .build()
        .expect("the fixture relationship must build")
}

async fn compiled(model: SemanticModel) -> SemanticManifest {
    let fs = fixture_fs(&CONSTITUENTS).await;
    let (manifest, warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect("the fixture model must compile");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    manifest
}

async fn refused(model: SemanticModel, names: &[&str]) -> CompileError {
    let fs = fixture_fs(names).await;
    let (error, _) = compile(model, None, &fs, test_provenance())
        .await
        .expect_err("the fixture model must be refused");
    error.kind
}

fn names(manifest: &SemanticManifest) -> Vec<&str> {
    let mut all: Vec<&str> = manifest
        .semantic_sources
        .values()
        .map(|entry| entry.name.as_str())
        .collect();
    all.sort_unstable();
    all
}

fn units(manifest: &SemanticManifest) -> Vec<&str> {
    let mut all: Vec<&str> = manifest
        .semantic_sources
        .values()
        .filter(|entry| matches!(entry.role_definition, RoleDefinition::Joinset { .. }))
        .map(|entry| entry.name.as_str())
        .collect();
    all.sort_unstable();
    all
}

fn operands<'m>(manifest: &'m SemanticManifest, name: &str) -> (&'m str, &'m str, PhysicalExprId) {
    match &by_name(manifest, name).role_definition {
        RoleDefinition::Joinset {
            from,
            to,
            on_expr_id,
            ..
        } => (
            manifest.semantic_sources[from].name.as_str(),
            manifest.semantic_sources[to].name.as_str(),
            on_expr_id.expect("a keyed unit has a condition"),
        ),
        other => panic!("`{name}` must be a `Joinset`, got {other:?}"),
    }
}

fn equality(left: PhysicalExpr, right: PhysicalExpr) -> PhysicalExpr {
    Expr::BinaryOp {
        op: BinaryOpKind::Eq,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn conjunction(left: PhysicalExpr, right: PhysicalExpr) -> PhysicalExpr {
    Expr::BinaryOp {
        op: BinaryOpKind::And,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn equi(manifest: &SemanticManifest, from: &str, to: &str, key: &str) -> PhysicalExpr {
    equality(
        column(semantic_alias(manifest, from), key),
        column(semantic_alias(manifest, to), key),
    )
}

fn chain_model() -> SemanticModel {
    model(
        vec![joinset(
            "sales",
            &CONSTITUENTS,
            vec![
                related("r1", "orders", "customers", "customer_id"),
                related("r2", "orders", "regions", "region"),
            ],
        )],
        Vec::new(),
    )
}

#[tokio::test]
async fn a_three_constituent_chain_enumerates_three_units() {
    let manifest = compiled(chain_model()).await;
    assert_eq!(
        units(&manifest),
        vec!["sales", "sales.r1", "sales.r2"],
        "the two pairs and the whole"
    );
    assert_eq!(
        names(&manifest),
        vec![
            "sales",
            "sales.customers",
            "sales.orders",
            "sales.r1",
            "sales.r2",
            "sales.regions",
        ],
        "three constituents and three units, nothing else"
    );
    assert!(
        by_name(&manifest, "sales").public,
        "the unit applying every relationship keeps the authored `public`"
    );
    for derived in ["sales.r1", "sales.r2"] {
        let unit = by_name(&manifest, derived);
        assert!(!unit.public, "`{derived}` is private");
        assert_eq!(
            unit.originators,
            vec![by_name(&manifest, "sales").id.clone()],
            "`{derived}` is reachable only through its public origin"
        );
    }
    assert!(
        manifest.validate().is_ok(),
        "the artifact satisfies its own validator"
    );
}

#[tokio::test]
async fn a_cumulative_unit_joins_its_two_immediate_operands_on_its_own_condition() {
    let manifest = compiled(chain_model()).await;
    let (from, to, on) = operands(&manifest, "sales");
    assert_eq!(
        (from, to),
        ("sales.r1", "sales.regions"),
        "the two immediate operands, ordered by the causing relationship's direction: \
         `r2` runs from `orders`, which sits inside the left operand"
    );
    assert_eq!(
        stored_tree(&manifest, on),
        equi(&manifest, "sales.orders", "sales.regions", "region"),
        "its own condition, qualified by the constituents `r2` relates"
    );
    assert_ne!(
        semantic_alias(&manifest, "sales.orders"),
        semantic_alias(&manifest, "sales.r1"),
        "the qualifier is the constituent's, not the left operand's"
    );

    let (pair_from, pair_to, pair_on) = operands(&manifest, "sales.r1");
    assert_eq!((pair_from, pair_to), ("sales.orders", "sales.customers"));
    assert_eq!(
        stored_tree(&manifest, pair_on),
        equi(&manifest, "sales.orders", "sales.customers", "customer_id"),
    );
}

#[tokio::test]
async fn a_mandatory_relationship_removes_the_member_sets_that_detach_it() {
    let manifest = compiled(model(
        vec![joinset(
            "sales",
            &CONSTITUENTS,
            vec![
                related_with("r1", "orders", "customers", "customer_id", Integrity::None),
                related_with("r2", "orders", "regions", "region", Integrity::Enforced),
            ],
        )],
        Vec::new(),
    ))
    .await;
    assert_eq!(
        units(&manifest),
        vec!["sales", "sales.r1"],
        "`{{orders, regions}}` detaches `customers` from a mandatory relationship"
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn an_off_spine_relationship_is_an_extra_conjunct_of_one_condition() {
    let manifest = compiled(model(
        vec![joinset(
            "sales",
            &CONSTITUENTS,
            vec![
                related_with("r1", "orders", "customers", "customer_id", Integrity::None),
                related_with("r2", "orders", "regions", "region", Integrity::None),
                related_with("r3", "customers", "regions", "region", Integrity::None),
            ],
        )],
        Vec::new(),
    ))
    .await;
    assert_eq!(
        units(&manifest),
        vec!["sales", "sales.r1"],
        "one unit, plus the bracketing step it is realised through"
    );
    let (_, _, on) = operands(&manifest, "sales");
    assert_eq!(
        stored_tree(&manifest, on),
        conjunction(
            equi(&manifest, "sales.orders", "sales.regions", "region"),
            equi(&manifest, "sales.customers", "sales.regions", "region"),
        ),
        "the off-spine relationship is a conjunct, not a dropped filter"
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn two_relationships_over_one_pair_are_three_units() {
    let manifest = compiled(model(
        vec![joinset(
            "sales",
            &["orders", "customers"],
            vec![
                related("ra", "orders", "customers", "customer_id"),
                related("rb", "orders", "customers", "region"),
            ],
        )],
        Vec::new(),
    ))
    .await;
    assert_eq!(units(&manifest), vec!["sales", "sales.ra", "sales.rb"]);
    let (_, _, both) = operands(&manifest, "sales");
    assert_eq!(
        stored_tree(&manifest, both),
        conjunction(
            equi(&manifest, "sales.orders", "sales.customers", "customer_id"),
            equi(&manifest, "sales.orders", "sales.customers", "region"),
        ),
        "the unit applying both applies both"
    );
    let (_, _, only_a) = operands(&manifest, "sales.ra");
    assert_eq!(
        stored_tree(&manifest, only_a),
        equi(&manifest, "sales.orders", "sales.customers", "customer_id"),
    );
    let (_, _, only_b) = operands(&manifest, "sales.rb");
    assert_eq!(
        stored_tree(&manifest, only_b),
        equi(&manifest, "sales.orders", "sales.customers", "region"),
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn an_authored_joinset_duplicating_an_implicit_unit_is_accepted() {
    let manifest = compiled(model(
        vec![
            joinset(
                "sales",
                &CONSTITUENTS,
                vec![
                    related("r1", "orders", "customers", "customer_id"),
                    related("r2", "orders", "regions", "region"),
                ],
            ),
            joinset(
                "pair",
                &["orders", "customers"],
                vec![related("r1", "orders", "customers", "customer_id")],
            ),
        ],
        Vec::new(),
    ))
    .await;
    assert_eq!(
        units(&manifest),
        vec!["pair", "sales", "sales.r1", "sales.r2"]
    );
    assert!(
        by_name(&manifest, "pair").public,
        "the authored duplicate is public"
    );
    assert!(
        !by_name(&manifest, "sales.r1").public,
        "the implicit one it duplicates stays private"
    );
    assert_ne!(
        by_name(&manifest, "pair").id,
        by_name(&manifest, "sales.r1").id,
        "different names are different entries"
    );
    assert!(manifest.validate().is_ok());
}

#[tokio::test]
async fn a_joinset_whose_relationships_do_not_span_it_is_refused() {
    let error = refused(
        model(
            vec![joinset(
                "sales",
                &CONSTITUENTS,
                vec![related("r1", "orders", "customers", "customer_id")],
            )],
            Vec::new(),
        ),
        &CONSTITUENTS,
    )
    .await;
    match &error {
        CompileError::RelationshipResolution { relationship, .. } => {
            assert_eq!(relationship, "sales", "the Joinset is named");
        }
        other => panic!("expected RelationshipResolution, got {other:?}"),
    }
}

#[tokio::test]
async fn two_units_deriving_one_name_are_refused_rather_than_merged() {
    let members = ["a", "b", "c", "d"];
    let error = refused(
        model(
            vec![joinset(
                "sales",
                &members,
                vec![
                    related("r1", "a", "b", "customer_id"),
                    related("r2", "b", "c", "customer_id"),
                    related("r3", "a", "c", "region"),
                    related("r4", "c", "d", "customer_id"),
                ],
            )],
            Vec::new(),
        ),
        &members,
    )
    .await;
    match &error {
        CompileError::EnumeratedUnitNameCollision {
            joinset,
            derived_name,
            second,
            ..
        } => {
            assert_eq!(joinset, "sales", "the Joinset is named");
            assert_eq!(derived_name, "sales.r1.r2", "the contested name");
            assert!(
                second.contains(&"r2".to_owned()),
                "the colliding unit's relationships are named: {second:?}"
            );
        }
        other => panic!("expected EnumeratedUnitNameCollision, got {other:?}"),
    }
}

#[tokio::test]
async fn a_derived_name_an_authored_constituent_holds_is_refused() {
    let members = ["orders", "regions", "r1"];
    let error = refused(
        model(
            vec![joinset(
                "sales",
                &members,
                vec![
                    related("r1", "orders", "regions", "customer_id"),
                    related("r2", "orders", "r1", "region"),
                ],
            )],
            Vec::new(),
        ),
        &members,
    )
    .await;
    match &error {
        CompileError::EnumeratedUnitNameCollision {
            derived_name,
            first,
            ..
        } => {
            assert_eq!(derived_name, "sales.r1");
            assert_eq!(first, "sales.r1", "the constituent is the holder");
        }
        other => panic!("expected EnumeratedUnitNameCollision, got {other:?}"),
    }
}

#[tokio::test]
async fn exceeding_the_enumeration_ceiling_is_a_hard_error() {
    let members: Vec<String> = (0..6).map(|i| format!("c{i}")).collect();
    let borrowed: Vec<&str> = members.iter().map(String::as_str).collect();
    let mut relationships = Vec::new();
    for (i, left) in members.iter().enumerate() {
        for right in &members[i + 1..] {
            relationships.push(related(
                &format!("r_{left}_{right}"),
                left,
                right,
                "customer_id",
            ));
        }
    }
    let error = refused(
        model(vec![joinset("sales", &borrowed, relationships)], Vec::new()),
        &borrowed,
    )
    .await;
    match &error {
        CompileError::EnumerationCeilingExceeded { joinset, units, .. } => {
            assert_eq!(joinset, "sales", "the Joinset is named");
            assert!(
                *units > 2000,
                "the count reached is carried, and it is over the ceiling: {units}"
            );
        }
        other => panic!("expected EnumerationCeilingExceeded, got {other:?}"),
    }
}

#[tokio::test]
async fn an_alias_is_its_source_roles_letter_and_its_index_in_entity_id_order() {
    const PROBE: [&str; 3] = ["eu-orders", "eu_orders", "eu_orders_fb2f09"];
    let probe = model(
        vec![joinset(
            "sales",
            &PROBE,
            vec![
                related("r1", "eu-orders", "eu_orders_fb2f09", "customer_id"),
                related("r2", "eu_orders", "eu-orders", "customer_id"),
            ],
        )],
        Vec::new(),
    );
    let fs = fixture_fs(&PROBE).await;
    let (manifest, warnings) = compile(probe, None, &fs, test_provenance())
        .await
        .expect("ADR 0078's probe compiles");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");

    let table: Vec<(&str, &str, &str)> = manifest
        .semantic_sources
        .values()
        .map(|entry| (&entry.id[..8], entry.name.as_str(), entry.alias.as_str()))
        .collect();
    assert_eq!(
        table,
        vec![
            ("1d26bbb0", "sales.eu_orders_fb2f09", "ssd0"),
            ("2d72245f", "sales.eu_orders", "ssd1"),
            ("a1415b51", "sales.r1", "ssj2"),
            ("f855b64c", "sales.r2", "ssj3"),
            ("fb2f09ca", "sales.eu-orders", "ssd4"),
            ("fef513d0", "sales", "ssj5"),
        ],
        "ADR 0078's table, in `EntityId` order"
    );

    let (_, _, on) = operands(&manifest, "sales.r1");
    assert_eq!(
        stored_tree(&manifest, on),
        equality(column("ssd4", "customer_id"), column("ssd0", "customer_id")),
        "`r1` compares two constituents, not one column with itself"
    );
    assert!(
        manifest.validate().is_ok(),
        "the artifact satisfies its own validator"
    );
}
