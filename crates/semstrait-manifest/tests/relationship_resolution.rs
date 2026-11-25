use semstrait_common::diagnostic::Diagnostic;
use semstrait_ir::{Cardinality, JoinType, PhysicalExpr};
use semstrait_ir::{
    CrossFilter as StoredCrossFilter, Integrity as StoredIntegrity, JoinsetJoinType, SemanticRole,
};
use semstrait_manifest::{
    compile, CompileError, PhysicalExprId, RoleDefinition, SemanticBitmask, SemanticManifest,
    StoredExpr,
};
use semstrait_model::expr::{dim, measure};
use semstrait_model::{
    AiContext, CrossFilter, Dimension, DimensionSite, Integrity, Joinset, Measure, MeasureSite,
    NestedDataset, Relationship, SemanticMapping, SemanticModel, StorageFormat,
    UnresolvedSemanticExpr,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{BinaryOpKind, Expr, ExprOperationExt, Literal, NamedReference};

mod common;
use common::golden::{self, Golden};
use common::{
    agg, by_name, column, compile_diagnostic_with_on, constituent, decimal_18_2, dimension,
    fixture_fs, lag, lake_folder, link_free_tree, semantic_alias, signature, test_provenance,
    untyped,
};

fn renamed_constituent(column: &str, name: &str, names: &[&str]) -> NestedDataset {
    let mut mapping = SemanticMapping::builder().column("customer_id", column);
    for mapped in names {
        if *mapped != "customer_id" {
            mapping = mapping.column(*mapped, *mapped);
        }
    }
    NestedDataset::builder(name)
        .path(lake_folder(name))
        .format(StorageFormat::Parquet)
        .semantic_mapping(mapping.build())
        .build()
}

fn model_with(relationships: Vec<Relationship>) -> SemanticModel {
    model_over(constituent, Vec::new(), relationships)
}

fn model_over(
    constituent: fn(&str, &[&str]) -> NestedDataset,
    dimensions: Vec<Dimension>,
    relationships: Vec<Relationship>,
) -> SemanticModel {
    let mut names: Vec<&str> = vec!["customer_id", "region", "revenue"];
    names.extend(dimensions.iter().map(|dimension| dimension.name().as_str()));
    let mut joinset = Joinset::builder("sales")
        .dataset(constituent("customers", &names))
        .dataset(constituent("orders", &names))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("region"))
        .measure(MeasureSite::r#ref("revenue"));
    for dimension in &dimensions {
        joinset = joinset.dimension(DimensionSite::r#ref(dimension.name().as_str()));
    }
    for relationship in relationships {
        joinset = joinset.relationship(relationship);
    }
    SemanticModel::builder()
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
        .dimensions(dimensions)
        .measure(
            Measure::builder("revenue")
                .data_type(DataType::Double)
                .agg(agg("sum"))
                .build(),
        )
        .joinset(joinset.build())
        .build()
        .expect("the fixture model must build")
        .0
}

fn keyed_model_with(relationships: Vec<Relationship>) -> SemanticModel {
    let names = ["customer_id", "region", "revenue"];
    let mut joinset = Joinset::builder("sales")
        .dataset(constituent("customers", &names))
        .dataset(constituent("orders", &names))
        .dimension(DimensionSite::r#ref("customer_id").with_key())
        .dimension(DimensionSite::r#ref("region").with_key())
        .measure(MeasureSite::r#ref("revenue"));
    for relationship in relationships {
        joinset = joinset.relationship(relationship);
    }
    SemanticModel::builder()
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
        )
        .joinset(joinset.build())
        .build()
        .expect("the fixture model must build")
        .0
}

fn many_to_one(keys: &[&str]) -> Relationship {
    Relationship::builder()
        .name("orders_to_customers")
        .from("orders")
        .to("customers")
        .on(keys.iter().copied())
        .many_to_one()
        .build()
        .expect("the fixture relationship must build")
}

async fn compiled(model: SemanticModel) -> SemanticManifest {
    let fs = fixture_fs(&["orders", "customers"]).await;
    let (manifest, warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect("the fixture model must compile");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    manifest
}

async fn refused(model: SemanticModel) -> CompileError {
    refused_diagnostic(model).await.kind
}

async fn refused_diagnostic(model: SemanticModel) -> Diagnostic<CompileError> {
    let fs = fixture_fs(&["orders", "customers"]).await;
    compile_diagnostic_with_on(model, None, &fs).await
}

struct Join<'m> {
    from: &'m str,
    on_expr_id: Option<PhysicalExprId>,
    filter_expr_id: Option<PhysicalExprId>,
    join_type: JoinsetJoinType,
    cardinality: Cardinality,
    integrity: StoredIntegrity,
    cross_filter: StoredCrossFilter,
    description: Option<&'m str>,
}

impl Join<'_> {
    fn on(&self) -> PhysicalExprId {
        self.on_expr_id.expect("a keyed unit has a join condition")
    }
}

fn join_body<'m>(manifest: &'m SemanticManifest, name: &str) -> Join<'m> {
    match &by_name(manifest, name).role_definition {
        RoleDefinition::Joinset {
            from,
            on_expr_id,
            filter_expr_id,
            join_type,
            cardinality,
            integrity,
            cross_filter,
            description,
            ..
        } => Join {
            from,
            on_expr_id: *on_expr_id,
            filter_expr_id: *filter_expr_id,
            join_type: *join_type,
            cardinality: *cardinality,
            integrity: *integrity,
            cross_filter: *cross_filter,
            description: description.as_deref(),
        },
        other => panic!("`{name}` must be a `Joinset`, got {other:?}"),
    }
}

fn condition(manifest: &SemanticManifest, id: PhysicalExprId) -> PhysicalExpr {
    let entry = manifest
        .expressions
        .get(&id)
        .expect("a join condition's handle resolves in the store");
    assert!(
        entry.physical_sources.is_empty(),
        "a join condition names no physical source; got {:?}",
        entry.physical_sources
    );
    link_free_tree(manifest, id)
}

fn equality(left: PhysicalExpr, right: PhysicalExpr) -> PhysicalExpr {
    Expr::BinaryOp {
        op: BinaryOpKind::Eq,
        left: Box::new(left),
        right: Box::new(right),
    }
}

#[tokio::test]
async fn a_condition_leaf_names_its_constituents_semantics() {
    let relationship = Relationship::builder()
        .name("orders_to_customers")
        .from("orders")
        .to("customers")
        .on(["customer_id"])
        .many_to_one()
        .filter(
            dim("customer_id")
                .expect("a name")
                .eq(untyped(Literal::String("c1".to_owned()))),
        )
        .build()
        .expect("the fixture relationship must build");
    let manifest = compiled(model_over(
        |name, names| renamed_constituent("cust_key", name, names),
        Vec::new(),
        vec![relationship],
    ))
    .await;
    assert!(
        manifest.validate().is_ok(),
        "{:?}",
        manifest.validate().err()
    );
    let join = join_body(&manifest, "sales");
    let orders = semantic_alias(&manifest, "sales.orders");
    let customers = semantic_alias(&manifest, "sales.customers");
    assert_eq!(
        condition(&manifest, join.on()),
        equality(
            column(orders, "customer_id"),
            column(customers, "customer_id")
        ),
        "each side compares the semantics its constituent outputs"
    );
    let filter = join
        .filter_expr_id
        .expect("the relationship filter is stored");
    assert_eq!(
        condition(&manifest, filter),
        equality(
            column(orders, "customer_id"),
            Expr::Literal(Literal::String("c1".to_owned()))
        ),
        "both sides serve `customer_id`, so the leaf reads `from`"
    );
}

#[tokio::test]
async fn validate_refuses_a_compiled_condition_leaf_its_constituent_does_not_serve() {
    let mut manifest = compiled(model_with(vec![many_to_one(&["customer_id"])])).await;
    let on_expr_id = join_body(&manifest, "sales").on();
    let dropped = signature(&manifest, "sales.customers", "customer_id").clone();
    let bit = (manifest.semantic_definitions.by_name("customer_id"))
        .expect("the fixture declares `customer_id`")
        .bit_position;
    let customers = by_name(&manifest, "sales.customers").id.clone();
    let entry = (manifest.semantic_sources.get_mut(&customers))
        .expect("the compiled artifact holds `sales.customers`");
    entry.semantic_signature_ids.retain(|id| *id != dropped.id);
    entry.semantic_bitmask = SemanticBitmask::from_positions(
        (entry.semantic_bitmask.set_positions().into_iter()).filter(|position| *position != bit),
    );
    let leaf = NamedReference::qualified(entry.alias.as_str(), "customer_id").expect("a name");
    manifest.semantic_signatures.remove(&dropped.id);
    manifest.expressions.physical.remove(&dropped.expression_id);

    let faults = manifest
        .validate()
        .expect_err("the mutated artifact's condition reads an unserved semantics");
    assert_eq!(faults.len(), 1, "expected exactly one fault: {faults:?}");
    assert!(
        matches!(
            &faults[0].kind,
            CompileError::UnservedConditionLeaf { expression, column, semantic_source }
                if *expression == on_expr_id && *column == leaf && *semantic_source == customers
        ),
        "expected UnservedConditionLeaf, got {:?}",
        faults[0]
    );
}

#[tokio::test]
async fn validate_compares_a_condition_leaf_with_its_served_semantics_by_one_case_fold() {
    let greek = Dimension::builder("οδος")
        .data_type(DataType::String)
        .categorical()
        .build();
    let manifest = compiled(model_over(
        constituent,
        vec![greek],
        vec![many_to_one(&["οδος"])],
    ))
    .await;
    let on_expr_id = join_body(&manifest, "sales").on();
    let orders = semantic_alias(&manifest, "sales.orders");
    let customers = semantic_alias(&manifest, "sales.customers");
    assert_eq!(
        condition(&manifest, on_expr_id),
        equality(column(orders, "οδος"), column(customers, "οδος"))
    );
    let customers_id = by_name(&manifest, "sales.customers").id.clone();
    let left = NamedReference::qualified(orders, "οδος").expect("a name");

    for (stored, served) in [("ΟΔΟΣ", false), ("Οδος", true)] {
        let mut edited = manifest.clone();
        let leaf = NamedReference::qualified(customers, stored).expect("a name");
        let entry =
            (edited.expressions.physical.get_mut(&on_expr_id)).expect("the condition is stored");
        entry.expr = StoredExpr::from(Expr::BinaryOp {
            op: BinaryOpKind::Eq,
            left: Box::new(Expr::Column(left.clone())),
            right: Box::new(Expr::Column(leaf.clone())),
        });
        let faults: Vec<CompileError> = match edited.validate() {
            Ok(_) => Vec::new(),
            Err(faults) => faults.into_iter().map(|fault| fault.kind).collect(),
        };
        let expected = (!served).then(|| CompileError::UnservedConditionLeaf {
            expression: on_expr_id,
            column: leaf,
            semantic_source: customers_id.clone(),
        });
        assert_eq!(faults, Vec::from_iter(expected), "`{stored}`");
    }
}

#[tokio::test]
async fn a_declared_basis_qualifies_each_leaf_by_its_constituent() {
    let manifest = compiled(model_with(vec![many_to_one(&["customer_id"])])).await;
    let join = join_body(&manifest, "sales");
    let left = semantic_alias(&manifest, "sales.orders");
    let right = semantic_alias(&manifest, "sales.customers");
    assert_ne!(left, right, "two constituents hold two aliases");
    assert_eq!(
        condition(&manifest, join.on()),
        equality(column(left, "customer_id"), column(right, "customer_id")),
        "the `from` side is the left leaf and the `to` side the right one"
    );
    assert_eq!(
        join.filter_expr_id, None,
        "a basis with no authored `filter:` leaves no relationship filter"
    );
}

#[tokio::test]
async fn a_physical_column_named_with_a_dot_is_qualified_whole() {
    let manifest = compiled(model_over(
        |name, names| renamed_constituent("a.b", name, names),
        Vec::new(),
        vec![many_to_one(&["customer_id"])],
    ))
    .await;
    let join = join_body(&manifest, "sales");
    assert_eq!(
        condition(&manifest, join.on()),
        equality(
            column(semantic_alias(&manifest, "sales.orders"), "customer_id"),
            column(semantic_alias(&manifest, "sales.customers"), "customer_id"),
        ),
    );
    for dataset in ["sales.orders", "sales.customers"] {
        let realisation = signature(&manifest, dataset, "customer_id").expression_id;
        let names: Vec<&str> = manifest
            .expressions
            .columns(&realisation)
            .into_iter()
            .map(|(_, column)| column.name())
            .collect();
        assert_eq!(
            names,
            ["a.b"],
            "`{dataset}`'s leaf's last part is the column"
        );
    }
}

#[tokio::test]
async fn a_declared_conjunction_is_left_deep_in_authored_order() {
    let manifest = compiled(model_with(vec![many_to_one(&["region", "customer_id"])])).await;
    let join = join_body(&manifest, "sales");
    let left = semantic_alias(&manifest, "sales.orders");
    let right = semantic_alias(&manifest, "sales.customers");
    let expected = Expr::BinaryOp {
        op: BinaryOpKind::And,
        left: Box::new(equality(column(left, "region"), column(right, "region"))),
        right: Box::new(equality(
            column(left, "customer_id"),
            column(right, "customer_id"),
        )),
    };
    assert_eq!(
        condition(&manifest, join.on()),
        expected,
        "left-deep, in the order the author wrote"
    );
}

#[tokio::test]
async fn a_join_condition_over_a_linked_dimension_holds_no_link() {
    let joinset = Joinset::builder("sales")
        .dataset(constituent("customers", &["customer_id"]))
        .dataset(constituent("orders", &["customer_id"]))
        .dimension(DimensionSite::r#ref("customer_id"))
        .dimension(DimensionSite::r#ref("customer_key"))
        .relationship(many_to_one(&["customer_key"]))
        .build();
    let string_dimension = |name: &str| {
        Dimension::builder(name)
            .data_type(DataType::String)
            .categorical()
    };
    let model = SemanticModel::builder()
        .name("sales")
        .dimension(string_dimension("customer_id").build())
        .dimension(
            string_dimension("customer_key")
                .expr(dim("customer_id").expect("a name"))
                .build(),
        )
        .joinset(joinset)
        .build()
        .expect("the fixture model must build")
        .0;
    let manifest = compiled(model).await;
    assert!(
        manifest.validate().is_ok(),
        "{:?}",
        manifest.validate().err()
    );
    let join = join_body(&manifest, "sales");
    assert_eq!(
        condition(&manifest, join.on()),
        equality(
            column(semantic_alias(&manifest, "sales.orders"), "customer_key"),
            column(semantic_alias(&manifest, "sales.customers"), "customer_key"),
        )
    );
}

#[tokio::test]
async fn a_derived_conjunction_is_in_semantics_name_order() {
    let derived = Relationship::builder()
        .name("orders_to_customers")
        .from("orders")
        .to("customers")
        .on_auto()
        .many_to_one()
        .build()
        .expect("an `on: auto` relationship builds");
    let manifest = compiled(keyed_model_with(vec![derived])).await;
    let join = join_body(&manifest, "sales");
    let left = semantic_alias(&manifest, "sales.orders");
    let right = semantic_alias(&manifest, "sales.customers");
    let expected = Expr::BinaryOp {
        op: BinaryOpKind::And,
        left: Box::new(equality(
            column(left, "customer_id"),
            column(right, "customer_id"),
        )),
        right: Box::new(equality(column(left, "region"), column(right, "region"))),
    };
    assert_eq!(condition(&manifest, join.on()), expected);
}

#[tokio::test]
async fn a_derivation_that_finds_nothing_fails_rather_than_crossing() {
    let mut joinset = Joinset::builder("sales")
        .dataset(constituent("customers", &["revenue"]))
        .dataset(constituent("orders", &["revenue"]))
        .measure(MeasureSite::r#ref("revenue"));
    joinset = joinset.relationship(
        Relationship::builder()
            .name("orders_to_customers")
            .from("orders")
            .to("customers")
            .on_auto()
            .many_to_one()
            .build()
            .expect("an `on: auto` relationship builds"),
    );
    let model = SemanticModel::builder()
        .name("sales")
        .measure(
            Measure::builder("revenue")
                .data_type(DataType::Double)
                .agg(agg("sum"))
                .build(),
        )
        .joinset(joinset.build())
        .build()
        .expect("the fixture model must build")
        .0;

    match refused(model).await {
        CompileError::RelationshipResolution { relationship, .. } => {
            assert_eq!(relationship, "orders_to_customers");
        }
        other => panic!("expected RelationshipResolution, got {other:?}"),
    }
}

fn theta(filter: UnresolvedSemanticExpr) -> Relationship {
    Relationship::builder()
        .name("orders_to_customers")
        .from("orders")
        .to("customers")
        .on_cross()
        .many_to_many()
        .optional(semstrait_model::Optional::None)
        .cross_filter(CrossFilter::None)
        .filter(filter)
        .build()
        .expect("an `on: cross` relationship with a filter builds")
}

#[tokio::test]
async fn a_cross_basis_stores_a_cross_join_with_no_condition() {
    let filter = dim("region")
        .expect("a name")
        .eq(untyped(Literal::String("EU".to_owned())));
    let manifest = compiled(model_with(vec![theta(filter)])).await;
    assert!(
        manifest.validate().is_ok(),
        "{:?}",
        manifest.validate().err()
    );
    let stored = serde_json::to_value(&by_name(&manifest, "sales").role_definition)
        .expect("a role body serializes");
    assert_eq!(stored["content"]["join_type"], "cross", "{stored}");
    assert!(stored["content"]["on_expr_id"].is_null(), "{stored}");
    let join = join_body(&manifest, "sales");
    assert_eq!(join.cardinality, Cardinality::ManyToMany);
    let filter = join
        .filter_expr_id
        .expect("the relationship filter is stored");
    assert_eq!(
        condition(&manifest, filter),
        equality(
            column(semantic_alias(&manifest, "sales.orders"), "region"),
            Expr::Literal(Literal::String("EU".to_owned()))
        ),
        "the filter's leaf reads the semantics `from` serves"
    );
}

#[tokio::test]
async fn a_cross_basis_beside_a_keyed_one_leaves_the_keyed_join() {
    let crossed = Relationship::builder()
        .name("customers_cross_orders")
        .from("customers")
        .to("orders")
        .on_cross()
        .build()
        .expect("an `on: cross` relationship builds");
    let keyed = Relationship::builder()
        .name("orders_to_customers")
        .from("orders")
        .to("customers")
        .on(["customer_id"])
        .many_to_one()
        .optional(semstrait_model::Optional::Left)
        .build()
        .expect("a keyed relationship builds");
    let manifest = compiled(model_with(vec![crossed, keyed])).await;
    assert!(
        manifest.validate().is_ok(),
        "{:?}",
        manifest.validate().err()
    );
    let join = join_body(&manifest, "sales");
    assert_eq!(join.join_type, JoinsetJoinType::Join(JoinType::Left));
    assert_eq!(join.cardinality, Cardinality::ManyToOne);
    assert_eq!(
        join.from,
        by_name(&manifest, "sales.orders").id,
        "`Left` preserves `orders`"
    );
    assert_eq!(
        condition(&manifest, join.on()),
        equality(
            column(semantic_alias(&manifest, "sales.orders"), "customer_id"),
            column(semantic_alias(&manifest, "sales.customers"), "customer_id"),
        ),
        "the cross basis adds no term"
    );
}

#[tokio::test]
async fn the_name_least_keyed_relationship_plans_a_unit_beside_a_cross() {
    let keyed = |name: &str, key: &str, optional| {
        Relationship::builder()
            .name(name)
            .from("orders")
            .to("customers")
            .on([key])
            .many_to_one()
            .optional(optional)
            .integrity(Integrity::None)
            .build()
            .expect("a keyed relationship builds")
    };
    let keyed_pair = || {
        vec![
            keyed("z_keyed", "region", semstrait_model::Optional::None),
            keyed("b_keyed", "customer_id", semstrait_model::Optional::Left),
        ]
    };
    let crossed = Relationship::builder()
        .name("a_cross")
        .from("orders")
        .to("customers")
        .on_cross()
        .integrity(Integrity::None)
        .build()
        .expect("an `on: cross` relationship builds");
    let mut with_cross = keyed_pair();
    with_cross.push(crossed);
    for (case, relationships) in [("alone", keyed_pair()), ("beside the cross", with_cross)] {
        let manifest = compiled(model_with(relationships)).await;
        assert_eq!(
            join_body(&manifest, "sales").join_type,
            JoinsetJoinType::Join(JoinType::Left),
            "`b_keyed` plans the unit {case}"
        );
    }
}

#[tokio::test]
async fn a_measure_named_as_a_relationship_filter_leaf_is_refused() {
    measure_in_relationship_filter_refusal()
        .await
        .assert_committed();
}

async fn measure_in_relationship_filter_refusal() -> Golden {
    let theta = theta(measure("revenue").expect("a name"));
    let diagnostic = refused_diagnostic(model_with(vec![theta])).await;
    match &diagnostic.kind {
        CompileError::MeasureOrMetricInRelationshipFilter {
            relationship,
            semantics,
            role,
        } => {
            assert_eq!(relationship, "orders_to_customers");
            assert_eq!(semantics, "revenue");
            assert_eq!(*role, SemanticRole::Measure);
        }
        other => panic!("expected MeasureOrMetricInRelationshipFilter, got {other:?}"),
    }
    Golden::message(
        "a_measure_named_as_a_relationship_filter_leaf_is_refused",
        diagnostic.to_string(),
    )
}

#[tokio::test]
async fn a_relationship_filter_is_typed_as_a_boolean_condition() {
    match refused(model_with(vec![theta(dim("region").expect("a name"))])).await {
        CompileError::TypeInferenceFailure { node, reason } => {
            assert!(
                node.contains("orders_to_customers"),
                "the relationship is named: {node}"
            );
            assert!(
                reason.contains("Boolean"),
                "the reason names Boolean: {reason}"
            );
        }
        other => panic!("expected the non-Boolean relationship filter's refusal, got {other:?}"),
    }

    let manifest = compiled(model_with(vec![theta(untyped(Literal::Null(
        DataType::String,
    )))]))
    .await;
    let filter = join_body(&manifest, "sales")
        .filter_expr_id
        .expect("the join keeps the relationship filter");
    assert_eq!(
        condition(&manifest, filter),
        Expr::Literal(Literal::Null(DataType::Boolean))
    );
}

#[tokio::test]
async fn a_relationship_filter_with_no_type_of_the_number_family_is_refused() {
    let scored = vec![
        dimension("score", DataType::Double),
        dimension("price", decimal_18_2()),
    ];
    let model = model_over(
        constituent,
        scored,
        vec![theta(
            dim("score").expect("a name") * dim("price").expect("a name"),
        )],
    );
    match refused(model).await {
        CompileError::TypeInferenceFailure { node, reason } => {
            assert_eq!(node, "orders_to_customers");
            assert_eq!(
                reason,
                "expected Boolean, found a value of the number family with no type"
            );
        }
        other => panic!("expected the relationship filter's refusal, got {other:?}"),
    }
}

#[tokio::test]
async fn an_authored_window_call_in_a_relationship_filter_is_refused() {
    let lagged = theta(
        lag(
            dim("region").expect("a name"),
            dim("customer_id").expect("a name"),
        )
        .eq(dim("region").expect("a name")),
    );
    match refused(model_with(vec![lagged])).await {
        CompileError::TypeInferenceFailure { node, reason } => {
            assert!(
                node.contains("orders_to_customers"),
                "the relationship is named: {node}"
            );
            assert!(
                reason.contains("window"),
                "the reason names the window: {reason}"
            );
        }
        other => panic!("expected the authored window's refusal, got {other:?}"),
    }
}

#[tokio::test]
async fn the_four_planning_fields_carry_the_resolved_defaults() {
    let manifest = compiled(model_with(vec![many_to_one(&["customer_id"])])).await;
    let join = join_body(&manifest, "sales");
    assert_eq!(join.join_type, JoinsetJoinType::Join(JoinType::Inner));
    assert_eq!(join.cardinality, Cardinality::ManyToOne);
    assert_eq!(join.integrity, StoredIntegrity::Assumed);
    assert_eq!(join.cross_filter, StoredCrossFilter::Left);
    assert_eq!(
        join.description, None,
        "no authored prose, so the straight unit carries none"
    );
}

#[tokio::test]
async fn a_straight_unit_carries_its_one_relationships_prose() {
    let documented = Relationship::builder()
        .name("orders_to_customers")
        .from("orders")
        .to("customers")
        .on(["customer_id"])
        .many_to_one()
        .description("every order belongs to one customer")
        .ai_context(
            AiContext::builder()
                .description("ask about this join")
                .build(),
        )
        .build()
        .expect("a documented relationship builds");
    let manifest = compiled(model_with(vec![documented])).await;
    assert_eq!(
        join_body(&manifest, "sales").description,
        Some("every order belongs to one customer")
    );
    let stored =
        serde_json::to_value(&by_name(&manifest, "sales").role_definition).expect("serialize");
    assert_eq!(
        stored["content"]["ai_context"]["description"], "ask about this join",
        "the manifest's own `AiContext` reaches a semantic source's artifact: {stored}"
    );
}

#[tokio::test]
async fn a_resolved_many_to_many_pair_may_not_carry_a_directional_cross_filter() {
    let crossed = Relationship::builder()
        .name("orders_to_customers")
        .from("orders")
        .to("customers")
        .on_cross()
        .cross_filter(CrossFilter::Right)
        .build()
        .expect("an `on: cross` relationship with no cardinality builds");
    match refused(model_with(vec![crossed])).await {
        CompileError::RelationshipManyToManyCrossFilterDirectional {
            relationship,
            cardinality,
            cross_filter,
        } => {
            assert_eq!(relationship, "orders_to_customers");
            assert_eq!(cardinality, Cardinality::ManyToMany);
            assert_eq!(cross_filter, StoredCrossFilter::Right);
        }
        other => panic!("expected RelationshipManyToManyCrossFilterDirectional, got {other:?}"),
    }
}

#[tokio::test]
async fn an_omitted_cardinality_under_a_non_cross_basis_is_refused() {
    let mut model = model_with(vec![many_to_one(&["customer_id"])]);
    for joinset in model.joinsets.values_mut() {
        for relationship in &mut joinset.body.relationships {
            relationship.cardinality = None;
        }
    }
    match refused(model).await {
        CompileError::RelationshipMissingCardinality { relationship } => {
            assert_eq!(relationship, "orders_to_customers");
        }
        other => panic!("expected RelationshipMissingCardinality, got {other:?}"),
    }
}

#[tokio::test]
async fn a_declared_key_no_side_exposes_is_refused() {
    match refused(model_with(vec![many_to_one(&["nowhere"])])).await {
        CompileError::RelationshipJoinKeyUnknownOnSide {
            relationship, key, ..
        } => {
            assert_eq!(relationship, "orders_to_customers");
            assert_eq!(key, "nowhere");
        }
        other => panic!("expected RelationshipJoinKeyUnknownOnSide, got {other:?}"),
    }
}

#[tokio::test]
async fn a_declared_key_that_is_not_a_dimension_is_refused() {
    match refused(model_with(vec![many_to_one(&["revenue"])])).await {
        CompileError::RelationshipJoinKeyNotADimension {
            relationship, key, ..
        } => {
            assert_eq!(relationship, "orders_to_customers");
            assert_eq!(key, "revenue");
        }
        other => panic!("expected RelationshipJoinKeyNotADimension, got {other:?}"),
    }
}

#[tokio::test]
async fn generate_goldens() {
    golden::generate(async { vec![measure_in_relationship_filter_refusal().await] }).await;
}
