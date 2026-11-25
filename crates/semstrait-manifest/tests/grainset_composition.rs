mod common;

use common::{
    agg, by_name, column, fixture_fs, identity_mapping, lake_folder, names, semantic_alias,
    stored_tree, test_provenance,
};
use std::collections::BTreeMap;

use semstrait_common::Granularity;
use semstrait_ir::TemporalType;
use semstrait_manifest::identity::mint_keyset_id;
use semstrait_manifest::{compile, CompileError, RoleDefinition, SemanticManifest};
use semstrait_model::{
    Dimension, DimensionSite, Grainset, Measure, MeasureSite, NestedDataset, ScdType,
    SemanticModel, StorageFormat, TemporalShape,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::ExprOperationExt;

const NAMES: [&str; 5] = ["region", "order_ts", "valid_from", "valid_to", "revenue"];

fn grain(name: &str, shape: Option<TemporalShape>) -> NestedDataset {
    let mut built = NestedDataset::builder(name)
        .path(lake_folder(name))
        .format(StorageFormat::Parquet)
        .semantic_mapping(identity_mapping(&NAMES));
    if let Some(shape) = shape {
        built = built.temporal(shape);
    }
    built.build()
}

fn daily(name: &str, at: Granularity) -> NestedDataset {
    grain(name, Some(TemporalShape::events("order_ts", Some(at))))
}

fn grainset(
    name: &str,
    grains: Vec<NestedDataset>,
    keyed: bool,
    own: Option<TemporalShape>,
) -> Grainset {
    let region = if keyed {
        DimensionSite::r#ref("region").with_key()
    } else {
        DimensionSite::r#ref("region")
    };
    let mut built = Grainset::builder(name)
        .dimension(region)
        .dimension(DimensionSite::r#ref("order_ts"))
        .dimension(DimensionSite::r#ref("valid_from"))
        .dimension(DimensionSite::r#ref("valid_to"))
        .measure(MeasureSite::r#ref("revenue"));
    for one in grains {
        built = built.dataset(one);
    }
    if let Some(shape) = own {
        built = built.temporal(shape);
    }
    built.build()
}

fn model(grainsets: Vec<Grainset>) -> SemanticModel {
    let mut built = SemanticModel::builder().name("sales");
    for axis in ["order_ts", "valid_from", "valid_to"] {
        built = built.dimension(
            Dimension::builder(axis)
                .data_type(DataType::Date)
                .categorical()
                .build(),
        );
    }
    built = built
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
    for one in grainsets {
        built = built.grainset(one);
    }
    built.build().expect("the fixture model must build").0
}

async fn compiled(model: SemanticModel, files: &[&str]) -> SemanticManifest {
    let fs = fixture_fs(files).await;
    let (manifest, warnings) = compile(model, None, &fs, test_provenance())
        .await
        .expect("the fixture model must compile");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    assert!(
        manifest.validate().is_ok(),
        "compile must not emit an artifact its own validator refuses: {:?}",
        manifest.validate()
    );
    manifest
}

async fn refused(model: SemanticModel, files: &[&str]) -> CompileError {
    let fs = fixture_fs(files).await;
    let (error, _) = compile(model, None, &fs, test_provenance())
        .await
        .expect_err("the fixture model must be refused");
    error.kind
}

fn semantics(manifest: &SemanticManifest, name: &str) -> semstrait_common::EntityId {
    manifest
        .semantic_definitions
        .by_name(name)
        .unwrap_or_else(|| panic!("`{name}` must be registered"))
        .id
        .clone()
}

#[tokio::test]
async fn each_grain_is_its_own_entry_referenced_by_the_grainset_arm() {
    let manifest = compiled(
        model(vec![grainset(
            "orders",
            vec![
                daily("daily", Granularity::Day),
                daily("monthly", Granularity::Month),
            ],
            true,
            None,
        )]),
        &["daily", "monthly"],
    )
    .await;

    let root = by_name(&manifest, "orders");
    let RoleDefinition::Grainset { grains, .. } = &root.role_definition else {
        panic!(
            "`orders` must hold the Grainset arm, not {:?}",
            root.role_definition
        );
    };
    let referenced: BTreeMap<&str, &str> = grains
        .iter()
        .map(|id| {
            let entry = manifest
                .semantic_sources
                .get(id.as_str())
                .unwrap_or_else(|| panic!("the Grainset names `{id}`, which is no entry"));
            (entry.name.as_str(), id.as_str())
        })
        .collect();
    assert_eq!(
        referenced.keys().copied().collect::<Vec<&str>>(),
        vec!["orders.daily", "orders.monthly"],
        "the arm names the two grains, scope-qualified by their parent"
    );
    assert!(
        !by_name(&manifest, "orders.daily").public,
        "an authored nested unit is private (ADR 0045)"
    );
}

#[tokio::test]
async fn one_cross_grain_unit_per_ordered_pair_finer_to_coarser() {
    let manifest = compiled(
        model(vec![grainset(
            "orders",
            vec![
                daily("monthly", Granularity::Month),
                daily("daily", Granularity::Day),
                daily("yearly", Granularity::Year),
            ],
            true,
            None,
        )]),
        &["daily", "monthly", "yearly"],
    )
    .await;

    assert_eq!(
        names(&manifest),
        vec![
            "orders",
            "orders.daily",
            "orders.daily_monthly",
            "orders.daily_yearly",
            "orders.monthly",
            "orders.monthly_yearly",
            "orders.yearly",
        ],
        "three grains yield three cross-grain units, each named `<grainset>.<finer>_<coarser>`"
    );

    let grain_ids: Vec<&str> = ["orders.daily", "orders.monthly", "orders.yearly"]
        .iter()
        .map(|name| by_name(&manifest, name).id.as_str())
        .collect();
    for unit in [
        "orders.daily_monthly",
        "orders.daily_yearly",
        "orders.monthly_yearly",
    ] {
        let RoleDefinition::Joinset { from, to, .. } = &by_name(&manifest, unit).role_definition
        else {
            panic!("`{unit}` must be a Joinset in the artifact");
        };
        for operand in [from, to] {
            assert!(
                grain_ids.contains(&operand.as_str()),
                "`{unit}` joins two grains, never another cross-grain unit"
            );
        }
        assert!(
            !by_name(&manifest, unit).public,
            "an implicit unit is private and reachable through its originator (ADR 0054)"
        );
    }
}

#[tokio::test]
async fn the_rollup_reads_the_finer_grain_as_its_from_side() {
    let manifest = compiled(
        model(vec![grainset(
            "orders",
            vec![
                daily("monthly", Granularity::Month),
                daily("daily", Granularity::Day),
            ],
            true,
            None,
        )]),
        &["daily", "monthly"],
    )
    .await;

    let RoleDefinition::Joinset { from, to, .. } =
        &by_name(&manifest, "orders.daily_monthly").role_definition
    else {
        panic!("the cross-grain unit must be a Joinset in the artifact");
    };
    assert_eq!(
        from.as_str(),
        by_name(&manifest, "orders.daily").id.as_str(),
        "the finer grain is the from side"
    );
    assert_eq!(
        to.as_str(),
        by_name(&manifest, "orders.monthly").id.as_str(),
        "the coarser grain is the to side"
    );
}

#[tokio::test]
async fn no_keyset_means_no_rollup_composition_and_no_diagnostic() {
    let manifest = compiled(
        model(vec![grainset(
            "orders",
            vec![
                daily("daily", Granularity::Day),
                daily("monthly", Granularity::Month),
            ],
            false,
            None,
        )]),
        &["daily", "monthly"],
    )
    .await;

    assert_eq!(
        names(&manifest),
        vec!["orders", "orders.daily", "orders.monthly"],
        "with nothing keyed there is no basis to derive, so no cross-grain unit exists"
    );
}

#[tokio::test]
async fn the_cross_grain_basis_reads_only_the_grains_guaranteed_keysets() {
    let manifest = compiled(
        model(vec![grainset(
            "orders",
            vec![
                daily("daily", Granularity::Day),
                daily("monthly", Granularity::Month),
            ],
            true,
            None,
        )]),
        &["daily", "monthly"],
    )
    .await;

    let RoleDefinition::Joinset {
        on_expr_id: Some(on_expr_id),
        filter_expr_id,
        ..
    } = &by_name(&manifest, "orders.daily_monthly").role_definition
    else {
        panic!("the cross-grain unit must be a keyed Joinset in the artifact");
    };
    assert!(
        filter_expr_id.is_none(),
        "no relationship means no authored `filter:` to lower"
    );
    let (daily, monthly) = (
        semantic_alias(&manifest, "orders.daily"),
        semantic_alias(&manifest, "orders.monthly"),
    );
    assert_eq!(
        stored_tree(&manifest, *on_expr_id),
        column(daily, "region").eq(column(monthly, "region")),
    );
}

#[tokio::test]
async fn a_grainset_publishes_the_one_shape_kind_its_grains_agree_on() {
    let manifest = compiled(
        model(vec![grainset(
            "orders",
            vec![
                daily("daily", Granularity::Day),
                daily("monthly", Granularity::Month),
            ],
            true,
            None,
        )]),
        &["daily", "monthly"],
    )
    .await;

    let family = by_name(&manifest, "orders")
        .temporal
        .as_ref()
        .expect("the family publishes the axis its grains agree on");
    assert_eq!(
        family.temporal_type,
        TemporalType::events(semantics(&manifest, "order_ts")),
        "the one shape kind reaches the family, resolved to its minted id"
    );
    assert_eq!(
        family.granularity, None,
        "distinct grains never agree, so a Grainset guarantees no single grain (ADR 0056)"
    );
}

#[tokio::test]
async fn two_shape_kinds_across_the_grains_are_refused() {
    let error = refused(
        model(vec![grainset(
            "orders",
            vec![
                daily("daily", Granularity::Day),
                grain(
                    "monthly",
                    Some(TemporalShape::scd(
                        ScdType::Type2,
                        "valid_from",
                        "valid_to",
                        Some(Granularity::Month),
                    )),
                ),
            ],
            true,
            None,
        )]),
        &["daily", "monthly"],
    )
    .await;

    match error {
        CompileError::ComposedTemporalDisagreement { shapes, .. } => assert_eq!(
            shapes.len(),
            2,
            "both contradicting claims are carried, not just that some pair disagrees"
        ),
        other => panic!("expected a shape disagreement, got {other:?}"),
    }
}

#[tokio::test]
async fn a_shape_stated_on_the_family_is_one_participant_not_an_override() {
    let error = refused(
        model(vec![grainset(
            "orders",
            vec![
                daily("daily", Granularity::Day),
                daily("monthly", Granularity::Month),
            ],
            true,
            Some(TemporalShape::snapshot("order_ts", None)),
        )]),
        &["daily", "monthly"],
    )
    .await;

    match error {
        CompileError::ComposedTemporalDisagreement { shapes, .. } => assert_eq!(
            shapes.len(),
            3,
            "both grains and the family's own claim are carried, the family's last"
        ),
        other => panic!("expected a shape disagreement, got {other:?}"),
    }
}

#[tokio::test]
async fn a_repeated_grain_across_the_constituents_is_reported() {
    let error = refused(
        model(vec![grainset(
            "orders",
            vec![
                daily("daily", Granularity::Day),
                daily("also_daily", Granularity::Day),
                daily("monthly", Granularity::Month),
            ],
            true,
            None,
        )]),
        &["daily", "also_daily", "monthly"],
    )
    .await;

    assert_eq!(
        error.to_string(),
        "grainset `orders`: children `daily` and `also_daily` have the same grain \
         (`order_ts` at day) and serve the same names"
    );
}

#[tokio::test]
async fn children_at_one_granularity_that_serve_different_names_compile() {
    let narrower = ["region", "order_ts", "revenue"];
    let manifest = compiled(
        model(vec![grainset(
            "orders",
            vec![
                daily("daily", Granularity::Day),
                common::dated_on("narrower", &narrower, "order_ts", Granularity::Day),
            ],
            true,
            None,
        )]),
        &["daily", "narrower"],
    )
    .await;

    assert_ne!(
        by_name(&manifest, "orders.daily").semantic_bitmask,
        by_name(&manifest, "orders.narrower").semantic_bitmask,
        "the two children differ by coverage mask"
    );
}

#[tokio::test]
async fn children_with_no_time_axis_repeat_a_grain_over_one_mask() {
    let error = refused(
        model(vec![grainset(
            "orders",
            vec![
                grain("plain", None),
                grain("also_plain", None),
                daily("daily", Granularity::Day),
            ],
            true,
            None,
        )]),
        &["plain", "also_plain", "daily"],
    )
    .await;

    assert_eq!(
        error.to_string(),
        "grainset `orders`: children `plain` and `also_plain` have the same grain \
         (no time axis) and serve the same names"
    );
}

#[tokio::test]
async fn children_with_no_time_axis_and_different_masks_compile() {
    let narrower = ["region", "revenue"];
    compiled(
        model(vec![grainset(
            "orders",
            vec![
                grain("plain", None),
                common::constituent("narrower", &narrower),
                daily("daily", Granularity::Day),
            ],
            true,
            None,
        )]),
        &["plain", "narrower", "daily"],
    )
    .await;
}

#[tokio::test]
async fn validate_refuses_a_stored_pair_with_one_grain_and_one_mask() {
    let mut manifest = two_grain_family().await;
    restate_granularity(&mut manifest, "orders.monthly", Granularity::Day);

    let faults = manifest.validate().expect_err("the stored pair repeats");
    assert!(
        faults.iter().any(|fault| matches!(
            &fault.kind,
            CompileError::GrainsetGrainRepeated { children, .. } if children.len() == 2
        )),
        "validate reports the repeated pair: {faults:?}"
    );
}

#[tokio::test]
async fn validate_reports_each_repeating_grainset() {
    let mut manifest = compiled(
        model(vec![
            two_grain_grainset("orders"),
            two_grain_grainset("returns"),
        ]),
        &["daily", "monthly"],
    )
    .await;
    for name in ["orders.monthly", "returns.monthly"] {
        restate_granularity(&mut manifest, name, Granularity::Day);
    }

    let faults = manifest.validate().expect_err("both pairs repeat");
    let repeated: Vec<&str> = faults
        .iter()
        .filter_map(|fault| match &fault.kind {
            CompileError::GrainsetGrainRepeated { grainset, .. } => Some(grainset.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(repeated, vec!["orders", "returns"], "{faults:?}");
}

#[tokio::test]
async fn a_grainsets_keysets_are_the_authors_and_nothing_is_derived() {
    let manifest = two_grain_family().await;

    let mut names: Vec<&str> = manifest
        .keysets
        .values()
        .map(|keyset| keyset.name.as_str())
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["default"; 4]);

    let family = by_name(&manifest, "orders");
    let RoleDefinition::Grainset { keysets, .. } = &family.role_definition else {
        panic!("`orders` must hold the Grainset arm");
    };
    assert_eq!(
        keysets,
        &vec![mint_keyset_id(&family.id, "default")],
        "the family carries the author's one keyset id and nothing derived"
    );
    assert!(
        manifest.validate().is_ok(),
        "an authored keyset inside the family's boundary must pass validate()"
    );
}

fn two_grain_grainset(name: &str) -> Grainset {
    grainset(
        name,
        vec![
            daily("daily", Granularity::Day),
            daily("monthly", Granularity::Month),
        ],
        true,
        None,
    )
}

fn two_grain_model() -> semstrait_model::SemanticModel {
    model(vec![two_grain_grainset("orders")])
}

fn restate_granularity(manifest: &mut SemanticManifest, name: &str, granularity: Granularity) {
    let id = by_name(manifest, name).id.clone();
    let entry = manifest
        .semantic_sources
        .get_mut(&id)
        .expect("the child is stored");
    let shape = entry.temporal.as_mut().expect("the child has a shape");
    shape.granularity = Some(granularity);
}

async fn two_grain_family() -> SemanticManifest {
    compiled(two_grain_model(), &["daily", "monthly"]).await
}

#[tokio::test]
async fn a_grainset_takes_g_and_its_cross_grain_unit_j() {
    let manifest = two_grain_family().await;

    let table: Vec<(&str, &str, &str)> = manifest
        .semantic_sources
        .values()
        .map(|entry| (&entry.id[..8], entry.name.as_str(), entry.alias.as_str()))
        .collect();
    assert_eq!(
        table,
        vec![
            ("4cb19992", "orders.daily_monthly", "ssj0"),
            ("802c2223", "orders.daily", "ssd1"),
            ("88a402f2", "orders.monthly", "ssd2"),
            ("cd3e4c16", "orders", "ssg3"),
        ],
        "the Grainset's `g`, each grain's `d` and the cross-grain unit's `j`, indexed in \
         `EntityId` order"
    );
}

#[tokio::test]
async fn a_grainset_family_owns_no_signature_for_a_name_a_child_maps_and_carries_its_keysets() {
    let manifest = two_grain_family().await;

    let family = by_name(&manifest, "orders");
    assert!(
        family.semantic_signature_ids.is_empty(),
        "a Grainset owns no signature for a name a child maps"
    );

    let RoleDefinition::Grainset { keysets, .. } = &family.role_definition else {
        panic!(
            "`orders` must hold the Grainset arm, not {:?}",
            family.role_definition
        );
    };
    assert_eq!(
        keysets,
        &vec![mint_keyset_id(&family.id, "default")],
        "one keyset id, minted on the family itself"
    );
    assert_eq!(
        manifest
            .keysets
            .get(keysets[0].as_str())
            .expect("an authored keyset id is in the roster")
            .name
            .as_str(),
        "default"
    );
}

#[tokio::test]
async fn a_root_grainset_whose_children_share_a_granularity_stores_it() {
    let narrower = ["region", "order_ts", "revenue"];
    let manifest = compiled(
        model(vec![grainset(
            "orders",
            vec![
                daily("daily", Granularity::Day),
                common::dated_on("narrow_daily", &narrower, "order_ts", Granularity::Day),
            ],
            true,
            None,
        )]),
        &["daily", "narrow_daily"],
    )
    .await;

    let family = by_name(&manifest, "orders")
        .temporal
        .as_ref()
        .expect("the family publishes the axis its grains agree on");
    assert_eq!(family.granularity, Some(Granularity::Day));
}
