mod common;

use common::{declares, leaf, leaf_with_axis_only, leaf_with_granularity, names_variant};
use semstrait_common::Granularity;
use semstrait_model::{
    ComplexExtras, Grainset, Joinset, NestedGrainset, NestedJoinset, NestedUnionset, SemanticModel,
    SemanticModelBuilder, Unionset, ValidateErrorKind,
};

fn admitted(model: SemanticModelBuilder) {
    let (_model, warnings) = model
        .build()
        .expect("a nested composition may declare its own granularity");
    assert!(warnings.is_empty());
}

fn refused_at_the_root(model: SemanticModelBuilder) {
    let err = model.build().unwrap_err();
    assert!(names_variant(
        &err,
        &ValidateErrorKind::TemporalGranularityOnComplex {
            data_kind: "root".to_string()
        }
    ));
}

#[test]
fn a_nested_unionset_under_a_joinset_admits_a_declared_granularity() {
    let inner = NestedUnionset::builder("u")
        .extras(declares(Granularity::Month))
        .dataset(leaf("u1"))
        .dataset(leaf("u2"))
        .build();
    let root = Joinset::builder("root")
        .extras(ComplexExtras::default())
        .dataset(leaf("d1"))
        .unionset(inner)
        .build();

    admitted(SemanticModel::builder().name("m").joinset(root));
}

#[test]
fn a_nested_joinset_under_a_unionset_admits_a_declared_granularity() {
    let inner = NestedJoinset::builder("j")
        .extras(declares(Granularity::Month))
        .dataset(leaf("j1"))
        .dataset(leaf("j2"))
        .build();
    let root = Unionset::builder("root")
        .extras(ComplexExtras::default())
        .dataset(leaf("d1"))
        .joinset(inner)
        .build();

    admitted(SemanticModel::builder().name("m").unionset(root));
}

#[test]
fn a_nested_grainset_under_a_joinset_admits_a_declared_granularity() {
    let inner = NestedGrainset::builder("g")
        .extras(declares(Granularity::Month))
        .dataset(leaf_with_granularity("g1", Granularity::Day))
        .dataset(leaf_with_granularity("g2", Granularity::Day))
        .build();
    let root = Joinset::builder("root")
        .extras(ComplexExtras::default())
        .dataset(leaf("d1"))
        .grainset(inner)
        .build();

    admitted(SemanticModel::builder().name("m").joinset(root));
}

#[test]
fn a_root_grainset_declaring_a_granularity_is_still_refused() {
    let root = Grainset::builder("root")
        .extras(declares(Granularity::Month))
        .dataset(leaf_with_granularity("d1", Granularity::Day))
        .dataset(leaf_with_granularity("d2", Granularity::Day))
        .build();

    refused_at_the_root(SemanticModel::builder().name("m").grainset(root));
}

#[test]
fn a_root_unionset_declaring_a_granularity_is_still_refused() {
    let root = Unionset::builder("root")
        .extras(declares(Granularity::Month))
        .dataset(leaf("d1"))
        .dataset(leaf("d2"))
        .build();

    refused_at_the_root(SemanticModel::builder().name("m").unionset(root));
}

#[test]
fn a_root_joinset_declaring_a_granularity_is_still_refused() {
    let root = Joinset::builder("root")
        .extras(declares(Granularity::Month))
        .dataset(leaf("d1"))
        .dataset(leaf("d2"))
        .build();

    refused_at_the_root(SemanticModel::builder().name("m").joinset(root));
}

#[test]
fn a_grainset_s_dataset_child_with_no_time_axis_compiles() {
    let root = Grainset::builder("root")
        .extras(ComplexExtras::default())
        .dataset(leaf_with_granularity("d1", Granularity::Day))
        .dataset(leaf("d2"))
        .build();

    admitted(SemanticModel::builder().name("m").grainset(root));
}

#[test]
fn a_grainset_s_dataset_child_with_a_time_axis_and_no_granularity_is_still_refused() {
    let root = Grainset::builder("root")
        .extras(ComplexExtras::default())
        .dataset(leaf_with_granularity("d1", Granularity::Day))
        .dataset(leaf_with_axis_only("d2"))
        .build();

    let err = SemanticModel::builder()
        .name("m")
        .grainset(root)
        .build()
        .unwrap_err();
    assert!(names_variant(
        &err,
        &ValidateErrorKind::GrainsetChildMissingGranularity {
            grainset: "root".to_string(),
            child: "d2".to_string(),
        }
    ));
}
