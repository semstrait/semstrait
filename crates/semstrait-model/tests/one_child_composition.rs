mod common;

use common::{leaf, leaf_with_granularity, names_variant};
use semstrait_common::diagnostic::{Diagnose, Diagnostic};
use semstrait_common::Granularity;
use semstrait_model::{
    ComplexExtras, Grainset, Joinset, ModelBuildErrorKind, NestedGrainset, NestedUnionset,
    SemanticModel, UnionMode, Unionset, ValidateErrorKind,
};

fn insufficient_with(parent: &str, child_count: usize) -> ValidateErrorKind {
    ValidateErrorKind::ComplexDataKindInsufficientChildren {
        parent: parent.to_string(),
        child_count,
    }
}

fn insufficient(parent: &str) -> ValidateErrorKind {
    insufficient_with(parent, 1)
}

fn one_constituent_joinset_refusal() -> Vec<Diagnostic<ModelBuildErrorKind>> {
    let joinset = Joinset::builder("ads")
        .extras(ComplexExtras::default())
        .dataset(leaf("only"))
        .build();
    SemanticModel::builder()
        .name("m")
        .joinset(joinset)
        .build()
        .unwrap_err()
}

#[test]
fn a_grainset_with_one_child_is_accepted() {
    let grainset = Grainset::builder("ads")
        .extras(ComplexExtras::default())
        .dataset(leaf_with_granularity("daily", Granularity::Day))
        .build();
    SemanticModel::builder()
        .name("m")
        .grainset(grainset)
        .build()
        .expect("a one-child Grainset passes model validation");
}

#[test]
fn a_unionset_of_mode_all_with_one_branch_is_accepted() {
    let unionset = Unionset::builder("ads")
        .extras(ComplexExtras::default())
        .mode(UnionMode::All)
        .dataset(leaf("only"))
        .build();
    SemanticModel::builder()
        .name("m")
        .unionset(unionset)
        .build()
        .expect("a one-branch Unionset of mode all passes model validation");
}

#[test]
fn a_nested_one_child_grainset_is_accepted() {
    let nested = NestedGrainset::builder("ads")
        .extras(ComplexExtras::default())
        .dataset(leaf_with_granularity("daily", Granularity::Day))
        .build();
    let unionset = Unionset::builder("u")
        .extras(ComplexExtras::default())
        .dataset(leaf("other"))
        .grainset(nested)
        .build();
    SemanticModel::builder()
        .name("m")
        .unionset(unionset)
        .build()
        .expect("a nested one-child Grainset passes model validation");
}

#[test]
fn a_unionset_of_mode_unique_with_one_branch_is_refused() {
    let unionset = Unionset::builder("ads")
        .extras(ComplexExtras::default())
        .mode(UnionMode::Unique)
        .dataset(leaf("only"))
        .build();
    let err = SemanticModel::builder()
        .name("m")
        .unionset(unionset)
        .build()
        .unwrap_err();
    assert!(names_variant(&err, &insufficient("ads")), "{err:?}");
}

#[test]
fn a_nested_unionset_of_mode_unique_with_one_branch_is_refused() {
    let nested = NestedUnionset::builder("inner")
        .extras(ComplexExtras::default())
        .mode(UnionMode::Unique)
        .dataset(leaf("only"))
        .build();
    let grainset = Grainset::builder("outer")
        .extras(ComplexExtras::default())
        .dataset(leaf("a"))
        .unionset(nested)
        .build();
    let err = SemanticModel::builder()
        .name("m")
        .grainset(grainset)
        .build()
        .unwrap_err();
    assert!(names_variant(&err, &insufficient("inner")), "{err:?}");
}

#[test]
fn a_joinset_with_one_constituent_is_refused() {
    let err = one_constituent_joinset_refusal();
    assert!(names_variant(&err, &insufficient("ads")), "{err:?}");
}

#[test]
fn the_refusal_text_holds_no_internal_label() {
    let diagnostics = one_constituent_joinset_refusal();
    let rendered = diagnostics
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(rendered.contains("`ads`"), "{rendered}");
    assert!(!rendered.contains("SR-10"), "{rendered}");
}

#[test]
fn a_grainset_with_no_child_is_refused_by_the_rule_that_holds_for_it() {
    let grainset = Grainset::builder("ads")
        .extras(ComplexExtras::default())
        .build();
    let err = SemanticModel::builder()
        .name("m")
        .grainset(grainset)
        .build()
        .unwrap_err();
    assert!(names_variant(&err, &insufficient_with("ads", 0)), "{err:?}");
}

#[test]
fn the_refusal_text_states_the_minimum_of_every_kind_that_raises_it() {
    assert_eq!(
        insufficient_with("ads", 0).message(),
        "complex data kind `ads` has 0 child(ren); a Grainset or a Unionset of mode `all` needs \
         at least 1, a Joinset or a Unionset of mode `unique` at least 2"
    );
}
