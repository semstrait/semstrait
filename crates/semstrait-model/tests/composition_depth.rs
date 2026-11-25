mod common;

use common::golden::{self, Golden};
use common::{declares, leaf, leaf_with_granularity, names_variant};
use semstrait_common::Granularity;
use semstrait_model::{
    ComplexExtras, Grainset, NestedGrainset, NestedJoinset, NestedUnionset, SemanticModel,
    ValidateErrorKind,
};

fn root_over(deepest: NestedGrainset) -> Grainset {
    let l3 = NestedUnionset::builder("l3")
        .extras(declares(Granularity::Month))
        .dataset(leaf("l3a"))
        .grainset(deepest)
        .build();
    let l2 = NestedJoinset::builder("l2")
        .extras(declares(Granularity::Month))
        .dataset(leaf("l2a"))
        .unionset(l3)
        .build();
    Grainset::builder("l1")
        .extras(ComplexExtras::default())
        .dataset(leaf_with_granularity("l1a", Granularity::Day))
        .joinset(l2)
        .build()
}

fn grainset_l4() -> NestedGrainset {
    NestedGrainset::builder("l4")
        .extras(declares(Granularity::Month))
        .dataset(leaf_with_granularity("l4a", Granularity::Day))
        .dataset(leaf_with_granularity("l4b", Granularity::Day))
        .build()
}

fn grainset_l4_with_a_fifth_level() -> NestedGrainset {
    let l5 = NestedJoinset::builder("l5")
        .extras(declares(Granularity::Month))
        .dataset(leaf("l5a"))
        .dataset(leaf("l5b"))
        .build();
    NestedGrainset::builder("l4")
        .extras(declares(Granularity::Month))
        .dataset(leaf_with_granularity("l4a", Granularity::Day))
        .joinset(l5)
        .build()
}

#[test]
fn four_nested_compositions_pass() {
    let (_model, warnings) = SemanticModel::builder()
        .name("m")
        .grainset(root_over(grainset_l4()))
        .build()
        .expect("four nested compositions pass model validation");
    assert!(warnings.is_empty());
}

#[test]
fn five_nested_compositions_are_refused() {
    five_deep_refusal().assert_committed();
}

fn five_deep_refusal() -> Golden {
    let err = SemanticModel::builder()
        .name("m")
        .grainset(root_over(grainset_l4_with_a_fifth_level()))
        .build()
        .unwrap_err();

    assert!(
        names_variant(
            &err,
            &ValidateErrorKind::CompositionNestingTooDeep {
                root: "l1".into(),
                depth: 5,
            }
        ),
        "{err:?}"
    );
    Golden::message("five_nested_compositions_are_refused", &err)
}

#[test]
fn generate_goldens() {
    golden::generate(|| vec![five_deep_refusal()]);
}
