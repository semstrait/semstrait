#[path = "../../semstrait-api/tests/common/mod.rs"]
mod common;

use std::time::Duration;

use semstrait::{load, Input, SemstraitBuilder};
use tokio::time::sleep;

use common::SALES;

#[tokio::test]
async fn two_loads_of_one_model_yaml_on_one_semstrait_return_the_cached_manifest() {
    let semstrait = SemstraitBuilder::default().build();

    let (first, _) = load(&semstrait, Input::ModelYaml(SALES.into()))
        .await
        .expect("the model compiles");
    let (second, _) = load(&semstrait, Input::ModelYaml(SALES.into()))
        .await
        .expect("the second load is a store hit");

    assert_eq!(second, first);

    sleep(Duration::from_millis(50)).await;
    let fresh = SemstraitBuilder::default().build();
    let (recompiled, _) = load(&fresh, Input::ModelYaml(SALES.into()))
        .await
        .expect("a fresh semstrait compiles the same yaml");

    assert_ne!(
        recompiled.created_at, first.created_at,
        "a genuine recompile after a delay carries a later created_at, so the equal created_at above is evidence of a cache hit, not of a coarse clock"
    );
}

#[tokio::test]
async fn load_of_a_changed_model_yaml_gives_a_different_model_id() {
    let semstrait = SemstraitBuilder::default().build();
    let changed = SALES.replace("sales", "sales_changed");

    let (original, _) = load(&semstrait, Input::ModelYaml(SALES.into()))
        .await
        .expect("the model compiles");
    let (recompiled, _) = load(&semstrait, Input::ModelYaml(changed))
        .await
        .expect("the changed model compiles");

    assert_ne!(recompiled.model_id, original.model_id);
}
