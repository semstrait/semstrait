mod common;

use common::golden::{asked, generate_when, Golden, GENERATOR, SWITCH};
use semstrait_common::diagnostic::Diagnose;
use semstrait_model::ParseErrorKind;
use tempfile::TempDir;

#[test]
fn the_generator_writes_only_when_asked_by_its_name_with_the_switch_set() {
    let runs: [(&[&str], bool, bool); 8] = [
        (&[], false, false),
        (&["--ignored"], false, false),
        (&["--include-ignored"], false, false),
        (&[GENERATOR], false, false),
        (&[], true, false),
        (&["--include-ignored"], true, false),
        (&[GENERATOR], true, true),
        (&["--exact", GENERATOR], true, true),
    ];
    let scratch = TempDir::new().expect("create temp dir");
    for (index, (args, switch, writes)) in runs.into_iter().enumerate() {
        let path = scratch.path().join(format!("run-{index}.error.txt"));
        let run = args.iter().map(|arg| (*arg).to_owned());
        generate_when(asked(run, switch), || {
            vec![Golden {
                path: path.clone(),
                rendered: "the rendered message".to_owned(),
            }]
        });
        assert_eq!(
            std::fs::read_to_string(&path).ok(),
            writes.then(|| "the rendered message".to_owned()),
            "arguments {args:?}, switch set: {switch}"
        );
    }
}

#[test]
fn the_generator_refuses_a_golden_whose_gate_fails_its_variant_assertion() {
    let scratch = TempDir::new().expect("create temp dir");
    let path = scratch.path().join("refused.error.txt");
    let outcome = std::panic::catch_unwind(|| {
        generate_when(true, || {
            let refusal = ParseErrorKind::RelationshipJoinKeysEmpty;
            assert!(
                matches!(
                    refusal,
                    ParseErrorKind::RelationshipJoinKeyDuplicated { .. }
                ),
                "the gate's variant assertion"
            );
            vec![Golden {
                path: path.clone(),
                rendered: refusal.message(),
            }]
        });
    });
    assert!(outcome.is_err(), "the failing gate fails the generator");
    assert!(!path.exists(), "a failing gate's golden is not written");
}

#[test]
fn a_missing_golden_fails_its_gate_naming_the_generator() {
    let scratch = TempDir::new().expect("create temp dir");
    let golden = Golden {
        path: scratch.path().join("missing.error.txt"),
        rendered: "the rendered message".to_owned(),
    };
    let failure = std::panic::catch_unwind(|| golden.assert_committed())
        .expect_err("a missing golden fails its gate");
    let message = failure
        .downcast_ref::<String>()
        .expect("a formatted failure");
    assert!(
        message.contains(&format!(
            "{SWITCH}=1 cargo test -p semstrait-model --test golden_generator {GENERATOR}"
        )),
        "{message}"
    );
}
