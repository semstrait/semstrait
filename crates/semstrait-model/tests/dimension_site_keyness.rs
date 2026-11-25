use semstrait_common::diagnostic::Diagnose;
use semstrait_model::{
    parse, DimensionSite, Keyness, ModelBuildErrorKind, ParseErrorKind, SemanticInterface,
    SemanticModel, ValidateErrorKind,
};
use std::collections::{BTreeMap, BTreeSet};

fn model_yaml(pools: &str, interface: &str) -> String {
    format!(
        "semantic_model:\n  name: keyness\n{pools}  datasets:\n    - name: orders\n      \
         extras:\n        semantic_mapping: auto\n{interface}"
    )
}

fn build(yaml: &str) -> SemanticModel {
    let (model, warnings) = build_with_warnings(yaml);
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    model
}

fn build_with_warnings(yaml: &str) -> (SemanticModel, Vec<ModelBuildErrorKind>) {
    let (model, warnings) = parse(yaml)
        .unwrap_or_else(|d| panic!("parse must succeed, got {d:?}"))
        .build()
        .unwrap_or_else(|d| panic!("build must succeed, got {d:?}"));
    (model, warnings.into_iter().map(|d| d.kind).collect())
}

fn parse_errors(yaml: &str) -> Vec<ParseErrorKind> {
    parse(yaml)
        .err()
        .unwrap_or_else(|| panic!("parse must fail for:\n{yaml}"))
        .into_iter()
        .map(|d| d.kind)
        .collect()
}

fn build_errors(yaml: &str) -> Vec<ModelBuildErrorKind> {
    parse(yaml)
        .unwrap_or_else(|d| panic!("parse must succeed, got {d:?}"))
        .build()
        .err()
        .unwrap_or_else(|| panic!("build must fail for:\n{yaml}"))
        .into_iter()
        .map(|d| d.kind)
        .collect()
}

fn complex_interface(model: &SemanticModel) -> &SemanticInterface {
    model
        .grainsets
        .values()
        .map(|k| &k.semantic_interface)
        .chain(model.unionsets.values().map(|k| &k.semantic_interface))
        .chain(model.joinsets.values().map(|k| &k.semantic_interface))
        .next()
        .expect("one complex data kind")
}

fn interface(model: &SemanticModel) -> &SemanticInterface {
    &model
        .datasets
        .values()
        .next()
        .expect("one dataset")
        .semantic_interface
}

fn membership(iface: &SemanticInterface, name: &str) -> BTreeSet<String> {
    iface
        .dimensions
        .values()
        .find(|s| s.name().as_ref() == name)
        .unwrap_or_else(|| panic!("no dimension site named `{name}`"))
        .keyness()
        .keyset_membership()
}

fn keysets(iface: &SemanticInterface) -> BTreeMap<String, Vec<String>> {
    iface
        .keysets()
        .into_iter()
        .map(|(k, members)| (k, members.iter().map(|n| n.as_str().to_owned()).collect()))
        .collect()
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|n| (*n).to_owned()).collect()
}

const POOLS: &str = "  dimensions:\n    - name: country\n      data_type: string\n      \
                     type: categorical\n";

fn inline_site(extra: &str) -> String {
    format!(
        "      dimensions:\n        - name: region\n          data_type: string\n          \
         type: categorical\n{extra}"
    )
}

fn ref_site(extra: &str) -> String {
    format!("      dimensions:\n        - ref: country\n{extra}")
}

const COMPANION: &str = "        - name: tenant\n          data_type: string\n          \
                         type: categorical\n          is_key: true\n          \
                         keysets: [surrogate]\n";

#[test]
fn every_legal_row_of_the_legality_table_yields_its_membership() {
    let rows: [(&str, &str, &[&str]); 7] = [
        ("absent `is_key`, no keyset", "", &[]),
        (
            "explicit `is_key: false`, no keyset",
            "          is_key: false\n",
            &[],
        ),
        (
            "`is_key: true`, no keyset",
            "          is_key: true\n",
            &["default"],
        ),
        (
            "`is_key: true` with the singular",
            "          is_key: true\n          keyset: natural\n",
            &["natural"],
        ),
        (
            "`is_key: true` with the plural",
            "          is_key: true\n          keysets: [natural, surrogate]\n",
            &["natural", "surrogate"],
        ),
        (
            "`is_key: true` with a repeated plural entry",
            "          is_key: true\n          keysets: [natural, natural]\n",
            &["natural"],
        ),
        (
            "explicit `keyset: default`",
            "          is_key: true\n          keyset: default\n",
            &["default"],
        ),
    ];

    for (label, keyness, expected) in rows {
        for (site_label, yaml, name) in [
            (
                "inline",
                model_yaml("", &format!("{}{COMPANION}", inline_site(keyness))),
                "region",
            ),
            (
                "ref",
                model_yaml(POOLS, &format!("{}{COMPANION}", ref_site(keyness))),
                "country",
            ),
        ] {
            let model = build(&yaml);
            assert_eq!(
                membership(interface(&model), name),
                set(expected),
                "{label} on a {site_label} site"
            );
        }
    }
}

#[test]
fn both_rejected_keyness_states_fail_at_parse_with_their_typed_kind() {
    const BOTH: &str = "          is_key: true\n          keyset: a\n          keysets: [b]\n";
    let cases: [(&str, &str, &str, &str); 4] = [
        ("both spellings, inline", BOTH, "Dimension", "both"),
        (
            "both spellings, ref",
            BOTH,
            "Dimension reference site",
            "both",
        ),
        (
            "singular without `is_key`",
            "          keyset: natural\n",
            "Dimension",
            "keyset",
        ),
        (
            "plural with `is_key: false`",
            "          is_key: false\n          keysets: [natural]\n",
            "Dimension",
            "keysets",
        ),
    ];

    for (label, keyness, carrier, marker) in cases {
        let yaml = if carrier == "Dimension reference site" {
            model_yaml(POOLS, &ref_site(keyness))
        } else {
            model_yaml("", &inline_site(keyness))
        };
        let errors = parse_errors(&yaml);
        let hit = errors.iter().any(|e| match e {
            ParseErrorKind::SemanticsSiteKeysetBothSpellings { carrier: got } => {
                marker == "both" && got == carrier
            }
            ParseErrorKind::SemanticsSiteKeysetWithoutIsKey {
                carrier: got,
                spelling,
            } => marker != "both" && got == carrier && spelling == marker,
            _ => false,
        });
        assert!(
            hit,
            "{label}: expected the typed keyness kind, got {errors:?}"
        );
    }
}

#[test]
fn a_keyness_spelling_that_names_no_keyset_fails_at_parse_with_its_typed_kind() {
    let empty_list: [(&str, &str); 3] = [
        (
            "activated",
            "          is_key: true\n          keysets: []\n",
        ),
        ("unactivated", "          keysets: []\n"),
        (
            "explicit `is_key: false`",
            "          is_key: false\n          keysets: []\n",
        ),
    ];
    for (label, keyness) in empty_list {
        let errors = parse_errors(&model_yaml("", &inline_site(keyness)));
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ParseErrorKind::SemanticsSiteKeysetListEmpty { carrier } if carrier == "Dimension"
            )),
            "`keysets: []`, {label}: expected SemanticsSiteKeysetListEmpty, got {errors:?}"
        );
    }

    let blank_name: [(&str, &str, &str); 4] = [
        (
            "empty singular",
            "          is_key: true\n          keyset: \"\"\n",
            "keyset",
        ),
        (
            "whitespace-only singular",
            "          is_key: true\n          keyset: \"  \"\n",
            "keyset",
        ),
        (
            "sole blank plural entry",
            "          is_key: true\n          keysets: [\"\"]\n",
            "keysets",
        ),
        (
            "blank beside a real name",
            "          is_key: true\n          keysets: [natural, \"\"]\n",
            "keysets",
        ),
    ];
    for (label, keyness, spelling) in blank_name {
        let errors = parse_errors(&model_yaml("", &inline_site(keyness)));
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ParseErrorKind::SemanticsSiteKeysetNameBlank {
                    carrier,
                    spelling: got,
                } if carrier == "Dimension" && got == spelling
            )),
            "{label}: expected SemanticsSiteKeysetNameBlank, got {errors:?}"
        );
    }

    for (label, keyness) in [
        (
            "`keysets: []`",
            "          is_key: true\n          keysets: []\n",
        ),
        (
            "a blank name",
            "          is_key: true\n          keyset: \"\"\n",
        ),
    ] {
        let errors = parse_errors(&model_yaml(POOLS, &ref_site(keyness)));
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ParseErrorKind::SemanticsSiteKeysetListEmpty { carrier }
                | ParseErrorKind::SemanticsSiteKeysetNameBlank { carrier, .. }
                    if carrier == "Dimension reference site"
            )),
            "{label} on a ref site: expected the typed kind, got {errors:?}"
        );
    }
}

#[test]
fn a_legal_keyness_state_draws_neither_rejection() {
    for keyness in [
        "",
        "          is_key: false\n",
        "          is_key: true\n",
        "          is_key: true\n          keyset: natural\n",
        "          is_key: true\n          keysets: [natural]\n",
    ] {
        build(&model_yaml("", &inline_site(keyness)));
        build(&model_yaml(POOLS, &ref_site(keyness)));
    }
}

#[test]
fn the_singular_merges_into_the_plural_and_default_materialises() {
    let model = build(&model_yaml(
        "",
        "      dimensions:\n        - name: alpha\n          data_type: string\n          \
         type: categorical\n          is_key: true\n        - name: bravo\n          \
         data_type: string\n          type: categorical\n          is_key: true\n          \
         keyset: natural\n        - name: charlie\n          data_type: string\n          \
         type: categorical\n",
    ));
    let iface = interface(&model);
    assert_eq!(membership(iface, "alpha"), set(&["default"]));
    assert_eq!(membership(iface, "bravo"), set(&["natural"]));
    assert_eq!(membership(iface, "charlie"), BTreeSet::new());
    assert_eq!(
        keysets(iface),
        BTreeMap::from([
            ("default".to_owned(), vec!["alpha".to_owned()]),
            ("natural".to_owned(), vec!["bravo".to_owned()]),
        ]),
        "a partial `default` is legal (US 11)"
    );
}

#[test]
fn explicit_default_and_implicit_default_are_indistinguishable() {
    let explicit = build(&model_yaml(
        "",
        &inline_site("          is_key: true\n          keyset: default\n"),
    ));
    let implicit = build(&model_yaml("", &inline_site("          is_key: true\n")));
    assert_eq!(
        keysets(interface(&explicit)),
        keysets(interface(&implicit)),
        "explicit `keyset: default` reconciles to the implicit one"
    );
}

#[test]
fn a_single_member_keyset_is_legal() {
    let model = build(&model_yaml(
        "",
        &inline_site("          is_key: true\n          keyset: surrogate\n"),
    ));
    assert_eq!(
        keysets(interface(&model)),
        BTreeMap::from([("surrogate".to_owned(), vec!["region".to_owned()])])
    );
}

#[test]
fn a_data_kind_with_no_is_key_anywhere_is_legal_and_declares_nothing() {
    let model = build(&model_yaml("", &inline_site("")));
    assert!(keysets(interface(&model)).is_empty());
}

#[test]
fn a_data_kind_may_declare_only_non_default_keysets() {
    let model = build(&model_yaml(
        "",
        "      dimensions:\n        - name: alpha\n          data_type: string\n          \
         type: categorical\n          is_key: true\n          keyset: natural\n        \
         - name: bravo\n          data_type: string\n          type: categorical\n          \
         is_key: true\n          keysets: [natural, surrogate]\n",
    ));
    let names: Vec<String> = keysets(interface(&model)).into_keys().collect();
    assert_eq!(names, vec!["natural", "surrogate"]);
}

#[test]
fn a_site_in_two_keysets_holds_the_same_relative_position_in_both() {
    let model = build(&model_yaml(
        "",
        "      dimensions:\n        - name: zulu\n          data_type: string\n          \
         type: categorical\n          is_key: true\n          keysets: [natural, surrogate]\n        \
         - name: mike\n          data_type: string\n          type: categorical\n          \
         is_key: true\n          keyset: surrogate\n        - name: alpha\n          \
         data_type: string\n          type: categorical\n          is_key: true\n          \
         keysets: [natural, surrogate]\n",
    ));
    let projected = keysets(interface(&model));
    assert_eq!(projected["natural"], vec!["alpha", "zulu"]);
    assert_eq!(projected["surrogate"], vec!["alpha", "mike", "zulu"]);

    let shared: Vec<&String> = projected["surrogate"]
        .iter()
        .filter(|n| projected["natural"].contains(n))
        .collect();
    assert_eq!(
        shared,
        projected["natural"].iter().collect::<Vec<_>>(),
        "a shared site must keep its relative position in both keysets"
    );
}

#[test]
fn two_keysets_over_identical_members_are_rejected() {
    let errors = build_errors(&model_yaml(
        "",
        &inline_site("          is_key: true\n          keysets: [natural, surrogate]\n"),
    ));
    assert!(
        errors.iter().any(|e| matches!(
            e,
            ModelBuildErrorKind::Validate(ValidateErrorKind::DataKindKeysetsIdenticalMembership {
                data_kind,
                keysets,
                members,
            }) if data_kind == "orders"
                && keysets == &vec!["natural".to_owned(), "surrogate".to_owned()]
                && members == &vec!["region".to_owned()]
        )),
        "expected DataKindKeysetsIdenticalMembership, got {errors:?}"
    );
}

#[test]
fn two_keysets_over_different_members_are_accepted() {
    build(&model_yaml(
        "",
        "      dimensions:\n        - name: alpha\n          data_type: string\n          \
         type: categorical\n          is_key: true\n          keysets: [natural, surrogate]\n        \
         - name: bravo\n          data_type: string\n          type: categorical\n          \
         is_key: true\n          keyset: surrogate\n",
    ));
}

#[test]
fn the_plural_spelling_round_trips_byte_exactly_on_both_site_forms() {
    for (label, yaml) in [
        (
            "inline",
            model_yaml(
                "",
                &format!(
                    "{}{COMPANION}",
                    inline_site(
                        "          is_key: true\n          keysets: [natural, surrogate]\n"
                    )
                ),
            ),
        ),
        (
            "ref",
            model_yaml(
                POOLS,
                &format!(
                    "{}{COMPANION}",
                    ref_site("          is_key: true\n          keysets: [natural, surrogate]\n")
                ),
            ),
        ),
        (
            "bare `is_key`",
            model_yaml("", &inline_site("          is_key: true\n")),
        ),
    ] {
        let model = build(&yaml);
        let dumped = serde_yaml::to_string(&model).expect("serialize");
        assert!(!dumped.contains('!'), "{label}: `!`-tag leak:\n{dumped}");
        let back: SemanticModel = serde_yaml::from_str(&dumped).expect("deserialize");
        assert_eq!(model, back, "{label}: round-trip changed the model");
    }
}

#[test]
fn the_singular_spelling_round_trips_semantically_as_the_plural() {
    for (label, yaml, name) in [
        (
            "inline",
            model_yaml(
                "",
                &inline_site("          is_key: true\n          keyset: natural\n"),
            ),
            "region",
        ),
        (
            "ref",
            model_yaml(
                POOLS,
                &ref_site("          is_key: true\n          keyset: natural\n"),
            ),
            "country",
        ),
    ] {
        let model = build(&yaml);
        let dumped = serde_yaml::to_string(&model).expect("serialize");
        assert!(
            dumped.contains("keysets:") && !dumped.contains("keyset: natural"),
            "{label}: the singular must re-serialise as the plural:\n{dumped}"
        );
        let back: SemanticModel = serde_yaml::from_str(&dumped).expect("deserialize");
        assert_eq!(
            membership(interface(&back), name),
            set(&["natural"]),
            "{label}: membership changed across the dump"
        );
    }
}

#[test]
fn membership_of_exactly_default_is_not_emitted_as_a_keysets_list() {
    let model = build(&model_yaml("", &inline_site("          is_key: true\n")));
    let dumped = serde_yaml::to_string(&model).expect("serialize");
    assert!(
        dumped.contains("is_key: true"),
        "activation elided:\n{dumped}"
    );
    assert!(
        !dumped.contains("keysets:"),
        "the reserved `default` must not be spelled out:\n{dumped}"
    );
}

#[test]
fn keyness_is_unrepresentable_off_the_dimension_site() {
    let cases: [(&str, &str, &str, &str); 5] = [
        (
            "pooled dimension",
            "is_key",
            "Dimension",
            "  dimensions:\n    - name: country\n      data_type: string\n      \
             type: categorical\n      is_key: true\n",
        ),
        (
            "pooled dimension, plural",
            "keysets",
            "Dimension",
            "  dimensions:\n    - name: country\n      data_type: string\n      \
             type: categorical\n      keysets: [natural]\n",
        ),
        (
            "pooled dimension, singular",
            "keyset",
            "Dimension",
            "  dimensions:\n    - name: country\n      data_type: string\n      \
             type: categorical\n      keyset: natural\n",
        ),
        (
            "pooled measure",
            "is_key",
            "Measure",
            "  dimensions:\n    - name: country\n      data_type: string\n      \
             type: categorical\n  measures:\n    - name: revenue\n      data_type: double\n      \
             agg: sum\n      is_key: true\n",
        ),
        (
            "pooled metric",
            "keyset",
            "Metric",
            "  dimensions:\n    - name: country\n      data_type: string\n      \
             type: categorical\n  metrics:\n    - name: margin\n      data_type: double\n      \
             expr: country\n      keyset: natural\n",
        ),
    ];
    for (label, key, carrier, pools) in cases {
        let errors = parse_errors(&model_yaml(pools, &ref_site("")));
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ParseErrorKind::UnknownField { field, parent }
                    if field == key && parent == carrier
            )),
            "{label}: expected UnknownField {{ `{key}`, `{carrier}` }}, got {errors:?}"
        );
    }
}

#[test]
fn a_measure_or_metric_site_cannot_spell_keyness() {
    const MEASURE_POOLS: &str =
        "  dimensions:\n    - name: country\n      data_type: string\n      \
         type: categorical\n  measures:\n    - name: revenue\n      \
         data_type: double\n      agg: sum\n  metrics:\n    \
         - name: margin\n      data_type: double\n      \
         expr:\n        divide: [revenue, revenue]\n";
    let cases: [(&str, &str, &str); 4] = [
        (
            "inline measure",
            "Measure",
            "      measures:\n        - name: clicks\n          data_type: long\n          \
             agg: sum\n          is_key: true\n",
        ),
        (
            "measure ref",
            "Measure reference site",
            "      measures:\n        - ref: revenue\n          is_key: true\n",
        ),
        (
            "inline metric",
            "Metric",
            "      metrics:\n        - name: aov\n          data_type: double\n          \
             expr: revenue\n          keyset: natural\n",
        ),
        (
            "metric ref",
            "Metric reference site",
            "      metrics:\n        - ref: margin\n          keysets: [natural]\n",
        ),
    ];
    for (label, carrier, block) in cases {
        let errors = parse_errors(&model_yaml(
            MEASURE_POOLS,
            &format!("{}{block}", ref_site("")),
        ));
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ParseErrorKind::UnknownField { parent, .. } if parent == carrier
            )),
            "{label}: expected an UnknownField on `{carrier}`, got {errors:?}"
        );
    }
}

#[test]
fn a_nested_form_carries_no_interface_to_hang_keyness_on() {
    let (model, warnings) = build_with_warnings(
        "semantic_model:\n  name: keyness\n  grainsets:\n    - name: roll\n      extras: {}\n      \
         datasets:\n        - name: a\n          extras:\n            temporal:\n              \
         events:\n                event_time: t\n              granularity: day\n          \
         is_key: true\n          keysets: [natural]\n          \
         dimensions:\n            - name: region\n              data_type: string\n              \
         type: categorical\n        - name: b\n          \
         extras:\n            temporal:\n              events:\n                \
         event_time: t\n              granularity: day\n",
    );
    for grainset in model.grainsets.values() {
        assert!(
            grainset.semantic_interface.keysets().is_empty(),
            "a nested form has no interface to declare a keyset on"
        );
    }
    for spelling in ["is_key", "keysets"] {
        assert!(
            warnings.iter().any(|w| matches!(
                w,
                ModelBuildErrorKind::Parse(ParseErrorKind::NestedDataKindKeynessIgnored {
                    data_kind,
                    spelling: got,
                }) if data_kind == "a" && got == spelling
            )),
            "`{spelling}:` on a nested form must warn, got {warnings:?}"
        );
    }
}

#[test]
fn a_typod_keyness_key_is_rejected_by_name() {
    let cases: [(&str, &str); 4] = [
        ("kyset", "          kyset: geo\n"),
        ("keysett", "          keysett: [geo]\n"),
        ("is_keys", "          is_keys: true\n"),
        (
            "keyests",
            "          is_key: true\n          keyests: [geo]\n",
        ),
    ];
    for (key, extra) in cases {
        let errors = parse_errors(&model_yaml("", &inline_site(extra)));
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ParseErrorKind::UnknownField { field, parent }
                    if field == key && parent == "Dimension"
            )),
            "`{key}`: expected UnknownField, got {errors:?}"
        );
    }
}

#[test]
fn a_typod_keyness_key_is_rejected_by_name_on_a_ref_site_too() {
    let errors = parse_errors(&model_yaml(POOLS, &ref_site("          kyset: geo\n")));
    assert!(
        errors.iter().any(|e| matches!(
            e,
            ParseErrorKind::UnknownField { field, parent }
                if field == "kyset" && parent == "Dimension reference site"
        )),
        "expected UnknownField on the reference site, got {errors:?}"
    );
}

#[test]
fn every_complex_public_form_carries_keyness_on_its_own_interface() {
    const CHILDREN: &str =
        "        - name: a\n          extras:\n            temporal:\n              \
         events:\n                event_time: country\n              \
         granularity: day\n        - name: b\n          extras:\n            \
         temporal:\n              events:\n                \
         event_time: country\n              granularity: day\n";
    const IFACE: &str =
        "      dimensions:\n        - ref: country\n          is_key: true\n          \
                         keyset: natural\n";
    for (label, block) in [
        (
            "grainset",
            format!(
                "  grainsets:\n    - name: roll\n      extras: {{}}\n{IFACE}      \
                 datasets:\n{CHILDREN}"
            ),
        ),
        (
            "unionset",
            format!(
                "  unionsets:\n    - name: all\n      extras: {{}}\n{IFACE}      mode: all\n      \
                 datasets:\n{CHILDREN}"
            ),
        ),
        (
            "joinset",
            format!(
                "  joinsets:\n    - name: joined\n      extras: {{}}\n{IFACE}      \
                 datasets:\n{CHILDREN}"
            ),
        ),
    ] {
        let yaml = format!("semantic_model:\n  name: keyness\n{POOLS}{block}");
        let (model, _warnings) = parse(&yaml)
            .unwrap_or_else(|d| panic!("{label}: parse must succeed, got {d:?}"))
            .build()
            .unwrap_or_else(|d| panic!("{label}: build must succeed, got {d:?}"));
        let iface = complex_interface(&model);
        assert_eq!(
            keysets(iface),
            BTreeMap::from([("natural".to_owned(), vec!["country".to_owned()])]),
            "{label}: complex forms inherit site keyness"
        );
    }
}

#[test]
fn a_key_expression_leaf_no_longer_parses() {
    let errors = parse_errors(&model_yaml(
        "",
        "      dimensions:\n        - name: region\n          data_type: string\n          \
         type: categorical\n          expr:\n            key: order_id\n",
    ));
    assert!(
        errors.iter().any(|e| matches!(
            e,
            ParseErrorKind::UnknownTag { tag } if tag == "key"
        )),
        "expected `key:` to be an unknown expression tag, got {errors:?}"
    );
}

#[test]
fn the_retired_keys_block_reaches_nothing_and_says_so() {
    for site in [
        "",
        "          expr: {lit: {decimal: 99999999999999999999}}\n",
    ] {
        let (model, warnings) = build_with_warnings(&model_yaml(
            "",
            &format!(
                "{}      keys:\n        primary:\n          fields: [region]\n",
                inline_site(site)
            ),
        ));
        assert!(keysets(interface(&model)).is_empty());
        assert!(
            warnings.iter().any(|w| matches!(
                w,
                ModelBuildErrorKind::Parse(ParseErrorKind::DataKindKeysBlockRetired { data_kind })
                    if data_kind == "orders"
            )),
            "a retired `keys:` block must warn, got {warnings:?}"
        );
    }
}

#[test]
fn a_data_kind_without_a_retired_spelling_draws_no_warning() {
    let (_model, warnings) = build_with_warnings(&model_yaml(
        "",
        &inline_site("          is_key: true\n          keyset: natural\n"),
    ));
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
}

#[test]
fn a_dimension_site_can_be_marked_a_key_from_code() {
    use semstrait_model::{Dataset, Dimension, DimensionType};
    use semstrait_sql::datatype::DataType;

    let inline = |name: &str| {
        DimensionSite::inline(
            Dimension::builder(name)
                .data_type(DataType::String)
                .dim_type(DimensionType::categorical())
                .build(),
        )
    };
    let dataset = Dataset::builder("orders")
        .table("warehouse.orders")
        .dimension(inline("alpha").with_key())
        .dimension(
            inline("bravo")
                .with_keysets(["natural", "surrogate"])
                .expect("named keysets"),
        )
        .dimension(
            DimensionSite::r#ref("country")
                .with_keysets(["natural"])
                .expect("named keysets"),
        )
        .build();

    assert_eq!(
        keysets(&dataset.semantic_interface),
        BTreeMap::from([
            ("default".to_owned(), vec!["alpha".to_owned()]),
            (
                "natural".to_owned(),
                vec!["bravo".to_owned(), "country".to_owned()]
            ),
            ("surrogate".to_owned(), vec!["bravo".to_owned()]),
        ])
    );
}

#[track_caller]
fn refused_from_code(names: &[&str]) -> [ModelBuildErrorKind; 2] {
    let constructed = Keyness::keysets_named(names.to_vec())
        .expect_err("a list naming no keyset is not a membership")
        .kind;
    let at_site = DimensionSite::r#ref("country")
        .with_keysets(names.to_vec())
        .expect_err("the site-level setter must propagate")
        .kind;
    [constructed, at_site]
}

#[test]
fn an_empty_keyset_list_is_refused_from_code_too() {
    for kind in refused_from_code(&[]) {
        assert_eq!(
            kind,
            ModelBuildErrorKind::Validate(ValidateErrorKind::KeysetListEmpty)
        );
        assert!(kind.message().contains("Keyness::key()"), "{kind:?}");
    }
}

#[test]
fn a_blank_keyset_name_is_refused_from_code_too() {
    for names in [&[""][..], &["  "], &["\t"], &["natural", ""]] {
        for kind in refused_from_code(names) {
            assert_eq!(
                kind,
                ModelBuildErrorKind::Validate(ValidateErrorKind::KeysetNameBlank),
                "{names:?}"
            );
        }
    }
}

#[test]
fn the_default_membership_spellings_stay_infallible() {
    assert_eq!(Keyness::key().keyset_membership(), set(&["default"]));
    assert_eq!(
        DimensionSite::r#ref("country")
            .with_key()
            .keyness()
            .keyset_membership(),
        set(&["default"])
    );
}
