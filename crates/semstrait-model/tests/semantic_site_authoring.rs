mod common;

use common::agg;
use semstrait_model::parse::parse_with_profile;
use semstrait_model::{
    parse, DimensionSite, EntityBTreeMap, IdentityProfile, MeasureSite, MetricSite,
    ModelBuildErrorKind, ParseErrorKind, SemanticModel, SemanticSite, ValidateErrorKind,
};

fn model_yaml(pools: &str, interface: &str) -> String {
    format!(
        "semantic_model:\n  name: sites\n{pools}  datasets:\n    - name: orders\n      \
         extras:\n        semantic_mapping: auto\n{interface}"
    )
}

fn build(yaml: &str) -> SemanticModel {
    let (model, warnings) = parse(yaml)
        .unwrap_or_else(|d| panic!("parse must succeed, got {d:?}"))
        .build()
        .unwrap_or_else(|d| panic!("build must succeed, got {d:?}"));
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
    model
}

fn parse_errors(yaml: &str) -> Vec<ParseErrorKind> {
    parse(yaml)
        .err()
        .unwrap_or_else(|| panic!("parse must fail for:\n{yaml}"))
        .into_iter()
        .map(|d| d.kind)
        .collect()
}

fn orders(model: &SemanticModel) -> &semstrait_model::Dataset {
    model.datasets.values().next().expect("one dataset")
}

fn site_names<I, R>(sites: &EntityBTreeMap<SemanticSite<I, R>>) -> Vec<String>
where
    I: semstrait_model::AuthoringBody,
    R: semstrait_model::AuthoringBody,
{
    let mut names: Vec<String> = sites
        .values()
        .map(|s| s.name().as_ref().to_owned())
        .collect();
    names.sort();
    names
}

const POOLS: &str = "  dimensions:\n    - name: country\n      data_type: string\n      \
                     type: categorical\n  measures:\n    - name: revenue\n      \
                     data_type: double\n      agg: sum\n  metrics:\n    - name: margin\n      \
                     data_type: double\n      expr:\n        divide: [revenue, revenue]\n";

fn six_forms() -> [(&'static str, &'static str, &'static str); 6] {
    [
        (
            "inline dimension",
            "",
            "      dimensions:\n        - name: region\n          data_type: string\n          \
             type: categorical\n",
        ),
        (
            "ref dimension",
            POOLS,
            "      dimensions:\n        - ref: country\n      measures:\n        - ref: revenue\n      \
             metrics:\n        - ref: margin\n",
        ),
        (
            "inline measure",
            "",
            "      measures:\n        - name: clicks\n          data_type: long\n          \
             agg: sum\n",
        ),
        (
            "ref measure with an expr override",
            POOLS,
            "      dimensions:\n        - ref: country\n      measures:\n        - ref: revenue\n          \
             expr:\n            multiply: [revenue, 2]\n      metrics:\n        - ref: margin\n",
        ),
        (
            "inline metric",
            "",
            "      metrics:\n        - name: aov\n          data_type: double\n          \
             expr:\n            divide: [revenue, orders]\n",
        ),
        (
            "ref metric with filter",
            POOLS,
            "      dimensions:\n        - ref: country\n      measures:\n        - ref: revenue\n      \
             metrics:\n        - ref: margin\n          filter:\n            eq: [country, 1]\n",
        ),
    ]
}

#[test]
fn every_site_form_round_trips_through_the_canonical_model() {
    for (label, pools, interface) in six_forms() {
        let model = build(&model_yaml(pools, interface));
        let yaml = serde_yaml::to_string(&model).expect("serialize");
        let back: SemanticModel = serde_yaml::from_str(&yaml).expect("deserialize");
        assert_eq!(model, back, "{label}: round-trip changed the model");
    }
}

#[test]
fn a_site_serializes_as_one_flat_mapping_with_no_tag_leak() {
    for (label, pools, interface) in six_forms() {
        let model = build(&model_yaml(pools, interface));
        let yaml = serde_yaml::to_string(&model).expect("serialize");
        assert!(!yaml.contains('!'), "{label}: `!`-tag leak:\n{yaml}");
        assert!(
            !yaml.contains("body:"),
            "{label}: internal body nesting leaked into the wire form:\n{yaml}"
        );
    }
}

#[test]
fn a_ref_site_keeps_the_ref_key_and_an_inline_site_keeps_name() {
    let model = build(&model_yaml(
        POOLS,
        "      dimensions:\n        - ref: country\n        - name: region\n          \
         data_type: string\n          type: categorical\n      measures:\n        - ref: revenue\n      \
         metrics:\n        - ref: margin\n",
    ));
    let dims = &orders(&model).semantic_interface.dimensions;
    assert_eq!(site_names(dims), vec!["country", "region"]);
    let country = dims
        .values()
        .find(|s| s.name().as_ref() == "country")
        .expect("country site");
    assert!(matches!(country, DimensionSite::Reference(_)));
    let region = dims
        .values()
        .find(|s| s.name().as_ref() == "region")
        .expect("region site");
    assert!(matches!(region, DimensionSite::Inline(_)));

    let yaml = serde_yaml::to_string(&model).expect("serialize");
    assert!(yaml.contains("ref: country"), "ref key elided:\n{yaml}");
    assert!(yaml.contains("name: region"), "name key elided:\n{yaml}");
}

#[test]
fn a_typod_key_is_rejected_by_name_at_every_body_level() {
    let cases: [(&str, &str, &str, &str); 10] = [
        (
            "SemanticBody level, inline",
            "exprr",
            "Dimension",
            "      dimensions:\n        - name: region\n          data_type: string\n          \
             type: categorical\n          exprr: 1\n",
        ),
        (
            "SemanticBody level, reference",
            "exprr",
            "Dimension reference site",
            "      dimensions:\n        - ref: country\n          exprr: 1\n",
        ),
        (
            "NamedSemanticBody level, inline",
            "data_typ",
            "Dimension",
            "      dimensions:\n        - name: region\n          data_typ: string\n          \
             type: categorical\n",
        ),
        (
            "NamedSemanticBody level, misspelled name",
            "naem",
            "Dimension",
            "      dimensions:\n        - naem: region\n          data_type: string\n          \
             type: categorical\n",
        ),
        (
            "Dimension leaf level",
            "typ",
            "Dimension",
            "      dimensions:\n        - name: region\n          data_type: string\n          \
             typ: categorical\n",
        ),
        (
            "Measure leaf level",
            "aggg",
            "Measure",
            "      measures:\n        - name: clicks\n          data_type: long\n          \
             aggg: sum\n",
        ),
        (
            "Measure reference level",
            "filtres",
            "Measure reference site",
            "      measures:\n        - ref: revenue\n          filtres: []\n",
        ),
        (
            "Metric leaf level",
            "additivty",
            "Metric",
            "      metrics:\n        - name: aov\n          data_type: double\n          \
             expr: revenue\n          additivty: full\n",
        ),
        (
            "Metric reference level",
            "exp",
            "Metric reference site",
            "      metrics:\n        - ref: margin\n          exp: revenue\n",
        ),
        (
            "wholly bogus key",
            "totally_bogus",
            "Dimension",
            "      dimensions:\n        - name: region\n          data_type: string\n          \
             type: categorical\n          totally_bogus: 1\n",
        ),
    ];

    for (label, key, carrier, interface) in cases {
        let errors = parse_errors(&model_yaml(POOLS, interface));
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ParseErrorKind::UnknownField { field, parent }
                    if field == key && parent == carrier
            )),
            "{label}: expected UnknownField {{ field: `{key}`, parent: `{carrier}` }}, got {errors:?}"
        );
    }
}

#[test]
fn a_root_pool_body_also_rejects_a_typod_key_by_name() {
    let yaml = model_yaml(
        "  dimensions:\n    - name: country\n      data_type: string\n      \
         type: categorical\n      bogus: 1\n",
        "      dimensions:\n        - ref: country\n",
    );
    let errors = parse_errors(&yaml);
    assert!(
        errors.iter().any(|e| matches!(
            e,
            ParseErrorKind::UnknownField { field, parent }
                if field == "bogus" && parent == "Dimension"
        )),
        "expected UnknownField for the pooled body, got {errors:?}"
    );
}

#[test]
fn a_reference_site_cannot_spell_a_non_overridable_field() {
    let cases = [
        (
            "description on a dimension ref",
            "description",
            "      dimensions:\n        - ref: country\n          description: nope\n",
        ),
        (
            "ai_context on a dimension ref",
            "ai_context",
            "      dimensions:\n        - ref: country\n          ai_context:\n            \
             synonyms: [x]\n",
        ),
        (
            "data_type on a dimension ref",
            "data_type",
            "      dimensions:\n        - ref: country\n          data_type: string\n",
        ),
        (
            "agg on a measure ref",
            "agg",
            "      measures:\n        - ref: revenue\n          agg: sum\n",
        ),
        (
            "additivity on a measure ref",
            "additivity",
            "      measures:\n        - ref: revenue\n          additivity: full\n",
        ),
        (
            "agg on a metric ref",
            "agg",
            "      metrics:\n        - ref: margin\n          agg: sum\n",
        ),
    ];
    for (label, key, interface) in cases {
        let errors = parse_errors(&model_yaml(POOLS, interface));
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ParseErrorKind::UnknownField { field, .. } if field == key)),
            "{label}: expected UnknownField for `{key}`, got {errors:?}"
        );
    }
}

#[test]
fn a_reference_site_can_spell_the_two_overridable_fields() {
    build(&model_yaml(
        POOLS,
        "      dimensions:\n        - ref: country\n          expr: country\n      \
         measures:\n        - ref: revenue\n          expr: revenue\n          \
         filter:\n            eq: [country, 1]\n      metrics:\n        - ref: margin\n          \
         expr:\n            divide: [revenue, revenue]\n",
    ));
}

#[test]
fn a_measure_carries_one_filter_predicate_and_no_filters_roster() {
    let model = build(&model_yaml(
        "",
        "      dimensions:\n        - name: status\n          data_type: string\n          \
         type: categorical\n      measures:\n        - name: revenue\n          \
         data_type: double\n          agg: sum\n          filter:\n            \
         eq: [status, {lit: paid}]\n",
    ));
    let site = orders(&model)
        .semantic_interface
        .measures
        .values()
        .next()
        .expect("one measure site");
    let SemanticSite::Inline(revenue) = site else {
        panic!("the measure is declared inline, got {site:?}")
    };
    assert!(revenue.filter.is_some());

    for (label, interface) in [
        (
            "Measure",
            "      measures:\n        - name: revenue\n          data_type: double\n          \
             agg: sum\n          filters:\n            - name: paid\n              \
             expr: status\n",
        ),
        (
            "Metric",
            "      metrics:\n        - name: aov\n          data_type: double\n          \
             expr: revenue\n          filters:\n            - name: paid\n              \
             expr: status\n",
        ),
        (
            "Measure reference site",
            "      measures:\n        - ref: revenue\n          filters:\n            - name: paid\n              \
             expr: status\n",
        ),
        (
            "Metric reference site",
            "      metrics:\n        - ref: margin\n          filters:\n            - name: paid\n              \
             expr: status\n",
        ),
    ] {
        let errors = parse_errors(&model_yaml(POOLS, interface));
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ParseErrorKind::UnknownField { field, .. } if field == "filters"
            )),
            "{label}: expected UnknownField for `filters`, got {errors:?}"
        );
    }
}

#[test]
fn an_aggregate_tag_in_a_body_is_a_parse_error_that_points_at_agg() {
    for (label, interface) in [
        (
            "measure expr",
            "      measures:\n        - name: revenue\n          data_type: double\n          \
             agg: sum\n          expr:\n            sum: amount\n",
        ),
        (
            "measure filter",
            "      measures:\n        - name: revenue\n          data_type: double\n          \
             agg: sum\n          filter:\n            gt: [{sum: amount}, {lit: 0}]\n",
        ),
        (
            "data kind filter expr",
            "      filters:\n        - name: big\n          expr:\n            \
             sum: amount\n",
        ),
    ] {
        let messages: Vec<String> = parse(&model_yaml("", interface))
            .err()
            .unwrap_or_else(|| panic!("{label}: parse must fail"))
            .into_iter()
            .map(|d| d.to_string())
            .collect();
        assert!(
            messages
                .iter()
                .any(|m| m.contains("sum") && m.contains("agg:")),
            "{label}: diagnostic must name the tag and `agg:`, got {messages:?}"
        );
    }
}

#[test]
fn name_together_with_ref_is_rejected() {
    for (expected, interface) in [
        (
            "Dimension",
            "      dimensions:\n        - name: region\n          ref: country\n          \
             data_type: string\n          type: categorical\n",
        ),
        (
            "Measure",
            "      measures:\n        - name: clicks\n          ref: revenue\n          \
             data_type: long\n          agg: sum\n",
        ),
        (
            "Metric",
            "      metrics:\n        - name: aov\n          ref: margin\n          \
             data_type: double\n          expr: revenue\n",
        ),
    ] {
        let errors = parse_errors(&model_yaml(POOLS, interface));
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ParseErrorKind::SemanticsSiteNameWithRef { carrier } if carrier == expected
            )),
            "{expected}: expected SemanticsSiteNameWithRef, got {errors:?}"
        );
    }
}

#[test]
fn each_spelling_alone_still_parses() {
    build(&model_yaml(
        POOLS,
        "      dimensions:\n        - ref: country\n        - name: region\n          \
         data_type: string\n          type: categorical\n      measures:\n        \
         - ref: revenue\n      metrics:\n        - ref: margin\n",
    ));
}

#[test]
fn a_site_without_an_id_is_minted_one_and_an_authored_id_survives() {
    const AUTHORED: &str = "0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e";
    let model = build(&model_yaml(
        POOLS,
        &format!(
            "      dimensions:\n        - ref: country\n        - name: region\n          \
             id: {AUTHORED}\n          data_type: string\n          type: categorical\n      \
             measures:\n        - ref: revenue\n      metrics:\n        - ref: margin\n      \
             filters:\n        - name: paid\n          expr:\n            eq: [country, 1]\n"
        ),
    ));
    let iface = &orders(&model).semantic_interface;

    let minted = iface
        .dimensions
        .values()
        .find(|s| s.name().as_ref() == "country")
        .expect("country site")
        .body()
        .id
        .clone()
        .expect("a reference site is minted an id");
    assert!(
        uuid::Uuid::parse_str(&minted).is_ok(),
        "minted site id is not UUID text: {minted}"
    );

    assert!(
        iface.dimensions.contains_key(AUTHORED),
        "authored site id absent from the id-keyed map: {:?}",
        iface.dimensions.keys().collect::<Vec<_>>()
    );

    let filter_id = iface
        .filters
        .values()
        .next()
        .expect("one filter")
        .id
        .clone()
        .expect("a filter is minted an id");
    assert!(iface.filters.contains_key(&filter_id));
}

const ID_LESS_MEMBERS: &str = "      dimensions:\n        - ref: country\n        - name: region\n          \
     data_type: string\n          type: categorical\n      measures:\n        - ref: revenue\n      \
     metrics:\n        - ref: margin\n      filters:\n        - name: paid\n          \
     expr:\n            eq: [country, 1]\n";

#[test]
fn the_strict_profile_reports_an_id_less_site_and_filter_rather_than_minting() {
    let err = parse_with_profile(
        &model_yaml(POOLS, ID_LESS_MEMBERS),
        IdentityProfile::StrictRequireProvided,
    )
    .expect("parse ok")
    .build()
    .expect_err("strict must refuse an id-less model");

    for path in [
        "orders.dimensions.country",
        "orders.dimensions.region",
        "orders.measures.revenue",
        "orders.metrics.margin",
        "orders.filters.paid",
    ] {
        assert!(
            err.iter().any(|d| matches!(
                &d.kind,
                ModelBuildErrorKind::Validate(ValidateErrorKind::MissingEntityId { path: got })
                    if got == path
            )),
            "expected MissingEntityId for `{path}`; got {err:?}"
        );
    }
}

#[test]
fn the_convenience_profile_still_mints_every_id_less_member() {
    let (model, warnings) = parse_with_profile(
        &model_yaml(POOLS, ID_LESS_MEMBERS),
        IdentityProfile::ConvenienceGenerateMissing,
    )
    .expect("parse ok")
    .build()
    .expect("convenience must mint");
    assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");

    let iface = &orders(&model).semantic_interface;
    assert_eq!(iface.dimensions.len(), 2);
    assert_eq!(iface.measures.len(), 1);
    assert_eq!(iface.metrics.len(), 1);
    assert_eq!(iface.filters.len(), 1);
    for site in iface.dimensions.values() {
        let id = site.body().id.as_deref().expect("minted");
        assert!(
            iface.dimensions.contains_key(id),
            "map key is the minted id"
        );
    }
    for filter in iface.filters.values() {
        assert!(filter.id.is_some(), "a filter mints like a site");
    }
}

#[test]
fn a_filter_has_no_reference_form() {
    let errors = parse_errors(&model_yaml(
        POOLS,
        "      dimensions:\n        - ref: country\n      filters:\n        - ref: paid\n",
    ));
    assert!(
        errors
            .iter()
            .any(|e| matches!(e, ParseErrorKind::UnknownField { field, .. } if field == "ref")),
        "expected `ref` to be an unknown filter key, got {errors:?}"
    );
}

#[test]
fn the_wire_projection_is_name_ordered_regardless_of_authored_order() {
    let model = build(&model_yaml(
        "",
        "      dimensions:\n        - name: zulu\n          data_type: string\n          \
         type: categorical\n        - name: mike\n          data_type: string\n          \
         type: categorical\n        - name: alpha\n          data_type: string\n          \
         type: categorical\n",
    ));
    let json = serde_json::to_value(&model).expect("serialize");
    let emitted: Vec<&str> = json["datasets"][0]["dimensions"]
        .as_array()
        .expect("dimensions array")
        .iter()
        .map(|d| d["name"].as_str().expect("name"))
        .collect();
    assert_eq!(emitted, ["alpha", "mike", "zulu"]);
}

#[test]
fn a_repeated_site_id_is_rejected_on_decode() {
    const SHARED: &str = "0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e";
    let errors = parse_errors(&model_yaml(
        "",
        &format!(
            "      dimensions:\n        - name: alpha\n          id: {SHARED}\n          \
             data_type: string\n          type: categorical\n        - name: bravo\n          \
             id: {SHARED}\n          data_type: string\n          type: categorical\n"
        ),
    ));
    assert!(
        errors.iter().any(|e| matches!(
            e,
            ParseErrorKind::YamlSyntax { message } if message.contains("duplicate entity id")
        )),
        "expected the dup-id gate to fire, got {errors:?}"
    );
}

#[test]
fn data_type_is_mandatory_on_every_inline_site() {
    for (label, interface) in [
        (
            "dimension",
            "      dimensions:\n        - name: region\n          type: categorical\n",
        ),
        (
            "measure",
            "      measures:\n        - name: clicks\n          agg: sum\n",
        ),
        (
            "metric",
            "      metrics:\n        - name: aov\n          expr: revenue\n",
        ),
    ] {
        let errors = parse_errors(&model_yaml("", interface));
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ParseErrorKind::SemanticsMissingDataType { .. })),
            "{label}: expected SemanticsMissingDataType, got {errors:?}"
        );
    }
}

#[test]
fn the_three_site_aliases_are_reachable_from_code() {
    use semstrait_model::{Dataset, Dimension, DimensionType, Measure, Metric};

    let dataset = Dataset::builder("orders")
        .table("warehouse.orders")
        .dimension(DimensionSite::inline(
            Dimension::builder("region")
                .data_type(semstrait_sql::datatype::DataType::String)
                .dim_type(DimensionType::categorical())
                .build(),
        ))
        .dimension(DimensionSite::r#ref("country"))
        .measure(MeasureSite::inline(
            Measure::builder("clicks")
                .data_type(semstrait_sql::datatype::DataType::BigInt)
                .agg(agg("sum"))
                .build(),
        ))
        .measure(MeasureSite::r#ref("revenue"))
        .metric(MetricSite::inline(
            Metric::builder("aov")
                .data_type(semstrait_sql::datatype::DataType::Double)
                .build(),
        ))
        .metric(MetricSite::r#ref("margin"))
        .build();

    let iface = &dataset.semantic_interface;
    assert_eq!(site_names(&iface.dimensions), vec!["country", "region"]);
    assert_eq!(site_names(&iface.measures), vec!["clicks", "revenue"]);
    assert_eq!(site_names(&iface.metrics), vec!["aov", "margin"]);
    for site in iface.dimensions.values() {
        assert!(
            site.body().id.is_some(),
            "the push-style setter must mint the site id"
        );
    }
}

#[test]
fn the_two_renamed_site_bodies_are_reachable_by_their_new_names() {
    use semstrait_model::{Dimension, DimensionDecl, DimensionType, Reference};
    use semstrait_sql::datatype::DataType;

    let declared: DimensionDecl = Dimension::builder("region")
        .data_type(DataType::String)
        .dim_type(DimensionType::categorical())
        .build()
        .into();
    let site = DimensionSite::inline(declared);
    assert_eq!(site.name().as_ref(), "region");
    assert!(
        site.keyness().keyset_membership().is_empty(),
        "a declaration promoted from the pool is not a key site"
    );

    let reference: Reference = Reference::new("country");
    assert_eq!(reference.entity_name().as_ref(), "country");
}

#[test]
fn the_data_kind_filter_carries_ai_context_and_the_measure_a_bare_predicate() {
    let yaml = model_yaml(
        "",
        "      dimensions:\n        - name: is_paid\n          data_type: bool\n          \
         type: binary\n      measures:\n        - name: revenue\n          \
         data_type: double\n          agg: sum\n          filter:\n            \
         eq: [is_paid, true]\n      filters:\n        \
         - name: paid_rows\n          expr:\n            eq: [is_paid, true]\n          \
         ai_context:\n            description: the paid slice of the rowset\n",
    );
    let model = build(&yaml);
    let iface = &orders(&model).semantic_interface;

    let data_kind_filter = iface.filters.values().next().expect("one DataKind filter");
    assert_eq!(
        data_kind_filter
            .ai_context
            .as_ref()
            .and_then(|a| a.description.as_deref()),
        Some("the paid slice of the rowset"),
    );

    let site = iface.measures.values().next().expect("one measure site");
    let SemanticSite::Inline(revenue) = site else {
        panic!("the measure is declared inline, got {site:?}")
    };
    assert!(revenue.filter.is_some(), "the aggregation predicate parsed");
}
