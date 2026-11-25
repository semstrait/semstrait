use crate::catalog::CatalogsConfig;
use crate::error::catalogs::CatalogsParseErrorKind;
use crate::error::parse::ParseErrorKind;
use crate::error::serde_message;
use crate::model::builder::SemanticModelBuilder;
use crate::model::identity::{validate_authored_id, IdentityProfile};
use crate::model::serde::digits::Digits;
use crate::model::types::data_kinds::{
    AnyDataKindRef, ComplexDataKind, ComplexDataKindRef, DataKind,
};
use crate::model::types::entities::{Relationship, SemanticSite};
use crate::yaml::env::{substitute_env_for_catalogs, substitute_env_for_model};
use crate::yaml::{retired, YamlRoot};
use semstrait_common::diagnostic::{Diagnostic, Diagnostics, Location, SourceId, Span};
use serde::Deserialize;

const DEFAULT_SOURCE: &str = "<inline>";

pub fn parse(input: &str) -> Result<SemanticModelBuilder, Diagnostics<ParseErrorKind>> {
    parse_with_source(input, DEFAULT_SOURCE)
}

pub fn parse_with_profile(
    input: &str,
    profile: IdentityProfile,
) -> Result<SemanticModelBuilder, Diagnostics<ParseErrorKind>> {
    parse_inner(input, DEFAULT_SOURCE, profile)
}

pub fn parse_with_source(
    input: &str,
    source: &str,
) -> Result<SemanticModelBuilder, Diagnostics<ParseErrorKind>> {
    parse_inner(input, source, IdentityProfile::default())
}

fn parse_inner(
    input: &str,
    source: &str,
    profile: IdentityProfile,
) -> Result<SemanticModelBuilder, Diagnostics<ParseErrorKind>> {
    let (builder, diagnostics) =
        lower_source_into(input, source, SemanticModelBuilder::with_profile(profile));
    if diagnostics.is_empty() {
        Ok(builder)
    } else {
        Err(diagnostics)
    }
}

pub(crate) fn lower_source_into(
    input: &str,
    source: &str,
    builder: SemanticModelBuilder,
) -> (SemanticModelBuilder, Diagnostics<ParseErrorKind>) {
    let expanded = match substitute_env_for_model(input) {
        Ok(s) => s,
        Err(diag) => return (builder, vec![diag]),
    };

    let yaml = Digits::new(serde_yaml::Deserializer::from_str(&expanded));
    let root = match YamlRoot::deserialize(yaml) {
        Ok(r) => r,
        Err(e) => return (builder, vec![Diagnostic::new(classify_yaml_error(&e))]),
    };

    let mut builder = root.lower_into(source, builder);

    for warning in retired::retired_data_kind_spellings(&expanded, source) {
        builder.push_parse_warning(warning);
    }

    let mut diagnostics: Diagnostics<ParseErrorKind> = Vec::new();
    check_identifiers(&builder, source, &mut diagnostics);

    (builder, diagnostics)
}

pub fn parse_catalogs(
    input: &str,
) -> Result<
    (CatalogsConfig, Diagnostics<CatalogsParseErrorKind>),
    Diagnostics<CatalogsParseErrorKind>,
> {
    parse_catalogs_with_source(input, DEFAULT_SOURCE)
}

pub fn parse_catalogs_with_source(
    input: &str,
    _source: &str,
) -> Result<
    (CatalogsConfig, Diagnostics<CatalogsParseErrorKind>),
    Diagnostics<CatalogsParseErrorKind>,
> {
    let expanded = match substitute_env_for_catalogs(input) {
        Ok(s) => s,
        Err(diag) => return Err(vec![diag]),
    };

    let cfg: CatalogsConfig = match serde_yaml::from_str(&expanded) {
        Ok(c) => c,
        Err(e) => {
            let kind = classify_catalogs_yaml_error(&e);
            return Err(vec![Diagnostic::new(kind)]);
        }
    };

    Ok((cfg, Vec::new()))
}

fn classify_yaml_error(e: &serde_yaml::Error) -> ParseErrorKind {
    let msg = e.to_string();

    if let Some(kind) = serde_message::classify(&msg) {
        return kind;
    }

    if let Some(field) = serde_message::backticked_after(&msg, "missing field `") {
        match field.as_str() {
            "agg" => {
                return ParseErrorKind::MeasureMissingAgg {
                    carrier: "Measure".to_string(),
                    name: "<unknown>".to_string(),
                };
            }
            "data_type" => {
                return ParseErrorKind::SemanticsMissingDataType {
                    carrier: "<unknown>".to_string(),
                    name: "<unknown>".to_string(),
                };
            }
            _ => {}
        }
    }

    ParseErrorKind::YamlSyntax { message: msg }
}

fn classify_catalogs_yaml_error(e: &serde_yaml::Error) -> CatalogsParseErrorKind {
    let msg = e.to_string();
    if msg.starts_with("unknown field `") {
        if let Some(rest) = msg.strip_prefix("unknown field `") {
            if let Some(end) = rest.find('`') {
                let field = &rest[..end];
                return CatalogsParseErrorKind::UnknownField {
                    field: field.to_string(),
                    parent: "<unknown>".to_string(),
                };
            }
        }
    }

    if msg.starts_with("missing field `") {
        if let Some(rest) = msg.strip_prefix("missing field `") {
            if let Some(end) = rest.find('`') {
                let field = &rest[..end];
                return CatalogsParseErrorKind::CatalogMissingField {
                    alias: "<unknown>".to_string(),
                    field: field.to_string(),
                };
            }
        }
    }

    CatalogsParseErrorKind::YamlSyntax { message: msg }
}

fn is_valid_identifier(raw: &str) -> bool {
    let mut chars = raw.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

const INVALID_IDENT_REASON: &str = "must match `[A-Za-z_][A-Za-z0-9_]*`";

fn push_invalid(raw: &str, source: &str, diagnostics: &mut Diagnostics<ParseErrorKind>) {
    if !is_valid_identifier(raw) {
        let loc = Location::new(SourceId::named(source), Span::unknown());
        diagnostics.push(
            Diagnostic::new(ParseErrorKind::InvalidIdentifier {
                raw: raw.to_string(),
                reason: INVALID_IDENT_REASON.to_string(),
            })
            .with_location(loc),
        );
    }
}

fn check_identifiers(
    builder: &SemanticModelBuilder,
    source: &str,
    diagnostics: &mut Diagnostics<ParseErrorKind>,
) {
    for d in builder.datasets.values() {
        check_any_data_kind(AnyDataKindRef::Dataset(d), source, diagnostics);
    }
    for g in builder.grainsets.values() {
        check_any_data_kind(AnyDataKindRef::Grainset(g), source, diagnostics);
    }
    for u in builder.unionsets.values() {
        check_any_data_kind(AnyDataKindRef::Unionset(u), source, diagnostics);
    }
    for j in builder.joinsets.values() {
        check_any_data_kind(AnyDataKindRef::Joinset(j), source, diagnostics);
    }

    for d in builder.dimensions.values() {
        push_invalid(d.name().as_str(), source, diagnostics);
    }
    for m in builder.measures.values() {
        push_invalid(m.name().as_str(), source, diagnostics);
    }
    for m in builder.metrics.values() {
        push_invalid(m.name().as_str(), source, diagnostics);
    }

    for r in builder.relationships.values() {
        push_invalid(&r.name, source, diagnostics);
    }

    for id in builder.id_locations.keys() {
        push_invalid_entity_id(id, source, diagnostics);
    }

    if let Some(id) = &builder.id {
        push_invalid_entity_id(id, source, diagnostics);
    }
}

fn push_invalid_entity_id(id: &str, source: &str, diagnostics: &mut Diagnostics<ParseErrorKind>) {
    if let Err(reason) = validate_authored_id(id) {
        let loc = Location::new(SourceId::named(source), Span::unknown());
        diagnostics.push(
            Diagnostic::new(ParseErrorKind::InvalidEntityId {
                raw: id.to_string(),
                reason,
            })
            .with_location(loc),
        );
    }
}

fn check_any_data_kind(
    any: AnyDataKindRef<'_>,
    source: &str,
    diagnostics: &mut Diagnostics<ParseErrorKind>,
) {
    match any {
        AnyDataKindRef::Dataset(d) => {
            push_invalid(&d.body.base.name, source, diagnostics);
            check_interface(&d.semantic_interface, source, diagnostics);
        }
        AnyDataKindRef::Grainset(g) => {
            push_invalid(&g.body.base.name, source, diagnostics);
            check_interface(&g.semantic_interface, source, diagnostics);
            check_complex_children(ComplexDataKindRef::Grainset(g), source, diagnostics);
        }
        AnyDataKindRef::Unionset(u) => {
            push_invalid(&u.body.base.name, source, diagnostics);
            check_interface(&u.semantic_interface, source, diagnostics);
            check_complex_children(ComplexDataKindRef::Unionset(u), source, diagnostics);
        }
        AnyDataKindRef::Joinset(j) => {
            push_invalid(&j.body.base.name, source, diagnostics);
            check_interface(&j.semantic_interface, source, diagnostics);
            check_complex_children(ComplexDataKindRef::Joinset(j), source, diagnostics);
            check_joinset_local_relationships(&j.body.relationships, source, diagnostics);
        }
        _ => {}
    }
}

fn check_complex_children(
    parent: ComplexDataKindRef<'_>,
    source: &str,
    diagnostics: &mut Diagnostics<ParseErrorKind>,
) {
    for child in parent.children_ref() {
        push_invalid(DataKind::name(&child), source, diagnostics);

        match child {
            crate::model::types::data_kinds::NestedDataKindRef::Grainset(ng) => {
                check_complex_children(ComplexDataKindRef::NestedGrainset(ng), source, diagnostics);
            }
            crate::model::types::data_kinds::NestedDataKindRef::Unionset(nu) => {
                check_complex_children(ComplexDataKindRef::NestedUnionset(nu), source, diagnostics);
            }
            crate::model::types::data_kinds::NestedDataKindRef::Joinset(nj) => {
                check_complex_children(ComplexDataKindRef::NestedJoinset(nj), source, diagnostics);
                check_joinset_local_relationships(&nj.body.relationships, source, diagnostics);
            }
            _ => {}
        }
    }
}

fn check_joinset_local_relationships(
    rels: &[Relationship],
    source: &str,
    diagnostics: &mut Diagnostics<ParseErrorKind>,
) {
    for r in rels {
        push_invalid(&r.name, source, diagnostics);
        if let Some(id) = &r.id {
            push_invalid_entity_id(id, source, diagnostics);
        }
    }
}

fn check_interface(
    iface: &crate::model::types::entities::SemanticInterface,
    source: &str,
    diagnostics: &mut Diagnostics<ParseErrorKind>,
) {
    for name in iface
        .dimensions
        .values()
        .filter(|s| matches!(s, SemanticSite::Inline(_)))
        .map(SemanticSite::name)
    {
        push_invalid(name.as_str(), source, diagnostics);
    }
    for name in iface
        .measures
        .values()
        .filter(|s| matches!(s, SemanticSite::Inline(_)))
        .map(SemanticSite::name)
    {
        push_invalid(name.as_str(), source, diagnostics);
    }
    for name in iface
        .metrics
        .values()
        .filter(|s| matches!(s, SemanticSite::Inline(_)))
        .map(SemanticSite::name)
    {
        push_invalid(name.as_str(), source, diagnostics);
    }
    for f in iface.filters.values() {
        push_invalid(f.name.as_str(), source, diagnostics);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::build::ModelBuildErrorKind;
    use crate::error::validate::ValidateErrorKind;

    #[test]
    fn parse_minimal_model() {
        let yaml = r#"
semantic_model:
  name: tiny
"#;
        let builder = parse(yaml).expect("parse should succeed");
        assert_eq!(builder.name_view(), "tiny");
        let err = builder.build().unwrap_err();
        assert!(err.iter().any(|d| matches!(
            d.kind,
            ModelBuildErrorKind::Validate(ValidateErrorKind::EmptyModel)
        )));
    }

    #[test]
    fn parse_unknown_top_level_block() {
        let yaml = r#"
semantic_model:
  name: x
  bogus_block: 1
"#;
        let err = parse(yaml).unwrap_err();
        assert!(err.iter().any(|d| matches!(
            d.kind,
            ParseErrorKind::UnknownField { .. } | ParseErrorKind::YamlSyntax { .. }
        )));
    }

    #[test]
    fn parse_dataset_with_single_dimension() {
        let yaml = r#"
semantic_model:
  name: ana
  datasets:
    - name: orders
      dimensions:
        - name: country
          data_type: string
          type: categorical
"#;
        let builder = parse(yaml).expect("parse ok");
        let (m, _diags) = builder.build().expect("build ok");
        let ds = match m.find_public("orders").expect("orders present") {
            crate::model::types::data_kinds::PublicDataKindRef::Dataset(d) => d,
            other => panic!("expected dataset, got {other:?}"),
        };
        assert_eq!(ds.body.base.name, "orders");
        assert_eq!(ds.semantic_interface.dimensions.len(), 1);
    }

    #[test]
    fn parse_then_build_emits_duplicate_data_kind_name() {
        let yaml = r#"
semantic_model:
  name: dup
  datasets:
    - name: x
  grainsets:
    - name: x
      datasets:
        - name: a
        - name: b
"#;
        let builder = parse(yaml).expect("parse ok");
        let err = builder.build().unwrap_err();
        assert!(err.iter().any(|d| matches!(
            &d.kind,
            ModelBuildErrorKind::Validate(ValidateErrorKind::DuplicateDataKindName { .. })
        )));
    }

    #[test]
    fn parse_substitutes_env_var() {
        std::env::set_var("SEMSTRAIT_TEST_VAR", "ana");
        let yaml = r#"
semantic_model:
  name: ${SEMSTRAIT_TEST_VAR}
"#;
        let builder = parse(yaml).expect("parse ok");
        assert_eq!(builder.name_view(), "ana");
        std::env::remove_var("SEMSTRAIT_TEST_VAR");
    }

    #[test]
    fn parse_emits_invalid_entity_id_for_malformed_authored_id() {
        let yaml = r#"
semantic_model:
  name: bad-id
  dimensions:
    - name: country
      id: not-a-uuid
      data_type: string
      type: categorical
"#;
        let err = parse(yaml).unwrap_err();
        assert!(
            err.iter().any(|d| matches!(
                &d.kind,
                ParseErrorKind::InvalidEntityId { raw, .. } if raw == "not-a-uuid"
            )),
            "expected InvalidEntityId for `not-a-uuid`; got {err:?}"
        );
    }

    #[test]
    fn parse_emits_invalid_entity_id_for_a_joinset_local_relationship() {
        let yaml = r#"
semantic_model:
  name: ana
  joinsets:
    - name: orders_with_customers
      extras: {}
      datasets:
        - name: orders_l
          extras:
            semantic_mapping: auto
        - name: customers_r
          extras:
            semantic_mapping: auto
      relationships:
        - name: l_to_r
          id: not-a-uuid
          from: orders_l
          to: customers_r
          on: [customer_id]
          cardinality: many_to_one
"#;
        let err = parse(yaml).unwrap_err();
        assert!(
            err.iter().any(|d| matches!(
                &d.kind,
                ParseErrorKind::InvalidEntityId { raw, .. } if raw == "not-a-uuid"
            )),
            "expected InvalidEntityId for `not-a-uuid`; got {err:?}"
        );
    }

    #[test]
    fn parse_accepts_a_canonical_id_on_a_joinset_local_relationship() {
        let yaml = r#"
semantic_model:
  name: ana
  joinsets:
    - name: orders_with_customers
      extras: {}
      datasets:
        - name: orders_l
          extras:
            semantic_mapping: auto
        - name: customers_r
          extras:
            semantic_mapping: auto
      relationships:
        - name: l_to_r
          id: 0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e
          from: orders_l
          to: customers_r
          on: [customer_id]
          cardinality: many_to_one
"#;
        parse(yaml).expect("a canonical authored id parses");
    }

    #[test]
    fn convenience_profile_derives_an_idless_entitys_id_from_its_name() {
        let yaml = r#"
semantic_model:
  name: mint
  datasets:
    - name: orders
      dimensions:
        - name: country
          data_type: string
          type: categorical
"#;
        let (model, _diags) = parse(yaml).expect("parse ok").build().expect("build ok");
        let ds = model.datasets.values().next().expect("one dataset");
        let id = ds
            .body
            .base
            .id
            .as_deref()
            .expect("convenience profile minted an id");
        assert!(
            crate::model::identity::validate_authored_id(id).is_ok(),
            "minted id is not canonical UUID text: {id}"
        );
        assert_eq!(id, "cd3e4c16-3f42-5527-9ebf-35556686870a");

        let site = ds
            .semantic_interface
            .dimensions
            .values()
            .next()
            .expect("one site");
        assert_eq!(
            site.body().id.as_deref(),
            Some("03f07794-dc7e-5d6b-9ff8-b92b35a79fc3")
        );
    }

    #[test]
    fn strict_profile_emits_missing_entity_id_for_idless_entity() {
        let yaml = r#"
semantic_model:
  name: strict
  dimensions:
    - name: country
      data_type: string
      type: categorical
"#;
        let err = parse_with_profile(yaml, IdentityProfile::StrictRequireProvided)
            .expect("parse ok")
            .build()
            .unwrap_err();
        assert!(
            err.iter().any(|d| matches!(
                &d.kind,
                ModelBuildErrorKind::Validate(ValidateErrorKind::MissingEntityId { path })
                    if path == "country"
            )),
            "expected MissingEntityId for `country`; got {err:?}"
        );
    }

    #[test]
    fn parse_unset_env_var_is_fatal() {
        let yaml = r#"
semantic_model:
  name: ${THIS_VAR_DOES_NOT_EXIST_xyz}
"#;
        let err = parse(yaml).unwrap_err();
        assert_eq!(err.len(), 1);
        assert!(matches!(err[0].kind, ParseErrorKind::UnsetEnvVar { .. }));
    }

    fn idless_model_yaml(name: &str) -> String {
        format!(
            r#"
semantic_model:
  name: {name}
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
"#
        )
    }

    fn root_id(yaml: &str) -> String {
        let (model, _diags) = parse(yaml).expect("parse ok").build().expect("build ok");
        model.id.expect("root id resolved after build")
    }

    #[test]
    fn two_parses_of_the_same_idless_model_derive_the_same_root_id() {
        let yaml = idless_model_yaml("analytics-v1");
        let a = root_id(&yaml);
        let b = root_id(&yaml);
        assert_eq!(a, b, "two independent parses must derive the same root id");
    }

    #[test]
    fn root_id_derivation_keys_on_name_not_content() {
        let a = root_id(&idless_model_yaml("analytics-v1"));
        let b = root_id(&idless_model_yaml("analytics-v2"));
        assert_ne!(a, b, "differing `name` must derive a different root id");

        let same_name_a = idless_model_yaml("analytics-v3");
        let same_name_b = r#"
semantic_model:
  name: analytics-v3
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
    - name: refunds
      extras:
        semantic_mapping: auto
"#;
        let c = root_id(&same_name_a);
        let d = root_id(same_name_b);
        assert_eq!(
            c, d,
            "differing content under the same `name` must derive the same root id"
        );
    }

    #[test]
    fn root_id_derivation_preserves_authored_name_case() {
        let mixed_case = root_id(&idless_model_yaml("Analytics"));
        let lower_case = root_id(&idless_model_yaml("analytics"));
        assert_ne!(
            mixed_case, lower_case,
            "names differing only in case must derive different root ids"
        );

        assert_eq!(
            mixed_case,
            crate::model::identity::derive_root_id("Analytics")
        );
        assert_eq!(
            lower_case,
            crate::model::identity::derive_root_id("analytics")
        );

        let padded = root_id(
            r#"
semantic_model:
  name: "  Analytics  "
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
"#,
        );
        assert_eq!(
            padded,
            crate::model::identity::derive_root_id("  Analytics  "),
            "leading/trailing whitespace in the authored name must survive into the seed"
        );
    }

    #[test]
    fn authored_root_id_round_trips_verbatim_and_wins_over_derivation() {
        const AUTHORED: &str = "0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e";
        let yaml = format!(
            r#"
semantic_model:
  id: "{AUTHORED}"
  name: analytics-v1
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
"#
        );
        let (model, _diags) = parse(&yaml).expect("parse ok").build().expect("build ok");
        assert_eq!(model.id.as_deref(), Some(AUTHORED));
        assert_ne!(
            model.id.as_deref(),
            Some(crate::model::identity::derive_root_id("analytics-v1").as_str())
        );
    }

    #[test]
    fn root_id_equal_to_a_contained_entity_id_builds_clean() {
        const SHARED: &str = "0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e";
        let yaml = format!(
            r#"
semantic_model:
  id: "{SHARED}"
  name: analytics-v1
  dimensions:
    - name: country
      id: "{SHARED}"
      data_type: string
      type: categorical
  datasets:
    - name: orders
      dimensions:
        - ref: country
      extras:
        semantic_mapping: auto
"#
        );
        let (model, _diags) = parse(&yaml).expect("parse ok").build().expect("build ok");
        assert_eq!(model.id.as_deref(), Some(SHARED));
    }

    #[test]
    fn v4_root_id_parses_clean() {
        const V4: &str = "550e8400-e29b-41d4-a716-446655440000";
        let yaml = format!(
            r#"
semantic_model:
  id: "{V4}"
  name: analytics-v1
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
"#
        );
        let (model, _diags) = parse(&yaml).expect("parse ok").build().expect("build ok");
        assert_eq!(model.id.as_deref(), Some(V4));
    }

    #[test]
    fn a_second_authored_root_id_cannot_re_key_the_model() {
        const FIRST: &str = "0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e";
        const SECOND: &str = "550e8400-e29b-41d4-a716-446655440000";
        let yaml = idless_model_yaml("analytics-v1");
        let model = parse(&yaml)
            .expect("parse ok")
            .id(FIRST)
            .id(SECOND)
            .build()
            .map(|(model, _diags)| model)
            .expect("build ok");
        assert_eq!(model.id.as_deref(), Some(FIRST));
    }

    #[test]
    fn missing_root_id_is_flagged_under_strict_and_derived_under_convenience() {
        let yaml = idless_model_yaml("analytics-v1");
        let err = parse_with_profile(&yaml, IdentityProfile::StrictRequireProvided)
            .expect("parse ok")
            .build()
            .unwrap_err();
        assert!(
            err.iter().any(|d| matches!(
                &d.kind,
                ModelBuildErrorKind::Validate(ValidateErrorKind::MissingEntityId { path })
                    if path == "<model root>"
            )),
            "expected MissingEntityId for the model root; got {err:?}"
        );

        let (lenient, _diags) = parse(&yaml).expect("parse ok").build().expect("build ok");
        assert_eq!(
            lenient.id,
            Some(crate::model::identity::derive_root_id("analytics-v1")),
            "convenience must derive the root id"
        );
    }

    #[test]
    fn v4_entity_id_parses_clean() {
        const V4: &str = "550e8400-e29b-41d4-a716-446655440000";
        let yaml = format!(
            r#"
semantic_model:
  name: analytics-v1
  dimensions:
    - name: country
      id: "{V4}"
      data_type: string
      type: categorical
  datasets:
    - name: orders
      dimensions:
        - ref: country
      extras:
        semantic_mapping: auto
"#
        );
        let (model, _diags) = parse(&yaml).expect("parse ok").build().expect("build ok");
        assert!(
            model.dimensions.contains_key(V4),
            "v4 entity id absent from the id-keyed map: {:?}",
            model.dimensions.keys().collect::<Vec<_>>()
        );
    }

    #[test]
    fn malformed_root_ids_yield_invalid_entity_id() {
        let cases = [
            ("00000000-0000-0000-0000-000000000000", "nil"),
            ("0197CB6E-A63A-7D53-8D6F-35F4B1D67D4E", "uppercase"),
            ("{0197cb6e-a63a-7d53-8d6f-35f4b1d67d4e}", "braced"),
            ("0197cb6e-a63a-7d53-8d6f-35f4b1d67d4", "wrong-length"),
        ];
        for (raw, label) in cases {
            let yaml = format!(
                r#"
semantic_model:
  id: "{raw}"
  name: analytics-v1
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
"#
            );
            let err = parse(&yaml).unwrap_err();
            assert!(
                err.iter().any(|d| matches!(
                    &d.kind,
                    ParseErrorKind::InvalidEntityId { raw: got, .. } if got == raw
                )),
                "case `{label}` ({raw}): expected InvalidEntityId; got {err:?}"
            );
        }
    }

    #[test]
    fn a_derived_root_id_persists_itself_into_the_model_bytes() {
        #[derive(serde::Serialize)]
        struct Persisted<'a> {
            semantic_model: &'a crate::model::SemanticModel,
        }

        let yaml = idless_model_yaml("analytics-v1");
        let (model, _diags) = parse(&yaml).expect("parse ok").build().expect("build ok");
        let derived = model.id.clone().expect("root id resolved");

        let bytes = serde_yaml::to_string(&Persisted {
            semantic_model: &model,
        })
        .expect("model serializes");
        assert!(
            bytes.contains(&format!("id: {derived}")),
            "derived root id absent from the persisted bytes:\n{bytes}"
        );

        let (reloaded, _diags) = parse(&bytes)
            .expect("re-parse ok")
            .build()
            .expect("re-build ok");
        assert_eq!(
            reloaded.id.as_deref(),
            Some(derived.as_str()),
            "re-read must treat the once-derived id as authored"
        );

        let renamed = bytes.replace("name: analytics-v1", "name: analytics-v2");
        let (after_rename, _diags) = parse(&renamed)
            .expect("re-parse ok")
            .build()
            .expect("re-build ok");
        assert_eq!(
            after_rename.id.as_deref(),
            Some(derived.as_str()),
            "a rename after the first save must not re-key the model"
        );
    }
}
