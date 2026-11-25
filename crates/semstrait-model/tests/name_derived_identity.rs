use semstrait_model::{parse, ModelBuildErrorKind, SemanticModel, ValidateErrorKind};

fn document(dataset: &str, dimension: &str) -> String {
    format!(
        r#"
semantic_model:
  name: ana
  dimensions:
    - name: {dimension}
      data_type: string
      type: categorical
  datasets:
    - name: {dataset}
      extras:
        semantic_mapping: auto
      dimensions:
        - ref: {dimension}
"#
    )
}

fn build(yaml: &str) -> SemanticModel {
    let (model, _warnings) = parse(yaml)
        .expect("parse ok")
        .build()
        .expect("the fixture document must build clean");
    model
}

fn ids(model: &SemanticModel) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for (id, dimension) in model.dimensions.iter() {
        out.push((format!("dimension {}", dimension.name()), id.clone()));
    }
    for (id, dataset) in model.datasets.iter() {
        out.push((format!("dataset {}", dataset.body.base.name), id.clone()));
        for (site_id, site) in dataset.semantic_interface.dimensions.iter() {
            out.push((
                format!("site {}.dimensions.{}", dataset.body.base.name, site.name()),
                site_id.clone(),
            ));
        }
    }
    out.sort();
    out
}

#[test]
fn one_document_read_twice_yields_one_set_of_ids() {
    let yaml = document("orders", "region");
    assert_eq!(ids(&build(&yaml)), ids(&build(&yaml)));
}

#[test]
fn renaming_an_entity_moves_its_id() {
    let before = ids(&build(&document("orders", "region")));
    let after = ids(&build(&document("purchases", "territory")));

    let before_ids: Vec<&String> = before.iter().map(|(_, id)| id).collect();
    for (entity, id) in &after {
        assert!(
            !before_ids.contains(&id),
            "{entity} kept its id across a rename, so its id is not derived \
             from its name"
        );
    }
    let site_before = ids(&build(&document("orders", "region")));
    let site_after = ids(&build(&document("shipments", "region")));
    let site_id = |set: &[(String, String)]| {
        set.iter()
            .find(|(entity, _)| entity.starts_with("site "))
            .expect("one site")
            .1
            .clone()
    };
    assert_ne!(site_id(&site_before), site_id(&site_after));
}

#[test]
fn an_authored_id_pins_identity_across_a_rename() {
    const PINNED_DATASET: &str = "0197cb6e-a63a-7d53-8d6f-35f4b1d67d4f";
    const PINNED_SITE: &str = "0197cb6e-a63a-7d53-8d6f-35f4b1d67d50";
    let authored = |dataset: &str| {
        format!(
            r#"
semantic_model:
  name: ana
  dimensions:
    - name: region
      data_type: string
      type: categorical
  datasets:
    - name: {dataset}
      id: {PINNED_DATASET}
      extras:
        semantic_mapping: auto
      dimensions:
        - ref: region
          id: {PINNED_SITE}
"#
        )
    };

    for dataset in ["orders", "purchases"] {
        let model = build(&authored(dataset));
        let entry = model.datasets.get(PINNED_DATASET).unwrap_or_else(|| {
            panic!("the authored dataset id must survive the rename to `{dataset}`")
        });
        assert_eq!(entry.body.base.name, dataset);
        assert!(
            entry
                .semantic_interface
                .dimensions
                .contains_key(PINNED_SITE),
            "the authored site id must survive too"
        );
    }
}

#[test]
fn two_interface_members_may_not_claim_one_path() {
    let yaml = r#"
semantic_model:
  name: ana
  dimensions:
    - name: region
      data_type: string
      type: categorical
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
      dimensions:
        - ref: region
        - ref: region
          is_key: true
"#;
    let err = parse(yaml).expect("parse ok").build().unwrap_err();
    assert!(
        err.iter().any(|d| matches!(
            &d.kind,
            ModelBuildErrorKind::Validate(ValidateErrorKind::InterfaceMemberPathNotUnique {
                data_kind,
                collection,
                name,
                occurrences,
            }) if data_kind == "orders"
                && collection == "dimensions"
                && name == "region"
                && occurrences.len() == 2
        )),
        "expected the path-uniqueness fault naming the data kind, the \
         collection and both occurrences' locations, got {err:?}"
    );
}

#[test]
fn one_name_in_two_data_kinds_is_two_members() {
    let yaml = r#"
semantic_model:
  name: ana
  dimensions:
    - name: region
      data_type: string
      type: categorical
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
      dimensions:
        - ref: region
    - name: shipments
      extras:
        semantic_mapping: auto
      dimensions:
        - ref: region
"#;
    let model = build(yaml);
    let site_ids: Vec<&String> = model
        .datasets
        .values()
        .flat_map(|d| d.semantic_interface.dimensions.keys())
        .collect();
    assert_eq!(site_ids.len(), 2);
    assert_ne!(site_ids[0], site_ids[1], "two paths are two ids");
}

#[test]
fn a_filter_and_a_dimension_may_share_a_name_in_one_data_kind() {
    let yaml = r#"
semantic_model:
  name: ana
  dimensions:
    - name: region
      data_type: string
      type: categorical
  datasets:
    - name: orders
      extras:
        semantic_mapping: auto
      dimensions:
        - ref: region
      filters:
        - name: region
          expr:
            eq: [region, region]
"#;
    let model = build(yaml);
    let dataset = model.datasets.values().next().expect("one dataset");
    let site = dataset
        .semantic_interface
        .dimensions
        .keys()
        .next()
        .expect("one site");
    let filter = dataset
        .semantic_interface
        .filters
        .keys()
        .next()
        .expect("one filter");
    assert_ne!(site, filter, "the collection is identity-bearing");
}
