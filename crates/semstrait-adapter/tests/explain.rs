mod support;

use std::error::Error;

use semstrait_adapter::trace_registry;
use semstrait_common::Granularity;
use semstrait_ir::{
    Cardinality, Delivery, GranularityGroup, PlanNode, ProjectNode, SemAnnotation, SourceRole,
};
use substrait_explain::{format_with_registry, OutputOptions, Parser};
use support::{
    explained, leaf, name, one_expression_of_each_kind, orders, paid_by_region, Faithful,
};

fn optimizations(text: &str) -> Vec<(&str, &str)> {
    let mut relation = "";
    let mut lines = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.starts_with("+ Opt:") {
            lines.push((relation, line));
        } else if let Some((name, _)) = line.split_once('[') {
            relation = name;
        }
    }
    lines
}

#[test]
fn the_explain_text_has_one_opt_line_for_a_traced_scan_and_one_for_a_traced_aggregate(
) -> Result<(), Box<dyn Error>> {
    let text = explained(paid_by_region()?, &Faithful)?;

    assert_eq!(
        optimizations(&text),
        vec![
            (
                "Aggregate",
                "+ Opt:Trace[scope_root='orders', compositions=0, granularity_fit=&NotSelected]"
            ),
            (
                "Read",
                "+ Opt:Trace[semantic_source='orders', source_role=&Dataset, physical_source='orders_table']"
            ),
        ],
        "{text}"
    );
    Ok(())
}

#[test]
fn the_traces_named_typed_arguments_parse_back_into_the_same_text() -> Result<(), Box<dyn Error>> {
    let scan = PlanNode::from(orders()?).with_trace([SemAnnotation::SemanticSource {
        semantic_source: "orders.detail".into(),
        source_role: SourceRole::Joinset,
        originators: vec!["orders".into()],
        physical_source: None,
        cardinality: Some(Cardinality::ManyToOne),
        delivery: Some(Delivery::Rows),
        answered_by: Some("orders.answer".into()),
        pruned: vec!["orders.x".into(), "orders.y".into()],
    }]);
    let projected = ProjectNode::new(scan, vec![(name("region"), leaf(&["region"]))])?;
    let traced = PlanNode::from(projected).with_trace([SemAnnotation::Rank {
        scope_root: "orders".into(),
        compositions: 2,
        granularity_fit: GranularityGroup::Selected,
        granularity: Some(Granularity::Month),
    }]);

    let text = explained(traced, &Faithful)?;
    assert_eq!(
        optimizations(&text),
        vec![
            (
                "Project",
                "+ Opt:Trace[scope_root='orders', compositions=2, granularity_fit=&Selected, granularity=&Month]"
            ),
            (
                "Read",
                "+ Opt:Trace[semantic_source='orders.detail', source_role=&Joinset, originators=('orders',), cardinality=&ManyToOne, delivery=&Rows, answered_by='orders.answer', pruned=('orders.x', 'orders.y')]"
            ),
        ],
        "{text}"
    );

    let registry = trace_registry();
    let parsed = (Parser::new().with_extension_registry(registry.clone()))
        .parse_plan(&text)
        .expect("the named, typed arguments parse back");
    let (formatted, errors) = format_with_registry(&parsed, &OutputOptions::default(), &registry);
    assert!(errors.is_empty(), "{errors:?}");
    assert_eq!(formatted.trim(), text.trim());
    Ok(())
}

#[test]
fn explain_returns_its_text_where_substrait_explain_reports_a_format_error(
) -> Result<(), Box<dyn Error>> {
    let text = explained(one_expression_of_each_kind()?, &Faithful)?;

    assert!(text.contains("Project["), "{text}");
    assert!(text.contains("!{"), "{text}");
    Ok(())
}
