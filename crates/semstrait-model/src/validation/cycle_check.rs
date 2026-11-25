use crate::error::validate::ValidateErrorKind;
use crate::model::entity_btree_map::NamedEntity;
use crate::model::types::entities::{Dimension, Measure, Metric};
use crate::model::SemanticModel;
use crate::UnresolvedSemanticExpr;
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};
use semstrait_common::{EntityId, Visitor};
use semstrait_sql::expr::{Expr, SemanticLeaf};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;

pub(super) fn check_all_cycles(model: &SemanticModel, diags: &mut Diagnostics<ValidateErrorKind>) {
    check_carrier_cycles("Dimension", &model.dimensions, Dimension::expr, diags);
    check_carrier_cycles("Measure", &model.measures, Measure::expr, diags);
    check_carrier_cycles("Metric", &model.metrics, Metric::expr, diags);
}

fn check_carrier_cycles<T: NamedEntity>(
    carrier: &'static str,
    pool: &BTreeMap<EntityId, T>,
    get_expr: impl Fn(&T) -> Option<&UnresolvedSemanticExpr>,
    diags: &mut Diagnostics<ValidateErrorKind>,
) {
    let graph: BTreeMap<&str, BTreeSet<String>> = pool
        .values()
        .map(|entity| {
            let refs = match get_expr(entity) {
                Some(expr) => collect_references(expr),
                None => BTreeSet::new(),
            };
            (entity.entity_name(), refs)
        })
        .collect();

    let mut visited: BTreeSet<String> = BTreeSet::new();
    for start in graph.keys() {
        if visited.contains(*start) {
            continue;
        }
        if let Some(cycle) = find_cycle_from(start, &graph, &mut visited) {
            diags.push(Diagnostic::new(
                ValidateErrorKind::CyclicSemanticsReference {
                    carrier: carrier.to_owned(),
                    cycle,
                },
            ));
        }
    }
}

fn find_cycle_from(
    start: &str,
    graph: &BTreeMap<&str, BTreeSet<String>>,
    visited: &mut BTreeSet<String>,
) -> Option<Vec<String>> {
    let mut grey: BTreeSet<String> = BTreeSet::new();
    let mut path: Vec<String> = Vec::new();
    let mut stack: Vec<(String, usize)> = vec![(start.to_owned(), 0)];
    grey.insert(start.to_owned());
    path.push(start.to_owned());

    while let Some((node, idx)) = stack.last().cloned() {
        let neighbors: Vec<String> = graph
            .get(node.as_str())
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();

        if idx >= neighbors.len() {
            stack.pop();
            grey.remove(&node);
            visited.insert(node.clone());
            path.pop();
            continue;
        }

        stack.last_mut().unwrap().1 += 1;
        let neighbor = &neighbors[idx];

        if grey.contains(neighbor) {
            let cycle_start = path.iter().position(|n| n == neighbor).unwrap();
            let cycle: Vec<String> = path[cycle_start..].to_vec();
            return Some(rotate_to_smallest(cycle));
        }
        if visited.contains(neighbor) {
            continue;
        }
        if graph.contains_key(neighbor.as_str()) {
            grey.insert(neighbor.clone());
            path.push(neighbor.clone());
            stack.push((neighbor.clone(), 0));
        }
    }
    None
}

fn rotate_to_smallest(cycle: Vec<String>) -> Vec<String> {
    let min_idx = cycle
        .iter()
        .enumerate()
        .min_by_key(|(_, name)| (*name).clone())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut out: Vec<String> = cycle[min_idx..].to_vec();
    out.extend_from_slice(&cycle[..min_idx]);
    out
}

fn collect_references(expr: &UnresolvedSemanticExpr) -> BTreeSet<String> {
    use semstrait_common::Tree;
    let mut collector = NameCollector {
        names: BTreeSet::new(),
    };
    let _ = expr.apply(&mut collector);
    collector.names
}

struct NameCollector {
    names: BTreeSet<String>,
}

impl Visitor<UnresolvedSemanticExpr> for NameCollector {
    type Output = ();

    fn f_down(&mut self, node: &UnresolvedSemanticExpr) -> ControlFlow<()> {
        if let Expr::Leaf(
            SemanticLeaf::Field(name)
            | SemanticLeaf::Dimension(name, _)
            | SemanticLeaf::Measure(name, _)
            | SemanticLeaf::Metric(name, _),
        ) = node
        {
            self.names.insert(name.name().to_owned());
        }
        ControlFlow::Continue(())
    }

    fn f_up(&mut self, _node: &UnresolvedSemanticExpr) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{col, dim, field, lit, measure};

    #[test]
    fn collect_references_finds_all_kinds() {
        let refs = collect_references(&(field("a").expect("a name") + dim("b").expect("a name")));
        assert!(refs.contains("a"));
        assert!(refs.contains("b"));
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn collect_references_skips_literal_and_column() {
        let refs = collect_references(&(lit(1i64) + col("c").expect("a name")));
        assert!(refs.is_empty());
    }

    #[test]
    fn rotate_to_smallest_picks_lex_min() {
        let rotated = rotate_to_smallest(vec!["c".into(), "a".into(), "b".into()]);
        assert_eq!(rotated, vec!["a".to_string(), "b".into(), "c".into()]);
    }

    #[test]
    fn no_cycle_in_acyclic_graph() {
        let mut graph: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
        graph.insert("a", ["b".to_owned()].into_iter().collect());
        graph.insert("b", ["c".to_owned()].into_iter().collect());
        graph.insert("c", BTreeSet::new());
        let mut visited: BTreeSet<String> = BTreeSet::new();
        assert!(find_cycle_from("a", &graph, &mut visited).is_none());
    }

    #[test]
    fn detects_self_loop() {
        let mut graph: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
        graph.insert("a", ["a".to_owned()].into_iter().collect());
        let mut visited: BTreeSet<String> = BTreeSet::new();
        let cycle = find_cycle_from("a", &graph, &mut visited).expect("should detect self-loop");
        assert_eq!(cycle, vec!["a".to_string()]);
    }

    #[test]
    fn detects_two_node_cycle() {
        let mut graph: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
        graph.insert("a", ["b".to_owned()].into_iter().collect());
        graph.insert("b", ["a".to_owned()].into_iter().collect());
        let mut visited: BTreeSet<String> = BTreeSet::new();
        let cycle = find_cycle_from("a", &graph, &mut visited).expect("should detect 2-cycle");
        assert_eq!(cycle, vec!["a".to_string(), "b".into()]);
    }

    #[test]
    fn detects_three_node_cycle_rotated_to_smallest() {
        let mut graph: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
        graph.insert("c", ["a".to_owned()].into_iter().collect());
        graph.insert("a", ["b".to_owned()].into_iter().collect());
        graph.insert("b", ["c".to_owned()].into_iter().collect());
        let mut visited: BTreeSet<String> = BTreeSet::new();
        let cycle = find_cycle_from("c", &graph, &mut visited).expect("should detect 3-cycle");
        assert_eq!(cycle, vec!["a".to_string(), "b".into(), "c".into()]);
    }

    #[test]
    fn ignores_cross_carrier_refs() {
        let mut graph: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
        graph.insert("a", ["b".to_owned()].into_iter().collect());
        let mut visited: BTreeSet<String> = BTreeSet::new();
        assert!(find_cycle_from("a", &graph, &mut visited).is_none());
    }

    #[test]
    fn check_all_cycles_finds_metric_cycle() {
        use crate::model::types::entities::Metric;
        use semstrait_sql::datatype::DataType;

        let a = Metric::builder("a")
            .data_type(DataType::Integer)
            .agg(crate::model::types::entities::Agg::of_name("sum").expect("a catalog aggregate"))
            .expr(measure("b").expect("a name"))
            .build();
        let b = Metric::builder("b")
            .data_type(DataType::Integer)
            .agg(crate::model::types::entities::Agg::of_name("sum").expect("a catalog aggregate"))
            .expr(measure("a").expect("a name"))
            .build();

        let mut model = SemanticModel {
            name: "m".to_owned(),
            ..Default::default()
        };
        model
            .metrics
            .insert(a.entity_id().expect("builder mints id").to_owned(), a);
        model
            .metrics
            .insert(b.entity_id().expect("builder mints id").to_owned(), b);

        let mut diags: Diagnostics<ValidateErrorKind> = Vec::new();
        check_all_cycles(&model, &mut diags);

        let cycle_diag = diags.iter().find_map(|d| match &d.kind {
            ValidateErrorKind::CyclicSemanticsReference { carrier, cycle } => {
                Some((carrier.clone(), cycle.clone()))
            }
            _ => None,
        });
        let (carrier, cycle) = cycle_diag.expect("expected one CyclicSemanticsReference");
        assert_eq!(carrier, "Metric");
        assert_eq!(cycle, vec!["a".to_string(), "b".into()]);
    }

    #[test]
    fn check_all_cycles_skips_acyclic_model() {
        use crate::model::types::entities::Metric;
        use semstrait_sql::datatype::DataType;

        let a = Metric::builder("a")
            .data_type(DataType::Integer)
            .agg(crate::model::types::entities::Agg::of_name("sum").expect("a catalog aggregate"))
            .expr(measure("b").expect("a name"))
            .build();
        let b = Metric::builder("b")
            .data_type(DataType::Integer)
            .agg(crate::model::types::entities::Agg::of_name("sum").expect("a catalog aggregate"))
            .build();

        let mut model = SemanticModel {
            name: "m".to_owned(),
            ..Default::default()
        };
        model
            .metrics
            .insert(a.entity_id().expect("builder mints id").to_owned(), a);
        model
            .metrics
            .insert(b.entity_id().expect("builder mints id").to_owned(), b);

        let mut diags: Diagnostics<ValidateErrorKind> = Vec::new();
        check_all_cycles(&model, &mut diags);
        assert!(!diags
            .iter()
            .any(|d| matches!(&d.kind, ValidateErrorKind::CyclicSemanticsReference { .. })));
    }

    #[test]
    fn builder_pipeline_rejects_cyclic_metrics() {
        use crate::error::build::ModelBuildErrorKind;
        use crate::model::types::entities::{Agg, Metric};
        use semstrait_sql::datatype::DataType;

        let cyclic = |name: &str, refs: &str| {
            Metric::builder(name)
                .data_type(DataType::Integer)
                .agg(Agg::of_name("sum").expect("a catalog aggregate"))
                .expr(measure(refs).expect("a name"))
                .build()
        };

        let diags = SemanticModel::builder()
            .name("m")
            .metric(cyclic("a", "b"))
            .metric(cyclic("b", "a"))
            .build()
            .expect_err("a cyclic model must fail to build");

        assert!(
            diags.iter().any(|d| matches!(
                &d.kind,
                ModelBuildErrorKind::Validate(ValidateErrorKind::CyclicSemanticsReference {
                    carrier,
                    cycle,
                }) if carrier == "Metric" && *cycle == vec!["a".to_string(), "b".into()]
            )),
            "expected a CyclicSemanticsReference over [a, b]; got {diags:?}"
        );
    }
}
