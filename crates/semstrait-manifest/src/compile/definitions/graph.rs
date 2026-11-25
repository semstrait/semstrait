use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::EntityId;
use semstrait_common::SemanticsName;

use crate::error::CompileError;

use super::super::types::RecursionState;

type ReferenceGraph = BTreeMap<SemanticsName, BTreeSet<SemanticsName>>;

pub(crate) fn topological_order(
    graph: &ReferenceGraph,
    recursion: &mut RecursionState,
) -> Result<Vec<SemanticsName>, CompileError> {
    *recursion = RecursionState::new();
    let mut sccs: Vec<Vec<SemanticsName>> = Vec::new();

    for root in graph.keys() {
        if recursion.tarjan_index(root).is_none() {
            strongconnect(root, graph, recursion, &mut sccs);
        }
    }

    for scc in &sccs {
        if is_cyclic_scc(scc, graph) {
            let mut cycle: Vec<EntityId> = scc.iter().map(|n| n.as_str().to_owned()).collect();
            cycle.sort();
            return Err(CompileError::CyclicReference { cycle });
        }
    }

    let order = sccs.into_iter().filter_map(|mut scc| scc.pop()).collect();
    Ok(order)
}

fn strongconnect(
    start: &SemanticsName,
    graph: &ReferenceGraph,
    recursion: &mut RecursionState,
    sccs: &mut Vec<Vec<SemanticsName>>,
) {
    let mut work: Vec<(SemanticsName, usize)> = Vec::new();
    let mut scc_stack: Vec<SemanticsName> = Vec::new();

    recursion.tarjan_visit(start.clone());
    scc_stack.push(start.clone());
    work.push((start.clone(), 0));

    while let Some((node, next_succ)) = work.last().cloned() {
        let successors = graph.get(&node);
        let succ_at = successors.and_then(|s| s.iter().nth(next_succ));

        if let Some(successor) = succ_at {
            if let Some(frame) = work.last_mut() {
                frame.1 += 1;
            }
            if recursion.tarjan_index(successor).is_none() {
                recursion.tarjan_visit(successor.clone());
                scc_stack.push(successor.clone());
                work.push((successor.clone(), 0));
            } else if recursion.tarjan_on_stack(successor) {
                if let Some(idx) = recursion.tarjan_index(successor) {
                    recursion.tarjan_relax_low_link(&node, idx);
                }
            }
            continue;
        }

        work.pop();
        if let (Some(node_low), Some((parent, _))) = (recursion.tarjan_low_link(&node), work.last())
        {
            recursion.tarjan_relax_low_link(&parent.clone(), node_low);
        }

        if recursion.tarjan_index(&node) == recursion.tarjan_low_link(&node) {
            let mut component = Vec::new();
            while let Some(member) = scc_stack.pop() {
                recursion.tarjan_pop(&member);
                let is_root = member == node;
                component.push(member);
                if is_root {
                    break;
                }
            }
            sccs.push(component);
        }
    }
}

fn is_cyclic_scc(scc: &[SemanticsName], graph: &ReferenceGraph) -> bool {
    match scc {
        [only] => graph.get(only).is_some_and(|succ| succ.contains(only)),
        _ => scc.len() > 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> EntityId {
        s.to_owned()
    }

    fn name(s: &str) -> SemanticsName {
        SemanticsName::from(s)
    }

    fn ref_graph(edges: &[(&str, &[&str])]) -> ReferenceGraph {
        edges
            .iter()
            .map(|(from, tos)| (name(from), tos.iter().map(|t| name(t)).collect()))
            .collect()
    }

    #[test]
    fn acyclic_chain_sorts_dependencies_first() {
        let graph = ref_graph(&[("a", &["b"]), ("b", &["c"]), ("c", &[])]);
        let mut rec = RecursionState::new();
        let order = topological_order(&graph, &mut rec).expect("acyclic");
        let pos = |n: &str| order.iter().position(|x| x.as_str() == n).expect("present");
        assert!(pos("c") < pos("b"));
        assert!(pos("b") < pos("a"));
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn diamond_topological_order_is_consistent_and_stable() {
        let graph = ref_graph(&[("a", &["b", "c"]), ("b", &["d"]), ("c", &["d"]), ("d", &[])]);
        let mut rec = RecursionState::new();
        let order = topological_order(&graph, &mut rec).expect("acyclic");
        let pos = |n: &str| order.iter().position(|x| x.as_str() == n).expect("present");
        assert!(pos("d") < pos("b") && pos("d") < pos("c"));
        assert!(pos("b") < pos("a") && pos("c") < pos("a"));

        let mut rec2 = RecursionState::new();
        let order2 = topological_order(&graph, &mut rec2).expect("acyclic");
        assert_eq!(order, order2, "topological order is stable across runs");
    }

    #[test]
    fn self_loop_is_cyclic_reference() {
        let graph = ref_graph(&[("a", &["a"])]);
        let mut rec = RecursionState::new();
        assert_eq!(
            topological_order(&graph, &mut rec),
            Err(CompileError::CyclicReference {
                cycle: vec![id("a")]
            })
        );
    }

    #[test]
    fn two_cycle_is_cyclic_reference_with_sorted_members() {
        let graph = ref_graph(&[("a", &["b"]), ("b", &["a"])]);
        let mut rec = RecursionState::new();
        assert_eq!(
            topological_order(&graph, &mut rec),
            Err(CompileError::CyclicReference {
                cycle: vec![id("a"), id("b")]
            })
        );
    }

    #[test]
    fn longer_cycle_is_detected() {
        let graph = ref_graph(&[("a", &["b"]), ("b", &["c"]), ("c", &["a"])]);
        let mut rec = RecursionState::new();
        match topological_order(&graph, &mut rec) {
            Err(CompileError::CyclicReference { cycle }) => {
                assert_eq!(cycle, vec![id("a"), id("b"), id("c")]);
            }
            other => panic!("expected CyclicReference, got {other:?}"),
        }
    }

    #[test]
    fn cycle_among_otherwise_acyclic_nodes_is_detected() {
        let graph = ref_graph(&[("root", &["x"]), ("x", &["y"]), ("y", &["x"])]);
        let mut rec = RecursionState::new();
        assert!(matches!(
            topological_order(&graph, &mut rec),
            Err(CompileError::CyclicReference { .. })
        ));
    }

    #[test]
    fn empty_reference_graph_sorts_to_empty() {
        let graph = ReferenceGraph::new();
        let mut rec = RecursionState::new();
        assert_eq!(topological_order(&graph, &mut rec), Ok(Vec::new()));
    }

    #[test]
    fn disconnected_acyclic_components_all_appear() {
        let graph = ref_graph(&[("a", &["b"]), ("b", &[]), ("c", &["d"]), ("d", &[])]);
        let mut rec = RecursionState::new();
        let order = topological_order(&graph, &mut rec).expect("acyclic");
        assert_eq!(order.len(), 4);
        let pos = |n: &str| order.iter().position(|x| x.as_str() == n).expect("present");
        assert!(pos("b") < pos("a"));
        assert!(pos("d") < pos("c"));
    }
}
