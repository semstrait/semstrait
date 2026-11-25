use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::EntityId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SemanticNodeId(u32);

impl SemanticNodeId {
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct SemanticEdgeId(u32);

impl SemanticEdgeId {
    pub const fn from_raw(value: u32) -> Self {
        Self(value)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticGraph {
    pub nodes: BTreeMap<SemanticNodeId, SemanticNode>,
    pub edges: BTreeMap<SemanticEdgeId, SemanticEdge>,
    pub outgoing_edges: BTreeMap<SemanticNodeId, Vec<SemanticEdgeId>>,
}

impl SemanticGraph {
    pub fn new(
        nodes: BTreeMap<SemanticNodeId, SemanticNode>,
        edges: BTreeMap<SemanticEdgeId, SemanticEdge>,
    ) -> Self {
        let mut outgoing_edges: BTreeMap<SemanticNodeId, Vec<SemanticEdgeId>> = BTreeMap::new();
        for (edge_id, edge) in &edges {
            outgoing_edges.entry(edge.from).or_default().push(*edge_id);
        }
        Self {
            nodes,
            edges,
            outgoing_edges,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticNode {
    pub payload: SemanticNodePayload,
    pub interface_id: Option<EntityId>,
}

impl SemanticNode {
    pub fn new(payload: SemanticNodePayload, interface_id: Option<EntityId>) -> Self {
        Self {
            payload,
            interface_id,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticNodePayload {
    DataKind { data_kind_id: EntityId },
    Semantic { semantic_id: EntityId },
    Source { source_id: EntityId },
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticEdge {
    pub from: SemanticNodeId,
    pub to: SemanticNodeId,
    pub edge_type: SemanticEdgeType,
}

impl SemanticEdge {
    pub fn new(from: SemanticNodeId, to: SemanticNodeId, edge_type: SemanticEdgeType) -> Self {
        Self {
            from,
            to,
            edge_type,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticEdgeType {
    Relationship,
    Composition,
    Binding,
    DependsOn,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticGraphFragment {
    pub key: SegmentKey,
    pub graph: SemanticGraph,
    pub touched_sources: Vec<EntityId>,
}

impl SemanticGraphFragment {
    pub fn new(key: SegmentKey, graph: SemanticGraph, touched_sources: Vec<EntityId>) -> Self {
        Self {
            key,
            graph,
            touched_sources,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SegmentKey {
    pub manifest_epoch: u64,
    pub datakind_ids: BTreeSet<EntityId>,
    pub requested_semantics: BTreeSet<EntityId>,
    pub constraints_fingerprint: [u8; 32],
}

impl SegmentKey {
    pub fn new(
        manifest_epoch: u64,
        datakind_ids: BTreeSet<EntityId>,
        requested_semantics: BTreeSet<EntityId>,
        constraints_fingerprint: [u8; 32],
    ) -> Self {
        Self {
            manifest_epoch,
            datakind_ids,
            requested_semantics,
            constraints_fingerprint,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticInterfaceBitmask {
    pub words: Vec<u64>,
}

impl SemanticInterfaceBitmask {
    pub fn new(words: Vec<u64>) -> Self {
        Self { words }
    }
}

pub trait SemanticGraphRead {
    fn node(&self, id: SemanticNodeId) -> Option<&SemanticNode>;
    fn edge(&self, id: SemanticEdgeId) -> Option<&SemanticEdge>;
    fn outgoing(&self, id: SemanticNodeId) -> &[SemanticEdgeId];
}

impl SemanticGraphRead for SemanticGraph {
    fn node(&self, id: SemanticNodeId) -> Option<&SemanticNode> {
        self.nodes.get(&id)
    }

    fn edge(&self, id: SemanticEdgeId) -> Option<&SemanticEdge> {
        self.edges.get(&id)
    }

    fn outgoing(&self, id: SemanticNodeId) -> &[SemanticEdgeId] {
        self.outgoing_edges.get(&id).map_or(&[], Vec::as_slice)
    }
}

pub trait SemanticGraphFragmentRead {
    fn key(&self) -> &SegmentKey;
    fn graph(&self) -> &SemanticGraph;
    fn touched_sources(&self) -> &[EntityId];
}

impl SemanticGraphFragmentRead for SemanticGraphFragment {
    fn key(&self) -> &SegmentKey {
        &self.key
    }

    fn graph(&self) -> &SemanticGraph {
        &self.graph
    }

    fn touched_sources(&self) -> &[EntityId] {
        &self.touched_sources
    }
}

pub trait SemanticInterfaceBitmaskOps {
    fn is_subset_of(&self, other: &Self) -> bool;
    fn intersects(&self, other: &Self) -> bool;
}

impl SemanticInterfaceBitmaskOps for SemanticInterfaceBitmask {
    fn is_subset_of(&self, other: &Self) -> bool {
        self.words.iter().enumerate().all(|(i, &w)| {
            let o = other.words.get(i).copied().unwrap_or(0);
            w & !o == 0
        })
    }

    fn intersects(&self, other: &Self) -> bool {
        self.words
            .iter()
            .zip(&other.words)
            .any(|(&a, &b)| a & b != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eid(s: &str) -> EntityId {
        s.to_string()
    }

    fn datakind_node(id: &str) -> SemanticNode {
        SemanticNode::new(
            SemanticNodePayload::DataKind {
                data_kind_id: eid(id),
            },
            None,
        )
    }

    #[test]
    fn node_and_edge_ids_round_trip_raw() {
        assert_eq!(SemanticNodeId::from_raw(7).as_u32(), 7);
        assert_eq!(SemanticEdgeId::from_raw(42).as_u32(), 42);
    }

    #[test]
    fn node_ids_order_by_index() {
        assert!(SemanticNodeId::from_raw(1) < SemanticNodeId::from_raw(2));
        assert!(SemanticEdgeId::from_raw(10) < SemanticEdgeId::from_raw(11));
    }

    #[test]
    fn new_derives_outgoing_sorted_by_edge_id() {
        let n0 = SemanticNodeId::from_raw(0);
        let n1 = SemanticNodeId::from_raw(1);
        let n2 = SemanticNodeId::from_raw(2);

        let mut nodes = BTreeMap::new();
        nodes.insert(n0, datakind_node("dk0"));
        nodes.insert(n1, datakind_node("dk1"));
        nodes.insert(n2, datakind_node("dk2"));

        let mut edges = BTreeMap::new();
        edges.insert(
            SemanticEdgeId::from_raw(2),
            SemanticEdge::new(n0, n2, SemanticEdgeType::Binding),
        );
        edges.insert(
            SemanticEdgeId::from_raw(5),
            SemanticEdge::new(n0, n1, SemanticEdgeType::Relationship),
        );
        edges.insert(
            SemanticEdgeId::from_raw(0),
            SemanticEdge::new(n0, n1, SemanticEdgeType::Relationship),
        );
        edges.insert(
            SemanticEdgeId::from_raw(3),
            SemanticEdge::new(n1, n0, SemanticEdgeType::Relationship),
        );
        edges.insert(
            SemanticEdgeId::from_raw(1),
            SemanticEdge::new(n1, n2, SemanticEdgeType::Composition),
        );

        let g = SemanticGraph::new(nodes, edges);

        assert_eq!(
            g.outgoing(n0),
            &[
                SemanticEdgeId::from_raw(0),
                SemanticEdgeId::from_raw(2),
                SemanticEdgeId::from_raw(5),
            ],
            "n0 adjacency must be ascending by SemanticEdgeId"
        );
        assert_eq!(
            g.outgoing(n1),
            &[SemanticEdgeId::from_raw(1), SemanticEdgeId::from_raw(3)]
        );
        assert_eq!(g.outgoing(n2), &[] as &[SemanticEdgeId]);
        assert!(!g.outgoing_edges.contains_key(&n2));
    }

    #[test]
    fn read_node_and_edge_hit_and_miss() {
        let n0 = SemanticNodeId::from_raw(0);
        let n1 = SemanticNodeId::from_raw(1);
        let mut nodes = BTreeMap::new();
        nodes.insert(n0, datakind_node("dk0"));
        nodes.insert(n1, datakind_node("dk1"));
        let mut edges = BTreeMap::new();
        edges.insert(
            SemanticEdgeId::from_raw(0),
            SemanticEdge::new(n0, n1, SemanticEdgeType::Relationship),
        );
        let g = SemanticGraph::new(nodes, edges);

        assert!(g.node(n0).is_some());
        assert!(g.node(SemanticNodeId::from_raw(99)).is_none());
        assert!(g.edge(SemanticEdgeId::from_raw(0)).is_some());
        assert!(g.edge(SemanticEdgeId::from_raw(99)).is_none());
        assert_eq!(
            g.outgoing(SemanticNodeId::from_raw(99)),
            &[] as &[SemanticEdgeId]
        );
    }

    #[test]
    fn node_payload_variants_construct() {
        let dk = SemanticNodePayload::DataKind {
            data_kind_id: eid("dk"),
        };
        let sem = SemanticNodePayload::Semantic {
            semantic_id: eid("sem"),
        };
        let src = SemanticNodePayload::Source {
            source_id: eid("src"),
        };
        assert_ne!(dk, sem);
        assert_ne!(src, dk);
    }

    #[test]
    fn node_carries_optional_interface_id() {
        let with = SemanticNode::new(
            SemanticNodePayload::Semantic {
                semantic_id: eid("s"),
            },
            Some(eid("iface")),
        );
        let without = SemanticNode::new(
            SemanticNodePayload::Semantic {
                semantic_id: eid("s"),
            },
            None,
        );
        assert_eq!(with.interface_id, Some(eid("iface")));
        assert_eq!(without.interface_id, None);
        assert_ne!(with, without);
    }

    #[test]
    fn reserved_depends_on_edge_type_exists() {
        let e = SemanticEdge::new(
            SemanticNodeId::from_raw(0),
            SemanticNodeId::from_raw(1),
            SemanticEdgeType::DependsOn,
        );
        assert_eq!(e.edge_type, SemanticEdgeType::DependsOn);
    }

    fn sample_segment_key() -> SegmentKey {
        SegmentKey::new(
            7,
            BTreeSet::from([eid("dk0")]),
            BTreeSet::from([eid("sem0"), eid("sem1")]),
            [0u8; 32],
        )
    }

    #[test]
    fn fragment_read_trait_exposes_parts() {
        let g = SemanticGraph::new(BTreeMap::new(), BTreeMap::new());
        let frag = SemanticGraphFragment::new(
            sample_segment_key(),
            g.clone(),
            vec![eid("src0"), eid("src1")],
        );

        assert_eq!(frag.key().manifest_epoch, 7);
        assert_eq!(*frag.graph(), g);
        assert_eq!(frag.touched_sources(), &[eid("src0"), eid("src1")]);
    }

    #[test]
    fn segment_key_equality_is_structural() {
        let a = sample_segment_key();
        let b = sample_segment_key();
        assert_eq!(a, b, "same components ⟹ equal cache key");

        let mut c = sample_segment_key();
        c.manifest_epoch = 8;
        assert_ne!(a, c, "epoch participates in identity");
    }

    #[test]
    fn segment_key_is_hashable_and_orderable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(sample_segment_key());
        assert!(set.contains(&sample_segment_key()));

        let mut lower = sample_segment_key();
        lower.manifest_epoch = 1;
        let mut higher = sample_segment_key();
        higher.manifest_epoch = 2;
        assert!(lower < higher);
    }

    #[test]
    fn bitmask_subset_same_length() {
        let sub = SemanticInterfaceBitmask::new(vec![0b0101]);
        let sup = SemanticInterfaceBitmask::new(vec![0b1111]);
        assert!(sub.is_subset_of(&sup));
        assert!(!sup.is_subset_of(&sub));
    }

    #[test]
    fn bitmask_subset_handles_unequal_lengths() {
        let sub = SemanticInterfaceBitmask::new(vec![0b0001]);
        let sup = SemanticInterfaceBitmask::new(vec![0b0011, 0xFF]);
        assert!(sub.is_subset_of(&sup));

        let longer = SemanticInterfaceBitmask::new(vec![0b0001, 0b0001]);
        let shorter = SemanticInterfaceBitmask::new(vec![0b0001]);
        assert!(!longer.is_subset_of(&shorter));
        assert!(shorter.is_subset_of(&longer));
    }

    #[test]
    fn bitmask_empty_is_subset_of_everything() {
        let empty = SemanticInterfaceBitmask::new(vec![]);
        let any = SemanticInterfaceBitmask::new(vec![0b1010]);
        assert!(empty.is_subset_of(&any));
        assert!(empty.is_subset_of(&empty));
        let zero_words = SemanticInterfaceBitmask::new(vec![0, 0]);
        assert!(zero_words.is_subset_of(&any));
    }

    #[test]
    fn bitmask_intersects() {
        let a = SemanticInterfaceBitmask::new(vec![0b0110]);
        let b = SemanticInterfaceBitmask::new(vec![0b0100]);
        let c = SemanticInterfaceBitmask::new(vec![0b1001]);
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn bitmask_intersects_handles_unequal_lengths() {
        let a = SemanticInterfaceBitmask::new(vec![0b0001, 0xFF]);
        let b = SemanticInterfaceBitmask::new(vec![0b0001]);
        assert!(a.intersects(&b));

        let c = SemanticInterfaceBitmask::new(vec![0b0010, 0xFF]);
        let d = SemanticInterfaceBitmask::new(vec![0b0001]);
        assert!(!c.intersects(&d));
    }
}
