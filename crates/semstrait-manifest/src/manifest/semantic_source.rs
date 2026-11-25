use semstrait_common::DataKindName;
use semstrait_common::EntityId;
use semstrait_ir::TemporalShape;
use semstrait_ir::{Cardinality, CrossFilter, Integrity, JoinsetJoinType, MultisetOp};
use serde::{Deserialize, Serialize};

use super::bitmap::SemanticBitmask;
use super::expression::PhysicalExprId;
use super::physical_source::{StorageTarget, Uri};
use super::semantic_definition::AiContext;
use super::source_alias::SourceAlias;

pub(crate) const SEPARATOR: char = '/';

mod multiset_mode {
    use semstrait_ir::MultisetOp;
    use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S: Serializer>(
        mode: &MultisetOp,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match mode {
            MultisetOp::UnionAll => "all",
            MultisetOp::UnionDistinct => "unique",
            other => unreachable!("semstrait-ir added a MultisetOp variant: {other:?}"),
        }
        .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<MultisetOp, D::Error> {
        match String::deserialize(deserializer)?.as_str() {
            "all" => Ok(MultisetOp::UnionAll),
            "unique" => Ok(MultisetOp::UnionDistinct),
            other => Err(de::Error::custom(format_args!(
                "union mode: unknown variant `{other}`"
            ))),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticSource {
    pub id: EntityId,

    pub name: DataKindName,

    pub alias: SourceAlias,

    pub public: bool,

    pub semantic_bitmask: SemanticBitmask,

    pub temporal: Option<TemporalShape>,

    pub role_definition: RoleDefinition,

    pub originators: Vec<EntityId>,

    pub semantic_signature_ids: Vec<EntityId>,
}

impl SemanticSource {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        id: impl Into<EntityId>,
        name: DataKindName,
        alias: SourceAlias,
        public: bool,
        semantic_bitmask: SemanticBitmask,
        temporal: Option<TemporalShape>,
        role_definition: RoleDefinition,
        originators: Vec<EntityId>,
    ) -> Self {
        Self {
            id: id.into(),
            name,
            alias,
            public,
            semantic_bitmask,
            temporal,
            role_definition,
            originators,
            semantic_signature_ids: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_semantic_signature_ids(mut self, mut ids: Vec<EntityId>) -> Self {
        ids.sort();
        ids.dedup();
        self.semantic_signature_ids = ids;
        self
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "content")]
pub enum RoleDefinition {
    Dataset {
        source_id: EntityId,
        selector: Selector,
    },

    Joinset {
        from: EntityId,
        to: EntityId,
        on_expr_id: Option<PhysicalExprId>,
        filter_expr_id: Option<PhysicalExprId>,
        join_type: JoinsetJoinType,
        cardinality: Cardinality,
        integrity: Integrity,
        cross_filter: CrossFilter,
        description: Option<String>,
        ai_context: Option<AiContext>,
    },

    Unionset {
        branches: Vec<EntityId>,
        #[serde(with = "multiset_mode")]
        mode: MultisetOp,
        null_fill: bool,
    },

    Grainset {
        grains: Vec<EntityId>,
        keysets: Vec<EntityId>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "content")]
pub enum Selector {
    Prefix,
    Pattern(Uri),
    Uris(Vec<Uri>),
    Table,
}

impl Selector {
    #[must_use]
    pub fn reads(&self, target: &StorageTarget) -> bool {
        matches!(
            (self, target),
            (Self::Table, StorageTarget::Table { .. })
                | (
                    Self::Prefix | Self::Pattern(_) | Self::Uris(_),
                    StorageTarget::Path { .. }
                )
        )
    }

    #[must_use]
    pub fn selects(&self, base_path: &Uri, object: &Uri) -> bool {
        let beneath = object.beneath(base_path).is_some_and(|rest| !debris(rest));
        match self {
            Self::Prefix => beneath,
            Self::Pattern(glob) => beneath && glob_match(glob.as_str(), object.as_str()),
            Self::Uris(named) => named.binary_search(object).is_ok(),
            Self::Table => false,
        }
    }
}

pub(crate) fn debris(rest: &str) -> bool {
    rest.split(SEPARATOR).any(|name| {
        name.starts_with('.')
            || (name.starts_with('_') && !name.contains('='))
            || name.ends_with("._COPYING_")
    })
}

pub(crate) fn glob_match(pattern: &str, candidate: &str) -> bool {
    let mut segments = candidate.split(SEPARATOR);
    pattern.split(SEPARATOR).all(|expected| {
        segments
            .next()
            .is_some_and(|actual| segment_match(expected, actual))
    }) && segments.next().is_none()
}

fn segment_match(pattern: &str, candidate: &str) -> bool {
    let expected: Vec<char> = pattern.chars().collect();
    let actual: Vec<char> = candidate.chars().collect();
    let (mut p, mut c) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while c < actual.len() {
        match expected.get(p) {
            Some('*') => {
                star = Some((p, c));
                p += 1;
            }
            Some('?') => {
                p += 1;
                c += 1;
            }
            Some(literal) if actual.get(c) == Some(literal) => {
                p += 1;
                c += 1;
            }
            _ => match star {
                Some((star_p, star_c)) => {
                    p = star_p + 1;
                    c = star_c + 1;
                    star = Some((star_p, c));
                }
                None => return false,
            },
        }
    }
    (expected.get(p..)).is_some_and(|rest| rest.iter().all(|&ch| ch == '*'))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceRole {
    Dataset,
    Unionset,
    Grainset,
    Joinset,
}

impl SourceRole {
    pub(super) const fn letter(self) -> char {
        match self {
            Self::Dataset => 'd',
            Self::Unionset => 'u',
            Self::Grainset => 'g',
            Self::Joinset => 'j',
        }
    }
}

impl RoleDefinition {
    pub(crate) const fn source_role(&self) -> SourceRole {
        match self {
            Self::Dataset { .. } => SourceRole::Dataset,
            Self::Joinset { .. } => SourceRole::Joinset,
            Self::Unionset { .. } => SourceRole::Unionset,
            Self::Grainset { .. } => SourceRole::Grainset,
        }
    }

    #[must_use]
    pub fn dataset(source_id: impl Into<EntityId>, selector: Selector) -> Self {
        Self::Dataset {
            source_id: source_id.into(),
            selector,
        }
    }

    #[must_use]
    pub fn unionset(mode: MultisetOp, branches: Vec<EntityId>, null_fill: bool) -> Self {
        Self::Unionset {
            branches,
            mode,
            null_fill,
        }
    }

    #[must_use]
    pub fn grainset(grains: Vec<EntityId>, keysets: Vec<EntityId>) -> Self {
        Self::Grainset { grains, keysets }
    }

    #[must_use]
    pub fn operands(&self) -> Vec<&EntityId> {
        match self {
            Self::Dataset { .. } => Vec::new(),
            Self::Joinset { from, to, .. } => vec![from, to],
            Self::Unionset { branches, .. } => branches.iter().collect(),
            Self::Grainset { grains, .. } => grains.iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use semstrait_ir::JoinType;

    fn entry(role_definition: RoleDefinition) -> SemanticSource {
        SemanticSource::new(
            "src-id",
            DataKindName::from("orders"),
            SourceAlias::from("orders"),
            true,
            SemanticBitmask::from_positions([0, 1]),
            None,
            role_definition,
            Vec::new(),
        )
    }

    #[test]
    fn a_dataset_arms_reference_is_not_an_operand() {
        let leaf = entry(RoleDefinition::dataset("src.orders", Selector::Table));
        assert!(leaf.role_definition.operands().is_empty());
        let RoleDefinition::Dataset { source_id, .. } = &leaf.role_definition else {
            panic!("built a Dataset arm");
        };
        assert_eq!(source_id, "src.orders");
    }

    #[test]
    fn every_composing_arm_reports_its_operands_in_order() {
        let union = RoleDefinition::unionset(
            MultisetOp::UnionAll,
            vec!["dk.a".to_owned(), "dk.b".to_owned()],
            true,
        );
        assert_eq!(union.operands(), vec!["dk.a", "dk.b"]);

        let grain =
            RoleDefinition::grainset(vec!["dk.month".to_owned(), "dk.day".to_owned()], Vec::new());
        assert_eq!(grain.operands(), vec!["dk.month", "dk.day"]);

        let join = RoleDefinition::Joinset {
            from: "dk.a".to_owned(),
            to: "dk.b".to_owned(),
            on_expr_id: Some(PhysicalExprId::new(uuid::Uuid::from_u128(7))),
            filter_expr_id: None,
            join_type: JoinsetJoinType::Join(JoinType::Inner),
            cardinality: Cardinality::ManyToOne,
            integrity: Integrity::Assumed,
            cross_filter: CrossFilter::None,
            description: None,
            ai_context: None,
        };
        assert_eq!(join.operands(), vec!["dk.a", "dk.b"]);
    }

    #[test]
    fn each_arm_round_trips_under_its_own_tag() {
        for (role, tag) in [
            (
                RoleDefinition::dataset("src.orders", Selector::Table),
                "dataset",
            ),
            (
                RoleDefinition::unionset(MultisetOp::UnionDistinct, vec!["dk.a".to_owned()], true),
                "unionset",
            ),
            (
                RoleDefinition::grainset(vec!["dk.day".to_owned()], Vec::new()),
                "grainset",
            ),
        ] {
            let source = entry(role);
            let json = serde_json::to_string(&source).expect("serialize SemanticSource");
            assert!(json.contains(&format!(r#""type":"{tag}""#)), "got: {json}");
            let back: SemanticSource = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, source);
        }
    }

    #[test]
    fn a_selector_selects_the_objects_of_the_listing_that_it_reads() {
        let (prefix, table) = (Selector::Prefix, Selector::Table);
        let pattern = Selector::Pattern("s3://lake/orders/*.parquet".into());
        let uris = Selector::Uris(vec![
            "s3://lake/orders/_meta.parquet".into(),
            "s3://lake/orders/b.parquet".into(),
        ]);
        let (orders, staging) = ("s3://lake/orders/", "s3://lake/_stage/");
        for (selector, base_path, object, selects) in [
            (&prefix, orders, "s3://lake/orders/a.parquet", true),
            (&prefix, orders, "s3://lake/orders/eu/a.parquet", true),
            (&prefix, orders, "s3://lake/orders/.a.parquet", false),
            (&prefix, orders, "s3://lake/orders/.tmp/a.parquet", false),
            (&prefix, orders, "s3://lake/orders/_SUCCESS", false),
            (&prefix, orders, "s3://lake/orders/a._COPYING_", false),
            (&prefix, orders, "s3://lake/orders/_k=v/a.parquet", true),
            (&prefix, orders, "s3://lake/returns/a.parquet", false),
            (&prefix, staging, "s3://lake/_stage/a.parquet", true),
            (&pattern, orders, "s3://lake/orders/a.parquet", true),
            (&pattern, orders, "s3://lake/orders/a.csv", false),
            (&pattern, orders, "s3://lake/orders/eu/a.parquet", false),
            (&pattern, orders, "s3://lake/orders/_a.parquet", false),
            (&pattern, orders, "s3://lake/orders/.a.parquet", false),
            (&pattern, orders, "s3://lake/returns/a.parquet", false),
            (&uris, orders, "s3://lake/orders/_meta.parquet", true),
            (&uris, orders, "s3://lake/orders/a.parquet", false),
            (&uris, orders, "s3://lake/orders/b.parquet", true),
            (&table, orders, "s3://lake/orders/a.parquet", false),
        ] {
            assert_eq!(
                selector.selects(&base_path.into(), &object.into()),
                selects,
                "{selector:?} over {base_path} at {object}"
            );
        }
    }

    #[test]
    fn a_wildcard_never_crosses_a_separator() {
        assert!(glob_match("orders/*.parquet", "orders/a.parquet"));
        assert!(
            !glob_match("orders/*.parquet", "orders/2024/a.parquet"),
            "a wildcard must not swallow a separator, or expansion would recurse"
        );
    }

    #[test]
    fn a_wildcard_in_a_middle_segment_expands_one_level() {
        assert!(glob_match(
            "lake/*/orders.parquet",
            "lake/eu/orders.parquet"
        ));
        assert!(!glob_match(
            "lake/*/orders.parquet",
            "lake/eu/west/orders.parquet"
        ));
    }

    #[test]
    fn a_single_char_wildcard_matches_one_character() {
        assert!(glob_match("part-?.parquet", "part-0.parquet"));
        assert!(!glob_match("part-?.parquet", "part-10.parquet"));
        assert!(!glob_match("a?b", "a/b"), "`?` must not match a separator");
    }

    #[test]
    fn literal_and_empty_matches_behave() {
        assert!(glob_match("orders/a.parquet", "orders/a.parquet"));
        assert!(!glob_match("orders/a.parquet", "orders/b.parquet"));
        assert!(glob_match("orders/*", "orders/"), "`*` may match empty");
        assert!(glob_match("*", "a"));
    }

    #[test]
    fn several_wildcards_compose() {
        assert!(glob_match("*/part-*.parquet", "eu/part-0.parquet"));
        assert!(!glob_match("*/part-*.parquet", "eu/west/part-0.parquet"));
    }
}
