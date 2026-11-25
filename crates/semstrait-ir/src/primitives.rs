use serde::{Deserialize, Serialize};

use semstrait_common::ValidateError;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Name(String);

impl Name {
    pub fn new(s: impl Into<String>) -> Result<Self, ValidateError> {
        let s = s.into();
        if s.is_empty() {
            return Err(ValidateError::EmptyName);
        }
        Ok(Self(s))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn into_string(self) -> String {
        self.0
    }

    pub const PLAN_PREFIX: &'static str = "__plan_";

    pub const RESERVED_PREFIXES: [&'static str; 3] = ["__semstrait_", Self::PLAN_PREFIX, "__agg_"];

    pub fn folded(name: &str) -> impl Iterator<Item = char> + '_ {
        name.chars().flat_map(char::to_lowercase)
    }

    pub fn reserved_prefix(name: &str) -> Option<&'static str> {
        Self::RESERVED_PREFIXES
            .into_iter()
            .find(|prefix| name.starts_with(prefix))
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cardinality {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Integrity {
    Enforced,
    Assumed,
    None,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossFilter {
    None,
    Left,
    Right,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinsetJoinType {
    Join(JoinType),
    Cross,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn name_round_trips_through_as_str_and_into_string() {
        let n = Name::new("amount").unwrap();
        assert_eq!(n.as_str(), "amount");
        assert_eq!(n.clone().into_string(), "amount");
    }

    #[test]
    fn name_rejects_empty() {
        let r = Name::new("");
        assert!(matches!(r, Err(ValidateError::EmptyName)));
    }

    #[test]
    fn name_is_case_sensitive() {
        let a = Name::new("Amount").unwrap();
        let b = Name::new("amount").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn name_admits_reserved_prefix_at_this_layer() {
        for raw in ["__semstrait_x", "__plan_x", "__agg_x"] {
            assert!(
                Name::new(raw).is_ok(),
                "Name::new must admit reserved-prefix `{raw}` at the IR layer"
            );
        }
    }

    #[test]
    fn a_reserved_prefix_is_found_with_its_case_kept() {
        for (name, prefix) in [
            ("__plan_x", Some("__plan_")),
            ("__agg_total", Some("__agg_")),
            ("__semstrait_id", Some("__semstrait_")),
            ("__PLAN_x", None),
            ("__Agg_total", None),
            ("__SemStrait_id", None),
            ("_plan_x", None),
            ("__plan", None),
            ("__planx", None),
            ("x__plan_", None),
            ("amount", None),
        ] {
            assert_eq!(Name::reserved_prefix(name), prefix, "`{name}`");
        }
    }

    #[test]
    fn name_admits_unicode_and_punctuation_at_this_layer() {
        for raw in ["x.y", "amount $", "ψ", "1col"] {
            assert!(Name::new(raw).is_ok(), "Name::new must admit `{raw}`");
        }
    }

    #[test]
    fn name_serde_json_roundtrip() {
        let n = Name::new("revenue").unwrap();
        let json = serde_json::to_string(&n).unwrap();
        let back: Name = serde_json::from_str(&json).unwrap();
        assert_eq!(n, back);
    }

    #[test]
    fn name_hash_is_deterministic() {
        let a = Name::new("x").unwrap();
        let b = Name::new("x").unwrap();
        let mut set: HashSet<Name> = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }

    #[test]
    fn join_type_serde_json_roundtrip() {
        for j in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
        ] {
            let json = serde_json::to_string(&j).unwrap();
            let back: JoinType = serde_json::from_str(&json).unwrap();
            assert_eq!(j, back);
        }
    }

    #[test]
    fn cardinality_serde_json_roundtrip() {
        for c in [
            Cardinality::OneToOne,
            Cardinality::OneToMany,
            Cardinality::ManyToOne,
            Cardinality::ManyToMany,
        ] {
            let json = serde_json::to_string(&c).unwrap();
            let back: Cardinality = serde_json::from_str(&json).unwrap();
            assert_eq!(c, back);
        }
    }
}
