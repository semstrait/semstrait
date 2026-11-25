use semstrait_common::str_newtype;
use semstrait_common::EntityId;
use serde::{Deserialize, Serialize};

use crate::identity::mint_keyset_id;

str_newtype!(KeysetName);

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Keyset {
    pub id: EntityId,
    pub name: KeysetName,
}

impl Keyset {
    #[must_use]
    pub fn new(semantic_source_id: &str, name: impl Into<KeysetName>) -> Self {
        let name = name.into();
        let id = mint_keyset_id(semantic_source_id, name.as_str());
        Self { id, name }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_mints_the_id_from_the_owning_source_and_the_name() {
        let keyset = Keyset::new("dk.orders", "natural");
        assert_eq!(
            keyset.id,
            mint_keyset_id("dk.orders", "natural"),
            "the id is the seed-derived mint, not a caller-supplied value"
        );
        assert_eq!(keyset.name, KeysetName::from("natural"));
    }

    #[test]
    fn the_same_name_on_two_sources_is_two_entries() {
        let one = Keyset::new("dk.orders", "default");
        let other = Keyset::new("dk.returns", "default");
        assert_ne!(one.id, other.id);
        assert_eq!(one.name, other.name, "the authored name is unchanged");
    }

    #[test]
    fn keyset_name_is_transparent_on_the_wire() {
        let json = serde_json::to_string(&KeysetName::from("natural")).expect("serialize");
        assert_eq!(json, "\"natural\"");
        let back: KeysetName = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, KeysetName::new("natural"));
        assert_eq!(back.as_str(), "natural");
        assert_eq!(back.to_string(), "natural");
        assert_eq!(back.clone().into_string(), "natural");
        assert_eq!(AsRef::<str>::as_ref(&back), "natural");
    }

    #[test]
    fn serde_json_round_trip() {
        let keyset = Keyset::new("dk.orders", "surrogate");
        let json = serde_json::to_string(&keyset).expect("serialize keyset");
        let back: Keyset = serde_json::from_str(&json).expect("deserialize keyset");
        assert_eq!(keyset, back);
    }
}
