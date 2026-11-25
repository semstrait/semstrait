use semstrait_common::EntityId;
use serde::de::{Deserializer, Error as _};
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};

pub(crate) trait NamedEntity {
    fn entity_id(&self) -> Option<&str>;
    fn entity_name(&self) -> &str;
}

pub(crate) fn reject_duplicate_authored_ids<T: NamedEntity, E: serde::de::Error>(
    items: &[T],
) -> Result<(), E> {
    let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for id in items.iter().filter_map(NamedEntity::entity_id) {
        if !seen.insert(id) {
            return Err(E::custom(format!("duplicate entity id `{id}`")));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityBTreeMap<T>(BTreeMap<EntityId, T>);

impl<T> Default for EntityBTreeMap<T> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<T> EntityBTreeMap<T> {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn into_inner(self) -> BTreeMap<EntityId, T> {
        self.0
    }
}

impl<T> From<BTreeMap<EntityId, T>> for EntityBTreeMap<T> {
    fn from(map: BTreeMap<EntityId, T>) -> Self {
        Self(map)
    }
}

impl<T> Deref for EntityBTreeMap<T> {
    type Target = BTreeMap<EntityId, T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for EntityBTreeMap<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Serialize + NamedEntity> Serialize for EntityBTreeMap<T> {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let mut values: Vec<&T> = self.0.values().collect();
        values.sort_by(|a, b| a.entity_name().cmp(b.entity_name()));
        let mut seq = ser.serialize_seq(Some(values.len()))?;
        for v in values {
            seq.serialize_element(v)?;
        }
        seq.end()
    }
}

impl<'de, T: Deserialize<'de> + NamedEntity> Deserialize<'de> for EntityBTreeMap<T> {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let items = Vec::<T>::deserialize(de)?;
        reject_duplicate_authored_ids::<T, D::Error>(&items)?;
        let mut map = BTreeMap::new();
        for item in items {
            let id = item
                .entity_id()
                .ok_or_else(|| D::Error::custom("entity is missing an `id`"))?
                .to_owned();
            map.insert(id, item);
        }
        Ok(Self(map))
    }
}

#[cfg(feature = "schemars")]
impl<T: schemars::JsonSchema> schemars::JsonSchema for EntityBTreeMap<T> {
    fn inline_schema() -> bool {
        <Vec<T>>::inline_schema()
    }
    fn schema_name() -> std::borrow::Cow<'static, str> {
        <Vec<T>>::schema_name()
    }
    fn schema_id() -> std::borrow::Cow<'static, str> {
        <Vec<T>>::schema_id()
    }
    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        <Vec<T>>::json_schema(generator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Probe {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<EntityId>,
        name: String,
    }

    impl NamedEntity for Probe {
        fn entity_id(&self) -> Option<&str> {
            self.id.as_deref()
        }
        fn entity_name(&self) -> &str {
            &self.name
        }
    }

    fn entity_btree_map(pairs: &[(&str, &str)]) -> EntityBTreeMap<Probe> {
        pairs
            .iter()
            .map(|(id, name)| {
                (
                    (*id).to_owned(),
                    Probe {
                        id: Some((*id).to_owned()),
                        name: (*name).to_owned(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>()
            .into()
    }

    #[test]
    fn serializes_in_name_order_not_id_order() {
        let m = entity_btree_map(&[("id-9", "alpha"), ("id-1", "zulu"), ("id-5", "mike")]);
        let json = serde_json::to_value(&m).expect("serialize");
        let names: Vec<&str> = json
            .as_array()
            .expect("seq")
            .iter()
            .map(|e| e["name"].as_str().expect("name"))
            .collect();
        assert_eq!(names, ["alpha", "mike", "zulu"]);
    }

    #[test]
    fn deserialize_rekeys_by_inline_id() {
        let m = entity_btree_map(&[("id-9", "alpha"), ("id-1", "zulu")]);
        let json = serde_json::to_string(&m).expect("serialize");
        let back: EntityBTreeMap<Probe> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, m);
        assert_eq!(back.get("id-9").expect("keyed by id").name, "alpha");
    }

    #[test]
    fn round_trip_is_byte_identical_regardless_of_insertion_order() {
        let a = entity_btree_map(&[("id-9", "alpha"), ("id-1", "zulu"), ("id-5", "mike")]);
        let b = entity_btree_map(&[("id-5", "mike"), ("id-9", "alpha"), ("id-1", "zulu")]);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap()
        );
    }

    #[test]
    fn deserialize_rejects_duplicate_ids() {
        let json = r#"[{"id":"dup","name":"a"},{"id":"dup","name":"b"}]"#;
        let err = serde_json::from_str::<EntityBTreeMap<Probe>>(json).unwrap_err();
        assert!(
            err.to_string().contains("duplicate entity id"),
            "expected SR-12 dup-id rejection, got: {err}"
        );
    }

    #[test]
    fn deserialize_rejects_entity_missing_id() {
        let json = r#"[{"name":"unminted"}]"#;
        let err = serde_json::from_str::<EntityBTreeMap<Probe>>(json).unwrap_err();
        assert!(
            err.to_string().contains("missing an `id`"),
            "expected residual-None rejection, got: {err}"
        );
    }
}
