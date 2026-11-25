use crate::error::build::ModelBuildErrorKind;
use crate::error::serde_message;
use crate::error::validate::ValidateErrorKind;
use crate::model::types::entities::semantic_body::gated_value;
use semstrait_common::diagnostic::Diagnostic;
use serde::de::DeserializeOwned;
use serde::ser::SerializeMap;
use serde::{Deserializer, Serialize, Serializer};
use serde_yaml::{Mapping, Value};
use std::collections::BTreeSet;

pub const DEFAULT_KEYSET: &str = "default";

#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct Keyness {
    #[cfg_attr(
        feature = "schemars",
        schemars(
            default,
            description = "Marks this site as a key; required before naming a `keyset:` or `keysets:`."
        )
    )]
    is_key: bool,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "Shorthand for a site in exactly one keyset; merged into `keysets:`."
        )
    )]
    keyset: Option<String>,

    #[cfg_attr(
        feature = "schemars",
        schemars(
            description = "Names the keysets a site belongs to, when it is in more than one."
        )
    )]
    keysets: Option<BTreeSet<String>>,
}

impl Keyness {
    pub fn keyset_membership(&self) -> BTreeSet<String> {
        if !self.is_key {
            return BTreeSet::new();
        }
        let mut names: BTreeSet<String> = self.keysets.clone().unwrap_or_default();
        names.extend(self.keyset.clone());
        if names.is_empty() {
            names.insert(DEFAULT_KEYSET.to_owned());
        }
        names
    }

    pub fn key() -> Self {
        Self {
            is_key: true,
            keyset: None,
            keysets: None,
        }
    }

    pub fn keysets_named(
        names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<Self, Diagnostic<ModelBuildErrorKind>> {
        let names: BTreeSet<String> = names.into_iter().map(Into::into).collect();
        if names.is_empty() {
            return Err(Diagnostic::new(ModelBuildErrorKind::Validate(
                ValidateErrorKind::KeysetListEmpty,
            )));
        }
        if names.iter().any(|name| blank(name)) {
            return Err(Diagnostic::new(ModelBuildErrorKind::Validate(
                ValidateErrorKind::KeysetNameBlank,
            )));
        }
        Ok(Self {
            is_key: true,
            keyset: None,
            keysets: Some(names),
        })
    }
}

impl Serialize for Keyness {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let membership = self.keyset_membership();
        let names_only_default = membership.len() == 1 && membership.contains(DEFAULT_KEYSET);
        let emit_keysets = !membership.is_empty() && !names_only_default;
        let mut map =
            serializer.serialize_map(Some(usize::from(self.is_key) + usize::from(emit_keysets)))?;
        if self.is_key {
            map.serialize_entry("is_key", &true)?;
        }
        if emit_keysets {
            map.serialize_entry("keysets", &membership)?;
        }
        map.end()
    }
}

pub(crate) fn decode_keyed_site<'de, D, T>(
    deserializer: D,
    carrier: &'static str,
    fields: &'static [&'static str],
) -> Result<(Keyness, T), D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    use serde::de::Error as _;

    let mut value = gated_value::<D>(deserializer, carrier, fields)?;
    let keyness = take_keyness::<D::Error>(&mut value, carrier)?;
    let body = serde_yaml::from_value(value).map_err(D::Error::custom)?;
    Ok((keyness, body))
}

fn take_keyness<E: serde::de::Error>(
    value: &mut Value,
    carrier: &'static str,
) -> Result<Keyness, E> {
    let Value::Mapping(map) = value else {
        return Ok(Keyness::default());
    };
    let is_key = take_field::<bool, E>(map, "is_key", carrier)?.unwrap_or(false);
    let keyset = take_field::<String, E>(map, "keyset", carrier)?;
    let keysets = take_field::<BTreeSet<String>, E>(map, "keysets", carrier)?;

    if keyset.is_some() && keysets.is_some() {
        return Err(E::custom(serde_message::keyset_both_spellings(carrier)));
    }
    if keysets.as_ref().is_some_and(BTreeSet::is_empty) {
        return Err(E::custom(serde_message::keyset_list_empty(carrier)));
    }
    if let Some(spelling) = blank_name_spelling(keyset.as_deref(), keysets.as_ref()) {
        return Err(E::custom(serde_message::keyset_name_blank(
            spelling, carrier,
        )));
    }
    if !is_key {
        if let Some(spelling) = keyset
            .as_ref()
            .map(|_| "keyset")
            .or_else(|| keysets.as_ref().map(|_| "keysets"))
        {
            return Err(E::custom(serde_message::keyset_without_is_key(
                spelling, carrier,
            )));
        }
    }
    Ok(Keyness {
        is_key,
        keyset,
        keysets,
    })
}

fn blank(name: &str) -> bool {
    name.trim().is_empty()
}

fn blank_name_spelling(
    keyset: Option<&str>,
    keysets: Option<&BTreeSet<String>>,
) -> Option<&'static str> {
    if keyset.is_some_and(blank) {
        return Some("keyset");
    }
    keysets
        .is_some_and(|names| names.iter().any(|n| blank(n)))
        .then_some("keysets")
}

fn take_field<T: DeserializeOwned, E: serde::de::Error>(
    map: &mut Mapping,
    key: &str,
    carrier: &str,
) -> Result<Option<T>, E> {
    match map.remove(key) {
        None => Ok(None),
        Some(value) => serde_yaml::from_value(value)
            .map(Some)
            .map_err(|e| E::custom(format!("{carrier}: `{key}`: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[track_caller]
    fn membership(k: &Keyness) -> Vec<String> {
        k.keyset_membership().into_iter().collect()
    }

    #[test]
    fn the_accessor_reaches_all_three_authored_states() {
        assert!(membership(&Keyness::default()).is_empty());
        assert_eq!(membership(&Keyness::key()), vec!["default"]);
        assert_eq!(
            membership(&Keyness::keysets_named(["natural", "surrogate"]).expect("named keysets")),
            vec!["natural", "surrogate"]
        );
        let singular = Keyness {
            is_key: true,
            keyset: Some("natural".to_owned()),
            keysets: None,
        };
        assert_eq!(membership(&singular), vec!["natural"]);
    }

    #[test]
    fn an_unactivated_site_has_no_membership_whatever_it_names() {
        let unactivated = Keyness {
            is_key: false,
            keyset: None,
            keysets: Some(BTreeSet::from(["natural".to_owned()])),
        };
        assert!(membership(&unactivated).is_empty());
    }
}
