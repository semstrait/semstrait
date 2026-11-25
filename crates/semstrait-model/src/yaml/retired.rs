use crate::error::parse::ParseErrorKind;
use crate::model::serde::digits::Digits;
use semstrait_common::diagnostic::{Diagnostic, Diagnostics, Location, SourceId, Span};
use serde::Deserialize;
use serde_yaml::Value;

const DATA_KIND_COLLECTIONS: [&str; 4] = ["datasets", "grainsets", "unionsets", "joinsets"];

const KEYS_BLOCK: &str = "keys";

const KEYNESS_KEYS: [&str; 3] = ["is_key", "keyset", "keysets"];

pub(crate) fn retired_data_kind_spellings(
    expanded: &str,
    source: &str,
) -> Diagnostics<ParseErrorKind> {
    let Ok(root) = Value::deserialize(Digits::new(serde_yaml::Deserializer::from_str(expanded)))
    else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let mut pending: Vec<(&Value, bool)> = child_entries(root.get("semantic_model"))
        .map(|entry| (entry, false))
        .collect();
    while let Some((entry, nested)) = pending.pop() {
        let name = entry
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("<unnamed>");
        if entry.get(KEYS_BLOCK).is_some() {
            out.push(warning(
                ParseErrorKind::DataKindKeysBlockRetired {
                    data_kind: name.to_owned(),
                },
                source,
            ));
        }
        if nested {
            for spelling in KEYNESS_KEYS {
                if entry.get(spelling).is_some() {
                    out.push(warning(
                        ParseErrorKind::NestedDataKindKeynessIgnored {
                            data_kind: name.to_owned(),
                            spelling: spelling.to_owned(),
                        },
                        source,
                    ));
                }
            }
        }
        pending.extend(child_entries(Some(entry)).map(|child| (child, true)));
    }
    out
}

fn child_entries(carrier: Option<&Value>) -> impl Iterator<Item = &Value> {
    DATA_KIND_COLLECTIONS
        .into_iter()
        .filter_map(move |collection| carrier?.get(collection)?.as_sequence())
        .flatten()
        .filter(|entry| entry.is_mapping())
}

fn warning(kind: ParseErrorKind, source: &str) -> Diagnostic<ParseErrorKind> {
    Diagnostic::new(kind).with_location(Location::new(SourceId::named(source), Span::unknown()))
}
