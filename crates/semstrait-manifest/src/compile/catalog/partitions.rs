use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use percent_encoding::percent_decode_str;
use semstrait_model::PartitionDef;
use semstrait_sql::expr::Literal;

use super::SEPARATOR;
use crate::compile::sources::LeafSource;
use crate::error::CompileError;
use crate::manifest::physical_source::{
    HivePartition, HivePartitions, KeyValue, StorageTarget, Uri,
};
use crate::manifest::semantic_definition::SemanticDefinition;
use crate::manifest::semantic_source::Selector;

const HIVE_DEFAULT_PARTITION: &str = "__HIVE_DEFAULT_PARTITION__";

pub(super) fn discovered(
    dataset: &str,
    entry: &str,
    base_path: &Uri,
    objects: &[Uri],
) -> Result<Option<HivePartitions>, CompileError> {
    let mut by_folder: BTreeMap<&str, Vec<Uri>> = BTreeMap::new();
    for object in objects {
        let folder = partition_folder(object, base_path).unwrap_or_default();
        by_folder.entry(folder).or_default().push(object.clone());
    }
    let direct = by_folder.remove("");
    if by_folder.is_empty() {
        return Ok(None);
    }
    let discovery = Discovery {
        dataset,
        entry,
        base_path,
    };
    let mut partitions: Vec<HivePartition> = Vec::new();
    for (folder, mut listing) in by_folder {
        let key_values = discovery.key_values(folder)?;
        if let Some(first) = partitions.first() {
            discovery.same_keys(&first.key_values, &key_values, folder)?;
        }
        listing.sort();
        partitions.push(HivePartition {
            key_values,
            listing,
        });
    }
    if let Some(object) = direct.and_then(|objects| objects.into_iter().min()) {
        return Err(CompileError::ObjectBesidePartitions {
            dataset: dataset.to_owned(),
            entry: entry.to_owned(),
            object: object.into_string(),
        });
    }
    Ok(Some(HivePartitions { partitions }))
}

pub(crate) fn partition_folder<'u>(object: &'u Uri, base_path: &Uri) -> Option<&'u str> {
    let rest = object.beneath(base_path)?;
    Some(
        (rest.rfind(SEPARATOR))
            .and_then(|at| rest.get(..=at))
            .unwrap_or_default(),
    )
}

pub(crate) enum FolderFault<'f> {
    WithoutKey(&'f str),
    Empty(&'f str),
    Repeated(&'f str, String),
}

pub(crate) fn folder_key_values(folder: &str) -> Result<Vec<KeyValue>, FolderFault<'_>> {
    let mut key_values: Vec<KeyValue> = Vec::new();
    let mut end = 0;
    for name in folder.split_inclusive(SEPARATOR) {
        end += name.len();
        let at = folder.get(..end).unwrap_or(folder);
        let name = name.strip_suffix(SEPARATOR).unwrap_or(name);
        let (key, value) = name.split_once('=').ok_or(FolderFault::WithoutKey(at))?;
        let (key, value) = (decoded(key), decoded(value));
        if key.is_empty() || value.is_empty() {
            return Err(FolderFault::Empty(at));
        }
        if let Some(held) = (key_values.iter()).find(|held| held.key.eq_ignore_ascii_case(&key)) {
            return Err(FolderFault::Repeated(at, held.key.clone()));
        }
        key_values.push(KeyValue {
            key: key.into_owned(),
            value: (value != HIVE_DEFAULT_PARTITION).then(|| value.into_owned()),
        });
    }
    Ok(key_values)
}

pub(in crate::compile) fn check_claims(
    claims: &[PartitionDef],
    read: &LeafSource,
    target: &StorageTarget,
    dimensions: &BTreeMap<&str, (&SemanticDefinition, Option<&str>)>,
) -> Result<(), CompileError> {
    let dataset = read.dataset.as_str();
    let partitions = match (&read.selector, target) {
        (
            Selector::Prefix,
            StorageTarget::Path {
                partition_def: Some(hive),
                ..
            },
        ) => hive.partitions.as_slice(),
        _ => &[],
    };
    let keys: Vec<&str> = (partitions.first()).map_or_else(Vec::new, |first| {
        (first.key_values.iter())
            .map(|held| held.key.as_str())
            .collect()
    });
    let values_of = |column: &str| {
        let at = keys
            .iter()
            .position(|key| key.eq_ignore_ascii_case(column))?;
        Some((partitions.iter()).filter_map(move |partition| partition.key_values.get(at)))
    };

    let mut claimed = BTreeSet::new();
    for claim in claims {
        let dimension = claim.dimension();
        let Some((_, mapped)) = dimensions.get(dimension) else {
            return Err(CompileError::PartitionEntryUnknownDimension {
                dataset: dataset.to_owned(),
                dimension: dimension.to_owned(),
            });
        };
        if !claimed.insert(dimension) {
            return Err(CompileError::PartitionEntryRepeated {
                dataset: dataset.to_owned(),
                dimension: dimension.to_owned(),
            });
        }
        let Some(column) = mapped else {
            return Err(CompileError::PartitionEntryNotColumn {
                dataset: dataset.to_owned(),
                dimension: dimension.to_owned(),
            });
        };
        let Some(values) = values_of(column) else {
            return Err(CompileError::PartitionEntryInapplicable {
                dataset: dataset.to_owned(),
                entry: read.entry.clone(),
                dimension: dimension.to_owned(),
                key: (*column).to_owned(),
                keys: keys.iter().map(|key| (*key).to_owned()).collect(),
            });
        };
        let Some(listed) = claim.values() else {
            continue;
        };
        for KeyValue { key, value } in values {
            let value = value.as_deref().unwrap_or(HIVE_DEFAULT_PARTITION);
            if !listed.iter().any(|held| held == value) {
                return Err(CompileError::PartitionValueNotListed {
                    dataset: dataset.to_owned(),
                    entry: read.entry.clone(),
                    dimension: dimension.to_owned(),
                    key: key.clone(),
                    value: value.to_owned(),
                });
            }
        }
    }

    for (dimension, (definition, column)) in dimensions {
        let (Some(column), Some(declared)) = (column, &definition.data_type) else {
            continue;
        };
        let values = values_of(column).into_iter().flatten();
        for value in values.filter_map(|held| held.value.as_ref()) {
            if Literal::String(value.clone()).read_as(declared).is_err() {
                return Err(CompileError::PartitionValueNotCast {
                    dataset: dataset.to_owned(),
                    entry: read.entry.clone(),
                    dimension: (*dimension).to_owned(),
                    value: value.clone(),
                    data_type: declared.clone(),
                });
            }
        }
    }
    Ok(())
}

struct Discovery<'s> {
    dataset: &'s str,
    entry: &'s str,
    base_path: &'s Uri,
}

impl Discovery<'_> {
    fn key_values(&self, folder: &str) -> Result<Vec<KeyValue>, CompileError> {
        folder_key_values(folder).map_err(|fault| {
            let (dataset, entry) = (self.dataset.to_owned(), self.entry.to_owned());
            let folder = |at: &str| format!("{}{at}", self.base_path);
            match fault {
                FolderFault::WithoutKey(at) => CompileError::PartitionFolderWithoutKey {
                    dataset,
                    entry,
                    folder: folder(at),
                },
                FolderFault::Empty(at) => CompileError::PartitionKeyValueEmpty {
                    dataset,
                    entry,
                    folder: folder(at),
                },
                FolderFault::Repeated(at, key) => CompileError::PartitionKeyRepeated {
                    dataset,
                    entry,
                    folder: folder(at),
                    key,
                },
            }
        })
    }

    fn same_keys(
        &self,
        first: &[KeyValue],
        keys: &[KeyValue],
        folder: &str,
    ) -> Result<(), CompileError> {
        let names = |keys: &[KeyValue]| keys.iter().map(|held| held.key.clone()).collect();
        if first.len() != keys.len()
            || !(first.iter().zip(keys))
                .all(|(one, other)| one.key.eq_ignore_ascii_case(&other.key))
        {
            return Err(CompileError::PartitionKeySequencesDiffer {
                dataset: self.dataset.to_owned(),
                entry: self.entry.to_owned(),
                folder: format!("{}{folder}", self.base_path),
                keys: names(keys),
                first: names(first),
            });
        }
        match (first.iter().zip(keys)).find(|(one, other)| one.key != other.key) {
            Some((one, other)) => Err(CompileError::PartitionKeySpelledTwice {
                dataset: self.dataset.to_owned(),
                entry: self.entry.to_owned(),
                first: one.key.clone(),
                second: other.key.clone(),
            }),
            None => Ok(()),
        }
    }
}

fn decoded(text: &str) -> Cow<'_, str> {
    percent_decode_str(text)
        .decode_utf8()
        .unwrap_or(Cow::Borrowed(text))
}
