use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

use semstrait_catalog::{
    CatalogProvider, CatalogProviderErrorKind, FileFormat, NamespaceRef, ResolvedTable, TableRef,
};
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};
use semstrait_common::io::{FileEntry, FileSystem, Path};
use semstrait_common::DataKindName;
use semstrait_common::{Dialect, EntityId};
use semstrait_model::{LeafExtras, StorageConfig, StorageFormat};
use semstrait_sql::datatype::DataType;

use super::sources::{LeafSource, ResolvedSource, SourceRoster};
use crate::error::CompileError;
use crate::manifest::physical_source::{
    HivePartitions, NotNormal, StorageProperties, StorageTarget, Uri,
};
use crate::manifest::semantic_source::{debris, glob_match, Selector, SEPARATOR};

pub(crate) mod partitions;

const WILDCARDS: [char; 2] = ['*', '?'];

#[allow(
    clippy::result_large_err,
    reason = "compile's ratified rich diagnostic return shape is intentional"
)]
pub(crate) async fn fetch_physical_sources(
    leaves: &[(DataKindName, &LeafExtras)],
    catalog: Option<&dyn CatalogProvider>,
    fs: &dyn FileSystem,
) -> Result<(SourceRoster, Diagnostics<CompileError>), Diagnostic<CompileError>> {
    let decls = (leaves.iter())
        .filter_map(|(name, extras)| Some((name, extras.storage.as_ref()?)))
        .map(|(name, storage)| StorageDecl::new(name.as_str(), storage))
        .collect::<Result<Vec<_>, _>>()
        .map_err(Diagnostic::new)?;

    let mut roster = SourceRoster::default();
    let mut selected: BTreeMap<EntityId, BTreeSet<Uri>> = BTreeMap::new();
    let mut partitions: BTreeMap<EntityId, HivePartitions> = BTreeMap::new();
    for decl in decls {
        for read in decl.reads(fs).await.map_err(Diagnostic::new)? {
            let id = read.storage.source_id();
            let resolved = match roster.resolved.entry(id.clone()) {
                Entry::Occupied(held) => held.into_mut(),
                Entry::Vacant(slot) => {
                    slot.insert(resolve_source(&id, read.storage, catalog).await?)
                }
            };
            if !resolved.storage.reads_a_supported_format() {
                return Err(Diagnostic::new(CompileError::UnsupportedSourceFormat {
                    dataset: decl.dataset.clone(),
                    entry: read.entry,
                    format: resolved.storage.format,
                }));
            }
            let name = resolved.storage.target.name().to_owned();
            selected.entry(id.clone()).or_default().extend(read.objects);
            if let Some(discovered) = read.partitions {
                partitions.insert(id.clone(), discovered);
            }
            let declared = roster
                .by_dataset
                .entry(decl.dataset.as_str().into())
                .or_default();
            if !declared.iter().any(|source| source.id == id) {
                declared.push(LeafSource {
                    id,
                    name,
                    selector: read.selector,
                    dataset: decl.dataset.clone(),
                    entry: read.entry,
                });
            }
        }
    }
    for (id, objects) in selected {
        if let Some(StorageTarget::Path {
            listings,
            partition_def,
            ..
        }) = roster
            .resolved
            .get_mut(&id)
            .map(|source| &mut source.storage.target)
        {
            *listings = objects.into_iter().collect();
            *partition_def = partitions.remove(&id);
        }
    }

    Ok((roster, Vec::new()))
}

#[derive(Debug)]
struct StorageDecl {
    dataset: String,
    entries: Entries,
}

#[derive(Debug)]
enum Entries {
    Paths {
        paths: Vec<(String, Uri)>,
        format: Option<FileFormat>,
    },
    Tables(Vec<(String, TableRef)>),
}

#[derive(Debug)]
struct Read {
    entry: String,
    storage: StorageProperties,
    selector: Selector,
    objects: Vec<Uri>,
    partitions: Option<HivePartitions>,
}

#[derive(Debug)]
struct Selection {
    base_path: Uri,
    selector: Selector,
    objects: Vec<Uri>,
    partitions: Option<HivePartitions>,
}

impl StorageDecl {
    fn new(dataset: &str, storage: &StorageConfig) -> Result<Self, CompileError> {
        guard_storage_arms(dataset, storage)?;
        let entries = if storage.tables.is_empty() {
            let mut written = BTreeSet::new();
            let paths = (storage.paths.iter()).map(|entry| {
                let uri = normal(dataset, entry)?;
                if !written.insert(uri.clone()) {
                    return Err(CompileError::DuplicateStorageUri {
                        dataset: dataset.to_owned(),
                        uri: uri.into_string(),
                    });
                }
                Ok((entry.clone(), uri))
            });
            Entries::Paths {
                paths: paths.collect::<Result<_, _>>()?,
                format: storage.format.and_then(authored_format),
            }
        } else {
            let tables = (storage.tables.iter()).map(|entry| {
                if entry.contains(WILDCARDS) {
                    return Err(CompileError::TableGlob {
                        dataset: dataset.to_owned(),
                        entry: entry.clone(),
                    });
                }
                Ok((entry.clone(), table_ref_from_fqn(entry)))
            });
            Entries::Tables(tables.collect::<Result<_, _>>()?)
        };
        Ok(Self {
            dataset: dataset.to_owned(),
            entries,
        })
    }

    async fn reads(&self, fs: &dyn FileSystem) -> Result<Vec<Read>, CompileError> {
        match &self.entries {
            Entries::Tables(tables) => Ok((tables.iter())
                .map(|(entry, table)| Read {
                    entry: entry.clone(),
                    storage: StorageProperties::bare(StorageTarget::table(table.clone())),
                    selector: Selector::Table,
                    objects: Vec::new(),
                    partitions: None,
                })
                .collect()),
            Entries::Paths { paths, format } => self.path_reads(paths, *format, fs).await,
        }
    }

    async fn path_reads(
        &self,
        paths: &[(String, Uri)],
        format: Option<FileFormat>,
        fs: &dyn FileSystem,
    ) -> Result<Vec<Read>, CompileError> {
        let mut by_base: BTreeMap<Uri, Read> = BTreeMap::new();
        let mut readers: BTreeMap<Uri, &str> = BTreeMap::new();
        for (entry, uri) in paths {
            for Selection {
                base_path,
                selector,
                objects,
                partitions,
            } in self.select(entry, uri, fs).await?
            {
                if !base_path.is_normal() {
                    return Err(CompileError::BasePathNotNormal {
                        dataset: self.dataset.clone(),
                        entry: entry.clone(),
                        base_path: base_path.into_string(),
                    });
                }
                for object in &objects {
                    if let Some(held) = readers.insert(object.clone(), entry) {
                        return Err(self.read_twice(held, entry));
                    }
                }
                match by_base.entry(base_path) {
                    Entry::Vacant(slot) => {
                        let target = StorageTarget::path(slot.key().clone(), Vec::new());
                        let storage = StorageProperties::new(target, format, Vec::new());
                        slot.insert(Read {
                            entry: entry.clone(),
                            storage,
                            selector,
                            objects,
                            partitions,
                        });
                    }
                    Entry::Occupied(mut held) => {
                        let held = held.get_mut();
                        let (Selector::Uris(named), Selector::Uris(more)) =
                            (&mut held.selector, selector)
                        else {
                            return Err(self.read_twice(&held.entry, entry));
                        };
                        named.extend(more);
                        named.sort();
                        held.objects.extend(objects);
                    }
                }
            }
        }
        Ok(by_base.into_values().collect())
    }

    async fn select(
        &self,
        entry: &str,
        uri: &Uri,
        fs: &dyn FileSystem,
    ) -> Result<Vec<Selection>, CompileError> {
        let path = uri.as_str();
        let (folder, object) = (path.rfind(SEPARATOR))
            .and_then(|at| path.split_at_checked(at + SEPARATOR.len_utf8()))
            .unwrap_or(("", path));
        let glob = path.contains(WILDCARDS);
        if !glob && !object.is_empty() {
            self.named_object(entry, uri, fs).await?;
            let named = vec![uri.clone()];
            return Ok(vec![Selection {
                base_path: Uri::from(folder),
                selector: Selector::Uris(named.clone()),
                objects: named,
                partitions: None,
            }]);
        }
        let mut folders: BTreeMap<Uri, Vec<Uri>> = BTreeMap::new();
        for listed in self.list(entry, listing_prefix(path), fs).await? {
            let uri = listed.path.as_str();
            let Some((base_path, rest)) = matched_folder(folder, uri) else {
                continue;
            };
            if !skipped(rest, listed.size) && (object.is_empty() || glob_match(object, rest)) {
                folders
                    .entry(base_path.into())
                    .or_default()
                    .push(uri.into());
            }
        }
        if folders.is_empty() {
            if glob {
                return Err(CompileError::SourcePatternMatchedNothing {
                    dataset: self.dataset.clone(),
                    pattern: entry.to_owned(),
                });
            }
            folders.insert(uri.clone(), Vec::new());
        }
        (folders.into_iter())
            .map(|(base_path, objects)| {
                let (selector, partitions) = match object {
                    "" => (
                        Selector::Prefix,
                        partitions::discovered(&self.dataset, entry, &base_path, &objects)?,
                    ),
                    object => (
                        Selector::Pattern(format!("{base_path}{object}").into()),
                        None,
                    ),
                };
                Ok(Selection {
                    base_path,
                    selector,
                    objects,
                    partitions,
                })
            })
            .collect()
    }

    async fn named_object(
        &self,
        entry: &str,
        uri: &Uri,
        fs: &dyn FileSystem,
    ) -> Result<(), CompileError> {
        let listed = self.list(entry, uri.as_str(), fs).await?;
        let size = (listed.iter())
            .find(|listed| listed.path.as_str() == uri.as_str())
            .map(|object| object.size);
        let (dataset, entry, uri) = (self.dataset.clone(), entry.to_owned(), uri.to_string());
        match size {
            Some(0) => Err(CompileError::StorageObjectEmpty {
                dataset,
                entry,
                uri,
            }),
            Some(_) => Ok(()),
            None => Err(CompileError::StorageObjectMissing {
                dataset,
                entry,
                uri,
            }),
        }
    }

    async fn list(
        &self,
        entry: &str,
        prefix: &str,
        fs: &dyn FileSystem,
    ) -> Result<Vec<FileEntry>, CompileError> {
        fs.list(&Path::new(prefix))
            .await
            .map_err(|kind| CompileError::SourceListingFailed {
                dataset: self.dataset.clone(),
                entry: entry.to_owned(),
                reason: describe(&kind),
            })
    }

    fn read_twice(&self, first: &str, second: &str) -> CompileError {
        CompileError::StorageReadTwice {
            dataset: self.dataset.clone(),
            first: first.to_owned(),
            second: second.to_owned(),
        }
    }
}

fn normal(dataset: &str, entry: &str) -> Result<Uri, CompileError> {
    Uri::normal(entry).map_err(|fault| match fault {
        NotNormal::Relative => CompileError::RelativeStoragePath {
            dataset: dataset.to_owned(),
            entry: entry.to_owned(),
        },
        NotNormal::Scheme(scheme) => CompileError::UnsupportedStorageScheme {
            dataset: dataset.to_owned(),
            entry: entry.to_owned(),
            scheme,
        },
    })
}

fn guard_storage_arms(dataset_name: &str, storage: &StorageConfig) -> Result<(), CompileError> {
    if storage.paths.is_empty() || storage.tables.is_empty() {
        return Ok(());
    }
    Err(CompileError::StorageMixesPathsAndTables {
        dataset: dataset_name.to_owned(),
        reason: format!(
            "`paths:` reaches {} file source(s), whose schema is assumed from the model, and \
             `tables:` reaches {} catalog source(s), whose schema is observed from a provider; \
             one leaf cannot carry both provenances, so split it into one Dataset per storage \
             kind and compose them",
            storage.paths.len(),
            storage.tables.len(),
        ),
    })
}

fn authored_format(format: StorageFormat) -> Option<FileFormat> {
    match format {
        StorageFormat::Parquet => Some(FileFormat::Parquet),
        StorageFormat::Csv => Some(FileFormat::Csv),
        StorageFormat::Json => Some(FileFormat::Json),
        StorageFormat::Orc => Some(FileFormat::Orc),
        StorageFormat::Avro => Some(FileFormat::Avro),
        _ => None,
    }
}

fn listing_prefix(pattern: &str) -> &str {
    let first = pattern.find(WILDCARDS).unwrap_or(pattern.len());
    let end = pattern[..first]
        .rfind(SEPARATOR)
        .map_or(0, |index| index + SEPARATOR.len_utf8());
    &pattern[..end]
}

fn matched_folder<'p>(folder: &str, path: &'p str) -> Option<(&'p str, &'p str)> {
    let depth = folder.matches(SEPARATOR).count().checked_sub(1)?;
    let (end, _) = path.match_indices(SEPARATOR).nth(depth)?;
    let (candidate, rest) = path.split_at_checked(end + SEPARATOR.len_utf8())?;
    (!rest.is_empty() && glob_match(folder, candidate)).then_some((candidate, rest))
}

fn skipped(rest: &str, size: u64) -> bool {
    size == 0 || debris(rest)
}

#[allow(
    clippy::result_large_err,
    reason = "compile's ratified rich diagnostic return shape is intentional"
)]
async fn resolve_source(
    id: &EntityId,
    storage: StorageProperties,
    catalog: Option<&dyn CatalogProvider>,
) -> Result<ResolvedSource, Diagnostic<CompileError>> {
    if let (Some(table), Some(provider)) = (storage.target.table_ref(), catalog) {
        let resolved = provider
            .resolve_table(table)
            .await
            .map_err(|kind| catalog_fetch_error(id, &storage.locator(), kind))?;
        return Ok(observed_source(id, resolved));
    }
    Ok(ResolvedSource {
        id: id.clone(),
        storage,
        native: None,
    })
}

fn observed_source(id: &EntityId, resolved: ResolvedTable) -> ResolvedSource {
    let target = StorageTarget::table_at(resolved.table, resolved.location);
    ResolvedSource {
        id: id.clone(),
        storage: StorageProperties::new(target, Some(resolved.format), resolved.partitions),
        native: Some(resolved.schema),
    }
}

pub(super) fn native_to_canonical(native: &str, dialect: Option<Dialect>) -> Option<DataType> {
    DataType::from_native_name(native, dialect)
}

fn catalog_fetch_error(
    id: &EntityId,
    locator: &str,
    kind: CatalogProviderErrorKind,
) -> Diagnostic<CompileError> {
    let error = match kind {
        CatalogProviderErrorKind::TableNotFound { .. } => CompileError::CatalogSourceMissing {
            source_id: id.clone(),
            locator: locator.to_owned(),
        },
        other => CompileError::CatalogFetch {
            source_id: id.clone(),
            locator: locator.to_owned(),
            reason: describe(&other),
        },
    };
    Diagnostic::new(error)
}

fn describe<K: semstrait_common::diagnostic::Diagnose>(kind: &K) -> String {
    kind.message()
}

pub(crate) fn table_ref_from_fqn(fqn: &str) -> TableRef {
    let mut parts: Vec<&str> = fqn.split('.').filter(|s| !s.is_empty()).collect();
    let name = parts.pop().unwrap_or(fqn);
    TableRef::new(NamespaceRef::new(parts.iter().copied()), name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_listing_prefix_is_the_deepest_literal_folder() {
        assert_eq!(
            listing_prefix("s3://lake/orders/*.parquet"),
            "s3://lake/orders/"
        );
        assert_eq!(listing_prefix("s3://lake/*/orders/"), "s3://lake/");
        assert_eq!(
            listing_prefix("s3://lake/orders/"),
            "s3://lake/orders/",
            "a folder with no wildcard lists itself"
        );
    }

    #[test]
    fn a_folder_glob_matches_the_folder_an_object_lies_beneath() {
        let folder = "s3://lake/*/orders/";
        assert_eq!(
            matched_folder(folder, "s3://lake/eu/orders/a.parquet"),
            Some(("s3://lake/eu/orders/", "a.parquet"))
        );
        assert_eq!(
            matched_folder(folder, "s3://lake/eu/orders/2024/a.parquet"),
            Some(("s3://lake/eu/orders/", "2024/a.parquet")),
            "an object deeper in the folder lies beneath it too"
        );
        assert_eq!(
            matched_folder(folder, "s3://lake/eu/returns/a.parquet"),
            None
        );
        assert_eq!(matched_folder(folder, "s3://lake/eu/orders/"), None);
    }

    #[test]
    fn every_authored_format_maps_onto_the_catalog_vocabulary() {
        assert_eq!(
            authored_format(StorageFormat::Parquet),
            Some(FileFormat::Parquet)
        );
        assert_eq!(authored_format(StorageFormat::Csv), Some(FileFormat::Csv));
        assert_eq!(authored_format(StorageFormat::Json), Some(FileFormat::Json));
        assert_eq!(authored_format(StorageFormat::Orc), Some(FileFormat::Orc));
        assert_eq!(authored_format(StorageFormat::Avro), Some(FileFormat::Avro));
    }

    #[test]
    fn the_storage_arm_guard_fires_only_on_a_genuine_mix() {
        let path = || vec!["a.parquet".to_owned()];
        let table = || vec!["db.t".to_owned()];
        let paths_only = StorageConfig::builder().paths(path()).build();
        let tables_only = StorageConfig::builder().tables(table()).build();
        let both = StorageConfig::builder()
            .paths(path())
            .tables(table())
            .build();
        assert!(guard_storage_arms("orders", &paths_only).is_ok());
        assert!(guard_storage_arms("orders", &tables_only).is_ok());
        assert!(matches!(
            guard_storage_arms("orders", &both),
            Err(CompileError::StorageMixesPathsAndTables { .. })
        ));
    }
}
