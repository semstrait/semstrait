use std::collections::{BTreeMap, BTreeSet};

use semstrait_common::DataKindName;
use semstrait_common::{Dialect, EntityId};
use sha2::{Digest, Sha256};

use crate::error::CompileError;

use super::catalog::native_to_canonical;
use super::expressions::columns::ColumnTypes;
use crate::manifest::expression::{ManifestExpressions, PhysicalExprId};
use crate::manifest::physical_source::{
    PhysicalSource, SchemaId, SchemaProvenance, SourceColumn, SourceSchema, StorageProperties,
};
use crate::manifest::semantic_signature::SemanticSignature;
use crate::manifest::semantic_source::Selector;
use crate::manifest::source_alias::SourceAliases;

#[derive(Debug, Clone)]
pub(crate) struct ResolvedSource {
    pub(crate) id: EntityId,
    pub(crate) storage: StorageProperties,
    pub(super) native: Option<semstrait_catalog::Schema>,
}

impl ResolvedSource {
    pub(crate) fn locator(&self) -> String {
        self.storage.locator()
    }

    #[cfg(test)]
    pub(crate) fn model_fed_for_test(storage: StorageProperties) -> Self {
        Self {
            id: storage.source_id(),
            storage,
            native: None,
        }
    }
}

#[derive(Debug)]
pub(crate) struct LeafSource {
    pub(crate) id: EntityId,
    pub(crate) name: String,
    pub(crate) selector: Selector,
    pub(crate) dataset: String,
    pub(crate) entry: String,
}

#[derive(Debug, Default)]
pub(crate) struct SourceRoster {
    pub(crate) resolved: BTreeMap<EntityId, ResolvedSource>,
    pub(crate) by_dataset: BTreeMap<DataKindName, Vec<LeafSource>>,
}
fn guard_unique_column_names(id: &EntityId, schema: &SourceSchema) -> Result<(), CompileError> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for column in &schema.columns {
        if !seen.insert(column.name.as_str()) {
            return Err(CompileError::DuplicateSourceColumn {
                physical_source_id: id.clone(),
                column: column.name.clone(),
            });
        }
    }
    Ok(())
}

fn mint_schema_id(schema: &SourceSchema) -> SchemaId {
    let mut hasher = Sha256::new();
    for column in &schema.columns {
        update_len_prefixed(&mut hasher, column.name.as_bytes());
        match &column.data_type {
            Some(data_type) => {
                hasher.update([0x01]);
                update_len_prefixed(&mut hasher, data_type.to_string().as_bytes());
            }
            None => hasher.update([0x00]),
        }
        match column.nullable {
            Some(nullable) => hasher.update([0x01, u8::from(nullable)]),
            None => hasher.update([0x00]),
        }
    }
    let digest: [u8; 32] = hasher.finalize().into();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{byte:02x}");
    }
    SchemaId::from(hex)
}

fn update_len_prefixed(hasher: &mut Sha256, bytes: &[u8]) {
    let len = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    hasher.update(len.to_le_bytes());
    hasher.update(bytes);
}

#[derive(Debug, Clone)]
struct ColumnDemand {
    expr_id: PhysicalExprId,
}

type Demand<'a> = BTreeMap<&'a EntityId, BTreeMap<&'a str, ColumnDemand>>;

pub(crate) fn complete_sources(
    roster: &BTreeMap<EntityId, ResolvedSource>,
    aliases: &SourceAliases,
    signatures: &BTreeMap<EntityId, SemanticSignature>,
    expressions: &ManifestExpressions,
    column_types: &BTreeMap<EntityId, ColumnTypes>,
    dialect: Option<Dialect>,
) -> Result<BTreeMap<EntityId, PhysicalSource>, CompileError> {
    let demand = column_demand(signatures, expressions);
    let mut sources = BTreeMap::new();
    for (id, resolved) in roster {
        let empty = BTreeMap::new();
        let wanted = demand.get(id).unwrap_or(&empty);
        let provenance = match &resolved.native {
            Some(native) => observed_provenance(id, native, wanted, dialect)?,
            None => assumed_provenance(wanted, column_types.get(id)),
        };
        guard_unique_column_names(id, provenance.schema())?;
        let schema_id = mint_schema_id(provenance.schema());
        sources.insert(
            id.clone(),
            PhysicalSource::new(
                id.clone(),
                resolved.storage.target.name(),
                aliases.physical_alias(id)?.clone(),
                resolved.storage.clone(),
                provenance,
                schema_id,
            ),
        );
    }
    Ok(sources)
}

fn column_demand<'a>(
    signatures: &'a BTreeMap<EntityId, SemanticSignature>,
    expressions: &'a ManifestExpressions,
) -> Demand<'a> {
    let mut demand: Demand<'a> = BTreeMap::new();
    let handles = signatures.values().flat_map(|signature| {
        std::iter::once(signature.expression_id).chain(signature.filter_expr_id)
    });
    for handle in handles {
        let Some(entry) = expressions.get(&handle) else {
            continue;
        };
        let columns = expressions.columns(&handle);
        for source_id in &entry.physical_sources {
            let bucket = demand.entry(source_id).or_default();
            for (expr_id, column) in &columns {
                bucket.insert(column.name(), ColumnDemand { expr_id: *expr_id });
            }
        }
    }
    demand
}

fn observed_provenance(
    id: &EntityId,
    native: &semstrait_catalog::Schema,
    wanted: &BTreeMap<&str, ColumnDemand>,
    dialect: Option<Dialect>,
) -> Result<SchemaProvenance, CompileError> {
    let observed: BTreeSet<&str> = native
        .columns
        .iter()
        .map(|column| column.name.as_str())
        .collect();
    if let Some(missing) = wanted.keys().find(|name| !observed.contains(**name)) {
        return Err(CompileError::ColumnNotInSourceSchema {
            column: (*missing).to_owned(),
            physical_source_id: id.clone(),
        });
    }

    let mut columns = Vec::with_capacity(wanted.len());
    for column in &native.columns {
        let Some(demanded) = wanted.get(column.name.as_str()) else {
            continue;
        };
        let data_type = native_to_canonical(&column.data_type, dialect).ok_or_else(|| {
            CompileError::UnmappableNativeType {
                column: column.name.clone(),
                expr_id: Some(demanded.expr_id),
                physical_source_id: id.clone(),
                native_type: column.data_type.clone(),
            }
        })?;
        columns.push(SourceColumn {
            name: column.name.clone(),
            data_type: Some(data_type),
            nullable: Some(column.nullable),
        });
    }

    Ok(SchemaProvenance::catalog_derived(
        SourceSchema { columns },
        native.clone(),
    ))
}

fn assumed_provenance(
    wanted: &BTreeMap<&str, ColumnDemand>,
    types: Option<&ColumnTypes>,
) -> SchemaProvenance {
    SchemaProvenance::model_fed(SourceSchema {
        columns: wanted
            .keys()
            .map(|name| SourceColumn {
                name: (*name).to_owned(),
                data_type: types.and_then(|types| types.read_type(name)).cloned(),
                nullable: None,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use semstrait_sql::datatype::DataType;
    use semstrait_sql::expr::{BinaryOpKind, Expr, NamedReference};

    use crate::identity::mint_signature_id;
    use crate::manifest::expression::ExprRoot;

    fn column(name: &str, data_type: DataType) -> SourceColumn {
        SourceColumn {
            name: name.to_owned(),
            data_type: Some(data_type),
            nullable: Some(true),
        }
    }

    #[test]
    fn a_duplicate_column_name_is_refused() {
        let schema = SourceSchema {
            columns: vec![
                column("amount", DataType::BigInt),
                column("region", DataType::String),
                column("amount", DataType::BigInt),
            ],
        };
        match guard_unique_column_names(&"src-1".to_owned(), &schema) {
            Err(CompileError::DuplicateSourceColumn {
                physical_source_id,
                column,
            }) => {
                assert_eq!(physical_source_id, "src-1");
                assert_eq!(column, "amount");
            }
            other => panic!("expected DuplicateSourceColumn, got {other:?}"),
        }
    }

    #[test]
    fn distinct_column_names_pass() {
        let schema = SourceSchema {
            columns: vec![
                column("amount", DataType::BigInt),
                column("region", DataType::String),
            ],
        };
        assert!(guard_unique_column_names(&"src-1".to_owned(), &schema).is_ok());
    }

    #[test]
    fn a_schema_id_is_stable_for_the_same_columns() {
        let schema = SourceSchema {
            columns: vec![column("id", DataType::BigInt)],
        };
        assert_eq!(mint_schema_id(&schema), mint_schema_id(&schema));
    }

    #[test]
    fn a_schema_id_moves_on_reorder() {
        let a = SourceSchema {
            columns: vec![column("a", DataType::BigInt), column("b", DataType::BigInt)],
        };
        let b = SourceSchema {
            columns: vec![column("b", DataType::BigInt), column("a", DataType::BigInt)],
        };
        assert_ne!(mint_schema_id(&a), mint_schema_id(&b));
    }

    #[test]
    fn a_schema_id_length_prefix_disambiguates_fields() {
        let ab = SourceSchema {
            columns: vec![column("ab", DataType::String)],
        };
        let a = SourceSchema {
            columns: vec![column("a", DataType::String)],
        };
        assert_ne!(mint_schema_id(&ab), mint_schema_id(&a));
    }

    #[test]
    fn a_schema_id_keys_on_the_canonical_type() {
        let bigint = SourceSchema {
            columns: vec![column("id", DataType::BigInt)],
        };
        let integer = SourceSchema {
            columns: vec![column("id", DataType::Integer)],
        };
        assert_ne!(mint_schema_id(&bigint), mint_schema_id(&integer));
    }

    #[test]
    fn an_empty_schema_still_mints_an_id() {
        let empty = mint_schema_id(&SourceSchema {
            columns: Vec::new(),
        });
        assert_eq!(empty.as_str().len(), 64, "SHA-256 rendered as hex");
        assert_ne!(
            empty,
            mint_schema_id(&SourceSchema {
                columns: vec![column("id", DataType::BigInt)]
            })
        );
    }

    #[test]
    fn an_unobserved_field_never_collides_with_an_observed_one() {
        let unobserved = SourceSchema {
            columns: vec![SourceColumn {
                name: "id".to_owned(),
                data_type: None,
                nullable: None,
            }],
        };
        for observed in [
            DataType::Boolean,
            DataType::Integer,
            DataType::String,
            DataType::Decimal {
                precision: 18,
                scale: 2,
            },
        ] {
            for nullable in [Some(true), Some(false), None] {
                let typed = SourceSchema {
                    columns: vec![SourceColumn {
                        name: "id".to_owned(),
                        data_type: Some(observed.clone()),
                        nullable,
                    }],
                };
                assert_ne!(
                    mint_schema_id(&unobserved),
                    mint_schema_id(&typed),
                    "an unobserved type must not mint the id of `{observed}`"
                );
            }
        }
    }

    #[test]
    fn an_absent_nullable_is_distinct_from_both_booleans() {
        let with = |nullable| SourceSchema {
            columns: vec![SourceColumn {
                name: "id".to_owned(),
                data_type: Some(DataType::BigInt),
                nullable,
            }],
        };
        let absent = mint_schema_id(&with(None));
        assert_ne!(absent, mint_schema_id(&with(Some(true))));
        assert_ne!(absent, mint_schema_id(&with(Some(false))));
    }

    fn demanded(raw: u128) -> ColumnDemand {
        ColumnDemand {
            expr_id: PhysicalExprId::new(uuid::Uuid::from_u128(raw)),
        }
    }

    #[test]
    fn an_untranslatable_spelling_fails_the_compile_with_its_spelling() {
        let native = semstrait_catalog::Schema::new(vec![
            semstrait_catalog::SchemaColumn::new("region", "timestamptz", true),
            semstrait_catalog::SchemaColumn::new("revenue", "double", false),
        ]);
        let wanted = BTreeMap::from([("region", demanded(1)), ("revenue", demanded(1))]);
        match observed_provenance(&"src-1".to_owned(), &native, &wanted, None) {
            Err(CompileError::UnmappableNativeType {
                column,
                native_type,
                expr_id,
                physical_source_id,
            }) => {
                assert_eq!(column, "region");
                assert_eq!(native_type, "timestamptz", "the spelling is not lost");
                assert!(expr_id.is_some(), "the reading expression is known here");
                assert_eq!(physical_source_id, "src-1");
            }
            other => panic!("expected UnmappableNativeType, got {other:?}"),
        }
    }

    #[test]
    fn an_iceberg_spelling_translates_through_the_dialect_the_provider_names() {
        let native = semstrait_catalog::Schema::new(vec![semstrait_catalog::SchemaColumn::new(
            "region",
            "timestamptz",
            true,
        )]);
        let wanted = BTreeMap::from([("region", demanded(1))]);
        let provenance = observed_provenance(
            &"src-1".to_owned(),
            &native,
            &wanted,
            Some(semstrait_common::Dialect::Iceberg),
        )
        .expect("Iceberg's dialect translates `timestamptz`");
        assert_eq!(
            provenance.schema().columns,
            vec![SourceColumn {
                name: "region".to_owned(),
                data_type: Some(DataType::Timestamp { precision: 6 }),
                nullable: Some(true),
            }]
        );
    }

    #[test]
    fn a_column_the_provider_never_reported_is_refused() {
        let native = semstrait_catalog::Schema::new(vec![semstrait_catalog::SchemaColumn::new(
            "region",
            "timestamptz",
            true,
        )]);
        let wanted = BTreeMap::from([("region", demanded(1)), ("missing", demanded(1))]);
        match observed_provenance(&"src-1".to_owned(), &native, &wanted, None) {
            Err(CompileError::ColumnNotInSourceSchema {
                column,
                physical_source_id,
            }) => {
                assert_eq!(column, "missing");
                assert_eq!(physical_source_id, "src-1");
            }
            other => panic!("expected ColumnNotInSourceSchema, got {other:?}"),
        }
    }

    #[test]
    fn the_model_fed_arm_observes_nothing_and_says_nothing() {
        let wanted = BTreeMap::from([("revenue", demanded(7)), ("region", demanded(7))]);
        let provenance = assumed_provenance(&wanted, Some(&ColumnTypes::default()));
        assert_eq!(
            provenance.schema().columns,
            vec![
                SourceColumn {
                    name: "region".to_owned(),
                    data_type: None,
                    nullable: None,
                },
                SourceColumn {
                    name: "revenue".to_owned(),
                    data_type: None,
                    nullable: None,
                },
            ],
            "name order, and no type or nullability invented from a declaration"
        );
    }

    const SRC: &str = "src.orders";

    fn one_row_pool(
        semantic: &str,
        expr: semstrait_ir::PhysicalExpr,
    ) -> (BTreeMap<EntityId, SemanticSignature>, ManifestExpressions) {
        let signature_id = mint_signature_id("dk.orders", semantic);
        let mut pool = ManifestExpressions::default();
        let root = pool
            .emit(
                &expr,
                &signature_id,
                ExprRoot::SignatureExpression,
                [SRC.to_owned()],
            )
            .expect("emit");
        let signature = SemanticSignature::new(signature_id, semantic, root);
        (BTreeMap::from([(signature.id.clone(), signature)]), pool)
    }

    fn column_ref(name: &str) -> semstrait_ir::PhysicalExpr {
        Expr::Column(NamedReference::unqualified(name).expect("a name"))
    }

    #[test]
    fn a_row_that_declares_no_type_still_demands_its_columns() {
        let (signatures, pool) = one_row_pool("sem.recent", column_ref("region"));
        let demand = column_demand(&signatures, &pool);
        let wanted = demand
            .get(&SRC.to_owned())
            .expect("the row's expression names this source");
        assert_eq!(
            wanted.keys().copied().collect::<Vec<_>>(),
            vec!["region"],
            "the column a filter reads is demanded like any other"
        );
    }

    #[test]
    fn a_filter_only_column_reaches_the_observed_schema_with_its_type() {
        let (signatures, pool) = one_row_pool("sem.recent", column_ref("region"));
        let demand = column_demand(&signatures, &pool);
        let native = semstrait_catalog::Schema::new(vec![
            semstrait_catalog::SchemaColumn::new("region", "string", false),
            semstrait_catalog::SchemaColumn::new("revenue", "double", true),
        ]);
        let provenance = observed_provenance(
            &SRC.to_owned(),
            &native,
            demand.get(&SRC.to_owned()).expect("demand for this source"),
            None,
        )
        .expect("the provider reported the column");
        assert_eq!(
            provenance.schema().columns,
            vec![SourceColumn {
                name: "region".to_owned(),
                data_type: Some(DataType::String),
                nullable: Some(false),
            }],
            "observed, not assumed and not dropped"
        );
    }

    #[test]
    fn a_filter_only_column_the_provider_lacks_is_refused() {
        let (signatures, pool) = one_row_pool("sem.recent", column_ref("region"));
        let demand = column_demand(&signatures, &pool);
        let native = semstrait_catalog::Schema::new(vec![semstrait_catalog::SchemaColumn::new(
            "revenue", "double", true,
        )]);
        match observed_provenance(
            &SRC.to_owned(),
            &native,
            demand.get(&SRC.to_owned()).expect("demand for this source"),
            None,
        ) {
            Err(CompileError::ColumnNotInSourceSchema { column, .. }) => {
                assert_eq!(column, "region");
            }
            other => panic!("expected ColumnNotInSourceSchema, got {other:?}"),
        }
    }

    #[test]
    fn a_multi_leaf_expression_fabricates_no_types_on_the_model_fed_arm() {
        let product = Expr::BinaryOp {
            op: BinaryOpKind::Multiply,
            left: Box::new(column_ref("price")),
            right: Box::new(column_ref("qty")),
        };
        let (signatures, pool) = one_row_pool("sem.revenue", product);
        let demand = column_demand(&signatures, &pool);
        let provenance = assumed_provenance(
            demand.get(&SRC.to_owned()).expect("demand for this source"),
            Some(&ColumnTypes::default()),
        );
        assert_eq!(
            provenance.schema().columns,
            vec![
                SourceColumn {
                    name: "price".to_owned(),
                    data_type: None,
                    nullable: None,
                },
                SourceColumn {
                    name: "qty".to_owned(),
                    data_type: None,
                    nullable: None,
                },
            ],
            "every leaf is demanded and none is given the definition's declared type"
        );
    }
}
