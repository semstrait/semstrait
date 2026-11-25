use semstrait_ir::{FilesScan, Reader, Schema, SchemaColumn, TableScan};
use semstrait_manifest::{Selector, StorageProperties, StorageTarget};

use crate::read_model::ReadModel;
use crate::strategy::{DatasetArm, Strategy, StrategyInput};
use crate::{Combine, Decision, FramePath, PhysicalRead, PhysicalSource, PlanError};

#[derive(Debug)]
pub(crate) struct DatasetStrategy;

impl Strategy<DatasetArm> for DatasetStrategy {
    fn decide(
        &self,
        read_model: &dyn ReadModel,
        arm: &DatasetArm,
        input: &StrategyInput,
    ) -> Result<Decision, PlanError> {
        Ok(Decision {
            stand_in: None,
            operands: Vec::new(),
            combine: Combine::Leaf,
            physical_read: Some(physical_read(read_model, arm, input)?),
        })
    }
}

fn physical_read(
    read_model: &dyn ReadModel,
    arm: &DatasetArm,
    input: &StrategyInput,
) -> Result<PhysicalRead, PlanError> {
    let source = &arm.physical_source;
    let reader = reader(&source.storage, &arm.selector, input)?;
    let columns = (input.read_leaves(read_model)?.into_iter())
        .map(|column| typed(source, &input.path, column))
        .collect::<Result<_, _>>()?;
    Ok(PhysicalRead {
        reader,
        schema: Schema { columns },
    })
}

fn reader(
    storage: &StorageProperties,
    selector: &Selector,
    input: &StrategyInput,
) -> Result<Option<Reader>, PlanError> {
    match (&storage.target, selector) {
        (StorageTarget::Table { table, .. }, Selector::Table) => {
            let names = [
                (table.catalog.as_ref())
                    .map_or("", |catalog| catalog.as_str())
                    .to_owned(),
                table.namespace.dotted(),
                table.name.clone(),
            ];
            Ok(Some(TableScan::named_table(names, storage.format).into()))
        }
        (
            StorageTarget::Path {
                base_path,
                partition_def: Some(_),
                ..
            },
            Selector::Prefix,
        ) => Ok(Some(
            FilesScan::new(storage.format, None, base_path.as_str(), Vec::new()).into(),
        )),
        (
            StorageTarget::Path {
                base_path,
                listings,
                ..
            },
            Selector::Prefix | Selector::Pattern(_) | Selector::Uris(_),
        ) => {
            let pattern = match selector {
                Selector::Pattern(glob) => Some(glob.to_string()),
                _ => None,
            };
            let paths: Vec<_> = (listings.iter())
                .filter(|object| selector.selects(base_path, object))
                .map(ToString::to_string)
                .collect();
            Ok((!paths.is_empty())
                .then(|| FilesScan::new(storage.format, pattern, base_path.as_str(), paths).into()))
        }
        _ => Err(input.broken()),
    }
}

pub(crate) fn typed(
    source: &PhysicalSource,
    path: &FramePath,
    name: String,
) -> Result<SchemaColumn, PlanError> {
    (source.columns.iter())
        .find(|column| column.name == name)
        .and_then(|column| {
            Some(SchemaColumn {
                name: column.name.clone(),
                data_type: column.data_type.clone()?,
                nullable: column.nullable.unwrap_or(true),
            })
        })
        .ok_or_else(|| PlanError::UnresolvedType {
            path: path.clone(),
            relation: source.relation.clone(),
            column: name,
        })
}
