use semstrait_catalog::FileFormat;

use crate::error::IrErrorKind;
use crate::expr::PhysicalExpr;
use crate::plan::common::{condition, NodeCommon};
use crate::types::{Schema, SchemaColumn};

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ScanNode {
    pub common: NodeCommon,
    pub schema: Schema,
    pub filter: Option<PhysicalExpr>,
    pub reader: Reader,
}

impl ScanNode {
    pub fn new(reader: impl Into<Reader>, schema: Schema) -> Result<Self, IrErrorKind> {
        schema.columns.iter().try_for_each(SchemaColumn::check)?;
        let reader = reader.into();
        if let Reader::VirtualTable(table) = &reader {
            table.check(&schema)?;
        }
        Ok(Self {
            common: NodeCommon::new(schema.columns.clone()),
            schema,
            filter: None,
            reader,
        })
    }

    pub fn with_filter(mut self, filter: PhysicalExpr) -> Result<Self, IrErrorKind> {
        condition(&self.schema, &filter)?;
        self.filter = Some(filter);
        Ok(self)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Reader {
    VirtualTable(VirtualTable),
    FilesScan(FilesScan),
    TableScan(TableScan),
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct VirtualTable {
    pub expressions: Vec<Vec<PhysicalExpr>>,
}

impl VirtualTable {
    pub fn new(expressions: Vec<Vec<PhysicalExpr>>) -> Self {
        Self { expressions }
    }

    fn check(&self, schema: &Schema) -> Result<(), IrErrorKind> {
        let no_fields = Schema {
            columns: Vec::new(),
        };
        let fields = schema.columns.len();
        (self.expressions.iter().enumerate()).try_for_each(|(row, values)| {
            if values.len() != fields {
                return Err(IrErrorKind::RowArity {
                    row,
                    fields,
                    got: values.len(),
                });
            }
            (values.iter().zip(&schema.columns)).try_for_each(|(value, field)| {
                match no_fields.type_of(value)? {
                    got if got.data_type != field.data_type => Err(IrErrorKind::RowType {
                        row,
                        field: field.name.clone(),
                        expected: field.data_type.clone(),
                        got: got.data_type,
                    }),
                    got if got.nullable && !field.nullable => Err(IrErrorKind::RowNull {
                        row,
                        field: field.name.clone(),
                    }),
                    _ => Ok(()),
                }
            })
        })
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct FilesScan {
    pub format: Option<FileFormat>,
    pub format_options: Option<Box<TextReadOptions>>,
    pub pattern: Option<String>,
    pub base_path: String,
    pub paths: Vec<String>,
}

impl FilesScan {
    pub fn new(
        format: Option<FileFormat>,
        pattern: Option<String>,
        base_path: impl Into<String>,
        paths: Vec<String>,
    ) -> Self {
        Self {
            format,
            format_options: None,
            pattern,
            base_path: base_path.into(),
            paths,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextReadOptions {
    pub field_delimiter: String,
    pub max_line_size: u64,
    pub quote: String,
    pub header_lines_to_skip: u64,
    pub escape: String,
    pub value_treated_as_null: Option<String>,
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableScan {
    #[non_exhaustive]
    NamedTable {
        names: [String; 3],
        format: Option<FileFormat>,
    },
}

impl TableScan {
    pub fn named_table(names: [String; 3], format: Option<FileFormat>) -> Self {
        Self::NamedTable { names, format }
    }
}

macro_rules! readers {
    ($($reader:ident),*) => {$(
        impl From<$reader> for Reader {
            fn from(reader: $reader) -> Self {
                Self::$reader(reader)
            }
        }
    )*};
}

readers!(VirtualTable, FilesScan, TableScan);
