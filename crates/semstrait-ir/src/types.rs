use semstrait_common::Tree;
use semstrait_sql::datatype::{CanonicalTypeSystem, DataType};
use semstrait_sql::expr::{type_tree, Expr, InputTypes, NamedReference, Typed};
use semstrait_sql::DocumentDialect;
use serde::{Deserialize, Serialize};

use crate::error::IrErrorKind;
use crate::expr::{Parameter, PhysicalExpr};
use crate::primitives::Name;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Schema {
    pub columns: Vec<SchemaColumn>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchemaColumn {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
}

impl Schema {
    pub fn position(&self, leaf: &NamedReference) -> Option<usize> {
        let joined = field_name(leaf);
        let binds = |field: &SchemaColumn| match leaf.parts() {
            [alias, .., name] if is_physical_alias(alias) => field.name == *name,
            _ => same_name(&field.name, &joined),
        };
        let mut bound = (self.columns.iter().enumerate()).filter(|(_, field)| binds(field));
        match (bound.next(), bound.next()) {
            (Some((position, _)), None) => Some(position),
            _ => None,
        }
    }

    pub fn type_of(&self, expr: &PhysicalExpr) -> Result<Typed, IrErrorKind> {
        if let Some(leaf) = self.unbound(expr) {
            return Err(IrErrorKind::UnboundLeaf {
                leaf: leaf.clone(),
                fields: self
                    .columns
                    .iter()
                    .map(|field| field.name.clone())
                    .collect(),
            });
        }
        type_tree(
            expr.clone(),
            self,
            DocumentDialect::Canonical,
            &CanonicalTypeSystem,
            None,
        )
        .map(|(_, typed)| typed)
        .map_err(|error| IrErrorKind::TypeRefused { error })
    }

    fn unbound<'e>(&self, expr: &'e PhysicalExpr) -> Option<&'e NamedReference> {
        match expr {
            Expr::Column(leaf) => self.position(leaf).is_none().then_some(leaf),
            _ => (expr.children().into_iter()).find_map(|operand| self.unbound(operand)),
        }
    }
}

impl SchemaColumn {
    pub(crate) fn check(&self) -> Result<(), IrErrorKind> {
        if self.data_type.is_valid() {
            Ok(())
        } else {
            Err(IrErrorKind::InvalidType {
                field: self.name.clone(),
                data_type: self.data_type.clone(),
            })
        }
    }
}

impl InputTypes<Parameter> for Schema {
    fn column(&self, column: &NamedReference) -> Typed {
        (self
            .position(column)
            .and_then(|position| self.columns.get(position)))
        .map_or_else(
            || Typed::new(DataType::Unknown, true),
            |field| Typed::new(field.data_type.clone(), field.nullable),
        )
    }

    fn leaf(&self, leaf: &Parameter) -> Typed {
        Typed::new(leaf.data_type.clone(), true)
    }
}

pub fn field_name(leaf: &NamedReference) -> String {
    leaf.parts().join(".")
}

pub(crate) fn same_name(left: &str, right: &str) -> bool {
    Name::folded(left).eq(Name::folded(right))
}

fn is_physical_alias(part: &str) -> bool {
    part.strip_prefix("ps")
        .is_some_and(|index| !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(name: &str, ty: DataType, nullable: bool) -> SchemaColumn {
        SchemaColumn {
            name: name.to_string(),
            data_type: ty,
            nullable,
        }
    }

    #[test]
    fn schema_column_construction_and_equality() {
        let a = col(
            "amount",
            DataType::Decimal {
                precision: 10,
                scale: 2,
            },
            false,
        );
        let b = col(
            "amount",
            DataType::Decimal {
                precision: 10,
                scale: 2,
            },
            false,
        );
        assert_eq!(a, b);

        let c = col(
            "amount",
            DataType::Decimal {
                precision: 10,
                scale: 2,
            },
            true,
        );
        assert_ne!(a, c, "nullable participates in equality");

        let d = col(
            "Amount",
            DataType::Decimal {
                precision: 10,
                scale: 2,
            },
            false,
        );
        assert_ne!(a, d, "name is case-sensitive (no normalization)");
    }

    #[test]
    fn schema_field_access_is_contract() {
        let s = Schema {
            columns: vec![
                col("id", DataType::BigInt, false),
                col("name", DataType::String, true),
            ],
        };
        assert_eq!(s.columns.len(), 2);
        assert_eq!(s.columns[0].name, "id");
        assert_eq!(s.columns[1].data_type, DataType::String);
        assert!(s.columns[1].nullable);
    }

    #[test]
    fn schema_empty_columns_is_admissible() {
        let s = Schema {
            columns: Vec::new(),
        };
        assert_eq!(s.columns.len(), 0);
    }

    #[test]
    fn schema_preserves_source_order() {
        let s = Schema {
            columns: vec![
                col("c0", DataType::Boolean, false),
                col("c1", DataType::Integer, false),
                col("c2", DataType::String, true),
            ],
        };
        let names: Vec<&str> = s.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["c0", "c1", "c2"]);
    }

    #[test]
    fn schema_admits_duplicate_column_names() {
        let s = Schema {
            columns: vec![
                col("x", DataType::Integer, false),
                col("x", DataType::Integer, false),
            ],
        };
        assert_eq!(s.columns.len(), 2);
        assert_eq!(s.columns[0], s.columns[1]);
    }

    #[test]
    fn schema_equality_is_structural_and_order_sensitive() {
        let a = Schema {
            columns: vec![
                col("a", DataType::Integer, false),
                col("b", DataType::String, true),
            ],
        };
        let b = a.clone();
        assert_eq!(a, b);

        let reordered = Schema {
            columns: vec![
                col("b", DataType::String, true),
                col("a", DataType::Integer, false),
            ],
        };
        assert_ne!(a, reordered, "column order participates in equality");
    }

    #[test]
    fn schema_hash_is_deterministic() {
        use std::collections::HashSet;
        let a = Schema {
            columns: vec![col("x", DataType::Integer, false)],
        };
        let b = a.clone();
        let mut set: HashSet<Schema> = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }

    #[test]
    fn schema_serde_json_roundtrip() {
        let s = Schema {
            columns: vec![
                col("id", DataType::BigInt, false),
                col("ts", DataType::Timestamp { precision: 6 }, true),
                col(
                    "price",
                    DataType::Decimal {
                        precision: 12,
                        scale: 4,
                    },
                    false,
                ),
            ],
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Schema = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn schema_column_serde_json_roundtrip_each_data_type() {
        let roster = [
            DataType::Boolean,
            DataType::TinyInt,
            DataType::SmallInt,
            DataType::Integer,
            DataType::BigInt,
            DataType::Real,
            DataType::Double,
            DataType::Decimal {
                precision: 10,
                scale: 2,
            },
            DataType::String,
            DataType::Binary,
            DataType::Date,
            DataType::Time { precision: 6 },
            DataType::Timestamp { precision: 6 },
            DataType::Interval,
        ];
        for dt in roster {
            let c = col("c", dt.clone(), false);
            let json = serde_json::to_string(&c).unwrap();
            let back: SchemaColumn = serde_json::from_str(&json).unwrap();
            assert_eq!(c, back, "roundtrip failed for column with type {dt}");
        }
    }
}
