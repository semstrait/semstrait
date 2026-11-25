use crate::{DriftKind, DriftReport, DriftStatus, Schema, TableRef};

pub fn diff_schema(table: TableRef, expected: &Schema, actual: &Schema) -> DriftReport {
    let mut details = Vec::new();
    let mut breaking = false;
    let mut changed = false;

    for ec in &expected.columns {
        match actual.columns.iter().find(|ac| ac.name == ec.name) {
            None => {
                details.push(DriftKind::ColumnRemoved {
                    name: ec.name.clone(),
                });
                breaking = true;
                changed = true;
            }
            Some(ac) => {
                if ac.data_type != ec.data_type {
                    details.push(DriftKind::ColumnRetyped {
                        name: ec.name.clone(),
                        expected: ec.data_type.clone(),
                        actual: ac.data_type.clone(),
                    });
                    breaking = true;
                    changed = true;
                }
                match (ec.nullable, ac.nullable) {
                    (true, false) => {
                        details.push(DriftKind::NullabilityTightened {
                            name: ec.name.clone(),
                        });
                        breaking = true;
                        changed = true;
                    }
                    (false, true) => {
                        details.push(DriftKind::NullabilityRelaxed {
                            name: ec.name.clone(),
                        });
                        changed = true;
                    }
                    _ => {}
                }
            }
        }
    }

    for ac in &actual.columns {
        if !expected.columns.iter().any(|ec| ec.name == ac.name) {
            details.push(DriftKind::ColumnAdded {
                name: ac.name.clone(),
            });
            changed = true;
        }
    }

    let status = if breaking {
        DriftStatus::Breaking
    } else if changed {
        DriftStatus::Compatible
    } else {
        DriftStatus::Unchanged
    };

    DriftReport::new(table, status, details)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NamespaceRef, SchemaColumn};

    fn table() -> TableRef {
        TableRef::new(NamespaceRef::new(["ns"]), "t")
    }

    fn schema(cols: &[(&str, &str, bool)]) -> Schema {
        Schema::new(
            cols.iter()
                .map(|(n, t, null)| SchemaColumn::new(*n, *t, *null))
                .collect(),
        )
    }

    #[test]
    fn identical_schemas_are_unchanged() {
        let s = schema(&[("id", "long", false), ("amount", "double", true)]);
        let report = diff_schema(table(), &s, &s);
        assert_eq!(report.status, DriftStatus::Unchanged);
        assert!(report.details.is_empty());
    }

    #[test]
    fn added_column_is_compatible() {
        let expected = schema(&[("id", "long", false)]);
        let actual = schema(&[("id", "long", false), ("note", "string", true)]);
        let report = diff_schema(table(), &expected, &actual);
        assert_eq!(report.status, DriftStatus::Compatible);
        assert_eq!(
            report.details,
            vec![DriftKind::ColumnAdded {
                name: "note".to_owned()
            }]
        );
    }

    #[test]
    fn removed_column_is_breaking() {
        let expected = schema(&[("id", "long", false), ("amount", "double", true)]);
        let actual = schema(&[("id", "long", false)]);
        let report = diff_schema(table(), &expected, &actual);
        assert_eq!(report.status, DriftStatus::Breaking);
        assert_eq!(
            report.details,
            vec![DriftKind::ColumnRemoved {
                name: "amount".to_owned()
            }]
        );
    }

    #[test]
    fn retype_is_breaking_by_native_string() {
        let expected = schema(&[("amount", "decimal(18,2)", true)]);
        let actual = schema(&[("amount", "double", true)]);
        let report = diff_schema(table(), &expected, &actual);
        assert_eq!(report.status, DriftStatus::Breaking);
        assert_eq!(
            report.details,
            vec![DriftKind::ColumnRetyped {
                name: "amount".to_owned(),
                expected: "decimal(18,2)".to_owned(),
                actual: "double".to_owned(),
            }]
        );
    }

    #[test]
    fn nullability_tightened_is_breaking_relaxed_is_compatible() {
        let report = diff_schema(
            table(),
            &schema(&[("c", "long", true)]),
            &schema(&[("c", "long", false)]),
        );
        assert_eq!(report.status, DriftStatus::Breaking);
        assert_eq!(
            report.details,
            vec![DriftKind::NullabilityTightened {
                name: "c".to_owned()
            }]
        );

        let report = diff_schema(
            table(),
            &schema(&[("c", "long", false)]),
            &schema(&[("c", "long", true)]),
        );
        assert_eq!(report.status, DriftStatus::Compatible);
        assert_eq!(
            report.details,
            vec![DriftKind::NullabilityRelaxed {
                name: "c".to_owned()
            }]
        );
    }

    #[test]
    fn column_order_does_not_affect_status() {
        let a = schema(&[("id", "long", false), ("amount", "double", true)]);
        let b = schema(&[("amount", "double", true), ("id", "long", false)]);
        assert_eq!(diff_schema(table(), &a, &b).status, DriftStatus::Unchanged);
    }
}
