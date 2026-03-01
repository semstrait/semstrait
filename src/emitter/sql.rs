//! SQL emitter
//!
//! Transforms a PlanNode tree into an ANSI SQL string.

use crate::plan::{
    PlanNode, Scan, Join, JoinType, CrossJoin, Filter, Aggregate, Project, Sort,
    SortDirection, Union, VirtualTable, LiteralValue,
    Expr, Column, Literal, AggregateExpr,
};
use crate::semantic_model::Aggregation;
use super::error::EmitError;

/// Emit a pretty-printed SQL string from a PlanNode.
///
/// If `output_names` is provided, the outermost SELECT will alias columns to
/// those semantic names. Otherwise physical names are used.
pub fn emit_sql(node: &PlanNode, output_names: Option<Vec<String>>) -> Result<String, EmitError> {
    if let Some(names) = output_names {
        let inner = emit_node(node, 1)?;
        let aliases: Vec<String> = names
            .iter()
            .enumerate()
            .map(|(i, name)| format!("col{} AS \"{}\"", i, name))
            .collect();
        Ok(format!("SELECT {}\nFROM (\n{}\n)", aliases.join(", "), inner))
    } else {
        let inner = emit_node(node, 0)?;
        Ok(inner)
    }
}

fn pad(indent: usize) -> String {
    "  ".repeat(indent)
}

// ---------------------------------------------------------------------------
// Node dispatch
// ---------------------------------------------------------------------------

fn emit_node(node: &PlanNode, indent: usize) -> Result<String, EmitError> {
    match node {
        PlanNode::Scan(scan) => emit_scan(scan, indent),
        PlanNode::Join(join) => emit_join(join, indent),
        PlanNode::CrossJoin(cross) => emit_cross_join(cross, indent),
        PlanNode::Filter(filter) => emit_filter(filter, indent),
        PlanNode::Aggregate(agg) => emit_aggregate(agg, indent),
        PlanNode::Project(proj) => emit_project(proj, indent),
        PlanNode::Sort(sort) => emit_sort(sort, indent),
        PlanNode::Union(union) => emit_union(union, indent),
        PlanNode::VirtualTable(vt) => emit_virtual_table(vt, indent),
    }
}

// ---------------------------------------------------------------------------
// Relation nodes
// ---------------------------------------------------------------------------

fn emit_scan(scan: &Scan, indent: usize) -> Result<String, EmitError> {
    let p = pad(indent);
    let cols = if scan.columns.is_empty() {
        "*".to_string()
    } else {
        scan.columns.join(", ")
    };

    let alias = scan.alias.as_deref().unwrap_or(&scan.table);
    if alias != scan.table {
        Ok(format!("{p}SELECT {cols}\n{p}FROM {table} AS {alias}",
            table = scan.table))
    } else {
        Ok(format!("{p}SELECT {cols}\n{p}FROM {table}",
            table = scan.table))
    }
}

fn emit_join(join: &Join, indent: usize) -> Result<String, EmitError> {
    let p = pad(indent);
    let left = emit_node(&join.left, indent + 1)?;
    let right = emit_node(&join.right, indent + 1)?;

    let join_kw = match join.join_type {
        JoinType::Inner => "INNER JOIN",
        JoinType::Left => "LEFT JOIN",
        JoinType::Right => "RIGHT JOIN",
        JoinType::Full => "FULL OUTER JOIN",
    };

    Ok(format!(
        "{p}SELECT *\n{p}FROM (\n{left}\n{p}) AS _left\n{p}{join_kw} (\n{right}\n{p}) AS _right\n{p}  ON {lk} = {rk}",
        lk = emit_column(&join.left_key),
        rk = emit_column(&join.right_key),
    ))
}

fn emit_cross_join(cross: &CrossJoin, indent: usize) -> Result<String, EmitError> {
    let p = pad(indent);
    let left = emit_node(&cross.left, indent + 1)?;
    let right = emit_node(&cross.right, indent + 1)?;
    Ok(format!(
        "{p}SELECT *\n{p}FROM (\n{left}\n{p}) AS _left\n{p}CROSS JOIN (\n{right}\n{p}) AS _right"
    ))
}

fn emit_filter(filter: &Filter, indent: usize) -> Result<String, EmitError> {
    let p = pad(indent);
    let input = emit_node(&filter.input, indent + 1)?;
    let predicate = emit_expr(&filter.predicate)?;
    Ok(format!(
        "{p}SELECT *\n{p}FROM (\n{input}\n{p}) AS _f\n{p}WHERE {predicate}"
    ))
}

fn emit_aggregate(agg: &Aggregate, indent: usize) -> Result<String, EmitError> {
    let p = pad(indent);
    let input = emit_node(&agg.input, indent + 1)?;

    let group_cols: Vec<String> = agg.group_by.iter().map(|c| emit_column(c)).collect();

    let agg_exprs: Vec<String> = agg.aggregates
        .iter()
        .map(|a| emit_aggregate_expr(a))
        .collect::<Result<Vec<_>, _>>()?;

    let mut select_items: Vec<String> = group_cols.clone();
    select_items.extend(agg_exprs);

    if group_cols.is_empty() {
        Ok(format!(
            "{p}SELECT {sel}\n{p}FROM (\n{input}\n{p})",
            sel = select_items.join(", "),
        ))
    } else {
        Ok(format!(
            "{p}SELECT {sel}\n{p}FROM (\n{input}\n{p})\n{p}GROUP BY {grp}",
            sel = select_items.join(", "),
            grp = group_cols.join(", "),
        ))
    }
}

fn emit_project(proj: &Project, indent: usize) -> Result<String, EmitError> {
    let p = pad(indent);
    let input = emit_node(&proj.input, indent + 1)?;

    let items: Vec<String> = proj.expressions
        .iter()
        .map(|pe| {
            let expr_sql = emit_expr(&pe.expr)?;
            Ok(format!("{} AS \"{}\"", expr_sql, pe.alias))
        })
        .collect::<Result<Vec<_>, EmitError>>()?;

    Ok(format!(
        "{p}SELECT {sel}\n{p}FROM (\n{input}\n{p})",
        sel = items.join(", "),
    ))
}

fn emit_sort(sort: &Sort, indent: usize) -> Result<String, EmitError> {
    let p = pad(indent);
    let input = emit_node(&sort.input, indent + 1)?;

    let keys: Vec<String> = sort.sort_keys
        .iter()
        .map(|k| {
            let dir = match k.direction {
                SortDirection::Ascending => "ASC",
                SortDirection::Descending => "DESC",
            };
            format!("{} {}", k.column, dir)
        })
        .collect();

    Ok(format!(
        "{p}SELECT *\n{p}FROM (\n{input}\n{p})\n{p}ORDER BY {keys}",
        keys = keys.join(", "),
    ))
}

fn emit_union(union: &Union, indent: usize) -> Result<String, EmitError> {
    if union.inputs.len() < 2 {
        return Err(EmitError::InvalidPlan(
            "Union requires at least 2 inputs".to_string(),
        ));
    }

    let parts: Vec<String> = union.inputs
        .iter()
        .map(|n| emit_node(n, indent))
        .collect::<Result<Vec<_>, _>>()?;

    let p = pad(indent);
    Ok(parts.join(&format!("\n{p}UNION ALL\n")))
}

fn emit_virtual_table(vt: &VirtualTable, indent: usize) -> Result<String, EmitError> {
    let p = pad(indent);

    if vt.rows.is_empty() {
        let nulls: Vec<String> = vt.columns
            .iter()
            .map(|c| format!("NULL AS \"{}\"", c))
            .collect();
        return Ok(format!("{p}SELECT {} WHERE 1=0", nulls.join(", ")));
    }

    let row_selects: Vec<String> = vt.rows
        .iter()
        .map(|row| {
            let items: Vec<String> = row
                .iter()
                .zip(vt.columns.iter())
                .map(|(val, col)| format!("{} AS \"{}\"", literal_value_to_sql(val), col))
                .collect();
            format!("{p}SELECT {}", items.join(", "))
        })
        .collect();

    Ok(row_selects.join(&format!("\n{p}UNION ALL\n")))
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

fn emit_expr(expr: &Expr) -> Result<String, EmitError> {
    match expr {
        Expr::Column(col) => Ok(emit_column(col)),
        Expr::Literal(lit) => Ok(emit_literal(lit)),
        Expr::BinaryOp { left, op, right } => {
            let l = emit_expr(left)?;
            let r = emit_expr(right)?;
            Ok(format!("{} {} {}", l, op.as_str(), r))
        }
        Expr::And(exprs) => {
            if exprs.is_empty() {
                return Ok("TRUE".to_string());
            }
            let parts: Vec<String> = exprs.iter().map(|e| emit_expr(e)).collect::<Result<Vec<_>, _>>()?;
            Ok(format!("({})", parts.join(" AND ")))
        }
        Expr::Or(exprs) => {
            if exprs.is_empty() {
                return Ok("FALSE".to_string());
            }
            let parts: Vec<String> = exprs.iter().map(|e| emit_expr(e)).collect::<Result<Vec<_>, _>>()?;
            Ok(format!("({})", parts.join(" OR ")))
        }
        Expr::In { expr, values } => {
            let needle = emit_expr(expr)?;
            let vals: Vec<String> = values.iter().map(|v| emit_expr(v)).collect::<Result<Vec<_>, _>>()?;
            Ok(format!("{} IN ({})", needle, vals.join(", ")))
        }
        Expr::Sql(s) => Ok(s.clone()),
        Expr::Add(a, b) => Ok(format!("({} + {})", emit_expr(a)?, emit_expr(b)?)),
        Expr::Subtract(a, b) => Ok(format!("({} - {})", emit_expr(a)?, emit_expr(b)?)),
        Expr::Multiply(a, b) => Ok(format!("({} * {})", emit_expr(a)?, emit_expr(b)?)),
        Expr::Divide(a, b) => Ok(format!(
            "(CAST({} AS DOUBLE) / CAST({} AS DOUBLE))",
            emit_expr(a)?,
            emit_expr(b)?,
        )),
        Expr::IsNull(inner) => Ok(format!("{} IS NULL", emit_expr(inner)?)),
        Expr::IsNotNull(inner) => Ok(format!("{} IS NOT NULL", emit_expr(inner)?)),
        Expr::Case { when_then, else_result } => {
            let mut sql = String::from("CASE");
            for (cond, then) in when_then {
                sql.push_str(&format!(" WHEN {} THEN {}", emit_expr(cond)?, emit_expr(then)?));
            }
            if let Some(el) = else_result {
                sql.push_str(&format!(" ELSE {}", emit_expr(el)?));
            }
            sql.push_str(" END");
            Ok(sql)
        }
        Expr::Coalesce(exprs) => {
            let parts: Vec<String> = exprs.iter().map(|e| emit_expr(e)).collect::<Result<Vec<_>, _>>()?;
            Ok(format!("COALESCE({})", parts.join(", ")))
        }
    }
}

fn emit_column(col: &Column) -> String {
    if col.table.is_empty() {
        col.name.clone()
    } else {
        format!("{}.{}", col.table, col.name)
    }
}

fn emit_literal(lit: &Literal) -> String {
    match lit {
        Literal::Null(type_name) => format!("CAST(NULL AS {})", sql_type_name(type_name)),
        Literal::Bool(b) => if *b { "TRUE".to_string() } else { "FALSE".to_string() },
        Literal::Int(i) => i.to_string(),
        Literal::Float(f) => format!("{}", f),
        Literal::String(s) => format!("'{}'", s.replace('\'', "''")),
    }
}

fn literal_value_to_sql(val: &LiteralValue) -> String {
    match val {
        LiteralValue::String(s) => format!("'{}'", s.replace('\'', "''")),
        LiteralValue::Int32(i) => i.to_string(),
        LiteralValue::Int64(i) => i.to_string(),
        LiteralValue::Float64(f) => format!("{}", f),
        LiteralValue::Bool(b) => if *b { "TRUE".to_string() } else { "FALSE".to_string() },
        LiteralValue::Null => "NULL".to_string(),
    }
}

fn sql_type_name(internal: &str) -> &str {
    match internal.to_lowercase().as_str() {
        "i8" | "i16" | "i32" | "int" | "integer" => "INTEGER",
        "i64" | "long" | "bigint" => "BIGINT",
        "f32" | "float" => "FLOAT",
        "f64" | "double" => "DOUBLE",
        "bool" | "boolean" => "BOOLEAN",
        "date" => "DATE",
        "timestamp" | "datetime" => "TIMESTAMP",
        "string" | "text" | "varchar" => "VARCHAR",
        _ => "VARCHAR",
    }
}

// ---------------------------------------------------------------------------
// Aggregates
// ---------------------------------------------------------------------------

fn emit_aggregate_expr(agg: &AggregateExpr) -> Result<String, EmitError> {
    let inner = emit_expr(&agg.expr)?;
    let func_sql = match agg.func {
        Aggregation::Sum => format!("SUM({})", inner),
        Aggregation::Avg => format!("AVG({})", inner),
        Aggregation::Count => format!("COUNT({})", inner),
        Aggregation::CountDistinct => format!("COUNT(DISTINCT {})", inner),
        Aggregation::Min => format!("MIN({})", inner),
        Aggregation::Max => format!("MAX({})", inner),
    };
    Ok(format!("{} AS \"{}\"", func_sql, agg.alias))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{BinaryOperator, ProjectExpr, SortKey};
    use crate::semantic_model::Schema;
    use crate::query::QueryRequest;
    use crate::selector::select_datasets;
    use crate::resolver::resolve_query;
    use crate::planner::plan_query;

    fn load_test_schema() -> Schema {
        Schema::from_file("test_data/steelwheels.yaml").unwrap()
    }

    // -- unit: scan -----------------------------------------------------------

    #[test]
    fn test_sql_scan() {
        let scan = PlanNode::Scan(
            Scan::new("public.orders")
                .with_columns(
                    vec!["id".into(), "amount".into()],
                    vec!["i32".into(), "f64".into()],
                ),
        );
        let sql = emit_sql(&scan, None).unwrap();
        assert_eq!(sql, "SELECT id, amount\nFROM public.orders");
    }

    #[test]
    fn test_sql_scan_with_alias() {
        let scan = PlanNode::Scan(
            Scan::new("public.orders")
                .with_alias("o")
                .with_columns(
                    vec!["id".into(), "amount".into()],
                    vec!["i32".into(), "f64".into()],
                ),
        );
        let sql = emit_sql(&scan, None).unwrap();
        assert_eq!(sql, "SELECT id, amount\nFROM public.orders AS o");
    }

    // -- unit: filter ---------------------------------------------------------

    #[test]
    fn test_sql_filter() {
        let scan = Scan::new("t")
            .with_alias("t")
            .with_columns(vec!["id".into(), "name".into()], vec!["i32".into(), "string".into()]);

        let filter = PlanNode::Filter(Filter {
            input: Box::new(PlanNode::Scan(scan)),
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Column(Column::new("t", "id"))),
                op: BinaryOperator::Eq,
                right: Box::new(Expr::Literal(Literal::Int(42))),
            },
        });
        let sql = emit_sql(&filter, None).unwrap();
        assert!(sql.contains("WHERE t.id = 42"));
        assert!(sql.contains("\n"));
    }

    // -- unit: aggregate ------------------------------------------------------

    #[test]
    fn test_sql_aggregate() {
        let scan = Scan::new("fact")
            .with_alias("fact")
            .with_columns(
                vec!["year".into(), "amount".into()],
                vec!["i32".into(), "f64".into()],
            );

        let agg = PlanNode::Aggregate(Aggregate {
            input: Box::new(PlanNode::Scan(scan)),
            group_by: vec![Column::new("fact", "year")],
            aggregates: vec![AggregateExpr {
                func: Aggregation::Sum,
                expr: Expr::Column(Column::new("fact", "amount")),
                alias: "total".into(),
            }],
        });
        let sql = emit_sql(&agg, None).unwrap();
        assert!(sql.contains("SUM(fact.amount) AS \"total\""));
        assert!(sql.contains("GROUP BY fact.year"));
        assert!(sql.contains("\n"));
    }

    // -- unit: project --------------------------------------------------------

    #[test]
    fn test_sql_project() {
        let scan = Scan::new("t")
            .with_alias("t")
            .with_columns(vec!["a".into(), "b".into()], vec!["i32".into(), "i32".into()]);

        let proj = PlanNode::Project(Project {
            input: Box::new(PlanNode::Scan(scan)),
            expressions: vec![
                ProjectExpr {
                    expr: Expr::Column(Column::new("t", "a")),
                    alias: "col_a".into(),
                },
                ProjectExpr {
                    expr: Expr::Literal(Literal::String("hello".into())),
                    alias: "greeting".into(),
                },
            ],
        });
        let sql = emit_sql(&proj, None).unwrap();
        assert!(sql.contains("t.a AS \"col_a\""));
        assert!(sql.contains("'hello' AS \"greeting\""));
    }

    // -- unit: sort -----------------------------------------------------------

    #[test]
    fn test_sql_sort() {
        let scan = Scan::new("t")
            .with_columns(vec!["year".into(), "amount".into()], vec!["i32".into(), "f64".into()]);

        let sort = PlanNode::Sort(Sort {
            input: Box::new(PlanNode::Scan(scan)),
            sort_keys: vec![SortKey {
                column: "year".into(),
                direction: SortDirection::Descending,
            }],
        });
        let sql = emit_sql(&sort, None).unwrap();
        assert!(sql.contains("ORDER BY year DESC"));
    }

    // -- unit: union ----------------------------------------------------------

    #[test]
    fn test_sql_union() {
        let s1 = PlanNode::Scan(
            Scan::new("a").with_columns(vec!["x".into()], vec!["i32".into()]),
        );
        let s2 = PlanNode::Scan(
            Scan::new("b").with_columns(vec!["x".into()], vec!["i32".into()]),
        );

        let u = PlanNode::Union(Union { inputs: vec![s1, s2] });
        let sql = emit_sql(&u, None).unwrap();
        assert!(sql.contains("UNION ALL"));
    }

    #[test]
    fn test_sql_union_requires_two() {
        let s1 = PlanNode::Scan(Scan::new("a").with_columns(vec!["x".into()], vec!["i32".into()]));
        let u = PlanNode::Union(Union { inputs: vec![s1] });
        assert!(emit_sql(&u, None).is_err());
    }

    // -- unit: virtual table --------------------------------------------------

    #[test]
    fn test_sql_virtual_table() {
        let vt = PlanNode::VirtualTable(VirtualTable {
            columns: vec!["name".into(), "value".into()],
            column_types: vec!["string".into(), "i32".into()],
            rows: vec![
                vec![LiteralValue::String("a".into()), LiteralValue::Int32(1)],
                vec![LiteralValue::String("b".into()), LiteralValue::Int32(2)],
            ],
        });
        let sql = emit_sql(&vt, None).unwrap();
        assert!(sql.contains("'a' AS \"name\""));
        assert!(sql.contains("UNION ALL"));
    }

    // -- unit: expressions ----------------------------------------------------

    #[test]
    fn test_sql_case_expr() {
        let expr = Expr::Case {
            when_then: vec![
                (
                    Expr::BinaryOp {
                        left: Box::new(Expr::Column(Column::new("t", "x"))),
                        op: BinaryOperator::Gt,
                        right: Box::new(Expr::Literal(Literal::Int(0))),
                    },
                    Expr::Literal(Literal::String("positive".into())),
                ),
            ],
            else_result: Some(Box::new(Expr::Literal(Literal::String("non-positive".into())))),
        };
        let sql = emit_expr(&expr).unwrap();
        assert!(sql.starts_with("CASE WHEN"));
        assert!(sql.contains("ELSE 'non-positive'"));
        assert!(sql.ends_with("END"));
    }

    #[test]
    fn test_sql_coalesce_expr() {
        let expr = Expr::Coalesce(vec![
            Expr::Column(Column::new("t", "a")),
            Expr::Literal(Literal::Int(0)),
        ]);
        let sql = emit_expr(&expr).unwrap();
        assert_eq!(sql, "COALESCE(t.a, 0)");
    }

    #[test]
    fn test_sql_in_expr() {
        let expr = Expr::In {
            expr: Box::new(Expr::Column(Column::new("t", "status"))),
            values: vec![
                Expr::Literal(Literal::String("active".into())),
                Expr::Literal(Literal::String("pending".into())),
            ],
        };
        let sql = emit_expr(&expr).unwrap();
        assert_eq!(sql, "t.status IN ('active', 'pending')");
    }

    #[test]
    fn test_sql_divide_casts_to_double() {
        let expr = Expr::Divide(
            Box::new(Expr::Column(Column::new("t", "a"))),
            Box::new(Expr::Column(Column::new("t", "b"))),
        );
        let sql = emit_expr(&expr).unwrap();
        assert!(sql.contains("CAST(t.a AS DOUBLE)"));
        assert!(sql.contains("CAST(t.b AS DOUBLE)"));
    }

    // -- integration: full pipeline -------------------------------------------

    #[test]
    fn test_sql_end_to_end() {
        let schema = load_test_schema();
        let model = schema.get_model("steelwheels").unwrap();
        let request = QueryRequest {
            model: "steelwheels".into(),
            dimensions: None,
            rows: Some(vec!["dates.year".into()]),
            columns: None,
            metrics: Some(vec!["sales".into()]),
            filter: None,
        };

        let selected = select_datasets(&schema, model, &["dates.year".into()], &["sales".into()])
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let resolved = resolve_query(&schema, &request, &selected).unwrap();
        let plan_node = plan_query(&resolved).unwrap();

        let sql = emit_sql(&plan_node, None).unwrap();
        assert!(sql.contains("SELECT"), "SQL should contain SELECT: {}", sql);
        assert!(sql.contains("FROM"), "SQL should contain FROM: {}", sql);
        assert!(sql.contains("GROUP BY"), "SQL should contain GROUP BY: {}", sql);
        assert!(sql.contains("SUM"), "SQL should contain SUM: {}", sql);
        assert!(sql.contains("\n"), "SQL should be multi-line: {}", sql);
    }

    #[test]
    fn test_sql_with_output_names() {
        let scan = PlanNode::Scan(
            Scan::new("t")
                .with_columns(vec!["a".into(), "b".into()], vec!["i32".into(), "i32".into()]),
        );
        let sql = emit_sql(&scan, Some(vec!["Alpha".into(), "Beta".into()])).unwrap();
        assert!(sql.contains("col0 AS \"Alpha\""));
        assert!(sql.contains("col1 AS \"Beta\""));
        assert!(sql.contains("  SELECT a, b"), "Inner SELECT should be indented:\n{}", sql);
        assert!(sql.contains("  FROM t"), "Inner FROM should be indented:\n{}", sql);
    }

    #[test]
    fn test_sql_indentation() {
        let scan = Scan::new("fact")
            .with_alias("fact")
            .with_columns(
                vec!["year".into(), "amount".into()],
                vec!["i32".into(), "f64".into()],
            );

        let agg = PlanNode::Aggregate(Aggregate {
            input: Box::new(PlanNode::Scan(scan)),
            group_by: vec![Column::new("fact", "year")],
            aggregates: vec![AggregateExpr {
                func: Aggregation::Sum,
                expr: Expr::Column(Column::new("fact", "amount")),
                alias: "total".into(),
            }],
        });
        let sql = emit_sql(&agg, None).unwrap();
        // Inner scan should be indented
        assert!(sql.contains("  SELECT year, amount"), "Inner SELECT should be indented:\n{}", sql);
        assert!(sql.contains("  FROM fact"), "Inner FROM should be indented:\n{}", sql);
    }
}
