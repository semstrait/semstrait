use semstrait::{Schema, SemanticModel, QueryRequest, plan::PlanNode};
use substrait::proto::Plan;
use super::ReproducibilityParams;

/// Print a comprehensive lineage report
pub fn print_lineage_report(
    _schema: &Schema,
    model: &SemanticModel,
    request: &QueryRequest,
    plan_node: &PlanNode,
    _substrait_plan: &Plan,
    repro_params: &ReproducibilityParams,
    snapshot_id: &str,
) {
    println!("\n📋 LINEAGE REPORT");
    println!("================");

    print_reproducibility_header(repro_params, snapshot_id);
    print_query_summary(request);
    print_metric_formulas(model, request);
    print_table_attribution(model, request);
    print_plan_tree(plan_node);
}

/// Print reproducibility header with snapshot ID and parameters
fn print_reproducibility_header(repro_params: &ReproducibilityParams, snapshot_id: &str) {
    println!("🔐 REPRODUCIBILITY");
    println!("  📸 Snapshot ID: {}", &snapshot_id[..16]); // Show first 16 chars
    println!("  📅 As-of: {}", repro_params.as_of);
    println!("  🌍 Timezone: {}", repro_params.timezone);
    println!("  💰 Currency: {} (FX rate: {:.4})", repro_params.currency, repro_params.fx_rate);
    println!("  📊 Attribution Window: {} days", repro_params.attribution_window);
    println!();
}

/// Print query summary
fn print_query_summary(request: &QueryRequest) {
    println!("\n🔍 Query Summary:");
    println!("  Model: {}", request.model);

    if let Some(ref rows) = request.rows {
        println!("  Dimensions: {}", rows.join(", "));
    }

    if let Some(ref cols) = request.columns {
        println!("  Columns: {}", cols.join(", "));
    }

    if let Some(ref metrics) = request.metrics {
        println!("  Metrics: {}", metrics.join(", "));
    }

    if let Some(ref filters) = request.filter {
        println!("  Filters: {} conditions", filters.len());
    }
}

/// Print metric formulas with their underlying expressions
fn print_metric_formulas(model: &SemanticModel, request: &QueryRequest) {
    if let Some(ref metric_names) = request.metrics {
        println!("\n📐 Metric Formulas:");

        for metric_name in metric_names {
            if let Some(metric) = model.get_metric(metric_name) {
                println!("  {} (type: {:?})", metric.name, metric.data_type.as_ref().unwrap_or(&semstrait::semantic_model::DataType::F64));

                // Simple display of the expression for now
                match &metric.expr {
                    semstrait::semantic_model::MetricExpr::MeasureRef(name) => {
                        println!("    Formula: {}", name);
                    }
                    semstrait::semantic_model::MetricExpr::Structured(node) => {
                        println!("    Formula: {:?}", node);
                    }
                }

                if let Some(label) = &metric.label {
                    println!("    Label: {}", label);
                }
                if let Some(desc) = &metric.description {
                    println!("    Description: {}", desc);
                }
                println!();
            }
        }
    }
}

/// Print table attribution and source information
fn print_table_attribution(model: &SemanticModel, _request: &QueryRequest) {
    println!("🏷️  Source Attribution:");

    // Show which tableGroups will be queried
    println!("  📊 Model: {} ({} table groups)", model.name, model.table_groups.len());

    for table_group in &model.table_groups {
        println!("  📁 TableGroup: {}", table_group.name);

        for table in &table_group.tables {
            println!("    📄 Table: {} ({} measures, {} dimensions)",
                table.table,
                table.measures.len(),
                table.dimensions.len()
            );

            // Show measure sources
            for measure_name in &table.measures {
                if let Some(measure) = table_group.measures.iter().find(|m| m.name == *measure_name) {
                    println!("      📊 {} → {:?}", measure.name, measure.expr);
                }
            }
        }
        println!();
    }
}

/// Print a pretty tree representation of the plan
fn print_plan_tree(plan_node: &PlanNode) {
    println!("🌳 Plan Tree:");

    fn print_node(node: &PlanNode, depth: usize) {
        let indent = "  ".repeat(depth);

        match node {
            PlanNode::Scan(scan) => {
                println!("{}📖 Read: {}", indent, scan.table);
                if !scan.columns.is_empty() {
                    println!("{}   Columns: {}", indent, scan.columns.join(", "));
                }
            }
            PlanNode::Project(project) => {
                println!("{}🔧 Project", indent);
                print_node(&project.input, depth + 1);
            }
            PlanNode::Aggregate(agg) => {
                println!("{}📊 Aggregate", indent);
                if !agg.group_by.is_empty() {
                    println!("{}   Group by: {} columns", indent, agg.group_by.len());
                }
                if !agg.aggregates.is_empty() {
                    println!("{}   Aggregates: {} measures", indent, agg.aggregates.len());
                }
                print_node(&agg.input, depth + 1);
            }
            PlanNode::Union(union) => {
                println!("{}🔗 Union", indent);
                for (i, input) in union.inputs.iter().enumerate() {
                    println!("{}   Branch {}:", indent, i + 1);
                    print_node(input, depth + 2);
                }
            }
            PlanNode::Filter(filter) => {
                println!("{}🔍 Filter", indent);
                print_node(&filter.input, depth + 1);
            }
            PlanNode::Sort(sort) => {
                println!("{}📋 Sort", indent);
                if !sort.sort_keys.is_empty() {
                    println!("{}   Sort keys: {} items", indent, sort.sort_keys.len());
                }
                print_node(&sort.input, depth + 1);
            }
            PlanNode::Join(join) => {
                println!("{}🔗 Join ({:?})", indent, join.join_type);
                println!("{}   Left:", indent);
                print_node(&join.left, depth + 2);
                println!("{}   Right:", indent);
                print_node(&join.right, depth + 2);
            }
            PlanNode::VirtualTable(vt) => {
                println!("{}💭 Virtual: {} rows", indent, vt.rows.len());
            }
        }
    }

    print_node(plan_node, 1);
}

/// Placeholder for reconciliation view - will implement when DataFusion is added
#[allow(dead_code)]
pub fn print_reconciliation(_batches: &[impl std::fmt::Debug]) {
    println!("\n🔍 RECONCILIATION VIEW");
    println!("=====================");
    println!("💡 (Reconciliation view not yet implemented - will show raw vs aggregated data)");
}