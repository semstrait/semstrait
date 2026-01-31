//! Shared test utilities for integration tests

use semstrait::{
    emit_plan, parser,
    QueryRequest, Schema,
};
use substrait::proto::Plan;

/// Load a test fixture from the tests/test_data directory
pub fn load_fixture(name: &str) -> Schema {
    let path = format!("tests/test_data/{}", name);
    parser::parse_file(&path)
        .unwrap_or_else(|e| panic!("Failed to load test data {}: {}", name, e))
}

/// Load a test fixture from the test_data directory (existing fixtures)
pub fn load_test_data(name: &str) -> Schema {
    let path = format!("test_data/{}", name);
    parser::parse_file(&path)
        .unwrap_or_else(|e| panic!("Failed to load test data {}: {}", name, e))
}

/// Run the full pipeline: schema + request → Substrait Plan
pub fn run_pipeline(schema: &Schema, request: &QueryRequest) -> Result<Plan, String> {
    use semstrait::planner::plan_semantic_query;
    
    // Get model
    let model = schema
        .get_model(&request.model)
        .ok_or_else(|| format!("Model '{}' not found", request.model))?;

    // Use plan_semantic_query which handles cross-tableGroup cases
    let plan_node = plan_semantic_query(schema, model, request)
        .map_err(|e| format!("Planning failed: {}", e))?;

    // Emit Substrait
    let substrait = emit_plan(&plan_node, None).map_err(|e| format!("Emission failed: {}", e))?;

    Ok(substrait)
}

// =============================================================================
// Plan Inspection Utilities
// =============================================================================

use substrait::proto::rel::RelType;

/// Count the number of ReadRel (table scans) in the plan
pub fn count_scans(plan: &Plan) -> usize {
    let mut count = 0;
    for plan_rel in &plan.relations {
        if let Some(rel_type) = &plan_rel.rel_type {
            count += count_scans_in_rel_type(rel_type);
        }
    }
    count
}

fn count_scans_in_rel_type(rel_type: &substrait::proto::plan_rel::RelType) -> usize {
    use substrait::proto::plan_rel::RelType as PlanRelType;
    match rel_type {
        PlanRelType::Rel(rel) => count_scans_in_rel(rel),
        PlanRelType::Root(root) => root.input.as_ref().map(|r| count_scans_in_rel(r)).unwrap_or(0),
    }
}

fn count_scans_in_rel(rel: &substrait::proto::Rel) -> usize {
    let Some(rel_type) = &rel.rel_type else {
        return 0;
    };

    match rel_type {
        RelType::Read(_) => 1,
        RelType::Filter(f) => f.input.as_ref().map(|r| count_scans_in_rel(r)).unwrap_or(0),
        RelType::Project(p) => p.input.as_ref().map(|r| count_scans_in_rel(r)).unwrap_or(0),
        RelType::Aggregate(a) => a.input.as_ref().map(|r| count_scans_in_rel(r)).unwrap_or(0),
        RelType::Sort(s) => s.input.as_ref().map(|r| count_scans_in_rel(r)).unwrap_or(0),
        RelType::Join(j) => {
            let left = j.left.as_ref().map(|r| count_scans_in_rel(r)).unwrap_or(0);
            let right = j.right.as_ref().map(|r| count_scans_in_rel(r)).unwrap_or(0);
            left + right
        }
        RelType::Set(s) => s.inputs.iter().map(count_scans_in_rel).sum(),
        _ => 0,
    }
}

/// Check if the plan contains any JoinRel nodes
pub fn has_join(plan: &Plan) -> bool {
    for plan_rel in &plan.relations {
        if let Some(rel_type) = &plan_rel.rel_type {
            if has_join_in_rel_type(rel_type) {
                return true;
            }
        }
    }
    false
}

fn has_join_in_rel_type(rel_type: &substrait::proto::plan_rel::RelType) -> bool {
    use substrait::proto::plan_rel::RelType as PlanRelType;
    match rel_type {
        PlanRelType::Rel(rel) => has_join_in_rel(rel),
        PlanRelType::Root(root) => root.input.as_ref().map(|r| has_join_in_rel(r)).unwrap_or(false),
    }
}

fn has_join_in_rel(rel: &substrait::proto::Rel) -> bool {
    let Some(rel_type) = &rel.rel_type else {
        return false;
    };

    match rel_type {
        RelType::Join(_) => true,
        RelType::Filter(f) => f.input.as_ref().map(|r| has_join_in_rel(r)).unwrap_or(false),
        RelType::Project(p) => p.input.as_ref().map(|r| has_join_in_rel(r)).unwrap_or(false),
        RelType::Aggregate(a) => a.input.as_ref().map(|r| has_join_in_rel(r)).unwrap_or(false),
        RelType::Sort(s) => s.input.as_ref().map(|r| has_join_in_rel(r)).unwrap_or(false),
        RelType::Set(s) => s.inputs.iter().any(has_join_in_rel),
        _ => false,
    }
}

/// Check if the plan contains a SetRel (UNION)
pub fn has_union(plan: &Plan) -> bool {
    for plan_rel in &plan.relations {
        if let Some(rel_type) = &plan_rel.rel_type {
            if has_union_in_rel_type(rel_type) {
                return true;
            }
        }
    }
    false
}

fn has_union_in_rel_type(rel_type: &substrait::proto::plan_rel::RelType) -> bool {
    use substrait::proto::plan_rel::RelType as PlanRelType;
    match rel_type {
        PlanRelType::Rel(rel) => has_union_in_rel(rel),
        PlanRelType::Root(root) => root.input.as_ref().map(|r| has_union_in_rel(r)).unwrap_or(false),
    }
}

fn has_union_in_rel(rel: &substrait::proto::Rel) -> bool {
    let Some(rel_type) = &rel.rel_type else {
        return false;
    };

    match rel_type {
        RelType::Set(_) => true,
        RelType::Filter(f) => f.input.as_ref().map(|r| has_union_in_rel(r)).unwrap_or(false),
        RelType::Project(p) => p.input.as_ref().map(|r| has_union_in_rel(r)).unwrap_or(false),
        RelType::Aggregate(a) => a.input.as_ref().map(|r| has_union_in_rel(r)).unwrap_or(false),
        RelType::Sort(s) => s.input.as_ref().map(|r| has_union_in_rel(r)).unwrap_or(false),
        RelType::Join(j) => {
            j.left.as_ref().map(|r| has_union_in_rel(r)).unwrap_or(false)
                || j.right.as_ref().map(|r| has_union_in_rel(r)).unwrap_or(false)
        }
        _ => false,
    }
}

/// Count the number of JoinRel nodes in the plan
pub fn count_joins(plan: &Plan) -> usize {
    let mut count = 0;
    for plan_rel in &plan.relations {
        if let Some(rel_type) = &plan_rel.rel_type {
            count += count_joins_in_rel_type(rel_type);
        }
    }
    count
}

fn count_joins_in_rel_type(rel_type: &substrait::proto::plan_rel::RelType) -> usize {
    use substrait::proto::plan_rel::RelType as PlanRelType;
    match rel_type {
        PlanRelType::Rel(rel) => count_joins_in_rel(rel),
        PlanRelType::Root(root) => root.input.as_ref().map(|r| count_joins_in_rel(r)).unwrap_or(0),
    }
}

fn count_joins_in_rel(rel: &substrait::proto::Rel) -> usize {
    let Some(rel_type) = &rel.rel_type else {
        return 0;
    };

    match rel_type {
        RelType::Join(j) => {
            1 + j.left.as_ref().map(|r| count_joins_in_rel(r)).unwrap_or(0)
                + j.right.as_ref().map(|r| count_joins_in_rel(r)).unwrap_or(0)
        }
        RelType::Filter(f) => f.input.as_ref().map(|r| count_joins_in_rel(r)).unwrap_or(0),
        RelType::Project(p) => p.input.as_ref().map(|r| count_joins_in_rel(r)).unwrap_or(0),
        RelType::Aggregate(a) => a.input.as_ref().map(|r| count_joins_in_rel(r)).unwrap_or(0),
        RelType::Sort(s) => s.input.as_ref().map(|r| count_joins_in_rel(r)).unwrap_or(0),
        RelType::Set(s) => s.inputs.iter().map(count_joins_in_rel).sum(),
        _ => 0,
    }
}

// =============================================================================
// Debug Utilities
// =============================================================================

/// Print the plan structure for debugging
#[allow(dead_code)]
pub fn debug_plan_structure(plan: &Plan) {
    println!("Plan has {} relations", plan.relations.len());
    for (i, plan_rel) in plan.relations.iter().enumerate() {
        println!("  Relation {}: {:?}", i, plan_rel.rel_type.as_ref().map(|r| {
            use substrait::proto::plan_rel::RelType as PlanRelType;
            match r {
                PlanRelType::Rel(_) => "Rel",
                PlanRelType::Root(_) => "Root",
            }
        }));
    }
}
