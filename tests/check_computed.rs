mod test_helpers;
use semstrait_manifest::{CompileSource, ManifestCompiler};
use semstrait_planner::{ResolvedQueryRequest, SemanticPlanner};
use semstrait_adapter::sql::{AnsiDialect, AnsiSqlEmitter, SqlEmitter};
use std::collections::HashMap;
use test_helpers::load_model;

#[tokio::test]
async fn check_computed() {
    let yaml = load_model("orders_computed_dim");
    let m = ManifestCompiler::new().compile(CompileSource::Yaml(yaml)).await.unwrap();
    let req = ResolvedQueryRequest {
        entity_name: "orders_daily".to_string(),
        dimensions: vec!["date".to_string(), "market".to_string()],
        measures: vec!["revenue".to_string()],
        filters: vec![], inline_filters: vec![], pending_inline_filters: vec![], grain: None, limit: None, order_by: vec![],
        session_variables: HashMap::new(),
    };
    let plan = SemanticPlanner::builder().build().plan(&req, &m).unwrap();
    let sql = AnsiSqlEmitter::new(AnsiDialect).emit(&plan).unwrap();
    println!("\n=== orders_computed_dim: market (UPPER) ===\n{}", sql);
}
