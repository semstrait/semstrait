//! Integration tests for the full planning pipeline.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indexmap::IndexMap;
    use semstrait_ir::{Expr, LogicalPlan, PlanNode};

    use crate::error::PlannerError;
    use crate::optimizer::OptimizerPass;
    use crate::planner::SemanticPlanner;
    use crate::request::{
        FilterOperator, FilterValue, OrderByClause, QueryFilter, ResolvedQueryRequest,
        SortDirection,
    };
    use crate::tests::helpers::*;

    // ========================================================================
    // Simple grainset planning
    // ========================================================================

    #[test]
    fn test_simple_grainset_plan() {
        let planner = SemanticPlanner::builder().build();
        let manifest = make_test_manifest();
        let request = make_test_request("orders", vec!["date"], vec!["revenue"]);

        let plan = planner.plan(&request, &manifest);
        assert!(plan.is_ok(), "planning should succeed: {:?}", plan.err());

        let plan = plan.unwrap();
        assert_eq!(
            plan.output_names,
            vec!["date".to_string(), "revenue".to_string()]
        );

        // Verify plan structure: Aggregate (L5 skipped) or Project -> Aggregate -> Project (L2) -> Scan
        let agg = match &plan.root {
            PlanNode::Aggregate(a) => a, // identity L5 skipped
            PlanNode::Project(proj) => match proj.input.as_ref() {
                PlanNode::Aggregate(a) => a,
                _ => panic!("Project input should be Aggregate"),
            },
            _ => panic!(
                "root should be Aggregate or Project, got {:?}",
                std::mem::discriminant(&plan.root)
            ),
        };

        // After layered refactor: Aggregate → Project (L2 rename) → Scan.
        assert!(
            matches!(agg.input.as_ref(), PlanNode::Project(_)),
            "Aggregate input should be Project (L2 rename), got {:?}",
            std::mem::discriminant(agg.input.as_ref())
        );

        // Verify GROUP BY has the date dimension.
        assert_eq!(agg.group_by.len(), 1);

        // Verify there is one aggregate measure.
        assert_eq!(agg.aggregates.len(), 1);

        if let PlanNode::Project(rename) = agg.input.as_ref() {
            if let PlanNode::Scan(scan) = rename.input.as_ref() {
                assert_eq!(scan.table_name, "orders_daily");
            } else {
                panic!("L2 rename input should be Scan");
            }
        }
    }

    #[test]
    fn test_grainset_plan_multiple_dims() {
        let planner = SemanticPlanner::builder().build();
        let manifest = make_test_manifest();
        let request = make_test_request(
            "orders",
            vec!["date", "region"],
            vec!["revenue"],
        );

        let plan = planner.plan(&request, &manifest).unwrap();
        assert_eq!(
            plan.output_names,
            vec!["date".to_string(), "region".to_string(), "revenue".to_string()]
        );
    }

    #[test]
    fn test_grainset_plan_kind_not_found() {
        let planner = SemanticPlanner::builder().build();
        let manifest = make_test_manifest();
        let request = make_test_request("nonexistent", vec!["date"], vec!["revenue"]);

        let result = planner.plan(&request, &manifest);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PlannerError::KindNotFound(_)
        ));
    }

    // ========================================================================
    // Constraint evaluation (integration)
    // ========================================================================

    #[test]
    fn test_constraint_violation_blocks_planning() {
        let planner = SemanticPlanner::builder().build();
        let manifest = make_test_manifest_with_constraints(
            Some(semstrait_manifest::DimensionConstraints {
                one_of: Some(vec!["date".to_string()]),
                none_of: None,
                all: None,
            }),
            None,
        );

        // Request without date — should fail constraint check.
        let request = make_test_request("orders", vec!["region"], vec!["revenue"]);
        let result = planner.plan(&request, &manifest);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PlannerError::ConstraintViolation { .. }
        ));
    }

    #[test]
    fn test_constraint_satisfied_allows_planning() {
        let planner = SemanticPlanner::builder().build();
        let manifest = make_test_manifest_with_constraints(
            Some(semstrait_manifest::DimensionConstraints {
                one_of: Some(vec!["date".to_string()]),
                none_of: None,
                all: None,
            }),
            None,
        );

        // Request with date — should pass.
        let request = make_test_request("orders", vec!["date"], vec!["revenue"]);
        let result = planner.plan(&request, &manifest);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Filter injection
    // ========================================================================

    #[test]
    fn test_user_filter_injection() {
        let planner = SemanticPlanner::builder().build();
        let manifest = make_test_manifest();

        let request = ResolvedQueryRequest {
            entity_name: "orders".to_string(),
            dimensions: vec!["date".to_string()],
            measures: vec!["revenue".to_string()],
            filters: vec![QueryFilter {
                field: "region".to_string(),
                operator: FilterOperator::Eq,
                values: vec![FilterValue::String("US".to_string())],
            }],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: HashMap::new(),
        };

        let plan = planner.plan(&request, &manifest).unwrap();

        // Root should be a Filter node wrapping the Project.
        assert!(
            matches!(&plan.root, PlanNode::Filter(_)),
            "root should be Filter when user filters are present, got {:?}",
            std::mem::discriminant(&plan.root)
        );

        if let PlanNode::Filter(filter) = &plan.root {
            // Check the predicate is region = 'US'.
            match &filter.predicate {
                Expr::BinaryOp(bin) => {
                    assert_eq!(
                        bin.op,
                        semstrait_ir::BinaryOp::Eq,
                        "should be equality filter"
                    );
                    assert!(
                        matches!(bin.left.as_ref(), Expr::Column(col) if col.name == "region"),
                        "left should be column 'region'"
                    );
                    assert!(
                        matches!(bin.right.as_ref(), Expr::Literal(semstrait_core::Literal::String { value }) if value == "US"),
                        "right should be string 'US'"
                    );
                }
                other => panic!("expected BinaryOp, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_multiple_user_filters() {
        let planner = SemanticPlanner::builder().build();
        let manifest = make_test_manifest();

        let request = ResolvedQueryRequest {
            entity_name: "orders".to_string(),
            dimensions: vec!["date".to_string()],
            measures: vec!["revenue".to_string()],
            filters: vec![
                QueryFilter {
                    field: "region".to_string(),
                    operator: FilterOperator::Eq,
                    values: vec![FilterValue::String("US".to_string())],
                },
                QueryFilter {
                    field: "revenue".to_string(),
                    operator: FilterOperator::Gt,
                    values: vec![FilterValue::Number(1000.0)],
                },
            ],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: HashMap::new(),
        };

        let plan = planner.plan(&request, &manifest).unwrap();

        // Should have Filter -> Filter -> Project -> ...
        assert!(matches!(&plan.root, PlanNode::Filter(_)));
        if let PlanNode::Filter(f1) = &plan.root {
            assert!(matches!(f1.input.as_ref(), PlanNode::Filter(_)));
        }
    }

    // ========================================================================
    // ORDER BY and LIMIT
    // ========================================================================

    #[test]
    fn test_order_by() {
        let planner = SemanticPlanner::builder().build();
        let manifest = make_test_manifest();

        let request = ResolvedQueryRequest {
            entity_name: "orders".to_string(),
            dimensions: vec!["date".to_string()],
            measures: vec!["revenue".to_string()],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: None,
            order_by: vec![OrderByClause {
                field: "revenue".to_string(),
                direction: SortDirection::Descending,
            }],
            session_variables: HashMap::new(),
        };

        let plan = planner.plan(&request, &manifest).unwrap();
        assert!(
            matches!(&plan.root, PlanNode::Sort(_)),
            "root should be Sort when ORDER BY is specified"
        );
    }

    #[test]
    fn test_limit() {
        let planner = SemanticPlanner::builder().build();
        let manifest = make_test_manifest();

        let request = ResolvedQueryRequest {
            entity_name: "orders".to_string(),
            dimensions: vec!["date".to_string()],
            measures: vec!["revenue".to_string()],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: Some(10),
            order_by: vec![],
            session_variables: HashMap::new(),
        };

        let plan = planner.plan(&request, &manifest).unwrap();
        assert!(
            matches!(&plan.root, PlanNode::Fetch(_)),
            "root should be Fetch when LIMIT is specified"
        );
        if let PlanNode::Fetch(fetch) = &plan.root {
            assert_eq!(fetch.count, Some(10));
            assert_eq!(fetch.offset, 0);
        }
    }

    // ========================================================================
    // Optimizer pass-through
    // ========================================================================

    #[test]
    fn test_optimizer_identity_pass_through() {
        // Build planner with NO optimizer passes (default).
        let planner = SemanticPlanner::builder().build();
        let manifest = make_test_manifest();
        let request = make_test_request("orders", vec!["date"], vec!["revenue"]);

        let plan = planner.plan(&request, &manifest).unwrap();
        // Should succeed and produce a valid plan.
        assert!(!plan.output_names.is_empty());
    }

    /// A counting pass that tracks how many times it was invoked.
    struct CountingPass {
        count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl OptimizerPass for CountingPass {
        fn name(&self) -> &str {
            "counting"
        }

        fn apply(&self, plan: LogicalPlan) -> Result<LogicalPlan, PlannerError> {
            self.count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(plan)
        }
    }

    // ========================================================================
    // Ad-hoc multi-entity join (Phase 4 integration)
    // ========================================================================

    fn make_multi_entity_manifest() -> semstrait_manifest::CompiledManifest {
        use semstrait_manifest::{
            Cardinality, CompiledDimension, CompiledMeasure, CompiledRelationship,
            DimensionType, JoinColumnPair, JoinType as ModelJoinType,
        };
        use semstrait_manifest::acceleration::{
            CompiledDataKind, CompiledSimpleKind, CompiledInterface,
            DatasetBinding, ResolvedColumnMapping,
        };

        // Entity: orders (date, region, revenue) + customer_id for join key.
        let mut orders_dims = IndexMap::new();
        for name in &["date", "region"] {
            orders_dims.insert(name.to_string(), CompiledDimension {
                name: name.to_string(),
                description: None,
                data_type: semstrait_core::DataType::String,
                dim_type: DimensionType::Categorical(
                    semstrait_manifest::CategoricalDimension { enum_values: None },
                ),
                expr: None,
                expr_source: None,
            });
        }
        // customer_id as a dimension (join key).
        orders_dims.insert("customer_id".to_string(), CompiledDimension {
            name: "customer_id".to_string(),
            description: None,
            data_type: semstrait_core::DataType::String,
            dim_type: DimensionType::Categorical(
                semstrait_manifest::CategoricalDimension { enum_values: None },
            ),
            expr: None,
            expr_source: None,
        });

        let mut orders_measures = IndexMap::new();
        orders_measures.insert("revenue".to_string(), CompiledMeasure {
            name: "revenue".to_string(),
            description: None,
            data_type: semstrait_core::DataType::Number,
            agg: semstrait_core::Aggregation::Sum,
            expr: semstrait_core::Expr::entity_ref("amount"),
            expr_source: "amount".to_string(),
            additivity: None,
            constraints: None,
            filters: vec![],
        });

        let orders_iface = CompiledInterface {
            name: "orders".to_string(),
            description: None,
            dimensions: orders_dims,
            measures: orders_measures,
            metrics: IndexMap::new(),
            keys: None,
            filters: vec![],
            temporal_dim: None,
        };
        let mut orders_phys = IndexMap::new();
        orders_phys.insert("date".to_string(), "order_date".to_string());
        orders_phys.insert("region".to_string(), "region_name".to_string());
        orders_phys.insert("customer_id".to_string(), "cust_id".to_string());
        orders_phys.insert("revenue".to_string(), "amount".to_string());
        let orders_binding = DatasetBinding {
            dataset_name: "orders_ds".to_string(),
            column_mapping: ResolvedColumnMapping {
                physical: orders_phys,
                literals: HashMap::new(),
                temporal: HashMap::new(),
                anchored: HashMap::new(),
            },
            resolved_sources: vec![],
        };
        let orders_dk = CompiledDataKind::Simple(Box::new(CompiledSimpleKind {
            interface: orders_iface,
            binding: orders_binding,
        }));

        // Entity: customers (customer_name, id).
        let mut cust_dims = IndexMap::new();
        cust_dims.insert("customer_name".to_string(), CompiledDimension {
            name: "customer_name".to_string(),
            description: None,
            data_type: semstrait_core::DataType::String,
            dim_type: DimensionType::Categorical(
                semstrait_manifest::CategoricalDimension { enum_values: None },
            ),
            expr: None,
            expr_source: None,
        });
        cust_dims.insert("id".to_string(), CompiledDimension {
            name: "id".to_string(),
            description: None,
            data_type: semstrait_core::DataType::String,
            dim_type: DimensionType::Categorical(
                semstrait_manifest::CategoricalDimension { enum_values: None },
            ),
            expr: None,
            expr_source: None,
        });

        let cust_iface = CompiledInterface {
            name: "customers".to_string(),
            description: None,
            dimensions: cust_dims,
            measures: IndexMap::new(),
            metrics: IndexMap::new(),
            keys: None,
            filters: vec![],
            temporal_dim: None,
        };
        let mut cust_phys = IndexMap::new();
        cust_phys.insert("customer_name".to_string(), "name".to_string());
        cust_phys.insert("id".to_string(), "customer_id".to_string());
        let cust_binding = DatasetBinding {
            dataset_name: "customers_ds".to_string(),
            column_mapping: ResolvedColumnMapping {
                physical: cust_phys,
                literals: HashMap::new(),
                temporal: HashMap::new(),
                anchored: HashMap::new(),
            },
            resolved_sources: vec![],
        };
        let cust_dk = CompiledDataKind::Simple(Box::new(CompiledSimpleKind {
            interface: cust_iface,
            binding: cust_binding,
        }));

        // Relationship: orders.customer_id -> customers.id
        let relationship = CompiledRelationship {
            name: "orders_customers".to_string(),
            from: "orders".to_string(),
            to: "customers".to_string(),
            join_type: ModelJoinType::Left,
            columns: vec![JoinColumnPair {
                from: "customer_id".to_string(),
                to: "id".to_string(),
            }],
            cardinality: Cardinality::ManyToOne,
        };

        let mut entities = IndexMap::new();
        entities.insert("orders".to_string(), orders_dk);
        entities.insert("customers".to_string(), cust_dk);

        let mut rel_graph = semstrait_manifest::RelationshipGraph::default();
        rel_graph.set_shortest_path("orders", "customers", vec![0]);
        rel_graph.set_shortest_path("customers", "orders", vec![0]);

        semstrait_manifest::CompiledManifest {
            version: 3,
            compiled_at: chrono::Utc::now(),
            source_hash: "test_multi".to_string(),
            entities,
            relationships: vec![relationship],
            relationship_graph: rel_graph,
            field_index: semstrait_manifest::FieldIndex::default(),
            semantic_graph: semstrait_manifest::SemanticGraph::default(),
            diagnostics: semstrait_manifest::CompileDiagnostics::default(),
            model_name: "test_multi".to_string(),
            model_description: None,
            catalog_snapshot: None,
        }
    }

    #[test]
    fn test_ad_hoc_multi_entity_two_datasets() {
        let manifest = make_multi_entity_manifest();
        // Ad-hoc query spanning orders + customers (no FROM).
        let request = ResolvedQueryRequest {
            entity_name: String::new(),
            dimensions: vec!["date".into(), "customer_name".into()],
            measures: vec!["revenue".into()],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: HashMap::new(),
        };

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);
        assert!(result.is_ok(), "multi-entity ad-hoc should succeed: {:?}", result.err());

        let plan = result.unwrap();
        assert_eq!(plan.output_names, vec!["date", "customer_name", "revenue"]);

        // Plan shape: Project -> Join(left_fragment, right_fragment)
        // The root should be a Project (final projection).
        assert!(
            matches!(&plan.root, PlanNode::Project(_)),
            "root should be Project, got {:?}",
            std::mem::discriminant(&plan.root)
        );

        // Verify a JoinNode exists in the plan.
        fn contains_join(node: &PlanNode) -> bool {
            match node {
                PlanNode::Join(_) => true,
                PlanNode::Project(n) => contains_join(&n.input),
                PlanNode::Filter(n) => contains_join(&n.input),
                PlanNode::Sort(n) => contains_join(&n.input),
                PlanNode::Fetch(n) => contains_join(&n.input),
                PlanNode::Aggregate(n) => contains_join(&n.input),
                PlanNode::Union(n) => n.inputs.iter().any(contains_join),
                PlanNode::Scan(_) => false,
            }
        }
        assert!(contains_join(&plan.root), "plan should contain a JoinNode");
    }

    #[test]
    fn test_ad_hoc_multi_entity_output_names() {
        let manifest = make_multi_entity_manifest();
        // Only select date + customer_name (no revenue) — should still resolve
        // to multi-entity since customer_name is on "customers".
        let request = ResolvedQueryRequest {
            entity_name: String::new(),
            dimensions: vec!["date".into(), "customer_name".into()],
            measures: vec![],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: HashMap::new(),
        };

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);
        assert!(result.is_ok(), "dim-only multi-entity should succeed: {:?}", result.err());

        let plan = result.unwrap();
        assert_eq!(plan.output_names, vec!["date", "customer_name"]);
    }

    #[test]
    fn test_ad_hoc_multi_entity_with_order_limit() {
        let manifest = make_multi_entity_manifest();
        let request = ResolvedQueryRequest {
            entity_name: String::new(),
            dimensions: vec!["date".into(), "customer_name".into()],
            measures: vec!["revenue".into()],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: Some(50),
            order_by: vec![OrderByClause {
                field: "revenue".to_string(),
                direction: SortDirection::Descending,
            }],
            session_variables: HashMap::new(),
        };

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);
        assert!(result.is_ok(), "multi-entity with order/limit should succeed: {:?}", result.err());

        let plan = result.unwrap();

        // Root should be Fetch (LIMIT) wrapping Sort (ORDER BY).
        assert!(
            matches!(&plan.root, PlanNode::Fetch(_)),
            "root should be Fetch when LIMIT is specified"
        );
        if let PlanNode::Fetch(fetch) = &plan.root {
            assert_eq!(fetch.count, Some(50));
            assert!(
                matches!(fetch.input.as_ref(), PlanNode::Sort(_)),
                "Fetch input should be Sort when ORDER BY is specified"
            );
        }
    }

    #[test]
    fn test_ad_hoc_multi_entity_with_user_filter() {
        let manifest = make_multi_entity_manifest();
        let request = ResolvedQueryRequest {
            entity_name: String::new(),
            dimensions: vec!["date".into(), "customer_name".into()],
            measures: vec!["revenue".into()],
            filters: vec![QueryFilter {
                field: "date".to_string(),
                operator: FilterOperator::Eq,
                values: vec![FilterValue::String("2024-01-01".to_string())],
            }],
            inline_filters: vec![],
            pending_inline_filters: vec![],
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: HashMap::new(),
        };

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);
        assert!(result.is_ok(), "multi-entity with filter should succeed: {:?}", result.err());

        let plan = result.unwrap();
        // Root should be a Filter node (user filter applied after join).
        assert!(
            matches!(&plan.root, PlanNode::Filter(_)),
            "root should be Filter when user filters are present"
        );
    }

    // ========================================================================
    // Ad-hoc + inline raw filters
    //
    // Exercise the deferred-lowering path: parser stashes raw_filters as
    // `PendingInlineFilter`s; `plan_ad_hoc` (single-entity) or
    // `build_ad_hoc_join_plan::build_entity_requests` (multi-entity) lowers
    // them into `CompiledFilter`s against the resolved interface(s). The
    // result rides the same scan-layer engine as named DataKind filters per
    // `docs/design/foundations/11_names_and_scopes.md §6.4.2`.
    // ========================================================================

    fn contains_filter_node(node: &PlanNode) -> bool {
        match node {
            PlanNode::Filter(_) => true,
            PlanNode::Project(n) => contains_filter_node(&n.input),
            PlanNode::Aggregate(n) => contains_filter_node(&n.input),
            PlanNode::Sort(n) => contains_filter_node(&n.input),
            PlanNode::Fetch(n) => contains_filter_node(&n.input),
            PlanNode::Join(n) => {
                contains_filter_node(&n.left) || contains_filter_node(&n.right)
            }
            PlanNode::Union(n) => n.inputs.iter().any(contains_filter_node),
            PlanNode::Scan(_) => false,
        }
    }

    fn count_filter_nodes(node: &PlanNode) -> usize {
        match node {
            PlanNode::Filter(n) => 1 + count_filter_nodes(&n.input),
            PlanNode::Project(n) => count_filter_nodes(&n.input),
            PlanNode::Aggregate(n) => count_filter_nodes(&n.input),
            PlanNode::Sort(n) => count_filter_nodes(&n.input),
            PlanNode::Fetch(n) => count_filter_nodes(&n.input),
            PlanNode::Join(n) => count_filter_nodes(&n.left) + count_filter_nodes(&n.right),
            PlanNode::Union(n) => n.inputs.iter().map(count_filter_nodes).sum(),
            PlanNode::Scan(_) => 0,
        }
    }

    /// Count FilterNodes that appear strictly underneath any Join in the
    /// plan tree (i.e. per-entity scan-side filters in multi-entity ad-hoc).
    /// Filters above the topmost Join are user filters / post-join filters
    /// and are excluded.
    fn count_filter_nodes_under_join(node: &PlanNode) -> usize {
        match node {
            PlanNode::Join(_) => count_filter_nodes(node),
            PlanNode::Project(n) => count_filter_nodes_under_join(&n.input),
            PlanNode::Aggregate(n) => count_filter_nodes_under_join(&n.input),
            PlanNode::Sort(n) => count_filter_nodes_under_join(&n.input),
            PlanNode::Fetch(n) => count_filter_nodes_under_join(&n.input),
            PlanNode::Filter(n) => count_filter_nodes_under_join(&n.input),
            PlanNode::Union(n) => n.inputs.iter().map(count_filter_nodes_under_join).sum(),
            PlanNode::Scan(_) => 0,
        }
    }

    fn adhoc_request_with_pending(
        select: Vec<&str>,
        pending: Vec<crate::request::PendingInlineFilter>,
    ) -> ResolvedQueryRequest {
        ResolvedQueryRequest {
            entity_name: String::new(),
            dimensions: select.into_iter().map(String::from).collect(),
            measures: vec![],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: pending,
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: HashMap::new(),
        }
    }

    fn pending(field: &str, op: &str, value: serde_json::Value) -> crate::request::PendingInlineFilter {
        crate::request::PendingInlineFilter {
            field: field.to_string(),
            operator: op.to_string(),
            value,
        }
    }

    #[test]
    fn test_ad_hoc_single_entity_with_inline_filter_on_dim() {
        // `region` is a String dimension on the single `orders` entity.
        // Ad-hoc resolution picks `orders` from the select fields and lowers
        // the pending filter against its interface.
        let manifest = make_test_manifest();
        let request = adhoc_request_with_pending(
            vec!["date", "region", "revenue"],
            vec![pending("region", "eq", serde_json::json!("US"))],
        );

        let planner = SemanticPlanner::builder().build();
        let result = planner.plan(&request, &manifest);
        assert!(
            result.is_ok(),
            "ad-hoc single-entity + inline filter should succeed: {:?}",
            result.err()
        );

        let plan = result.unwrap();
        assert!(
            contains_filter_node(&plan.root),
            "plan should contain a FilterNode from the inline filter"
        );
    }

    #[test]
    fn test_ad_hoc_single_entity_inline_filter_unknown_field() {
        // `nonexistent` is not on the `orders` interface — lowering must
        // surface InlineFilterError::FieldNotFound, which the planner
        // wraps as PlannerError::InlineFilterResolution.
        let manifest = make_test_manifest();
        let request = adhoc_request_with_pending(
            vec!["date", "revenue"],
            vec![pending("nonexistent", "eq", serde_json::json!("X"))],
        );

        let planner = SemanticPlanner::builder().build();
        let err = planner.plan(&request, &manifest).unwrap_err();
        match err {
            PlannerError::InlineFilterResolution(
                crate::inline_filter::InlineFilterError::FieldNotFound { .. },
            ) => {}
            other => panic!("expected FieldNotFound, got {:?}", other),
        }
    }

    #[test]
    fn test_ad_hoc_single_entity_inline_filter_bad_operator() {
        let manifest = make_test_manifest();
        let request = adhoc_request_with_pending(
            vec!["date", "region", "revenue"],
            vec![pending("region", "regex", serde_json::json!("US"))],
        );

        let planner = SemanticPlanner::builder().build();
        let err = planner.plan(&request, &manifest).unwrap_err();
        match err {
            PlannerError::InlineFilterResolution(
                crate::inline_filter::InlineFilterError::OperatorInvalid { .. },
            ) => {}
            other => panic!("expected OperatorInvalid, got {:?}", other),
        }
    }

    #[test]
    fn test_ad_hoc_single_entity_inline_filter_type_mismatch() {
        // `revenue` is Number; a string literal doesn't coerce.
        let manifest = make_test_manifest();
        let request = adhoc_request_with_pending(
            vec!["date", "revenue"],
            vec![pending("revenue", ">", serde_json::json!("not_a_number"))],
        );

        let planner = SemanticPlanner::builder().build();
        let err = planner.plan(&request, &manifest).unwrap_err();
        match err {
            PlannerError::InlineFilterResolution(
                crate::inline_filter::InlineFilterError::ValueTypeMismatch { .. },
            ) => {}
            other => panic!("expected ValueTypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_ad_hoc_multi_entity_inline_filter_per_owner() {
        // Two raw filters, each owned by a different entity:
        //   - `region` is on `orders`
        //   - `customer_name` is on `customers`
        // Each filter must be attributed to its owning entity and placed on
        // that entity's per-entity scan. The resulting plan contains exactly
        // two FilterNodes under the Join.
        let manifest = make_multi_entity_manifest();
        let request = ResolvedQueryRequest {
            entity_name: String::new(),
            dimensions: vec!["date".into(), "customer_name".into()],
            measures: vec!["revenue".into()],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![
                pending("region", "eq", serde_json::json!("EU")),
                pending("customer_name", "eq", serde_json::json!("Acme")),
            ],
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: HashMap::new(),
        };

        let planner = SemanticPlanner::builder().build();
        let plan = planner
            .plan(&request, &manifest)
            .expect("multi-entity ad-hoc with per-owner inline filters should succeed");

        // Each filter lands on its owning entity's scan branch, beneath the
        // top-level Join. We expect at least two FilterNodes under the Join.
        let under_join = count_filter_nodes_under_join(&plan.root);
        assert!(
            under_join >= 2,
            "expected at least 2 FilterNodes under the Join (one per owning entity), got {}",
            under_join
        );
    }

    #[test]
    fn test_ad_hoc_multi_entity_inline_filter_shared_key_anchor_tiebreak() {
        // `customer_id` is the foreign-key column on `orders`. `customers`
        // has the corresponding key as `id`, so `customer_id` is owned only
        // by `orders` in this manifest. The anchor-first tiebreak code path
        // still runs because attribution searches MatchedEntity.covered_*
        // left-to-right; we assert here that the filter lands on the
        // `orders` scan exactly once and does not duplicate.
        let manifest = make_multi_entity_manifest();
        let request = ResolvedQueryRequest {
            entity_name: String::new(),
            dimensions: vec!["date".into(), "customer_name".into()],
            measures: vec!["revenue".into()],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![pending(
                "customer_id",
                "eq",
                serde_json::json!("c-123"),
            )],
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: HashMap::new(),
        };

        let planner = SemanticPlanner::builder().build();
        let plan = planner
            .plan(&request, &manifest)
            .expect("multi-entity ad-hoc with shared-side key filter should succeed");

        // Exactly one FilterNode under the Join (placed on the `orders`
        // scan side), not duplicated across both branches.
        let under_join = count_filter_nodes_under_join(&plan.root);
        assert_eq!(
            under_join, 1,
            "shared-key inline filter should be placed once (anchor-first), got {} FilterNode(s) under Join",
            under_join
        );
    }

    #[test]
    fn test_ad_hoc_multi_entity_inline_filter_field_on_no_entity() {
        // `does_not_exist` is on neither entity. Attribution must fail with
        // FieldOnNoEntity wrapped as PlannerError::InlineFilterResolution.
        let manifest = make_multi_entity_manifest();
        let request = ResolvedQueryRequest {
            entity_name: String::new(),
            dimensions: vec!["date".into(), "customer_name".into()],
            measures: vec!["revenue".into()],
            filters: vec![],
            inline_filters: vec![],
            pending_inline_filters: vec![pending(
                "does_not_exist",
                "eq",
                serde_json::json!("X"),
            )],
            grain: None,
            limit: None,
            order_by: vec![],
            session_variables: HashMap::new(),
        };

        let planner = SemanticPlanner::builder().build();
        let err = planner.plan(&request, &manifest).unwrap_err();
        match err {
            PlannerError::InlineFilterResolution(
                crate::inline_filter::InlineFilterError::FieldOnNoEntity { .. },
            ) => {}
            other => panic!("expected FieldOnNoEntity, got {:?}", other),
        }
    }

    #[test]
    fn test_optimizer_custom_pass_invoked() {
        let count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let planner = SemanticPlanner::builder()
            .with_optimizer_pass(CountingPass {
                count: count.clone(),
            })
            .build();

        let manifest = make_test_manifest();
        let request = make_test_request("orders", vec!["date"], vec!["revenue"]);

        let plan = planner.plan(&request, &manifest);
        assert!(plan.is_ok());
        assert_eq!(
            count.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "optimizer pass should have been invoked exactly once"
        );
    }
}
