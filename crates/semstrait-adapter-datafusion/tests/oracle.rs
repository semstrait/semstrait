#![allow(
    clippy::result_large_err,
    reason = "a function in `plan`'s place returns what `plan` returns, its PlanError unboxed"
)]

mod common;

use std::error::Error;

use common::case::Case;
use common::declaration::Declaration;
use common::golden;
use common::refusal::Refusal;
use common::runner::{self, Planned};
use common::{
    aggregate, executed, files_read, leaf, name, plan, produced, relations, rules, scanned, wire,
    Rows,
};
use datafusion::arrow::datatypes::{DataType as ArrowType, Field, Fields};
use datafusion::scalar::ScalarValue;
use semstrait_catalog::FileFormat;
use semstrait_common::Diagnostic;
use semstrait_common::Granularity;
use semstrait_ir::{
    AggregateNode, FilesScan, IrErrorKind, PlanNode, Reader, SemAnnotation, SourceRole,
};
use semstrait_ir::{SortNode, TableScan};
use semstrait_manifest::SemanticManifest;
use semstrait_model::model::serde::parse_expr;
use semstrait_planner::{
    DatasetArm, Decision, DimensionModifier, Direction, OrderKey, PlanError, Planner, ReadModel,
    Request, RequestContext, Selection, Strategy, StrategyInput, Timezone, Weekday,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::{NullOrdering, SemanticLeaf, SortDir, SortKey};
use substrait::proto::expression::field_reference::{
    ReferenceType::DirectReference, RootType::RootReference,
};
use substrait::proto::expression::literal::LiteralType;
use substrait::proto::expression::reference_segment::ReferenceType::StructField;
use substrait::proto::expression::{FieldReference, RexType};
use substrait::proto::extensions::simple_extension_declaration::MappingType;
use substrait::proto::extensions::AdvancedExtension;
use substrait::proto::function_argument::ArgType::Value;
use substrait::proto::read_rel::local_files::file_or_files::PathType;
use substrait::proto::rel::RelType;
use substrait::proto::rel_common::EmitKind;
use substrait::proto::set_rel::SetOp;
use substrait::proto::{plan_rel, CrossRel, Expression, JoinRel, Plan, PlanRel, ProjectRel};
use substrait::proto::{ReadRel, Rel, RelRoot};
use substrait::version::version_with_producer;

macro_rules! cases {
    (@plan) => { runner::planned };
    (@plan $plan:expr) => { $plan };
    ($($test:ident: $case:literal $(by $plan:expr)?;)*) => {
        $(
            #[tokio::test]
            async fn $test() -> Result<(), Box<dyn Error>> {
                runner::run($case, cases!(@plan $($plan)?)).await?.assert_committed();
                Ok(())
            }
        )*

        const LISTED: &[&str] = &[$($case),*];

        #[tokio::test]
        async fn generate_goldens() {
            let gates: &[(&str, runner::PlanFn)] = &[$(($case, cases!(@plan $($plan)?))),*];
            golden::generate(async {
                let mut goldens = Vec::new();
                for &(case, plan) in gates {
                    goldens.push(runner::run(case, plan).await.expect(case));
                }
                goldens
            })
            .await;
        }
    };
}

cases! {
    harness_mechanics_table_read: "harness/mechanics/table_read" by table;
    harness_mechanics_files_read: "harness/mechanics/files_read" by files;
    harness_mechanics_plan_refusal: "harness/mechanics/plan_refusal" by refusing;
    harness_mechanics_table_read_planned: "harness/mechanics/table_read_planned";
    harness_mechanics_files_read_planned: "harness/mechanics/files_read_planned";
    refusal_missing_column_column_not_in_source: "refusal/missing_column/column_not_in_source";
    refusal_measure_without_agg_measure_missing_agg: "refusal/measure_without_agg/measure_missing_agg";
    request_elements_aggregate_filter_mirror: "request/elements/aggregate_filter_mirror";
    request_elements_named_filter: "request/elements/named_filter";
    request_elements_distinct_dimensions: "request/elements/distinct_dimensions";
    request_elements_narrow_projection: "request/elements/narrow_projection";
    request_elements_no_catalog_name: "request/elements/no_catalog_name";
    request_elements_measures_only_one_row: "request/elements/measures_only_one_row";
    request_elements_limit_and_offset: "request/elements/limit_and_offset";
    request_elements_offset_no_limit: "request/elements/offset_no_limit";
    request_elements_limit_past_max: "request/elements/limit_past_max";
    request_elements_nulls_sort_high_asc: "request/elements/nulls_sort_high_asc";
    request_elements_nulls_sort_high_desc: "request/elements/nulls_sort_high_desc";
    request_elements_where_on_dimension: "request/elements/where_on_dimension";
    request_elements_where_on_measure: "request/elements/where_on_measure";
    request_elements_where_on_metric: "request/elements/where_on_metric";
    resolution_scope_root_lower_case_label: "resolution/scope_root/lower_case_label";
    resolution_scope_root_upper_case_label: "resolution/scope_root/upper_case_label";
    resolution_scope_root_bucketed_time_axis: "resolution/scope_root/bucketed_time_axis";
    resolution_scope_root_selected_time_axis: "resolution/scope_root/selected_time_axis";
    resolution_scope_root_no_time_axis_selected: "resolution/scope_root/no_time_axis_selected";
    resolution_scope_root_fewer_semantics_wins: "resolution/scope_root/fewer_semantics_wins";
    resolution_scope_root_bucketed_coarsest_first: "resolution/scope_root/bucketed_coarsest_first";
    resolution_scope_root_axis_unselected_coarsest_first: "resolution/scope_root/axis_unselected_coarsest_first";
    resolution_scope_root_axis_selected_finest_first: "resolution/scope_root/axis_selected_finest_first";
    resolution_scope_root_finer_grain_excludes_coarse: "resolution/scope_root/finer_grain_excludes_coarse";
    resolution_scope_root_pinned_from_matches_pick: "resolution/scope_root/pinned_from_matches_pick";
    resolution_scope_root_composition_before_grain: "resolution/scope_root/composition_before_grain";
    resolution_scope_root_name_order_tie: "resolution/scope_root/name_order_tie";
    resolution_scope_root_pinned_first_in_order: "resolution/scope_root/pinned_first_in_order";
    read_files_empty_folder_grouped: "read/files/empty_folder_grouped";
    read_files_empty_folder_totals: "read/files/empty_folder_totals";
    read_files_one_uri_file: "read/files/one_uri_file";
    read_files_two_uri_files: "read/files/two_uri_files";
    read_types_catalog_string_cast: "read/types/catalog_string_cast";
    read_types_object_string_read_type: "read/types/object_string_read_type";
    read_types_projection_casts_to_declared: "read/types/projection_casts_to_declared";
    read_types_count_only_column_type: "read/types/count_only_column_type";
    read_types_widest_declared_type: "read/types/widest_declared_type";
    read_types_agreeing_types_no_cast: "read/types/agreeing_types_no_cast";
    read_types_catalog_type_cast: "read/types/catalog_type_cast";
    read_partitions_default_partition_null: "read/partitions/default_partition_null";
    read_partitions_dt_key_bucketed_month: "read/partitions/dt_key_bucketed_month";
    read_partitions_file_own_year_column: "read/partitions/file_own_year_column";
    read_partitions_folder_value_wins: "read/partitions/folder_value_wins";
    read_partitions_folder_distinct_count: "read/partitions/folder_distinct_count";
    read_partitions_month_key_partitions: "read/partitions/month_key_partitions";
    read_partitions_predicate_on_key: "read/partitions/predicate_on_key";
    read_partitions_base_path_key_column: "read/partitions/base_path_key_column";
    temporal_shape_default_week_start: "temporal/shape/default_week_start";
    temporal_shape_sunday_week_start: "temporal/shape/sunday_week_start";
    temporal_shape_repeated_local_hour: "temporal/shape/repeated_local_hour";
    temporal_shape_month_crosses_midnight: "temporal/shape/month_crosses_midnight";
    temporal_shape_timestamptz_month_bucket: "temporal/shape/timestamptz_month_bucket";
    temporal_shape_timestamptz_month_under_zone: "temporal/shape/timestamptz_month_under_zone";
    temporal_shape_timestamptz_predicate_utc: "temporal/shape/timestamptz_predicate_utc";
    temporal_shape_date_trunc_month: "temporal/shape/date_trunc_month";
    temporal_shape_date_trunc_month_fixed: "temporal/shape/date_trunc_month_fixed";
    temporal_shape_date_trunc_hour: "temporal/shape/date_trunc_hour";
    temporal_shape_date_trunc_hour_fixed: "temporal/shape/date_trunc_hour_fixed";
    temporal_shape_date_trunc_week: "temporal/shape/date_trunc_week";
    temporal_shape_date_trunc_week_fixed: "temporal/shape/date_trunc_week_fixed";
    temporal_shape_extract_date_parts: "temporal/shape/extract_date_parts";
    temporal_shape_extract_date_parts_fixed: "temporal/shape/extract_date_parts_fixed";
    temporal_shape_extract_timestamp_parts: "temporal/shape/extract_timestamp_parts";
    temporal_shape_extract_timestamp_fixed: "temporal/shape/extract_timestamp_fixed";
    temporal_shape_predicate_under_bucket: "temporal/shape/predicate_under_bucket";
    function_behavior_concat_null_operand: "function/behavior/concat_null_operand";
    function_behavior_concat_two_columns: "function/behavior/concat_two_columns";
    function_behavior_regexp_pattern_market: "function/behavior/regexp_pattern_market";
    function_behavior_eq_excludes_null: "function/behavior/eq_excludes_null";
    function_behavior_distinct_from_includes_null: "function/behavior/distinct_from_includes_null";
    additivity_semi_additive_latest_no_axes: "additivity/semi_additive/latest_no_axes";
    additivity_semi_additive_predicate_reads_finished: "additivity/semi_additive/predicate_reads_finished";
    additivity_semi_additive_latest_per_account: "additivity/semi_additive/latest_per_account";
    additivity_semi_additive_earliest_per_account: "additivity/semi_additive/earliest_per_account";
    additivity_semi_additive_quotient_at_marker: "additivity/semi_additive/quotient_at_marker";
    additivity_semi_additive_linear_at_marker: "additivity/semi_additive/linear_at_marker";
    additivity_semi_additive_marker_beside_neighbours: "additivity/semi_additive/marker_beside_neighbours";
    additivity_semi_additive_metric_output_predicate: "additivity/semi_additive/metric_output_predicate";
    additivity_semi_additive_latest_across_accounts: "additivity/semi_additive/latest_across_accounts";
    additivity_semi_additive_balance_and_cost_differ: "additivity/semi_additive/balance_and_cost_differ";
    additivity_semi_additive_account_latest_full_cost: "additivity/semi_additive/account_latest_full_cost";
    additivity_semi_additive_latest_non_null_or_all: "additivity/semi_additive/latest_non_null_or_all";
    additivity_semi_additive_net_marker_difference: "additivity/semi_additive/net_marker_difference";
    additivity_semi_additive_net_by_day_latest: "additivity/semi_additive/net_by_day_latest";
    additivity_semi_additive_closing_net_marker: "additivity/semi_additive/closing_net_marker";
    additivity_semi_additive_latest_per_account_through_joinset: "additivity/semi_additive/latest_per_account_through_joinset";
    additivity_non_additive_reaggregated_average: "additivity/non_additive/reaggregated_average";
    formula_metric_bonus_joined_to_buyers: "formula/metric/bonus_joined_to_buyers";
    formula_metric_value_not_repeated: "formula/metric/value_not_repeated";
    formula_metric_pushdown_value_not_repeated: "formula/metric/pushdown_value_not_repeated";
    formula_metric_grouped_reads_step_field: "formula/metric/grouped_reads_step_field";
    formula_metric_grouped_any_formula_passes: "formula/metric/grouped_any_formula_passes";
    formula_metric_pushdown_one_aggregate: "formula/metric/pushdown_one_aggregate";
    formula_metric_pushdown_no_else: "formula/metric/pushdown_no_else";
    formula_metric_pushdown_zero_else: "formula/metric/pushdown_zero_else";
    formula_metric_quotient_takes_parts: "formula/metric/quotient_takes_parts";
    formula_metric_constant_takes_parts: "formula/metric/constant_takes_parts";
    formula_metric_non_additive_takes_parts: "formula/metric/non_additive_takes_parts";
    formula_metric_avg_with_no_dimension: "formula/metric/avg_with_no_dimension";
    formula_metric_row_level_metric_measure: "formula/metric/row_level_metric_measure";
    formula_metric_term_before_bucket: "formula/metric/term_before_bucket";
    formula_metric_steps_join_not_distinct: "formula/metric/steps_join_not_distinct";
    formula_metric_predicate_above_step_two: "formula/metric/predicate_above_step_two";
    formula_metric_sum_beside_distinct_count: "formula/metric/sum_beside_distinct_count";
    formula_metric_fan_out_row_number_marker: "formula/metric/fan_out_row_number_marker";
    formula_metric_parts_under_fan_out: "formula/metric/parts_under_fan_out";
    formula_metric_case_by_position_channel: "formula/metric/case_by_position_channel";
    formula_metric_case_under_cast: "formula/metric/case_under_cast";
    formula_metric_case_inside_condition: "formula/metric/case_inside_condition";
    formula_metric_equal_results_two_parts: "formula/metric/equal_results_two_parts";
    formula_metric_condition_reads_measure: "formula/metric/condition_reads_measure";
    formula_metric_case_inside_later_branch: "formula/metric/case_inside_later_branch";
    formula_metric_case_inside_result: "formula/metric/case_inside_result";
    formula_metric_case_by_result_position: "formula/metric/case_by_result_position";
    formula_metric_two_metrics_two_grains: "formula/metric/two_metrics_two_grains";
    formula_metric_formula_grouped_by_dimension: "formula/metric/formula_grouped_by_dimension";
    formula_metric_pushdown_not_repeated: "formula/metric/pushdown_not_repeated";
    formula_metric_measure_and_metric_by_region: "formula/metric/measure_and_metric_by_region";
    formula_metric_measure_and_metric_total: "formula/metric/measure_and_metric_total";
    formula_metric_expr_reads_row_dimension: "formula/metric/expr_reads_row_dimension";
    formula_metric_constant_keeps_measure_link: "formula/metric/constant_keeps_measure_link";
    formula_metric_filter_narrows_linked_subtree: "formula/metric/filter_narrows_linked_subtree";
    formula_metric_linked_subtrees_join_difference: "formula/metric/linked_subtrees_join_difference";
    formula_metric_no_agg_as_dimension: "formula/metric/no_agg_as_dimension";
    formula_metric_formula_reads_linked_subtree: "formula/metric/formula_reads_linked_subtree";
    formula_metric_formula_reads_each_day: "formula/metric/formula_reads_each_day";
    formula_metric_tier_reads_label_field: "formula/metric/tier_reads_label_field";
    formula_metric_formula_grain_holds_column: "formula/metric/formula_grain_holds_column";
    formula_metric_group_by_row_metric: "formula/metric/group_by_row_metric";
    formula_metric_predicate_on_row_metric: "formula/metric/predicate_on_row_metric";
    formula_metric_field_by_position_no_copy: "formula/metric/field_by_position_no_copy";
    formula_metric_case_parts_under_fan_out: "formula/metric/case_parts_under_fan_out";
    formula_metric_no_else_adds_nothing: "formula/metric/no_else_adds_nothing";
    formula_metric_null_condition_second_result: "formula/metric/null_condition_second_result";
    formula_metric_grain_groups_other_bucket: "formula/metric/grain_groups_other_bucket";
    formula_metric_renamed_column_row_projection: "formula/metric/renamed_column_row_projection";
    formula_metric_case_parts_over_grainset: "formula/metric/case_parts_over_grainset";
    formula_metric_case_parts_in_union_branch: "formula/metric/case_parts_in_union_branch";
    formula_metric_metric_over_aggregates: "formula/metric/metric_over_aggregates";
    formula_metric_filter_on_inlined_aggregate: "formula/metric/filter_on_inlined_aggregate";
    formula_push_down_case_over_grainset_operand: "formula/push_down/case_over_grainset_operand";
    formula_push_down_pushdown_meets_grainset: "formula/push_down/pushdown_meets_grainset";
    formula_push_down_case_by_union_branch: "formula/push_down/case_by_union_branch";
    formula_push_down_case_inside_condition_branch: "formula/push_down/case_inside_condition_branch";
    formula_push_down_pushdown_to_union_root: "formula/push_down/pushdown_to_union_root";
    formula_push_down_case_over_union_in_joinset: "formula/push_down/case_over_union_in_joinset";
    formula_push_down_pushdown_through_union_branches: "formula/push_down/pushdown_through_union_branches";
    joinset_projection_coalesce_both_map: "joinset/projection/coalesce_both_map";
    joinset_projection_cross_filter_above: "joinset/projection/cross_filter_above";
    joinset_pruned_left_join_prunes_other_side: "joinset/pruned/left_join_prunes_other_side";
    joinset_pruned_bypassed_middle_operand: "joinset/pruned/bypassed_middle_operand";
    joinset_delivery_one_row_per_join_key: "joinset/delivery/one_row_per_join_key";
    joinset_delivery_no_match_coalesced_zero: "joinset/delivery/no_match_coalesced_zero";
    joinset_delivery_many_to_many_duplicate_rows: "joinset/delivery/many_to_many_duplicate_rows";
    joinset_delivery_both_sides_partial_below_join: "joinset/delivery/both_sides_partial_below_join";
    joinset_delivery_partial_meets_own_channel: "joinset/delivery/partial_meets_own_channel";
    joinset_delivery_non_additive_gives_rows: "joinset/delivery/non_additive_gives_rows";
    joinset_fan_out_case_condition_marker_parts: "joinset/fan_out/case_condition_marker_parts";
    joinset_fan_out_case_result_marker_parts: "joinset/fan_out/case_result_marker_parts";
    joinset_fan_out_grain_holds_fanned_key: "joinset/fan_out/grain_holds_fanned_key";
    joinset_fan_out_avg_reads_row_identity: "joinset/fan_out/avg_reads_row_identity";
    joinset_fan_out_keyset_gives_row_identity: "joinset/fan_out/keyset_gives_row_identity";
    joinset_fan_out_no_keyset_every_dimension: "joinset/fan_out/no_keyset_every_dimension";
    joinset_fan_out_marker_by_dataset_axis: "joinset/fan_out/marker_by_dataset_axis";
    joinset_fan_out_marker_ignores_coalesced_date: "joinset/fan_out/marker_ignores_coalesced_date";
    joinset_fan_out_repeated_snapshot_row_identity: "joinset/fan_out/repeated_snapshot_row_identity";
    joinset_fan_out_marker_per_order_not_line: "joinset/fan_out/marker_per_order_not_line";
    joinset_predicate_copy_below_inner_join: "joinset/predicate/copy_below_inner_join";
    joinset_predicate_two_copies_one_above_one_below: "joinset/predicate/two_copies_one_above_one_below";
    joinset_predicate_relationship_filter_above_join: "joinset/predicate/relationship_filter_above_join";
    joinset_predicate_named_filter_two_copies: "joinset/predicate/named_filter_two_copies";
    joinset_predicate_copy_pruned_with_operand: "joinset/predicate/copy_pruned_with_operand";
    joinset_predicate_copy_stays_as_condition: "joinset/predicate/copy_stays_as_condition";
    joinset_predicate_copy_below_left_join: "joinset/predicate/copy_below_left_join";
    joinset_predicate_predicate_above_null_filled_join: "joinset/predicate/predicate_above_null_filled_join";
    unionset_delivery_bucket_then_marker: "unionset/delivery/bucket_then_marker";
    unionset_delivery_marker_below_formula: "unionset/delivery/marker_below_formula";
    unionset_delivery_nondisjoint_avg_above_union: "unionset/delivery/nondisjoint_avg_above_union";
    unionset_delivery_nondisjoint_count_distinct: "unionset/delivery/nondisjoint_count_distinct";
    unionset_delivery_decimal_precision_kept: "unionset/delivery/decimal_precision_kept";
    unionset_delivery_null_filled_coalesced_count: "unionset/delivery/null_filled_coalesced_count";
    unionset_delivery_ungrouped_count_coalesced: "unionset/delivery/ungrouped_count_coalesced";
    unionset_delivery_mixed_type_count_cast: "unionset/delivery/mixed_type_count_cast";
    unionset_delivery_disjoint_branches_final_avg: "unionset/delivery/disjoint_branches_final_avg";
    unionset_delivery_ratio_over_filtered_branches: "unionset/delivery/ratio_over_filtered_branches";
    unionset_unique_keyset_distinguishes_rows: "unionset/unique/keyset_distinguishes_rows";
    unionset_unique_no_keyset_collapses_rows: "unionset/unique/no_keyset_collapses_rows";
    unionset_unique_distinct_keeps_different_days: "unionset/unique/distinct_keeps_different_days";
    unionset_pruned_branch_null_fill_unmapped: "unionset/pruned/branch_null_fill_unmapped";
    unionset_pruned_literal_folds_prunes_branch: "unionset/pruned/literal_folds_prunes_branch";
    unionset_pruned_every_branch_excluded_count: "unionset/pruned/every_branch_excluded_count";
    unionset_pruned_null_fill_equality_prunes_branch: "unionset/pruned/null_fill_equality_prunes_branch";
    unionset_pruned_null_fill_is_null_reads_branch: "unionset/pruned/null_fill_is_null_reads_branch";
    unionset_pruned_null_fill_negation_prunes_branch: "unionset/pruned/null_fill_negation_prunes_branch";
    unionset_pruned_null_fill_named_equality_prunes_branch: "unionset/pruned/null_fill_named_equality_prunes_branch";
    unionset_pruned_null_fill_named_is_null_reads_branch: "unionset/pruned/null_fill_named_is_null_reads_branch";
    unionset_pruned_null_fill_named_negation_prunes_branch: "unionset/pruned/null_fill_named_negation_prunes_branch";
    unionset_pruned_null_fill_or_keeps_branch_for_later_rows: "unionset/pruned/null_fill_or_keeps_branch_for_later_rows";
    unionset_pruned_every_branch_excluded_sum_null: "unionset/pruned/every_branch_excluded_sum_null";
    unionset_pruned_branch_not_servable_is_pruned: "unionset/pruned/branch_not_servable_is_pruned";
    unionset_pruned_every_branch_excluded_with_grainset_branch: "unionset/pruned/every_branch_excluded_with_grainset_branch";
    unionset_pruned_every_branch_a_grainset_excluded: "unionset/pruned/every_branch_a_grainset_excluded";
    unionset_filter_own_filter_through_composed_join: "unionset/filter/own_filter_through_composed_join";
    unionset_filter_own_filter_partial_aggregates: "unionset/filter/own_filter_partial_aggregates";
    unionset_filter_own_filter_over_rows: "unionset/filter/own_filter_over_rows";
    unionset_filter_branch_filter_and_union_filter: "unionset/filter/branch_filter_and_union_filter";
    unionset_filter_named_filter_in_each_branch: "unionset/filter/named_filter_in_each_branch";
    unionset_implicit_path_token_disjoint_branches: "unionset/implicit/path_token_disjoint_branches";
    unionset_implicit_predicate_excludes_every_leaf: "unionset/implicit/predicate_excludes_every_leaf";
    unionset_implicit_predicate_excludes_one_leaf: "unionset/implicit/predicate_excludes_one_leaf";
    unionset_implicit_two_tables_one_union: "unionset/implicit/two_tables_one_union";
    unionset_implicit_two_paths_not_disjoint: "unionset/implicit/two_paths_not_disjoint";
    unionset_implicit_predicate_over_union_columns: "unionset/implicit/predicate_over_union_columns";
    unionset_partials_u1_counts_count_by_channel: "unionset/partials/u1_counts_count_by_channel";
    unionset_partials_u1_counts_count_and_filtered_metric: "unionset/partials/u1_counts_count_and_filtered_metric";
    unionset_partials_u1_counts_bigint_count_by_channel: "unionset/partials/u1_counts_bigint_count_by_channel";
    unionset_partials_u1_counts_bigint_count_and_filtered_metric: "unionset/partials/u1_counts_bigint_count_and_filtered_metric";
    unionset_partials_all_orders_cast_count_by_channel: "unionset/partials/all_orders_cast_count_by_channel";
    unionset_partials_all_orders_cast_count_and_node: "unionset/partials/all_orders_cast_count_and_node";
    unionset_partials_all_orders_cast_count_and_filtered_metric: "unionset/partials/all_orders_cast_count_and_filtered_metric";
    unionset_partials_all_orders_cast_node_where_count: "unionset/partials/all_orders_cast_node_where_count";
    unionset_partials_all_orders_cast_integer_sum_past_its_type: "unionset/partials/all_orders_cast_integer_sum_past_its_type";
    unionset_partials_all_orders_cast_decimal_sum_past_its_type: "unionset/partials/all_orders_cast_decimal_sum_past_its_type";
    unionset_partials_all_orders_cast_bigint_count_by_channel: "unionset/partials/all_orders_cast_bigint_count_by_channel";
    unionset_partials_all_orders_cast_bigint_count_and_node: "unionset/partials/all_orders_cast_bigint_count_and_node";
    unionset_partials_all_orders_cast_bigint_count_and_filtered_metric: "unionset/partials/all_orders_cast_bigint_count_and_filtered_metric";
    unionset_partials_spend_two_levels_clicks_by_region: "unionset/partials/spend_two_levels_clicks_by_region";
    unionset_partials_joinset_over_unionset_cast_sum_by_region: "unionset/partials/joinset_over_unionset_cast_sum_by_region";
    grainset_routing_bypass_then_drill_across: "grainset/routing/bypass_then_drill_across";
    grainset_routing_operand_kept_alone_each_name_from_its_rank_first_child_filtered_above_the_join: "grainset/routing/operand_kept_alone_each_name_from_its_rank_first_child_filtered_above_the_join";
    grainset_routing_operand_of_a_built_join_one_child_answers_whole: "grainset/routing/operand_of_a_built_join_one_child_answers_whole";
    grainset_routing_nested_grainset_without_a_keyset_serves_a_measure_one_child_maps: "grainset/routing/nested_grainset_without_a_keyset_serves_a_measure_one_child_maps";
    grainset_routing_nested_grainset_without_a_keyset_cannot_compute_a_metric_over_rows_of_two_children: "grainset/routing/nested_grainset_without_a_keyset_cannot_compute_a_metric_over_rows_of_two_children";
    grainset_routing_drill_across_children_own_steps: "grainset/routing/drill_across_children_own_steps";
    grainset_routing_one_child_serves_whole: "grainset/routing/one_child_serves_whole";
    grainset_routing_measures_split_across_children: "grainset/routing/measures_split_across_children";
    grainset_routing_no_child_maps_all: "grainset/routing/no_child_maps_all";
    grainset_routing_demand_forces_the_join: "grainset/routing/demand_forces_the_join";
    grainset_routing_walk_picks_answering_child: "grainset/routing/walk_picks_answering_child";
    grainset_routing_constituent_frame_rebucketed: "grainset/routing/constituent_frame_rebucketed";
    grainset_routing_day_child_answers_alone: "grainset/routing/day_child_answers_alone";
    grainset_routing_coarsest_child_answers: "grainset/routing/coarsest_child_answers";
    grainset_routing_spanning_week_no_month_child: "grainset/routing/spanning_week_no_month_child";
    grainset_routing_coarsest_week_child_alone: "grainset/routing/coarsest_week_child_alone";
    grainset_routing_later_child_predicate_above: "grainset/routing/later_child_predicate_above";
    grainset_routing_first_child_predicate_above: "grainset/routing/first_child_predicate_above";
    grainset_routing_two_measures_two_children: "grainset/routing/two_measures_two_children";
    grainset_routing_drill_across_at_year: "grainset/routing/drill_across_at_year";
    grainset_routing_no_granularity_fixes_coarsest: "grainset/routing/no_granularity_fixes_coarsest";
    grainset_routing_no_time_axis_crossed_rows: "grainset/routing/no_time_axis_crossed_rows";
    grainset_routing_whole_demand_drills_by_name: "grainset/routing/whole_demand_drills_by_name";
    grainset_routing_finest_child_least_bucketed: "grainset/routing/finest_child_least_bucketed";
    grainset_routing_coarsest_at_or_finer: "grainset/routing/coarsest_at_or_finer";
    grainset_routing_shared_day_different_dimensions: "grainset/routing/shared_day_different_dimensions";
    grainset_routing_shared_day_different_dimensions_with_clicks: "grainset/routing/shared_day_different_dimensions_with_clicks";
    grainset_routing_each_name_from_its_rank_first_child: "grainset/routing/each_name_from_its_rank_first_child";
    grainset_routing_one_child_maps_a_metric_whole_so_it_answers_alone: "grainset/routing/one_child_maps_a_metric_whole_so_it_answers_alone";
    grainset_routing_no_child_maps_a_metric_whole_so_the_grainset_computes_it: "grainset/routing/no_child_maps_a_metric_whole_so_the_grainset_computes_it";
    grainset_routing_operand_kept_alone_computes_a_metric_no_child_maps_whole: "grainset/routing/operand_kept_alone_computes_a_metric_no_child_maps_whole";
    grainset_routing_a_computed_metric_with_its_own_filter_on_a_grainset: "grainset/routing/a_computed_metric_with_its_own_filter_on_a_grainset";
    grainset_routing_a_computed_metric_with_its_own_filter_on_an_operand_kept_alone: "grainset/routing/a_computed_metric_with_its_own_filter_on_an_operand_kept_alone";
    grainset_routing_a_metric_with_its_own_agg_on_a_grainset: "grainset/routing/a_metric_with_its_own_agg_on_a_grainset";
    grainset_routing_a_metric_with_its_own_agg_on_an_operand_kept_alone: "grainset/routing/a_metric_with_its_own_agg_on_an_operand_kept_alone";
    grainset_routing_a_chain_over_a_computed_metric_on_a_grainset: "grainset/routing/a_chain_over_a_computed_metric_on_a_grainset";
    grainset_routing_a_chain_over_a_computed_metric_on_an_operand_kept_alone: "grainset/routing/a_chain_over_a_computed_metric_on_an_operand_kept_alone";
    grainset_routing_one_child_maps_a_metric_whole_so_a_joinset_operand_kept_alone_answers_alone: "grainset/routing/one_child_maps_a_metric_whole_so_a_joinset_operand_kept_alone_answers_alone";
    grainset_routing_no_keyset_one_child_answers_whole: "grainset/routing/no_keyset_one_child_answers_whole";
    grainset_routing_shared_grain_other_mask: "grainset/routing/shared_grain_other_mask";
    grainset_routing_shared_grain_name_order_tie: "grainset/routing/shared_grain_name_order_tie";
    grainset_routing_shared_day_tie_name_first_joined_first: "grainset/routing/shared_day_tie_name_first_joined_first";
    grainset_routing_shared_day_tie_name_first_joined_first_swapped: "grainset/routing/shared_day_tie_name_first_joined_first_swapped";
    grainset_routing_shared_day_tie_null_group_holds_each_childs_value: "grainset/routing/shared_day_tie_null_group_holds_each_childs_value";
    grainset_routing_shared_day_tie_null_group_holds_each_childs_value_swapped: "grainset/routing/shared_day_tie_null_group_holds_each_childs_value_swapped";
    grainset_routing_by_date_full_join_keeps_every_group: "grainset/routing/by_date_full_join_keeps_every_group";
    grainset_routing_by_date_measure_predicate_above_the_join: "grainset/routing/by_date_measure_predicate_above_the_join";
    grainset_routing_by_date_three_children: "grainset/routing/by_date_three_children";
    grainset_routing_by_date_three_children_reversed: "grainset/routing/by_date_three_children_reversed";
    grainset_routing_no_time_axis_clicks: "grainset/routing/no_time_axis_clicks";
    grainset_routing_no_time_axis_campaign_clicks: "grainset/routing/no_time_axis_campaign_clicks";
    grainset_routing_mixed_time_axis_no_date: "grainset/routing/mixed_time_axis_no_date";
    grainset_routing_mixed_time_axis_date_at_month: "grainset/routing/mixed_time_axis_date_at_month";
    grainset_routing_own_metric_cpc_by_date: "grainset/routing/own_metric_cpc_by_date";
    grainset_routing_own_metric_cpc_cpm_by_date: "grainset/routing/own_metric_cpc_cpm_by_date";
    grainset_routing_own_metric_cpc_by_date_over_campaign_children: "grainset/routing/own_metric_cpc_by_date_over_campaign_children";
    grainset_routing_own_metric_predicate_above_the_join: "grainset/routing/own_metric_predicate_above_the_join";
    grainset_routing_own_agg_metric_one_dataset: "grainset/routing/own_agg_metric_one_dataset";
    grainset_routing_own_agg_metric_over_a_grainset: "grainset/routing/own_agg_metric_over_a_grainset";
    grainset_routing_own_agg_metric_reading_a_dimension_one_dataset: "grainset/routing/own_agg_metric_reading_a_dimension_one_dataset";
    grainset_routing_own_agg_metric_reading_a_dimension_over_a_grainset: "grainset/routing/own_agg_metric_reading_a_dimension_over_a_grainset";
    grainset_routing_own_agg_metric_beside_a_metric_without_agg_one_dataset: "grainset/routing/own_agg_metric_beside_a_metric_without_agg_one_dataset";
    grainset_routing_own_agg_metric_beside_a_metric_without_agg_over_a_grainset: "grainset/routing/own_agg_metric_beside_a_metric_without_agg_over_a_grainset";
    grainset_routing_own_agg_metric_read_by_a_metric_one_dataset: "grainset/routing/own_agg_metric_read_by_a_metric_one_dataset";
    grainset_routing_own_agg_metric_read_by_a_metric_over_a_grainset: "grainset/routing/own_agg_metric_read_by_a_metric_over_a_grainset";
    grainset_routing_own_agg_metric_at_a_finer_formula_grain_read_by_a_metric_one_dataset: "grainset/routing/own_agg_metric_at_a_finer_formula_grain_read_by_a_metric_one_dataset";
    grainset_routing_own_agg_metric_at_a_finer_formula_grain_read_by_a_metric_over_a_grainset: "grainset/routing/own_agg_metric_at_a_finer_formula_grain_read_by_a_metric_over_a_grainset";
    grainset_routing_own_agg_metric_over_an_own_agg_metric_one_dataset: "grainset/routing/own_agg_metric_over_an_own_agg_metric_one_dataset";
    grainset_routing_own_agg_metric_over_an_own_agg_metric_over_a_grainset: "grainset/routing/own_agg_metric_over_an_own_agg_metric_over_a_grainset";
    grainset_routing_own_agg_metric_read_late_under_a_filter_one_dataset: "grainset/routing/own_agg_metric_read_late_under_a_filter_one_dataset";
    grainset_routing_own_agg_metric_read_late_under_a_filter_over_a_grainset: "grainset/routing/own_agg_metric_read_late_under_a_filter_over_a_grainset";
    grainset_routing_own_agg_metric_beside_a_month_measure_from_a_child_that_lacks_region_over_a_grainset: "grainset/routing/own_agg_metric_beside_a_month_measure_from_a_child_that_lacks_region_over_a_grainset";
    grainset_routing_parts_metric_reading_a_dimension_one_dataset: "grainset/routing/parts_metric_reading_a_dimension_one_dataset";
    grainset_routing_parts_metric_beside_a_month_measure_from_a_child_that_lacks_region_over_a_grainset: "grainset/routing/parts_metric_beside_a_month_measure_from_a_child_that_lacks_region_over_a_grainset";
    grainset_routing_own_agg_metric_beside_a_month_measure_through_a_unionset_over_the_grainset: "grainset/routing/own_agg_metric_beside_a_month_measure_through_a_unionset_over_the_grainset";
    grainset_routing_own_agg_metric_beside_a_month_measure_through_a_joinset_over_the_grainset: "grainset/routing/own_agg_metric_beside_a_month_measure_through_a_joinset_over_the_grainset";
    grainset_routing_own_agg_metric_reading_the_time_axis_one_dataset: "grainset/routing/own_agg_metric_reading_the_time_axis_one_dataset";
    grainset_routing_own_agg_metric_reading_the_time_axis_over_a_grainset: "grainset/routing/own_agg_metric_reading_the_time_axis_over_a_grainset";
    grainset_routing_own_agg_metric_under_a_filter_read_by_a_metric_one_dataset: "grainset/routing/own_agg_metric_under_a_filter_read_by_a_metric_one_dataset";
    grainset_routing_own_agg_metric_under_a_filter_read_by_a_metric_over_a_grainset: "grainset/routing/own_agg_metric_under_a_filter_read_by_a_metric_over_a_grainset";
    grainset_routing_own_agg_metric_beside_the_same_metric_under_a_filter_over_a_grainset: "grainset/routing/own_agg_metric_beside_the_same_metric_under_a_filter_over_a_grainset";
    grainset_named_filter_web_at_month: "grainset/named_filter/web_at_month";
    grainset_named_filter_web_at_day: "grainset/named_filter/web_at_day";
    grainset_named_filter_web_total: "grainset/named_filter/web_total";
    grainset_named_filter_tv_total: "grainset/named_filter/tv_total";
    grainset_named_filter_tv_at_month: "grainset/named_filter/tv_at_month";
    grainset_named_filter_web_and_not_app: "grainset/named_filter/web_and_not_app";
    grainset_named_filter_known_at_month: "grainset/named_filter/known_at_month";
    grainset_named_filter_no_filter_at_month: "grainset/named_filter/no_filter_at_month";
    grainset_named_filter_partial_web_at_month: "grainset/named_filter_partial/web_at_month";
    grainset_named_filter_partial_no_filter_at_month: "grainset/named_filter_partial/no_filter_at_month";
    grainset_spend_named_filter_drill_across_web: "grainset/spend_named_filter/drill_across_web";
    grainset_spend_named_filter_budget_web: "grainset/spend_named_filter/budget_web";
    grainset_spend_named_filter_web_cpc2_at_month: "grainset/spend_named_filter/web_cpc2_at_month";
    grainset_spend_named_filter_web_cpc_with_its_own_filter: "grainset/spend_named_filter/web_cpc_with_its_own_filter";
    grainset_spend_named_filter_distinct_users_under_the_metrics_own_filter: "grainset/spend_named_filter/distinct_users_under_the_metrics_own_filter";
    grainset_spend_named_filter_web_cpc2_under_the_named_filter: "grainset/spend_named_filter/web_cpc2_under_the_named_filter";
    grainset_spend_named_filter_web_cpc_of_a_reads_a_metric_one_child_maps_whole: "grainset/spend_named_filter/web_cpc_of_a_reads_a_metric_one_child_maps_whole";
    grainset_spend_named_filter_web_only_cpc_on_a_dataset_sums_the_formula_over_the_channels_the_request_does_not_group_by: "grainset/spend_named_filter/web_only_cpc_on_a_dataset_sums_the_formula_over_the_channels_the_request_does_not_group_by";
    grainset_spend_named_filter_web_only_cpc_on_a_grainset_takes_the_parts_that_each_child_forms_on_its_own_rows: "grainset/spend_named_filter/web_only_cpc_on_a_grainset_takes_the_parts_that_each_child_forms_on_its_own_rows";
    grainset_spend_named_filter_channel_mix_cpc_on_a_dataset_takes_three_parts_that_each_read_two_dimensions: "grainset/spend_named_filter/channel_mix_cpc_on_a_dataset_takes_three_parts_that_each_read_two_dimensions";
    grainset_spend_named_filter_channel_mix_cpc_on_a_grainset_takes_the_same_three_parts: "grainset/spend_named_filter/channel_mix_cpc_on_a_grainset_takes_the_same_three_parts";
    grainset_spend_named_filter_nested_cpc_on_a_dataset_takes_a_key_column_for_each_case: "grainset/spend_named_filter/nested_cpc_on_a_dataset_takes_a_key_column_for_each_case";
    grainset_spend_named_filter_nested_cpc_on_a_grainset_takes_the_same_key_columns_in_each_child: "grainset/spend_named_filter/nested_cpc_on_a_grainset_takes_the_same_key_columns_in_each_child";
    grainset_spend_named_filter_web_cpc_or_users_on_a_dataset_counts_distinct_users_per_part: "grainset/spend_named_filter/web_cpc_or_users_on_a_dataset_counts_distinct_users_per_part";
    grainset_spend_named_filter_web_cpc_or_users_on_a_grainset_counts_distinct_users_per_part_inside_the_child: "grainset/spend_named_filter/web_cpc_or_users_on_a_grainset_counts_distinct_users_per_part_inside_the_child";
    grainset_spend_named_filter_web_avg_cpc_or_zero_on_a_dataset_takes_avg_cpc_per_part: "grainset/spend_named_filter/web_avg_cpc_or_zero_on_a_dataset_takes_avg_cpc_per_part";
    grainset_spend_named_filter_web_avg_cpc_or_zero_on_a_grainset_computes_avg_cpc_per_part: "grainset/spend_named_filter/web_avg_cpc_or_zero_on_a_grainset_computes_avg_cpc_per_part";
    grainset_spend_named_filter_early_cpc_on_a_dataset_reads_date_at_its_own_grain_at_month: "grainset/spend_named_filter/early_cpc_on_a_dataset_reads_date_at_its_own_grain_at_month";
    grainset_spend_named_filter_early_cpc_on_a_grainset_of_day_children_reads_date_at_its_own_grain_at_month: "grainset/spend_named_filter/early_cpc_on_a_grainset_of_day_children_reads_date_at_its_own_grain_at_month";
    grainset_spend_named_filter_web_only_cpc_through_a_joinset_over_the_grainset_takes_the_parts: "grainset/spend_named_filter/web_only_cpc_through_a_joinset_over_the_grainset_takes_the_parts";
    grainset_spend_named_filter_web_cpc2_through_a_joinset_over_the_grainset: "grainset/spend_named_filter/web_cpc2_through_a_joinset_over_the_grainset";
    grainset_spend_named_filter_web_cpc2_through_a_built_join_over_the_grainset: "grainset/spend_named_filter/web_cpc2_through_a_built_join_over_the_grainset";
    grainset_spend_named_filter_web_cpc2_per_target_reads_the_value_that_a_nested_grainset_computes: "grainset/spend_named_filter/web_cpc2_per_target_reads_the_value_that_a_nested_grainset_computes";
    grainset_spend_named_filter_web_only_cpc_per_target_in_eu_through_a_joinset_where_no_child_forms_the_part_key_under_the_filter: "grainset/spend_named_filter/web_only_cpc_per_target_in_eu_through_a_joinset_where_no_child_forms_the_part_key_under_the_filter";
    grainset_spend_named_filter_web_only_cpc_per_target_in_eu_by_region_through_a_joinset_where_no_child_forms_the_part_key_under_the_filter: "grainset/spend_named_filter/web_only_cpc_per_target_in_eu_by_region_through_a_joinset_where_no_child_forms_the_part_key_under_the_filter";
    grainset_spend_named_filter_avg_cpc_by_a_row_level_metric_on_a_dataset: "grainset/spend_named_filter/avg_cpc_by_a_row_level_metric_on_a_dataset";
    grainset_spend_named_filter_avg_cpc_by_a_row_level_metric_on_a_grainset: "grainset/spend_named_filter/avg_cpc_by_a_row_level_metric_on_a_grainset";
    grainset_spend_named_filter_web_cpc_avg_beside_a_measure_from_a_child_that_lacks_channel: "grainset/spend_named_filter/web_cpc_avg_beside_a_measure_from_a_child_that_lacks_channel";
    grainset_spend_named_filter_web_max_cpc_on_a_dataset: "grainset/spend_named_filter/web_max_cpc_on_a_dataset";
    grainset_spend_named_filter_web_max_cpc_on_a_grainset: "grainset/spend_named_filter/web_max_cpc_on_a_grainset";
    grainset_spend_named_filter_web_max_cpc_beside_a_measure_on_a_grainset: "grainset/spend_named_filter/web_max_cpc_beside_a_measure_on_a_grainset";
    grainset_spend_named_filter_web_max_cpc_of_a_on_a_dataset: "grainset/spend_named_filter/web_max_cpc_of_a_on_a_dataset";
    grainset_spend_named_filter_web_max_cpc_of_a_on_a_grainset: "grainset/spend_named_filter/web_max_cpc_of_a_on_a_grainset";
    grainset_spend_named_filter_web_label_max_cpc_on_a_dataset: "grainset/spend_named_filter/web_label_max_cpc_on_a_dataset";
    grainset_spend_named_filter_web_label_max_cpc_on_a_grainset: "grainset/spend_named_filter/web_label_max_cpc_on_a_grainset";
    grainset_spend_named_filter_web_label_max_cpc_beside_a_measure_on_a_dataset: "grainset/spend_named_filter/web_label_max_cpc_beside_a_measure_on_a_dataset";
    grainset_spend_named_filter_web_label_max_cpc_beside_a_measure_on_a_grainset: "grainset/spend_named_filter/web_label_max_cpc_beside_a_measure_on_a_grainset";
    grainset_spend_named_filter_web_label_max_cpc_of_a_on_a_dataset: "grainset/spend_named_filter/web_label_max_cpc_of_a_on_a_dataset";
    grainset_spend_named_filter_web_label_max_cpc_of_a_on_a_grainset: "grainset/spend_named_filter/web_label_max_cpc_of_a_on_a_grainset";
    grainset_spend_named_filter_web_label_max_cpc_of_a_beside_the_unfiltered_one_on_a_dataset: "grainset/spend_named_filter/web_label_max_cpc_of_a_beside_the_unfiltered_one_on_a_dataset";
    grainset_spend_named_filter_web_label_max_cpc_of_a_beside_the_unfiltered_one_on_a_grainset: "grainset/spend_named_filter/web_label_max_cpc_of_a_beside_the_unfiltered_one_on_a_grainset";
    grainset_spend_named_filter_web_label_max_cpc_of_a_through_a_joinset_over_the_grainset: "grainset/spend_named_filter/web_label_max_cpc_of_a_through_a_joinset_over_the_grainset";
    grainset_spend_named_filter_clicks_beside_avg_cpc_on_a_dataset: "grainset/spend_named_filter/clicks_beside_avg_cpc_on_a_dataset";
    grainset_spend_named_filter_clicks_beside_avg_cpc_on_a_grainset_fixes_month_once_for_every_group: "grainset/spend_named_filter/clicks_beside_avg_cpc_on_a_grainset_fixes_month_once_for_every_group";
    grainset_spend_named_filter_clicks_beside_avg_cpc_through_a_joinset_names_the_daily_child_on_the_clicks_group: "grainset/spend_named_filter/clicks_beside_avg_cpc_through_a_joinset_names_the_daily_child_on_the_clicks_group";
    grainset_spend_daily_named_filter_x_only_at_day: "grainset/spend_daily_named_filter/x_only_at_day";
    composition_union_named_filter_web_at_month: "composition/union_named_filter/web_at_month";
    composition_grainset_union_named_filter_web_at_month: "composition/grainset_union_named_filter/web_at_month";
    composition_union_spend_named_filter_x_only_at_month: "composition/union_spend_named_filter/x_only_at_month";
    composition_union_spend_named_filter_own_agg_metric_reads_the_union_again_per_day_under_a_month_demand: "composition/union_spend_named_filter/own_agg_metric_reads_the_union_again_per_day_under_a_month_demand";
    composition_union_spend_named_filter_own_agg_metric_reads_the_union_again_per_day_beside_a_measure_under_a_month_demand: "composition/union_spend_named_filter/own_agg_metric_reads_the_union_again_per_day_beside_a_measure_under_a_month_demand";
    composition_union_spend_named_filter_operand_kept_alone_reads_the_union_again_per_day_under_a_month_demand: "composition/union_spend_named_filter/operand_kept_alone_reads_the_union_again_per_day_under_a_month_demand";
    composition_union_spend_named_filter_grainset_child_reads_the_union_again_per_day_under_a_month_demand: "composition/union_spend_named_filter/grainset_child_reads_the_union_again_per_day_under_a_month_demand";
    composition_union_spend_named_filter_web_cpc2_on_a_unionset_branch_by_channel_and_campaign: "composition/union_spend_named_filter/web_cpc2_on_a_unionset_branch_by_channel_and_campaign";
    composition_union_spend_named_filter_web_cpc2_on_a_unionset_branch_by_channel_campaign_and_month: "composition/union_spend_named_filter/web_cpc2_on_a_unionset_branch_by_channel_campaign_and_month";
    composition_unique_union_join_field_extends_distinct: "composition/unique_union/join_field_extends_distinct";
    composition_unique_union_unbuilt_join_no_extension: "composition/unique_union/unbuilt_join_no_extension";
    composition_unique_union_filter_prunes_to_one_branch: "composition/unique_union/filter_prunes_to_one_branch";
    composition_unique_union_relationship_filter_extends_distinct: "composition/unique_union/relationship_filter_extends_distinct";
    composition_unique_union_keyset_extends_distinct: "composition/unique_union/keyset_extends_distinct";
    composition_delivery_nonadditive_takes_rows: "composition/delivery/nonadditive_takes_rows";
    composition_delivery_partial_aggregates_one_per_key: "composition/delivery/partial_aggregates_one_per_key";
    composition_fold_fold_stops_at_null_fill: "composition/fold/fold_stops_at_null_fill";
    composition_fold_fold_stops_regardless_of_predicate: "composition/fold/fold_stops_regardless_of_predicate";
    composition_fold_fold_through_branch_operands: "composition/fold/fold_through_branch_operands";
    composition_fold_branch_is_its_own_frame: "composition/fold/branch_is_its_own_frame";
    composition_fold_neither_side_folds_literal: "composition/fold/neither_side_folds_literal";
    composition_fold_one_side_folds_other_not: "composition/fold/one_side_folds_other_not";
    composition_fold_both_sides_same_literal: "composition/fold/both_sides_same_literal";
    composition_fold_excluded_branch_prunes_both_children: "composition/fold/excluded_branch_prunes_both_children";
    composition_fold_no_literal_keeps_branch_read: "composition/fold/no_literal_keeps_branch_read";
    composition_fold_renamed_branch_reads_first: "composition/fold/renamed_branch_reads_first";
    composition_one_child_grainset_answers_as_a_dataset: "composition/one_child/grainset_answers_as_a_dataset";
    composition_one_child_grainset_keeps_its_child_granularity: "composition/one_child/grainset_keeps_its_child_granularity";
    composition_one_child_grainset_nested_in_a_unionset: "composition/one_child/grainset_nested_in_a_unionset";
    composition_one_child_unionset_over_one_source_cost_and_clicks: "composition/one_child/unionset_over_one_source_cost_and_clicks";
    composition_one_child_unionset_over_one_source_with_a_metric: "composition/one_child/unionset_over_one_source_with_a_metric";
    composition_one_child_unionset_over_a_one_child_grainset_cost_and_clicks: "composition/one_child/unionset_over_a_one_child_grainset_cost_and_clicks";
    composition_one_child_unionset_stating_month_over_a_day_child_keeps_the_day_child: "composition/one_child/unionset_stating_month_over_a_day_child_keeps_the_day_child";
    composition_one_child_unionset_stating_day_over_a_month_child_keeps_the_month_child: "composition/one_child/unionset_stating_day_over_a_month_child_keeps_the_month_child";
    composition_one_child_unionset_stating_day_over_a_month_child_answers_a_day_request_from_the_sibling: "composition/one_child/unionset_stating_day_over_a_month_child_answers_a_day_request_from_the_sibling";
    composition_one_child_nested_member_whose_inner_grainset_cannot_serve_a_dimension_loses_to_a_dataset_member_campaign: "composition/one_child/nested_member_whose_inner_grainset_cannot_serve_a_dimension_loses_to_a_dataset_member_campaign";
    composition_one_child_nested_member_whose_inner_grainset_cannot_serve_a_dimension_loses_to_a_dataset_member_platform: "composition/one_child/nested_member_whose_inner_grainset_cannot_serve_a_dimension_loses_to_a_dataset_member_platform";
    composition_one_child_case_5_total_clicks_rolls_up_without_the_overlap: "composition/one_child/case_5_total_clicks_rolls_up_without_the_overlap";
    composition_one_child_case_6_siblings_on_disjoint_campaigns_add_up: "composition/one_child/case_6_siblings_on_disjoint_campaigns_add_up";
    composition_fold_reduced_leaves_not_disjoint: "composition/fold/reduced_leaves_not_disjoint";
    composition_grainset_constituent_constituent_answers_no_drill: "composition/grainset_constituent/constituent_answers_no_drill";
    composition_grainset_constituent_disjoint_branch_own_drill: "composition/grainset_constituent/disjoint_branch_own_drill";
    composition_grainset_constituent_disjoint_branch_each_name_from_its_rank_first_child_through_the_full_join: "composition/grainset_constituent/disjoint_branch_each_name_from_its_rank_first_child_through_the_full_join";
    composition_grainset_constituent_disjoint_branch_computes_a_metric_no_child_maps_whole: "composition/grainset_constituent/disjoint_branch_computes_a_metric_no_child_maps_whole";
    composition_grainset_constituent_disjoint_branch_null_fills_a_name_beside_a_computed_metric: "composition/grainset_constituent/disjoint_branch_null_fills_a_name_beside_a_computed_metric";
    composition_grainset_constituent_disjoint_branch_lowers_a_condition_onto_the_measures_its_grainset_serves: "composition/grainset_constituent/disjoint_branch_lowers_a_condition_onto_the_measures_its_grainset_serves";
    composition_grainset_constituent_disjoint_branch_computes_a_metric_with_its_own_agg: "composition/grainset_constituent/disjoint_branch_computes_a_metric_with_its_own_agg";
    composition_grainset_constituent_disjoint_branch_computes_a_chain_over_a_computed_metric: "composition/grainset_constituent/disjoint_branch_computes_a_chain_over_a_computed_metric";
    composition_grainset_constituent_disjoint_branch_gives_cost_and_clicks_by_name_so_the_unionset_computes_cpc_not_the_child_that_maps_it_whole: "composition/grainset_constituent/disjoint_branch_gives_cost_and_clicks_by_name_so_the_unionset_computes_cpc_not_the_child_that_maps_it_whole";
    composition_grainset_constituent_disjoint_branch_computes_an_own_agg_metric_reading_a_dimension: "composition/grainset_constituent/disjoint_branch_computes_an_own_agg_metric_reading_a_dimension";
    composition_grainset_constituent_disjoint_branch_reads_the_time_axis_per_day_for_an_own_agg_metric: "composition/grainset_constituent/disjoint_branch_reads_the_time_axis_per_day_for_an_own_agg_metric";
    composition_grainset_constituent_disjoint_branch_reads_the_time_axis_per_day_for_an_own_agg_metric_at_month: "composition/grainset_constituent/disjoint_branch_reads_the_time_axis_per_day_for_an_own_agg_metric_at_month";
    composition_grainset_constituent_disjoint_branch_reads_the_time_axis_per_day_for_an_own_agg_metric_under_a_dimension_filter: "composition/grainset_constituent/disjoint_branch_reads_the_time_axis_per_day_for_an_own_agg_metric_under_a_dimension_filter";
    composition_grainset_constituent_disjoint_branch_reads_the_time_axis_per_day_for_an_own_agg_metric_in_a_null_group: "composition/grainset_constituent/disjoint_branch_reads_the_time_axis_per_day_for_an_own_agg_metric_in_a_null_group";
    composition_grainset_constituent_disjoint_branch_finishes_a_linked_own_agg_metric_per_day_for_an_own_agg_metric_reading_the_time_axis: "composition/grainset_constituent/disjoint_branch_finishes_a_linked_own_agg_metric_per_day_for_an_own_agg_metric_reading_the_time_axis";
    composition_grainset_constituent_grainset_child_branch_reads_the_time_axis_per_day_for_an_own_agg_metric: "composition/grainset_constituent/grainset_child_branch_reads_the_time_axis_per_day_for_an_own_agg_metric";
    composition_grainset_constituent_grainset_child_branch_reads_the_time_axis_per_day_for_an_own_agg_metric_at_month: "composition/grainset_constituent/grainset_child_branch_reads_the_time_axis_per_day_for_an_own_agg_metric_at_month";
    composition_grainset_constituent_grainset_child_branch_reads_the_time_axis_per_day_for_an_own_agg_metric_under_a_dimension_filter: "composition/grainset_constituent/grainset_child_branch_reads_the_time_axis_per_day_for_an_own_agg_metric_under_a_dimension_filter";
    composition_grainset_constituent_grainset_child_branch_reads_the_time_axis_per_day_for_an_own_agg_metric_in_a_null_group: "composition/grainset_constituent/grainset_child_branch_reads_the_time_axis_per_day_for_an_own_agg_metric_in_a_null_group";
    composition_grainset_constituent_grainset_child_branch_finishes_a_linked_own_agg_metric_per_day_for_an_own_agg_metric_reading_the_time_axis: "composition/grainset_constituent/grainset_child_branch_finishes_a_linked_own_agg_metric_per_day_for_an_own_agg_metric_reading_the_time_axis";
    composition_grainset_constituent_disjoint_branch_reads_the_union_again_at_step_1s_grouping_for_an_own_agg_metric: "composition/grainset_constituent/disjoint_branch_reads_the_union_again_at_step_1s_grouping_for_an_own_agg_metric";
    composition_grainset_constituent_disjoint_branch_joins_the_union_read_again_at_step_1s_grouping_beside_a_measure: "composition/grainset_constituent/disjoint_branch_joins_the_union_read_again_at_step_1s_grouping_beside_a_measure";
    composition_grainset_constituent_disjoint_branch_filters_cost_by_the_channel_total_beside_a_union_read_again_at_step_1s_grouping: "composition/grainset_constituent/disjoint_branch_filters_cost_by_the_channel_total_beside_a_union_read_again_at_step_1s_grouping";
    composition_grainset_constituent_disjoint_branch_reads_a_measure_from_the_drill_across_of_the_union_read_again_at_step_1s_grouping: "composition/grainset_constituent/disjoint_branch_reads_a_measure_from_the_drill_across_of_the_union_read_again_at_step_1s_grouping";
    composition_grainset_constituent_disjoint_branch_reads_the_union_again_once_per_own_agg_metrics_step_1_grouping: "composition/grainset_constituent/disjoint_branch_reads_the_union_again_once_per_own_agg_metrics_step_1_grouping";
    composition_grainset_constituent_disjoint_branch_reads_the_union_again_once_per_step_1_grouping_for_a_formula_over_an_own_agg_metric: "composition/grainset_constituent/disjoint_branch_reads_the_union_again_once_per_step_1_grouping_for_a_formula_over_an_own_agg_metric";
    composition_grainset_constituent_disjoint_branch_filters_by_a_formula_over_a_metric_read_again_at_one_of_two_step_1_groupings: "composition/grainset_constituent/disjoint_branch_filters_by_a_formula_over_a_metric_read_again_at_one_of_two_step_1_groupings";
    composition_grainset_constituent_disjoint_branch_divides_its_own_aggregate_by_a_metric_read_again_at_one_of_two_step_1_groupings: "composition/grainset_constituent/disjoint_branch_divides_its_own_aggregate_by_a_metric_read_again_at_one_of_two_step_1_groupings";
    composition_grainset_constituent_disjoint_branch_reads_the_union_again_once_per_step_1_grouping_for_a_formula_over_two_own_agg_metrics: "composition/grainset_constituent/disjoint_branch_reads_the_union_again_once_per_step_1_grouping_for_a_formula_over_two_own_agg_metrics";
    composition_grainset_constituent_disjoint_branch_keeps_a_channel_that_one_of_two_reads_again_at_step_1s_grouping_holds: "composition/grainset_constituent/disjoint_branch_keeps_a_channel_that_one_of_two_reads_again_at_step_1s_grouping_holds";
    composition_grainset_constituent_disjoint_branch_joins_two_reads_again_at_step_1s_grouping_left_onto_the_frames_own_read: "composition/grainset_constituent/disjoint_branch_joins_two_reads_again_at_step_1s_grouping_left_onto_the_frames_own_read";
    composition_grainset_constituent_disjoint_branch_joins_a_later_subtree_left_after_two_reads_again_join_full: "composition/grainset_constituent/disjoint_branch_joins_a_later_subtree_left_after_two_reads_again_join_full";
    composition_grainset_constituent_disjoint_branch_filters_by_an_own_agg_metric_the_union_reads_again_at_step_1s_grouping: "composition/grainset_constituent/disjoint_branch_filters_by_an_own_agg_metric_the_union_reads_again_at_step_1s_grouping";
    composition_grainset_constituent_disjoint_branch_filters_by_an_own_agg_metric_its_grouping_finishes_keeping_the_null_campaign: "composition/grainset_constituent/disjoint_branch_filters_by_an_own_agg_metric_its_grouping_finishes_keeping_the_null_campaign";
    composition_grainset_constituent_disjoint_branch_filters_by_a_metric_read_again_beside_the_frames_own_aggregate: "composition/grainset_constituent/disjoint_branch_filters_by_a_metric_read_again_beside_the_frames_own_aggregate";
    composition_grainset_constituent_disjoint_branch_reads_the_union_again_for_the_metric_a_formula_links: "composition/grainset_constituent/disjoint_branch_reads_the_union_again_for_the_metric_a_formula_links";
    composition_grainset_constituent_disjoint_branch_filters_by_a_formula_over_the_metric_it_links_read_again: "composition/grainset_constituent/disjoint_branch_filters_by_a_formula_over_the_metric_it_links_read_again";
    composition_grainset_constituent_disjoint_branch_divides_the_frames_own_aggregate_by_a_metric_read_again: "composition/grainset_constituent/disjoint_branch_divides_the_frames_own_aggregate_by_a_metric_read_again";
    composition_grainset_constituent_disjoint_branch_divides_the_frames_own_aggregate_by_a_metric_read_again_beside_cpc: "composition/grainset_constituent/disjoint_branch_divides_the_frames_own_aggregate_by_a_metric_read_again_beside_cpc";
    composition_grainset_constituent_disjoint_branch_filters_by_a_formula_over_the_frames_own_aggregate_and_a_metric_read_again: "composition/grainset_constituent/disjoint_branch_filters_by_a_formula_over_the_frames_own_aggregate_and_a_metric_read_again";
    composition_grainset_constituent_disjoint_branch_reads_the_union_again_for_an_own_agg_metrics_link_at_the_readers_step_1_grouping: "composition/grainset_constituent/disjoint_branch_reads_the_union_again_for_an_own_agg_metrics_link_at_the_readers_step_1_grouping";
    composition_grainset_constituent_disjoint_branch_reads_the_union_again_for_an_own_agg_metrics_link_at_the_readers_step_1_grouping_beside_cpc: "composition/grainset_constituent/disjoint_branch_reads_the_union_again_for_an_own_agg_metrics_link_at_the_readers_step_1_grouping_beside_cpc";
    composition_grainset_constituent_disjoint_branch_reads_its_own_aggregate_beside_an_own_agg_metrics_link_at_the_readers_step_1_grouping: "composition/grainset_constituent/disjoint_branch_reads_its_own_aggregate_beside_an_own_agg_metrics_link_at_the_readers_step_1_grouping";
    composition_grainset_constituent_disjoint_branch_reads_its_own_aggregate_beside_an_own_agg_metrics_link_at_the_readers_step_1_grouping_beside_cpc: "composition/grainset_constituent/disjoint_branch_reads_its_own_aggregate_beside_an_own_agg_metrics_link_at_the_readers_step_1_grouping_beside_cpc";
    composition_grainset_constituent_disjoint_branch_reads_the_union_again_at_the_part_key_for_a_parts_metric: "composition/grainset_constituent/disjoint_branch_reads_the_union_again_at_the_part_key_for_a_parts_metric";
    composition_grainset_constituent_disjoint_branch_read_alone_reads_the_union_again_at_the_part_key_under_a_filter: "composition/grainset_constituent/disjoint_branch_read_alone_reads_the_union_again_at_the_part_key_under_a_filter";
    composition_grainset_constituent_disjoint_branch_reads_the_union_again_at_the_part_key_under_a_dimension_filter: "composition/grainset_constituent/disjoint_branch_reads_the_union_again_at_the_part_key_under_a_dimension_filter";
    composition_grainset_constituent_disjoint_branch_counts_distinct_users_under_a_metrics_own_filter: "composition/grainset_constituent/disjoint_branch_counts_distinct_users_under_a_metrics_own_filter";
    composition_grainset_constituent_disjoint_branch_applies_a_named_filter_in_each_child_beside_a_computed_metric: "composition/grainset_constituent/disjoint_branch_applies_a_named_filter_in_each_child_beside_a_computed_metric";
    composition_grainset_constituent_disjoint_branch_read_alone_under_a_filter_the_other_branch_null_fills: "composition/grainset_constituent/disjoint_branch_read_alone_under_a_filter_the_other_branch_null_fills";
    composition_grainset_constituent_disjoint_branch_read_alone_needs_no_dimension_that_folds_to_a_literal: "composition/grainset_constituent/disjoint_branch_read_alone_needs_no_dimension_that_folds_to_a_literal";
    composition_grainset_constituent_overlapping_branch_that_computes_a_metric_takes_partial_aggregates: "composition/grainset_constituent/overlapping_branch_that_computes_a_metric_takes_partial_aggregates";
    composition_grainset_constituent_outside_mask_null_fill: "composition/grainset_constituent/outside_mask_null_fill";
    composition_grainset_constituent_outside_mask_no_join: "composition/grainset_constituent/outside_mask_no_join";
    composition_grainset_constituent_branch_without_a_keyset_serves_a_measure_one_child_maps: "composition/grainset_constituent/branch_without_a_keyset_serves_a_measure_one_child_maps";
    composition_grainset_constituent_branch_without_a_keyset_is_pruned_for_a_metric_over_rows_of_two_children: "composition/grainset_constituent/branch_without_a_keyset_is_pruned_for_a_metric_over_rows_of_two_children";
    composition_grainset_constituent_operand_kept_alone_computes_an_own_agg_metric_reading_a_dimension: "composition/grainset_constituent/operand_kept_alone_computes_an_own_agg_metric_reading_a_dimension";
    composition_grainset_constituent_operand_kept_alone_reads_the_time_axis_per_day_for_an_own_agg_metric: "composition/grainset_constituent/operand_kept_alone_reads_the_time_axis_per_day_for_an_own_agg_metric";
    composition_grainset_constituent_operand_kept_alone_reads_the_time_axis_per_day_for_an_own_agg_metric_at_month: "composition/grainset_constituent/operand_kept_alone_reads_the_time_axis_per_day_for_an_own_agg_metric_at_month";
    composition_grainset_constituent_operand_kept_alone_reads_the_time_axis_per_day_for_an_own_agg_metric_under_a_dimension_filter: "composition/grainset_constituent/operand_kept_alone_reads_the_time_axis_per_day_for_an_own_agg_metric_under_a_dimension_filter";
    composition_grainset_constituent_operand_kept_alone_reads_the_time_axis_per_day_for_an_own_agg_metric_in_a_null_group: "composition/grainset_constituent/operand_kept_alone_reads_the_time_axis_per_day_for_an_own_agg_metric_in_a_null_group";
    composition_grainset_constituent_operand_kept_alone_finishes_a_linked_own_agg_metric_per_day_for_an_own_agg_metric_reading_the_time_axis: "composition/grainset_constituent/operand_kept_alone_finishes_a_linked_own_agg_metric_per_day_for_an_own_agg_metric_reading_the_time_axis";
    composition_grainset_constituent_operand_kept_alone_counts_distinct_users_under_a_metrics_own_filter: "composition/grainset_constituent/operand_kept_alone_counts_distinct_users_under_a_metrics_own_filter";
    composition_grainset_constituent_operand_kept_alone_applies_a_named_filter_in_each_child_beside_a_computed_metric: "composition/grainset_constituent/operand_kept_alone_applies_a_named_filter_in_each_child_beside_a_computed_metric";
    composition_grainset_constituent_operand_kept_alone_takes_a_late_own_agg_metric_in_place_beside_a_measure: "composition/grainset_constituent/operand_kept_alone_takes_a_late_own_agg_metric_in_place_beside_a_measure";
    composition_grainset_constituent_operand_kept_alone_computes_two_parts_metrics_an_own_agg_metric_and_a_measure_in_one_request: "composition/grainset_constituent/operand_kept_alone_computes_two_parts_metrics_an_own_agg_metric_and_a_measure_in_one_request";
    composition_grainset_constituent_operand_kept_alone_computes_a_parts_metric_by_two_dimensions_with_null_values: "composition/grainset_constituent/operand_kept_alone_computes_a_parts_metric_by_two_dimensions_with_null_values";
    composition_grainset_constituent_operand_kept_alone_groups_an_own_agg_metric_by_a_row_level_metric: "composition/grainset_constituent/operand_kept_alone_groups_an_own_agg_metric_by_a_row_level_metric";
    composition_grainset_constituent_operand_kept_alone_computes_an_own_agg_metric_over_an_own_agg_metric_at_its_formula_grain_beside_a_measure: "composition/grainset_constituent/operand_kept_alone_computes_an_own_agg_metric_over_an_own_agg_metric_at_its_formula_grain_beside_a_measure";
    composition_grainset_constituent_operand_kept_alone_routes_each_formula_group_on_its_own_beside_a_measure_from_a_child_that_lacks_gender: "composition/grainset_constituent/operand_kept_alone_routes_each_formula_group_on_its_own_beside_a_measure_from_a_child_that_lacks_gender";
    composition_grainset_constituent_operand_kept_alone_holds_a_unionset_read_alone_under_a_filter_the_other_branch_null_fills: "composition/grainset_constituent/operand_kept_alone_holds_a_unionset_read_alone_under_a_filter_the_other_branch_null_fills";
    composition_grainset_constituent_operand_kept_alone_holds_a_unionset_read_alone_with_no_dimension_that_folds_to_a_literal: "composition/grainset_constituent/operand_kept_alone_holds_a_unionset_read_alone_with_no_dimension_that_folds_to_a_literal";
    composition_grainset_constituent_grainset_child_holds_a_unionset_whose_one_grainset_branch_drills_across_with_no_literal: "composition/grainset_constituent/grainset_child_holds_a_unionset_whose_one_grainset_branch_drills_across_with_no_literal";
    composition_grainset_constituent_grainset_child_holds_a_unionset_beside_a_dataset_under_a_named_filter_both_map: "composition/grainset_constituent/grainset_child_holds_a_unionset_beside_a_dataset_under_a_named_filter_both_map";
    composition_grainset_constituent_grainset_child_holds_a_unionset_whose_branch_groups_by_a_part_keys_dimension_under_partial_aggregates: "composition/grainset_constituent/grainset_child_holds_a_unionset_whose_branch_groups_by_a_part_keys_dimension_under_partial_aggregates";
    composition_grainset_constituent_grainset_child_holds_a_unionset_whose_branch_groups_by_a_part_keys_dimension_under_a_dimension_filter: "composition/grainset_constituent/grainset_child_holds_a_unionset_whose_branch_groups_by_a_part_keys_dimension_under_a_dimension_filter";
    composition_grainset_constituent_operand_kept_alone_holds_a_grainset_whose_unionset_child_groups_by_a_part_keys_dimension: "composition/grainset_constituent/operand_kept_alone_holds_a_grainset_whose_unionset_child_groups_by_a_part_keys_dimension";
    composition_grainset_constituent_grainset_child_holds_a_unionset_whose_linked_reader_takes_the_grainsets_metric_filter_and_keeps_every_group: "composition/grainset_constituent/grainset_child_holds_a_unionset_whose_linked_reader_takes_the_grainsets_metric_filter_and_keeps_every_group";
    composition_grainset_constituent_operand_kept_alone_holds_a_grainset_whose_unionset_child_reads_a_linked_metric_plain_and_under_the_grainsets_metric_filter: "composition/grainset_constituent/operand_kept_alone_holds_a_grainset_whose_unionset_child_reads_a_linked_metric_plain_and_under_the_grainsets_metric_filter";
    composition_grainset_constituent_operand_kept_alone_holds_a_grainset_whose_unionset_child_maps_whole_a_linked_reader_beside_the_grainsets_metric_filter_and_the_linked_reader_plain: "composition/grainset_constituent/operand_kept_alone_holds_a_grainset_whose_unionset_child_maps_whole_a_linked_reader_beside_the_grainsets_metric_filter_and_the_linked_reader_plain";
    composition_grainset_constituent_grainset_child_holds_a_unionset_whose_linked_reader_takes_the_grainsets_metric_filter_under_a_predicate: "composition/grainset_constituent/grainset_child_holds_a_unionset_whose_linked_reader_takes_the_grainsets_metric_filter_under_a_predicate";
    composition_grainset_constituent_grainset_child_holds_a_unionset_that_reads_a_linked_metric_plain_and_under_the_grainsets_metric_filter: "composition/grainset_constituent/grainset_child_holds_a_unionset_that_reads_a_linked_metric_plain_and_under_the_grainsets_metric_filter";
    composition_grainset_constituent_grainset_child_holds_a_unionset_that_reads_a_linked_metric_and_the_metric_that_links_it_plain_beside_the_grainsets_metric_filter: "composition/grainset_constituent/grainset_child_holds_a_unionset_that_reads_a_linked_metric_and_the_metric_that_links_it_plain_beside_the_grainsets_metric_filter";
    composition_grainset_constituent_grainset_child_holds_a_unionset_that_maps_whole_a_linked_reader_beside_the_grainsets_metric_filter_and_the_linked_reader_plain: "composition/grainset_constituent/grainset_child_holds_a_unionset_that_maps_whole_a_linked_reader_beside_the_grainsets_metric_filter_and_the_linked_reader_plain";
    composition_grainset_constituent_grainset_child_holds_a_unionset_that_maps_whole_a_linked_reader_beside_the_grainsets_metric_filter_the_linked_reader_plain_and_its_link: "composition/grainset_constituent/grainset_child_holds_a_unionset_that_maps_whole_a_linked_reader_beside_the_grainsets_metric_filter_the_linked_reader_plain_and_its_link";
    composition_grainset_constituent_grainset_child_holds_a_unionset_that_maps_whole_a_linked_reader_under_its_own_filter_and_keeps_every_group: "composition/grainset_constituent/grainset_child_holds_a_unionset_that_maps_whole_a_linked_reader_under_its_own_filter_and_keeps_every_group";
    composition_grainset_constituent_grainset_child_holds_a_unionset_that_reads_again_at_step_1s_grouping_beside_an_own_agg_metric_the_grainset_computes: "composition/grainset_constituent/grainset_child_holds_a_unionset_that_reads_again_at_step_1s_grouping_beside_an_own_agg_metric_the_grainset_computes";
    composition_grainset_constituent_grainset_child_holds_a_unionset_that_reads_again_for_the_metric_a_formula_links_beside_metrics_over_its_names: "composition/grainset_constituent/grainset_child_holds_a_unionset_that_reads_again_for_the_metric_a_formula_links_beside_metrics_over_its_names";
    composition_grainset_constituent_grainset_child_holds_a_unionset_that_reads_again_for_an_own_agg_metrics_link_at_the_readers_step_1_grouping: "composition/grainset_constituent/grainset_child_holds_a_unionset_that_reads_again_for_an_own_agg_metrics_link_at_the_readers_step_1_grouping";
    composition_grainset_constituent_grainset_child_holds_a_unionset_that_reads_its_own_aggregate_beside_an_own_agg_metrics_link_at_the_readers_step_1_grouping: "composition/grainset_constituent/grainset_child_holds_a_unionset_that_reads_its_own_aggregate_beside_an_own_agg_metrics_link_at_the_readers_step_1_grouping";
    composition_grainset_constituent_grainset_child_holds_a_unionset_that_reads_again_at_the_part_key_beside_a_parts_metric_the_grainset_computes: "composition/grainset_constituent/grainset_child_holds_a_unionset_that_reads_again_at_the_part_key_beside_a_parts_metric_the_grainset_computes";
    composition_grainset_constituent_operand_kept_alone_holds_a_unionset_that_reads_again_at_step_1s_grouping_for_an_own_agg_metric: "composition/grainset_constituent/operand_kept_alone_holds_a_unionset_that_reads_again_at_step_1s_grouping_for_an_own_agg_metric";
    composition_grainset_constituent_operand_kept_alone_holds_a_unionset_that_filters_by_an_own_agg_metric_read_again_at_step_1s_grouping: "composition/grainset_constituent/operand_kept_alone_holds_a_unionset_that_filters_by_an_own_agg_metric_read_again_at_step_1s_grouping";
    composition_grainset_constituent_operand_kept_alone_holds_a_unionset_that_reads_again_once_per_step_1_grouping_for_a_formula_over_an_own_agg_metric: "composition/grainset_constituent/operand_kept_alone_holds_a_unionset_that_reads_again_once_per_step_1_grouping_for_a_formula_over_an_own_agg_metric";
    composition_grainset_constituent_operand_kept_alone_holds_a_unionset_that_divides_its_own_aggregate_by_a_metric_read_again: "composition/grainset_constituent/operand_kept_alone_holds_a_unionset_that_divides_its_own_aggregate_by_a_metric_read_again";
    composition_grainset_constituent_operand_kept_alone_holds_a_unionset_that_reads_again_for_an_own_agg_metrics_link_at_the_readers_step_1_grouping: "composition/grainset_constituent/operand_kept_alone_holds_a_unionset_that_reads_again_for_an_own_agg_metrics_link_at_the_readers_step_1_grouping";
    composition_grainset_constituent_operand_kept_alone_holds_a_unionset_that_reads_its_own_aggregate_beside_an_own_agg_metrics_link_at_the_readers_step_1_grouping: "composition/grainset_constituent/operand_kept_alone_holds_a_unionset_that_reads_its_own_aggregate_beside_an_own_agg_metrics_link_at_the_readers_step_1_grouping";
    composition_grainset_constituent_operand_kept_alone_holds_a_unionset_that_reads_again_at_the_part_key_for_a_parts_metric: "composition/grainset_constituent/operand_kept_alone_holds_a_unionset_that_reads_again_at_the_part_key_for_a_parts_metric";
    composition_grainset_constituent_operand_kept_alone_holds_a_unionset_that_prunes_two_of_three_branches_and_reads_the_grainset_again_at_the_part_key: "composition/grainset_constituent/operand_kept_alone_holds_a_unionset_that_prunes_two_of_three_branches_and_reads_the_grainset_again_at_the_part_key";
    composition_time_axis_root_grainset_to_union_child: "composition/time_axis/root_grainset_to_union_child";
    composition_time_axis_latest_day_by_account: "composition/time_axis/latest_day_by_account";
    composition_time_axis_kept_operand_marker_in_join: "composition/time_axis/kept_operand_marker_in_join";
    composition_time_axis_plain_sum_no_marker: "composition/time_axis/plain_sum_no_marker";
    composition_time_axis_earliest_day_by_account: "composition/time_axis/earliest_day_by_account";
    composition_time_axis_condition_and_rank_combine: "composition/time_axis/condition_and_rank_combine";
    composition_time_axis_partitioned_condition_and_rank: "composition/time_axis/partitioned_condition_and_rank";
    composition_time_axis_rank_reads_dropped_row: "composition/time_axis/rank_reads_dropped_row";
    composition_time_axis_union_month_day_values: "composition/time_axis/union_month_day_values";
    composition_time_axis_empty_branch_no_rows: "composition/time_axis/empty_branch_no_rows";
    composition_time_axis_null_date_own_group: "composition/time_axis/null_date_own_group";
    composition_time_axis_depth_three_stack: "composition/time_axis/depth_three_stack";
    composition_time_axis_literal_branches_prune_sibling: "composition/time_axis/literal_branches_prune_sibling";
    composition_time_axis_equated_dates_month_grain: "composition/time_axis/equated_dates_month_grain";
    composition_time_axis_year_coarser_than_month: "composition/time_axis/year_coarser_than_month";
    composition_time_axis_constituent_declares_own_year: "composition/time_axis/constituent_declares_own_year";
    composition_time_axis_plain_sum_answers_alone: "composition/time_axis/plain_sum_answers_alone";
    composition_time_axis_strategy_before_constituent_pick: "composition/time_axis/strategy_before_constituent_pick";
    composition_time_axis_target_once_per_bucket: "composition/time_axis/target_once_per_bucket";
    composition_time_axis_picked_child_sets_axis: "composition/time_axis/picked_child_sets_axis";
    composition_time_axis_sibling_branch_no_target: "composition/time_axis/sibling_branch_no_target";
    composition_time_axis_plain_sum_unchanged: "composition/time_axis/plain_sum_unchanged";
    composition_time_axis_day_values_single_target: "composition/time_axis/day_values_single_target";
    composition_time_axis_one_target_per_group: "composition/time_axis/one_target_per_group";
    composition_time_axis_left_operand_sets_month: "composition/time_axis/left_operand_sets_month";
    composition_time_axis_coalesced_date_no_day_axis: "composition/time_axis/coalesced_date_no_day_axis";
    composition_time_axis_root_union_derives_day: "composition/time_axis/root_union_derives_day";
    composition_time_axis_disjoint_branches_own_markers: "composition/time_axis/disjoint_branches_own_markers";
    composition_time_axis_no_partition_latest_instant: "composition/time_axis/no_partition_latest_instant";
}

fn summed_by(
    reader: impl Into<Reader>,
    dataset: &str,
    by: (&str, DataType),
    (total, summed, data_type): (&str, &str, DataType),
) -> Result<PlanNode, IrErrorKind> {
    let group = by.0;
    let read = scanned(reader, &[by, (summed, data_type)], None)?;
    let totals = vec![(name(total), aggregate("sum", vec![leaf(&[summed])], None))];
    let totals = AggregateNode::new(read, vec![(name(group), leaf(&[group]))], totals)?;
    Ok(
        PlanNode::from(totals).with_trace([SemAnnotation::SemanticSource {
            semantic_source: dataset.into(),
            source_role: SourceRole::Dataset,
            originators: Vec::new(),
            physical_source: None,
            cardinality: None,
            delivery: None,
            answered_by: None,
            pruned: Vec::new(),
        }]),
    )
}

fn table(_: &SemanticManifest, _: &Request, _: &RequestContext) -> Planned {
    let reader = TableScan::named_table(["", "warehouse", "orders"].map(String::from), None);
    let revenue = ("revenue", "amount", DataType::Double);
    let totals = summed_by(reader, "orders", ("region", DataType::String), revenue);
    let last_first = SortKey {
        expr: leaf(&["region"]),
        dir: SortDir::Desc {
            nulls: NullOrdering::First,
        },
    };
    let sorted = SortNode::new(totals.expect("the totals"), vec![last_first]);
    Ok((plan(sorted.expect("the sort")), Vec::new()))
}

fn files(_: &SemanticManifest, _: &Request, _: &RequestContext) -> Planned {
    let object = "s3://semstrait-e2e/events/part-0.parquet".to_owned();
    let folder = "s3://semstrait-e2e/events/";
    let reader = FilesScan::new(Some(FileFormat::Parquet), None, folder, vec![object]);
    let clicks = ("clicks", "clicks", DataType::BigInt);
    let totals = summed_by(reader, "events", ("channel", DataType::String), clicks);
    Ok((plan(totals.expect("the totals")), Vec::new()))
}

#[derive(Debug)]
struct Refusing;

impl Strategy<DatasetArm> for Refusing {
    fn decide(
        &self,
        _: &dyn ReadModel,
        _: &DatasetArm,
        input: &StrategyInput,
    ) -> Result<Decision, PlanError> {
        Err(PlanError::NoSignature {
            path: input.path.clone(),
            semantic_source: input.semantic_source.clone(),
            semantics: input.names.first().expect("a selected name").name.clone(),
        })
    }
}

fn refusing(manifest: &SemanticManifest, request: &Request, context: &RequestContext) -> Planned {
    let planner = Planner::new().with_dataset_strategy(Box::new(Refusing));
    planner.plan(manifest, request, context)
}

#[test]
fn a_variant_that_is_not_rendered_fails_its_case() -> Result<(), Box<dyn Error>> {
    let refusal: Refusal = serde_yaml::from_str("{stage: plan, variant: EmptySelect}")?;
    let refused = [Diagnostic::new(PlanError::EmptySelect)];
    let failure = refusal
        .compared("harness/mechanics/plan_refusal", &refused)
        .expect_err("no rendering");
    assert!(failure.to_string().contains("no rendering"), "{failure}");
    Ok(())
}

#[test]
fn a_row_of_another_width_fails_its_case() -> Result<(), Box<dyn Error>> {
    let case: Case = serde_yaml::from_str(
        "{request: {select: [region, revenue]}, expect: {rows: [[region, revenue], [us, 20.0, 99]]}}",
    )?;
    let fields = Fields::from(vec![
        Field::new("region", ArrowType::Utf8, true),
        Field::new("revenue", ArrowType::Float64, true),
    ]);
    let failure = case.rows(&fields).err().ok_or("a row of three values")?;
    assert!(failure.to_string().contains("3 values"), "{failure}");
    Ok(())
}

fn rooted(relation: RelType) -> Plan {
    let root = RelRoot {
        input: Some(Rel {
            rel_type: Some(relation),
        }),
        names: Vec::new(),
    };
    Plan {
        version: Some(version_with_producer("semstrait")),
        relations: vec![PlanRel {
            rel_type: Some(plan_rel::RelType::Root(root)),
        }],
        ..Plan::default()
    }
}

#[test]
#[should_panic(expected = "a relation that the wire check reads")]
fn the_wire_check_fails_a_relation_that_it_does_not_read() {
    wire("a hash join", &rooted(RelType::HashJoin(Box::default())));
}

#[test]
fn the_wire_check_reads_both_inputs_of_a_join() {
    let read = |trace: bool| {
        let advanced_extension = trace.then(AdvancedExtension::default);
        let read = ReadRel {
            advanced_extension,
            ..ReadRel::default()
        };
        Some(Box::new(Rel {
            rel_type: Some(RelType::Read(Box::new(read))),
        }))
    };
    for trace in [false, true] {
        let join = JoinRel {
            left: read(false),
            right: read(trace),
            ..JoinRel::default()
        };
        let cross = CrossRel {
            left: read(false),
            right: read(trace),
            ..CrossRel::default()
        };
        for relation in [
            RelType::Join(Box::new(join)),
            RelType::Cross(Box::new(cross)),
        ] {
            let checked = std::panic::catch_unwind(|| wire("a join", &rooted(relation)));
            assert_eq!(checked.is_err(), trace, "trace on the right input fails");
        }
    }
}

#[test]
fn the_case_builder_maps_every_request_part() -> Result<(), Box<dyn Error>> {
    let case: Case = serde_yaml::from_str(
        "
         request:
           from: orders
           select:
             - region
             - {dimension: day, granularity: month}
             - {dimension: channel}
             - {measure: revenue}
             - {metric: margin}
           where: {gt: [revenue, 10]}
           filters: [paid]
           order_by: [{name: region, direction: asc}, {name: revenue, direction: desc}]
           limit: 10
           offset: 5
         context: {timezone: Europe/Berlin, week_start: sunday}
         expect: {rows: [[region, day, channel, revenue, margin]]}",
    )?;
    let predicate = parse_expr::<SemanticLeaf>(&serde_yaml::from_str("{gt: [revenue, 10]}")?)?;
    let select = vec![
        Selection::Field("region".into()),
        Selection::Dimension {
            name: "day".into(),
            modifier: Some(DimensionModifier::Granularity(Granularity::Month)),
        },
        Selection::Dimension {
            name: "channel".into(),
            modifier: None,
        },
        Selection::Measure {
            name: "revenue".into(),
        },
        Selection::Metric {
            name: "margin".into(),
        },
    ];
    let order_by = vec![
        OrderKey::new("region", Direction::Asc),
        OrderKey::new("revenue", Direction::Desc),
    ];
    let request = Request::new(select)
        .with_from("orders")
        .with_predicate(predicate)
        .with_filters(vec!["paid".into()])
        .with_order_by(order_by)
        .with_limit(10)
        .with_offset(5);
    let context = RequestContext::default()
        .with_timezone(Timezone::new("Europe/Berlin")?)
        .with_week_start(Weekday::Sunday);

    assert_eq!(case.request()?, request);
    assert_eq!(case.context()?, context);
    assert_eq!(case.keys()?, [0, 3]);

    let defaults: Case = serde_yaml::from_str("request: {select: [region]}\nexpect: {rows: []}")?;
    assert_eq!(
        defaults.request()?,
        Request::new(vec![Selection::Field("region".into())])
    );
    assert_eq!(defaults.context()?, RequestContext::default());
    Ok(())
}

#[tokio::test]
async fn compiles_clean_against_its_declaration() -> Result<(), Box<dyn Error>> {
    runner::compiled("harness/harness").await?;
    Ok(())
}

#[tokio::test]
async fn the_commerce_model_compiles_clean_against_its_declaration() -> Result<(), Box<dyn Error>> {
    runner::compiled("models/commerce").await?;
    Ok(())
}

#[tokio::test]
async fn acme_v3_compiles_clean_against_its_declaration() -> Result<(), Box<dyn Error>> {
    runner::corpus("acme_eu_ad_platform_v3").await?;
    Ok(())
}

#[tokio::test]
async fn comprehensive_ecommerce_compiles_clean_against_its_declaration(
) -> Result<(), Box<dyn Error>> {
    runner::corpus("comprehensive_ecommerce").await?;
    Ok(())
}

#[tokio::test]
async fn e2e_full_coverage_compiles_clean_against_its_declaration() -> Result<(), Box<dyn Error>> {
    runner::corpus("e2e_full_coverage").await?;
    Ok(())
}

#[test]
fn the_listing_holds_each_case_file_and_no_other() -> Result<(), Box<dyn Error>> {
    runner::listing(LISTED)?;
    let (first, rest) = LISTED.split_first().ok_or("a listed case")?;
    let unlisted = runner::listing(rest)
        .err()
        .ok_or("a case file that no runner lists")?;
    assert!(unlisted.to_string().contains(first), "{unlisted}");
    let absent = [LISTED, &["harness/absent"]].concat();
    let missing = runner::listing(&absent)
        .err()
        .ok_or("a listed case with no file")?;
    assert!(missing.to_string().contains("harness/absent"), "{missing}");
    Ok(())
}

#[tokio::test]
async fn chains_compile_plan_and_the_producer() -> Result<(), Box<dyn Error>> {
    let (manifest, _) = runner::compiled("harness/sales").await?;
    let request = Request::new(vec![
        Selection::Field("channel".into()),
        Selection::Field("region_label".into()),
    ]);
    let planned = runner::planned(&manifest, &request, &RequestContext::default());
    runner::validated("chains_compile_plan_and_the_producer", planned)?;
    Ok(())
}

#[tokio::test]
async fn the_type_check_fails_a_column_declared_with_another_type() -> Result<(), Box<dyn Error>> {
    let (manifest, _) = runner::compiled("harness/harness").await?;
    let declarations = [
        (
            "amount",
            r#"
tables:
  - namespace: warehouse
    name: orders
    columns:
      - {name: region, type: string}
      - {name: amount, type: bigint}
"#,
        ),
        (
            "clicks",
            r#"
objects:
  - uri: s3://semstrait-e2e/events/part-0.parquet
    columns:
      - {name: channel, type: string}
      - {name: clicks, type: double}
"#,
        ),
    ];
    for (column, declaration) in declarations {
        let declaration: Declaration = serde_yaml::from_str(declaration)?;
        let refusal = declaration.check(&manifest).expect_err(column).to_string();
        assert!(
            refusal.contains(column),
            "the type check fails `{column}`: {refusal}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn a_partition_key_takes_its_values_from_the_folders() -> Result<(), Box<dyn Error>> {
    let (_, declared) = runner::compiled("read/partitioned").await?;
    let folder = "s3://semstrait-e2e/orders/";
    let reader = FilesScan::new(Some(FileFormat::Parquet), None, folder, Vec::new());
    let revenue = ("revenue", "amount", DataType::BigInt);
    let totals = summed_by(reader, "orders", ("year", DataType::Integer), revenue)?;
    let production = produced(&plan(totals), &rules())?;
    let (_, rows) = executed(&declared.session()?, &production).await?;
    let row = |year, revenue| {
        vec![
            ScalarValue::Int32(Some(year)),
            ScalarValue::Int64(Some(revenue)),
        ]
    };
    Rows::Multiset(vec![row(2025, 10), row(2026, 25)]).assert("partitioned", rows);
    Ok(())
}

fn shape(plan: &Plan) -> Vec<&RelType> {
    (relations(plan).into_iter())
        .flat_map(|relation| &relation.rel_type)
        .collect()
}

fn function(plan: &Plan, anchor: u32) -> Option<&str> {
    (plan.extensions.iter()).find_map(|declaration| match &declaration.mapping_type {
        Some(MappingType::ExtensionFunction(function)) if function.function_anchor == anchor => {
            Some(function.name.as_str())
        }
        _ => None,
    })
}

fn field(expression: &Expression) -> Option<i32> {
    let Some(RexType::Selection(reference)) = &expression.rex_type else {
        return None;
    };
    let FieldReference {
        reference_type: Some(DirectReference(segment)),
        root_type: Some(RootReference(_)),
    } = reference.as_ref()
    else {
        return None;
    };
    match &segment.reference_type {
        Some(StructField(field)) if field.child.is_none() => Some(field.field),
        _ => None,
    }
}

fn field_equals<'p>(plan: &'p Plan, filter: &'p Expression) -> Option<(i32, &'p LiteralType)> {
    let Some(RexType::ScalarFunction(call)) = &filter.rex_type else {
        return None;
    };
    let [left, right] = &call.arguments[..] else {
        return None;
    };
    let (Some(Value(left)), Some(Value(right))) = (&left.arg_type, &right.arg_type) else {
        return None;
    };
    let Some(RexType::Literal(literal)) = &right.rex_type else {
        return None;
    };
    let equal = function(plan, call.function_reference) == Some("equal");
    equal.then_some((field(left)?, literal.literal_type.as_ref()?))
}

fn emitted(project: &ProjectRel, field: i32) -> Option<&Expression> {
    let Some(EmitKind::Emit(emit)) = &project.common.as_ref()?.emit_kind else {
        return None;
    };
    let first = *emit.output_mapping.first()?;
    let alone = emit.output_mapping.len() == project.expressions.len()
        && (emit.output_mapping.iter().zip(first..)).all(|(&kept, at)| kept == at);
    if !alone {
        return None;
    }
    project.expressions.get(usize::try_from(field).ok()?)
}

#[tokio::test]
async fn the_filter_clause_is_the_measure_filter() -> Result<(), Box<dyn Error>> {
    use RelType::{Aggregate, Project, Read};
    let plan = runner::production("request/elements/aggregate_filter_mirror").await?;
    let shape = shape(&plan);
    let [Aggregate(aggregate), Project(input), Read(_)] = &shape[..] else {
        panic!("the aggregate over its projection over the scan: {shape:?}");
    };
    let filters: Vec<_> = (aggregate.measures.iter())
        .map(|measure| measure.filter.as_ref())
        .collect();
    let [Some(filter), None] = filters[..] else {
        panic!("web_revenue's filter, then revenue's none: {filters:?}");
    };
    let condition = field(filter).and_then(|at| emitted(input, at));
    let condition = condition.ok_or("web_revenue's filter is a field of the projection")?;
    let (_, literal) = field_equals(&plan, condition).ok_or("`equal($p, literal)`")?;
    assert_eq!(literal, &LiteralType::String("web".into()));
    Ok(())
}

fn orders_file(object: &str) -> PathType {
    PathType::UriFile(format!("s3://semstrait-e2e/orders/{object}"))
}

#[tokio::test]
async fn one_uri_folder_read_of_the_base_path() -> Result<(), Box<dyn Error>> {
    use RelType::{Aggregate, Project, Read};
    let plan = runner::production("read/partitions/base_path_key_column").await?;
    let shape = shape(&plan);
    let [Aggregate(_), Project(_), Read(read)] = &shape[..] else {
        panic!("the aggregate over its projection over one read: {shape:?}");
    };
    let base = PathType::UriFolder("s3://semstrait-e2e/orders/".into());
    assert_eq!(files_read(read), [base]);
    Ok(())
}

#[tokio::test]
async fn two_paths_are_two_uri_file_reads_under_one_union_all() -> Result<(), Box<dyn Error>> {
    use RelType::{Aggregate, Project, Read, Set};
    let plan = runner::production("read/files/two_uri_files").await?;
    let shape = shape(&plan);
    let [Aggregate(_), Project(_), Set(union), Read(a), Read(b)] = &shape[..] else {
        panic!("the aggregate over its projection over one union of two reads: {shape:?}");
    };
    assert_eq!(union.op(), SetOp::UnionAll);
    assert_eq!(files_read(a), [orders_file("a.parquet")]);
    assert_eq!(files_read(b), [orders_file("b.parquet")]);
    Ok(())
}

#[tokio::test]
async fn one_path_is_one_uri_file_read_with_no_union() -> Result<(), Box<dyn Error>> {
    use RelType::{Aggregate, Project, Read};
    let plan = runner::production("read/files/one_uri_file").await?;
    let shape = shape(&plan);
    let [Aggregate(_), Project(_), Read(read)] = &shape[..] else {
        panic!("the aggregate over its projection over one read: {shape:?}");
    };
    assert_eq!(files_read(read), [orders_file("a.parquet")]);
    Ok(())
}

#[tokio::test]
async fn one_projection_holds_the_rank_markers_and_one_aggregate_both_measures(
) -> Result<(), Box<dyn Error>> {
    use RelType::{Aggregate, Project};
    let plan = runner::production("additivity/semi_additive/quotient_at_marker").await?;
    let shape = shape(&plan);
    let [Project(_), Aggregate(aggregate), Project(marking), ..] = &shape[..] else {
        panic!("the formula over the aggregate over the markers' projection: {shape:?}");
    };
    let windows = |project: &ProjectRel| {
        (project.expressions.iter())
            .filter(|expression| matches!(expression.rex_type, Some(RexType::WindowFunction(_))))
            .count()
    };
    let aggregates = (shape.iter())
        .filter(|relation| matches!(relation, Aggregate(_)))
        .count();
    let holding = (shape.iter())
        .filter(|relation| matches!(relation, Project(project) if windows(project) > 0))
        .count();
    assert_eq!(
        (aggregates, holding, windows(marking)),
        (1, 1, 2),
        "one aggregate step, over the one projection, which holds both markers"
    );
    let rank = |at| match emitted(marking, at).and_then(|marker| marker.rex_type.as_ref()) {
        Some(RexType::WindowFunction(window)) => {
            function(&plan, window.function_reference) == Some("rank")
        }
        _ => false,
    };
    let markers: Vec<_> = (aggregate.measures.iter())
        .map(|measure| field_equals(&plan, measure.filter.as_ref()?))
        .collect();
    let [Some((a, LiteralType::I64(1))), Some((b, LiteralType::I64(1)))] = markers[..] else {
        panic!("a's and b's filters, each `equal($p, 1)`: {markers:?}");
    };
    assert!(
        a != b && rank(a) && rank(b),
        "each Measure reads its own `rank` marker: {markers:?}"
    );
    Ok(())
}
