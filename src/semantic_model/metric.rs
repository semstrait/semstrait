//! Metric types - derived calculations from measures

use serde::Deserialize;
use super::types::DataType;

/// A metric - calculation combining measures
#[derive(Debug, Deserialize)]
pub struct Metric {
    pub name: String,
    pub label: Option<String>,
    /// Human-readable description for UIs and LLMs
    pub description: Option<String>,
    /// Alternative names (for LLM query understanding)
    pub synonyms: Option<Vec<String>>,
    pub hidden: Option<bool>,
    pub format: Option<String>,
    /// Result data type. Defaults to F64 for metrics.
    #[serde(rename = "type")]
    pub data_type: Option<DataType>,
    /// Expression combining measures
    pub expr: MetricExpr,
}

/// Metric expression - references measures by name
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MetricExpr {
    /// Simple measure reference: "sales"
    MeasureRef(String),
    /// Structured expression
    Structured(MetricExprNode),
}

/// Expression node for metric calculations
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MetricExprNode {
    /// Reference a measure: { measure: "sales" }
    Measure(String),
    /// Literal number
    Literal(f64),
    /// Addition
    Add(Vec<MetricExprArg>),
    /// Subtraction
    Subtract(Vec<MetricExprArg>),
    /// Multiplication
    Multiply(Vec<MetricExprArg>),
    /// Division
    Divide(Vec<MetricExprArg>),
    /// CASE WHEN expression - for cross-datasetGroup metrics
    Case(MetricCaseExpr),
}

/// CASE WHEN expression for metrics
/// Used for cross-datasetGroup metrics that select different measures based on datasetGroup
#[derive(Debug, Clone, Deserialize)]
pub struct MetricCaseExpr {
    /// List of WHEN...THEN branches
    pub when: Vec<MetricCaseWhen>,
    /// Optional ELSE value (defaults to 0)
    #[serde(rename = "else")]
    pub else_value: Option<Box<MetricExprArg>>,
}

/// A single WHEN...THEN branch in a metric CASE expression
#[derive(Debug, Clone, Deserialize)]
pub struct MetricCaseWhen {
    /// The condition to evaluate
    pub condition: MetricCondition,
    /// The measure to use if condition is true
    pub then: MetricExprArg,
}

/// Condition expression for metric CASE WHEN
/// Currently supports datasetGroup.name comparisons for cross-datasetGroup metrics
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MetricCondition {
    /// Equal: eq: [a, b]
    Eq(Vec<MetricConditionArg>),
    /// Not equal: ne: [a, b]
    Ne(Vec<MetricConditionArg>),
}

/// Argument in a metric condition
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MetricConditionArg {
    /// String value (e.g., "datasetGroup.name" or "adwords")
    String(String),
    /// Literal number
    Number(f64),
}

/// Argument in a metric expression
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum MetricExprArg {
    /// Shorthand: measure name as string
    MeasureName(String),
    /// Literal number
    LiteralNumber(f64),
    /// Nested expression
    Node(MetricExprNode),
}

impl Metric {
    /// Get the result data type, defaulting to F64 for metrics
    pub fn data_type(&self) -> DataType {
        self.data_type.clone().unwrap_or(DataType::F64)
    }

    /// Check if this metric is a cross-datasetGroup metric
    /// 
    /// A cross-datasetGroup metric uses `datasetGroup.name` (or legacy `tableGroup.name`)
    /// in CASE conditions to select different measures based on the active datasetGroup.
    pub fn is_cross_dataset_group(&self) -> bool {
        match &self.expr {
            MetricExpr::Structured(MetricExprNode::Case(case_expr)) => {
                case_expr.when.iter().any(|w| w.condition.references_dataset_group())
            }
            _ => false,
        }
    }

    /// Extract datasetGroup-to-measure mappings from a cross-datasetGroup metric
    /// 
    /// Returns a vec of (datasetGroup_name, measure_name) tuples.
    /// Returns empty vec if not a cross-datasetGroup metric.
    pub fn dataset_group_measures(&self) -> Vec<(String, String)> {
        match &self.expr {
            MetricExpr::Structured(MetricExprNode::Case(case_expr)) => {
                case_expr.when.iter()
                    .filter_map(|w| {
                        let dataset_group = w.condition.dataset_group_value()?;
                        let measure = w.then.measure_name()?;
                        Some((dataset_group, measure))
                    })
                    .collect()
            }
            _ => vec![],
        }
    }

    // Backwards compatibility aliases
    
    /// Check if this metric is a cross-datasetGroup metric (legacy alias)
    #[deprecated(note = "Use is_cross_dataset_group instead")]
    pub fn is_cross_table_group(&self) -> bool {
        self.is_cross_dataset_group()
    }

    /// Extract datasetGroup-to-measure mappings (legacy alias)
    #[deprecated(note = "Use dataset_group_measures instead")]
    pub fn table_group_measures(&self) -> Vec<(String, String)> {
        self.dataset_group_measures()
    }
}

impl MetricCondition {
    /// Check if this condition references datasetGroup.name (or legacy tableGroup.name)
    pub fn references_dataset_group(&self) -> bool {
        match self {
            MetricCondition::Eq(args) | MetricCondition::Ne(args) => {
                args.iter().any(|arg| {
                    matches!(arg, MetricConditionArg::String(s) if s == "datasetGroup.name" || s == "tableGroup.name")
                })
            }
        }
    }

    /// Extract the datasetGroup name value from a condition like eq: [datasetGroup.name, "adwords"]
    pub fn dataset_group_value(&self) -> Option<String> {
        match self {
            MetricCondition::Eq(args) if args.len() == 2 => {
                // Check if one arg is "datasetGroup.name" or legacy "tableGroup.name" and get the other
                let has_dataset_group = args.iter().any(|a| {
                    matches!(a, MetricConditionArg::String(s) if s == "datasetGroup.name" || s == "tableGroup.name")
                });
                if has_dataset_group {
                    args.iter().find_map(|a| {
                        match a {
                            MetricConditionArg::String(s) if s != "datasetGroup.name" && s != "tableGroup.name" => Some(s.clone()),
                            _ => None,
                        }
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    // Backwards compatibility aliases

    /// Check if this condition references datasetGroup.name (legacy alias)
    #[deprecated(note = "Use references_dataset_group instead")]
    pub fn references_table_group(&self) -> bool {
        self.references_dataset_group()
    }

    /// Extract the datasetGroup name value (legacy alias)
    #[deprecated(note = "Use dataset_group_value instead")]
    pub fn table_group_value(&self) -> Option<String> {
        self.dataset_group_value()
    }
}

impl MetricExprArg {
    /// Get the measure name if this is a simple measure reference
    pub fn measure_name(&self) -> Option<String> {
        match self {
            MetricExprArg::MeasureName(name) => Some(name.clone()),
            MetricExprArg::Node(MetricExprNode::Measure(name)) => Some(name.clone()),
            _ => None,
        }
    }
}
