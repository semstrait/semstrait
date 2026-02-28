//! Data type definitions for the semantic model

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::str::FromStr;

/// Supported data types in the semantic model
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// Boolean
    Bool,
    /// Variable-length string
    String,
    /// Date (days since Unix epoch)
    Date,
    /// Timestamp (microseconds since Unix epoch)
    Timestamp,
    /// Fixed-point decimal with precision and scale
    Decimal { precision: u8, scale: u8 },
}

impl Default for DataType {
    fn default() -> Self {
        DataType::String
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::I8 => write!(f, "i8"),
            DataType::I16 => write!(f, "i16"),
            DataType::I32 => write!(f, "i32"),
            DataType::I64 => write!(f, "i64"),
            DataType::F32 => write!(f, "f32"),
            DataType::F64 => write!(f, "f64"),
            DataType::Bool => write!(f, "bool"),
            DataType::String => write!(f, "string"),
            DataType::Date => write!(f, "date"),
            DataType::Timestamp => write!(f, "timestamp"),
            DataType::Decimal { precision, scale } => write!(f, "decimal({}, {})", precision, scale),
        }
    }
}

/// Error when parsing a data type string
#[derive(Debug, Clone)]
pub struct ParseDataTypeError {
    pub input: String,
    pub message: String,
}

impl fmt::Display for ParseDataTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Invalid data type '{}': {}", self.input, self.message)
    }
}

impl std::error::Error for ParseDataTypeError {}

impl FromStr for DataType {
    type Err = ParseDataTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_lowercase();
        
        // Check for decimal type first
        if lower.starts_with("decimal(") && lower.ends_with(')') {
            return parse_decimal(&lower);
        }
        
        match lower.as_str() {
            "i8" => Ok(DataType::I8),
            "i16" => Ok(DataType::I16),
            "i32" | "int" | "integer" => Ok(DataType::I32),
            "i64" | "long" | "bigint" => Ok(DataType::I64),
            "f32" | "float" => Ok(DataType::F32),
            "f64" | "double" => Ok(DataType::F64),
            "bool" | "boolean" => Ok(DataType::Bool),
            "string" | "text" | "varchar" => Ok(DataType::String),
            "date" => Ok(DataType::Date),
            "timestamp" | "datetime" => Ok(DataType::Timestamp),
            _ => Err(ParseDataTypeError {
                input: s.to_string(),
                message: "unknown type".to_string(),
            }),
        }
    }
}

fn parse_decimal(s: &str) -> Result<DataType, ParseDataTypeError> {
    // Extract the part between parentheses: "decimal(31, 7)" -> "31, 7"
    let inner = &s[8..s.len() - 1];
    let parts: Vec<&str> = inner.split(',').map(|p| p.trim()).collect();
    
    if parts.len() != 2 {
        return Err(ParseDataTypeError {
            input: s.to_string(),
            message: "decimal requires precision and scale, e.g., decimal(31, 7)".to_string(),
        });
    }
    
    let precision: u8 = parts[0].parse().map_err(|_| ParseDataTypeError {
        input: s.to_string(),
        message: "invalid precision".to_string(),
    })?;
    
    let scale: u8 = parts[1].parse().map_err(|_| ParseDataTypeError {
        input: s.to_string(),
        message: "invalid scale".to_string(),
    })?;
    
    if precision == 0 || precision > 38 {
        return Err(ParseDataTypeError {
            input: s.to_string(),
            message: "precision must be between 1 and 38".to_string(),
        });
    }
    
    if scale > precision {
        return Err(ParseDataTypeError {
            input: s.to_string(),
            message: "scale cannot exceed precision".to_string(),
        });
    }
    
    Ok(DataType::Decimal { precision, scale })
}

// Custom deserialize from string
impl<'de> Deserialize<'de> for DataType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        DataType::from_str(&s).map_err(serde::de::Error::custom)
    }
}

// Serialize back to string
impl Serialize for DataType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

// ============================================================================
// Aggregation
// ============================================================================

/// Aggregation functions for measures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Aggregation {
    /// Sum of values
    Sum,
    /// Average of values
    Avg,
    /// Count of rows
    Count,
    /// Count of distinct values
    CountDistinct,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
}

impl Default for Aggregation {
    fn default() -> Self {
        Aggregation::Sum
    }
}

impl fmt::Display for Aggregation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Aggregation::Sum => write!(f, "sum"),
            Aggregation::Avg => write!(f, "avg"),
            Aggregation::Count => write!(f, "count"),
            Aggregation::CountDistinct => write!(f, "count_distinct"),
            Aggregation::Min => write!(f, "min"),
            Aggregation::Max => write!(f, "max"),
        }
    }
}

/// Error when parsing an aggregation string
#[derive(Debug, Clone)]
pub struct ParseAggregationError {
    pub input: String,
}

impl fmt::Display for ParseAggregationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unknown aggregation '{}'. Valid options: sum, avg, count, count_distinct, min, max", self.input)
    }
}

impl std::error::Error for ParseAggregationError {}

impl FromStr for Aggregation {
    type Err = ParseAggregationError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sum" => Ok(Aggregation::Sum),
            "avg" | "average" => Ok(Aggregation::Avg),
            "count" => Ok(Aggregation::Count),
            "count_distinct" | "countdistinct" | "distinct_count" | "distinctcount" => Ok(Aggregation::CountDistinct),
            "min" | "minimum" => Ok(Aggregation::Min),
            "max" | "maximum" => Ok(Aggregation::Max),
            _ => Err(ParseAggregationError { input: s.to_string() }),
        }
    }
}

impl<'de> Deserialize<'de> for Aggregation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Aggregation::from_str(&s).map_err(serde::de::Error::custom)
    }
}

impl Serialize for Aggregation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

// ============================================================================
// DataType methods
// ============================================================================

impl DataType {
    /// Check if this is a numeric type (integer or floating point)
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            DataType::I8
                | DataType::I16
                | DataType::I32
                | DataType::I64
                | DataType::F32
                | DataType::F64
                | DataType::Decimal { .. }
        )
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64
        )
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DataType::F32 | DataType::F64)
    }

    /// Check if this is a temporal type (date or timestamp)
    pub fn is_temporal(&self) -> bool {
        matches!(self, DataType::Date | DataType::Timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_types() {
        assert_eq!("i32".parse::<DataType>().unwrap(), DataType::I32);
        assert_eq!("I64".parse::<DataType>().unwrap(), DataType::I64);
        assert_eq!("string".parse::<DataType>().unwrap(), DataType::String);
        assert_eq!("bool".parse::<DataType>().unwrap(), DataType::Bool);
        assert_eq!("date".parse::<DataType>().unwrap(), DataType::Date);
        assert_eq!("timestamp".parse::<DataType>().unwrap(), DataType::Timestamp);
    }

    #[test]
    fn test_parse_aliases() {
        assert_eq!("int".parse::<DataType>().unwrap(), DataType::I32);
        assert_eq!("integer".parse::<DataType>().unwrap(), DataType::I32);
        assert_eq!("long".parse::<DataType>().unwrap(), DataType::I64);
        assert_eq!("bigint".parse::<DataType>().unwrap(), DataType::I64);
        assert_eq!("float".parse::<DataType>().unwrap(), DataType::F32);
        assert_eq!("double".parse::<DataType>().unwrap(), DataType::F64);
        assert_eq!("boolean".parse::<DataType>().unwrap(), DataType::Bool);
        assert_eq!("text".parse::<DataType>().unwrap(), DataType::String);
        assert_eq!("varchar".parse::<DataType>().unwrap(), DataType::String);
        assert_eq!("datetime".parse::<DataType>().unwrap(), DataType::Timestamp);
    }

    #[test]
    fn test_parse_decimal() {
        assert_eq!(
            "decimal(31, 7)".parse::<DataType>().unwrap(),
            DataType::Decimal { precision: 31, scale: 7 }
        );
        assert_eq!(
            "DECIMAL(10,2)".parse::<DataType>().unwrap(),
            DataType::Decimal { precision: 10, scale: 2 }
        );
    }

    #[test]
    fn test_parse_decimal_errors() {
        assert!("decimal(0, 0)".parse::<DataType>().is_err()); // precision = 0
        assert!("decimal(5, 10)".parse::<DataType>().is_err()); // scale > precision
        assert!("decimal(50, 2)".parse::<DataType>().is_err()); // precision > 38
        assert!("decimal(10)".parse::<DataType>().is_err()); // missing scale
    }

    #[test]
    fn test_parse_unknown() {
        assert!("foo".parse::<DataType>().is_err());
    }

    #[test]
    fn test_display() {
        assert_eq!(DataType::I32.to_string(), "i32");
        assert_eq!(DataType::Decimal { precision: 31, scale: 7 }.to_string(), "decimal(31, 7)");
    }

    #[test]
    fn test_serde_roundtrip() {
        let types = vec![
            DataType::I32,
            DataType::F64,
            DataType::String,
            DataType::Decimal { precision: 18, scale: 2 },
        ];
        
        for dt in types {
            let json = serde_json::to_string(&dt).unwrap();
            let parsed: DataType = serde_json::from_str(&json).unwrap();
            assert_eq!(dt, parsed);
        }
    }

    #[test]
    fn test_type_predicates() {
        assert!(DataType::I32.is_numeric());
        assert!(DataType::I32.is_integer());
        assert!(!DataType::I32.is_float());
        assert!(!DataType::I32.is_temporal());

        assert!(DataType::F64.is_numeric());
        assert!(!DataType::F64.is_integer());
        assert!(DataType::F64.is_float());

        assert!(DataType::Decimal { precision: 10, scale: 2 }.is_numeric());
        assert!(!DataType::Decimal { precision: 10, scale: 2 }.is_integer());

        assert!(DataType::Date.is_temporal());
        assert!(DataType::Timestamp.is_temporal());
        assert!(!DataType::Date.is_numeric());

        assert!(!DataType::String.is_numeric());
        assert!(!DataType::String.is_temporal());
        assert!(!DataType::Bool.is_numeric());
    }

    // ========================================================================
    // Aggregation tests
    // ========================================================================

    #[test]
    fn test_parse_aggregation() {
        assert_eq!("sum".parse::<Aggregation>().unwrap(), Aggregation::Sum);
        assert_eq!("SUM".parse::<Aggregation>().unwrap(), Aggregation::Sum);
        assert_eq!("avg".parse::<Aggregation>().unwrap(), Aggregation::Avg);
        assert_eq!("count".parse::<Aggregation>().unwrap(), Aggregation::Count);
        assert_eq!("count_distinct".parse::<Aggregation>().unwrap(), Aggregation::CountDistinct);
        assert_eq!("min".parse::<Aggregation>().unwrap(), Aggregation::Min);
        assert_eq!("max".parse::<Aggregation>().unwrap(), Aggregation::Max);
    }

    #[test]
    fn test_parse_aggregation_aliases() {
        assert_eq!("average".parse::<Aggregation>().unwrap(), Aggregation::Avg);
        assert_eq!("countdistinct".parse::<Aggregation>().unwrap(), Aggregation::CountDistinct);
        assert_eq!("distinct_count".parse::<Aggregation>().unwrap(), Aggregation::CountDistinct);
        assert_eq!("minimum".parse::<Aggregation>().unwrap(), Aggregation::Min);
        assert_eq!("maximum".parse::<Aggregation>().unwrap(), Aggregation::Max);
    }

    #[test]
    fn test_parse_aggregation_unknown() {
        assert!("foo".parse::<Aggregation>().is_err());
        assert!("median".parse::<Aggregation>().is_err());
    }

    #[test]
    fn test_aggregation_display() {
        assert_eq!(Aggregation::Sum.to_string(), "sum");
        assert_eq!(Aggregation::CountDistinct.to_string(), "count_distinct");
    }

    #[test]
    fn test_aggregation_serde_roundtrip() {
        let aggs = vec![
            Aggregation::Sum,
            Aggregation::Avg,
            Aggregation::Count,
            Aggregation::CountDistinct,
            Aggregation::Min,
            Aggregation::Max,
        ];
        
        for agg in aggs {
            let json = serde_json::to_string(&agg).unwrap();
            let parsed: Aggregation = serde_json::from_str(&json).unwrap();
            assert_eq!(agg, parsed);
        }
    }
}
