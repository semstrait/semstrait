use crate::error::serde_message;
use semstrait_sql::expr::{AggregateAllowed, FunctionRef};
use semstrait_sql::functions::{registry, AggregateFunction};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Agg {
    function: &'static AggregateFunction,
    distinct: bool,
}

const COUNT_DISTINCT: &str = "count_distinct";

impl Agg {
    pub fn of_name(name: &str) -> Option<Self> {
        let distinct = name.eq_ignore_ascii_case(COUNT_DISTINCT);
        match registry().get(if distinct { "count" } else { name })? {
            FunctionRef::Aggregate(function, AggregateAllowed) => Some(Self { function, distinct }),
            _ => None,
        }
    }

    pub fn function(self) -> &'static AggregateFunction {
        self.function
    }

    pub fn distinct(self) -> bool {
        self.distinct
    }

    pub fn name(self) -> &'static str {
        if self.distinct {
            COUNT_DISTINCT
        } else {
            self.function.declarations.canonical.name
        }
    }
}

impl Serialize for Agg {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for Agg {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let name = String::deserialize(deserializer)?;
        let agg = Self::of_name(&name)
            .ok_or_else(|| D::Error::custom(serde_message::unknown_aggregate(&name)))?;
        agg.function
            .check_value_call(agg.distinct)
            .map(|()| agg)
            .map_err(|error| {
                D::Error::custom(serde_message::call_refused(&name, &error.to_string()))
            })
    }
}

#[cfg(feature = "schemars")]
impl schemars::JsonSchema for Agg {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Agg".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "description": "A catalog aggregate's name or one of its aliases, such as `sum` or `every`, or `count_distinct`."
        })
    }
}
