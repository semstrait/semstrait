use std::collections::BTreeMap;
use std::error::Error;

use datafusion::arrow::datatypes::Fields;
use semstrait_common::Granularity;
use semstrait_common::{DataKindName, SemanticsName};
use semstrait_model::model::serde::parse_expr;
use semstrait_planner::{
    DimensionModifier, Direction, OrderKey, Request, RequestContext, Selection, Timezone,
    UnknownTimezone, Weekday,
};
use semstrait_sql::datatype::DataType;
use semstrait_sql::expr::SemanticLeaf;
use serde::Deserialize;
use serde_yaml::Value;

use super::data::{arrow_type, row};
use super::refusal::Refusal;
use super::{fixture, Rows};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    model: Option<String>,
    request: Option<RequestPart>,
    #[serde(default)]
    context: ContextPart,
    #[serde(default)]
    input_data: BTreeMap<String, Vec<Vec<Value>>>,
    types: Option<Vec<DataType>>,
    #[serde(with = "serde_yaml::with::singleton_map")]
    pub expect: Expect,
    sql: Option<String>,
    plan: Option<String>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Expect {
    Rows(Vec<Vec<Value>>),
    Error(Refusal),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestPart {
    from: Option<DataKindName>,
    select: Vec<Selected>,
    #[serde(rename = "where")]
    predicate: Option<Value>,
    #[serde(default)]
    filters: Vec<SemanticsName>,
    #[serde(default)]
    order_by: Vec<Key>,
    limit: Option<u64>,
    offset: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
enum Selected {
    Field(SemanticsName),
    Dimension {
        dimension: SemanticsName,
        granularity: Option<Granularity>,
    },
    Measure {
        measure: SemanticsName,
    },
    Metric {
        metric: SemanticsName,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Key {
    name: SemanticsName,
    direction: Order,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Order {
    Asc,
    Desc,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextPart {
    timezone: Option<String>,
    week_start: Option<WeekStart>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum WeekStart {
    Monday,
    Sunday,
}

impl Selected {
    fn name(&self) -> &SemanticsName {
        match self {
            Self::Field(name)
            | Self::Dimension {
                dimension: name, ..
            }
            | Self::Measure { measure: name }
            | Self::Metric { metric: name } => name,
        }
    }

    fn selection(&self) -> Selection {
        let name = self.name().clone();
        match self {
            Self::Field(_) => Selection::Field(name),
            Self::Dimension { granularity, .. } => Selection::Dimension {
                name,
                modifier: granularity.map(DimensionModifier::Granularity),
            },
            Self::Measure { .. } => Selection::Measure { name },
            Self::Metric { .. } => Selection::Metric { name },
        }
    }
}

pub fn split(id: &str) -> Result<(&str, &str), Box<dyn Error>> {
    id.rsplit_once('/')
        .ok_or_else(|| format!("{id}: a case id needs `<file>/<key>`").into())
}

impl Case {
    pub fn problem(file: &str) -> Result<BTreeMap<String, Self>, Box<dyn Error>> {
        let text = std::fs::read_to_string(fixture(&format!("{file}.yaml")))
            .map_err(|error| format!("{file}.yaml: {error}"))?;
        serde_yaml::from_str(&text).map_err(|error| format!("{file}.yaml: {error}").into())
    }

    pub fn read(case: &str) -> Result<Self, Box<dyn Error>> {
        let (file, key) = split(case)?;
        let mut cases = Self::problem(file).map_err(|error| format!("{case}: {error}"))?;
        cases
            .remove(key)
            .ok_or_else(|| format!("{case}: no case `{key}` in `{file}.yaml`").into())
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub fn input_data(&self) -> &BTreeMap<String, Vec<Vec<Value>>> {
        &self.input_data
    }

    pub fn field(&self, field: &str) -> Option<&str> {
        match field {
            "plan" => self.plan.as_deref(),
            "message" => self.message.as_deref(),
            _ => None,
        }
    }

    fn requested(&self) -> Result<&RequestPart, Box<dyn Error>> {
        Ok(self.request.as_ref().ok_or("the case holds no request")?)
    }

    pub fn request(&self) -> Result<Request, Box<dyn Error>> {
        let part = self.requested()?;
        let keys = (part.order_by.iter()).map(|key| {
            let direction = match key.direction {
                Order::Asc => Direction::Asc,
                Order::Desc => Direction::Desc,
            };
            OrderKey::new(key.name.clone(), direction)
        });
        let mut request = Request::new(part.select.iter().map(Selected::selection).collect())
            .with_filters(part.filters.clone())
            .with_order_by(keys.collect());
        if let Some(from) = &part.from {
            request = request.with_from(from.clone());
        }
        if let Some(predicate) = &part.predicate {
            request = request.with_predicate(parse_expr::<SemanticLeaf>(predicate)?);
        }
        if let Some(limit) = part.limit {
            request = request.with_limit(limit);
        }
        if let Some(offset) = part.offset {
            request = request.with_offset(offset);
        }
        Ok(request)
    }

    pub fn context(&self) -> Result<RequestContext, UnknownTimezone> {
        let mut context = RequestContext::default();
        if let Some(timezone) = &self.context.timezone {
            context = context.with_timezone(Timezone::new(timezone)?);
        }
        if let Some(week_start) = self.context.week_start {
            context = context.with_week_start(match week_start {
                WeekStart::Monday => Weekday::Monday,
                WeekStart::Sunday => Weekday::Sunday,
            });
        }
        Ok(context)
    }

    pub fn rows(&self, fields: &Fields) -> Result<Rows, Box<dyn Error>> {
        let Expect::Rows(expected) = &self.expect else {
            return Err("the case expects an error".into());
        };
        let (header, rows) = expected.split_first().ok_or("the rows' header")?;
        let names = self.names()?;
        if !(header.iter().map(Value::as_str)).eq(names.iter().copied().map(Some)) {
            return Err(format!("the header {header:?} is not the select names {names:?}").into());
        }
        if let Some(types) = &self.types {
            let types: Vec<_> = (types.iter().map(arrow_type)).collect::<Result<_, _>>()?;
            let output: Vec<_> = fields.iter().map(|field| field.data_type()).collect();
            if !output.iter().copied().eq(&types) {
                return Err(format!("the output types {output:?} are not {types:?}").into());
            }
        }
        let rows = (rows.iter())
            .map(|values| row(values, fields))
            .collect::<Result<_, _>>()?;
        Ok(Rows::OrderedBy(self.keys()?, rows))
    }

    pub fn names(&self) -> Result<Vec<&str>, Box<dyn Error>> {
        let select = &self.requested()?.select;
        Ok(select
            .iter()
            .map(|selected| selected.name().as_str())
            .collect())
    }

    pub fn keys(&self) -> Result<Vec<usize>, Box<dyn Error>> {
        let names = self.names()?;
        (self.requested()?.order_by.iter())
            .map(|key| {
                let position = names.iter().position(|name| *name == key.name.as_str());
                position.ok_or_else(|| format!("the key `{}` is not selected", key.name).into())
            })
            .collect()
    }
}
