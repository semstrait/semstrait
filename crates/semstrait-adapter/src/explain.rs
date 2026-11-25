#![allow(
    clippy::result_large_err,
    reason = "every stage returns its Diagnostic envelope unboxed"
)]

use prost::{Message, Name};
use semstrait_common::diagnostic::{Diagnostic, Diagnostics};
use semstrait_common::Granularity;
use semstrait_ir::{
    Cardinality, Delivery, GranularityGroup, SemAnnotation, SemanticPlan, SourceRole,
};
use substrait::proto::extensions::AdvancedExtension;
use substrait_explain::extensions::{
    Any, ArgsAccess, EnumValue, Explainable, ExtensionArgs, ExtensionContext, ExtensionError,
    ExtensionRegistry, ExtensionValue, TupleValue,
};
use substrait_explain::{format_with_registry, OutputOptions};

use crate::error::InvalidRewrite;
use crate::producer::{Production, SubstraitProducer};
use crate::rules::SubstraitRules;

#[allow(
    clippy::type_complexity,
    reason = "ADR 0005's fail-fast form, spelled out as a caller binds it"
)]
pub fn explain(
    plan: &SemanticPlan,
    rules: &dyn SubstraitRules,
) -> Result<
    (String, Diagnostics<InvalidRewrite>),
    (Diagnostic<InvalidRewrite>, Diagnostics<InvalidRewrite>),
> {
    let copy = (SubstraitProducer::new(rules).production(plan, Production::Explain))
        .map_err(|error| (error, Vec::new()))?;
    let registry = trace_registry();
    let (text, _format_errors) = format_with_registry(&copy, &OutputOptions::default(), &registry);
    Ok((text, Vec::new()))
}

#[must_use]
pub fn trace_registry() -> ExtensionRegistry {
    let mut registry = ExtensionRegistry::new();
    registry.register_optimization::<Trace>().ok();
    registry
}

pub(crate) fn trace(subjects: &[SemAnnotation]) -> Option<AdvancedExtension> {
    (!subjects.is_empty()).then(|| AdvancedExtension {
        optimization: (subjects.iter())
            .map(|subject| Any::new(Trace::type_url(), wire(subject).encode_to_vec()).into())
            .collect(),
        enhancement: None,
    })
}

#[derive(Clone, PartialEq, Message)]
struct Trace {
    #[prost(string, optional, tag = "1")]
    semantic_source: Option<String>,
    #[prost(string, optional, tag = "2")]
    source_role: Option<String>,
    #[prost(string, repeated, tag = "3")]
    originators: Vec<String>,
    #[prost(string, optional, tag = "4")]
    physical_source: Option<String>,
    #[prost(string, optional, tag = "5")]
    scope_root: Option<String>,
    #[prost(uint32, optional, tag = "6")]
    compositions: Option<u32>,
    #[prost(string, optional, tag = "7")]
    granularity_fit: Option<String>,
    #[prost(string, optional, tag = "8")]
    granularity: Option<String>,
    #[prost(string, optional, tag = "9")]
    cardinality: Option<String>,
    #[prost(string, optional, tag = "10")]
    delivery: Option<String>,
    #[prost(string, optional, tag = "11")]
    answered_by: Option<String>,
    #[prost(string, repeated, tag = "12")]
    pruned: Vec<String>,
}

impl Name for Trace {
    const PACKAGE: &'static str = "semstrait";
    const NAME: &'static str = "Trace";
}

fn source_role_text(value: &SourceRole) -> &'static str {
    match value {
        SourceRole::Dataset => "Dataset",
        SourceRole::Joinset => "Joinset",
        SourceRole::Unionset => "Unionset",
        SourceRole::Grainset => "Grainset",
        other => unreachable!("semstrait-ir added a SourceRole variant: {other:?}"),
    }
}

fn granularity_fit_text(value: &GranularityGroup) -> &'static str {
    match value {
        GranularityGroup::Requested => "Requested",
        GranularityGroup::Selected => "Selected",
        GranularityGroup::NotSelected => "NotSelected",
        other => unreachable!("semstrait-ir added a GranularityGroup variant: {other:?}"),
    }
}

fn granularity_text(value: Granularity) -> &'static str {
    match value {
        Granularity::Minute => "Minute",
        Granularity::Hour => "Hour",
        Granularity::Day => "Day",
        Granularity::Week => "Week",
        Granularity::Month => "Month",
        Granularity::Quarter => "Quarter",
        Granularity::Year => "Year",
        other => unreachable!("semstrait-common added a Granularity variant: {other:?}"),
    }
}

fn cardinality_text(value: Cardinality) -> &'static str {
    match value {
        Cardinality::OneToOne => "OneToOne",
        Cardinality::OneToMany => "OneToMany",
        Cardinality::ManyToOne => "ManyToOne",
        Cardinality::ManyToMany => "ManyToMany",
        other => unreachable!("semstrait-ir added a Cardinality variant: {other:?}"),
    }
}

fn delivery_text(value: Delivery) -> &'static str {
    match value {
        Delivery::Rows => "Rows",
        Delivery::PartialAggregates => "PartialAggregates",
        Delivery::FinalAggregates => "FinalAggregates",
        other => unreachable!("semstrait-ir added a Delivery variant: {other:?}"),
    }
}

fn parse_source_role(text: String) -> Result<String, ExtensionError> {
    match text.as_str() {
        "Dataset" | "Joinset" | "Unionset" | "Grainset" => Ok(text),
        other => Err(ExtensionError::InvalidArgument(format!(
            "source_role: unknown variant `{other}`"
        ))),
    }
}

fn parse_granularity_fit(text: String) -> Result<String, ExtensionError> {
    match text.as_str() {
        "Requested" | "Selected" | "NotSelected" => Ok(text),
        other => Err(ExtensionError::InvalidArgument(format!(
            "granularity_fit: unknown variant `{other}`"
        ))),
    }
}

fn parse_granularity(text: String) -> Result<String, ExtensionError> {
    match text.as_str() {
        "Minute" | "Hour" | "Day" | "Week" | "Month" | "Quarter" | "Year" => Ok(text),
        other => Err(ExtensionError::InvalidArgument(format!(
            "granularity: unknown variant `{other}`"
        ))),
    }
}

fn parse_cardinality(text: String) -> Result<String, ExtensionError> {
    match text.as_str() {
        "OneToOne" | "OneToMany" | "ManyToOne" | "ManyToMany" => Ok(text),
        other => Err(ExtensionError::InvalidArgument(format!(
            "cardinality: unknown variant `{other}`"
        ))),
    }
}

fn parse_delivery(text: String) -> Result<String, ExtensionError> {
    match text.as_str() {
        "Rows" | "PartialAggregates" | "FinalAggregates" => Ok(text),
        other => Err(ExtensionError::InvalidArgument(format!(
            "delivery: unknown variant `{other}`"
        ))),
    }
}

fn wire(subject: &SemAnnotation) -> Trace {
    match subject {
        SemAnnotation::Rank {
            scope_root,
            compositions,
            granularity_fit,
            granularity,
        } => Trace {
            scope_root: Some(scope_root.as_str().to_owned()),
            compositions: Some(*compositions),
            granularity_fit: Some(granularity_fit_text(granularity_fit).to_owned()),
            granularity: granularity.map(|value| granularity_text(value).to_owned()),
            ..Trace::default()
        },
        SemAnnotation::SemanticSource {
            semantic_source,
            source_role,
            originators,
            physical_source,
            cardinality,
            delivery,
            answered_by,
            pruned,
        } => Trace {
            semantic_source: Some(semantic_source.as_str().to_owned()),
            source_role: Some(source_role_text(source_role).to_owned()),
            originators: (originators.iter())
                .map(|name| name.as_str().to_owned())
                .collect(),
            physical_source: (physical_source.as_ref()).map(|value| value.as_str().to_owned()),
            cardinality: cardinality.map(|value| cardinality_text(value).to_owned()),
            delivery: delivery.map(|value| delivery_text(value).to_owned()),
            answered_by: (answered_by.as_ref()).map(|value| value.as_str().to_owned()),
            pruned: (pruned.iter())
                .map(|name| name.as_str().to_owned())
                .collect(),
            ..Trace::default()
        },
        _ => Trace::default(),
    }
}

impl Explainable for Trace {
    fn name() -> &'static str {
        Self::NAME
    }

    fn from_args(args: &mut ArgsAccess<'_>) -> Result<Self, ExtensionError> {
        Ok(Self {
            semantic_source: args.get_named::<String>("semantic_source")?,
            source_role: (args.get_named::<EnumValue>("source_role")?)
                .map(|EnumValue(text)| parse_source_role(text))
                .transpose()?,
            originators: (args.get_named_tuple::<String>("originators")?).unwrap_or_default(),
            physical_source: args.get_named::<String>("physical_source")?,
            scope_root: args.get_named::<String>("scope_root")?,
            compositions: (args.get_named::<i64>("compositions")?)
                .map(|value| {
                    u32::try_from(value).map_err(|_| {
                        ExtensionError::InvalidArgument(format!(
                            "compositions must fit in u32, got {value}"
                        ))
                    })
                })
                .transpose()?,
            granularity_fit: (args.get_named::<EnumValue>("granularity_fit")?)
                .map(|EnumValue(text)| parse_granularity_fit(text))
                .transpose()?,
            granularity: (args.get_named::<EnumValue>("granularity")?)
                .map(|EnumValue(text)| parse_granularity(text))
                .transpose()?,
            cardinality: (args.get_named::<EnumValue>("cardinality")?)
                .map(|EnumValue(text)| parse_cardinality(text))
                .transpose()?,
            delivery: (args.get_named::<EnumValue>("delivery")?)
                .map(|EnumValue(text)| parse_delivery(text))
                .transpose()?,
            answered_by: args.get_named::<String>("answered_by")?,
            pruned: (args.get_named_tuple::<String>("pruned")?).unwrap_or_default(),
        })
    }

    fn to_args(&self, _context: &ExtensionContext<'_>) -> Result<ExtensionArgs, ExtensionError> {
        let mut args = ExtensionArgs::default();
        if let Some(value) = &self.semantic_source {
            args.insert("semantic_source", value.clone());
        }
        if let Some(value) = &self.source_role {
            args.insert("source_role", ExtensionValue::Enum(value.clone()));
        }
        if !self.originators.is_empty() {
            args.insert("originators", tuple(&self.originators));
        }
        if let Some(value) = &self.physical_source {
            args.insert("physical_source", value.clone());
        }
        if let Some(value) = &self.scope_root {
            args.insert("scope_root", value.clone());
        }
        if let Some(value) = self.compositions {
            args.insert("compositions", i64::from(value));
        }
        if let Some(value) = &self.granularity_fit {
            args.insert("granularity_fit", ExtensionValue::Enum(value.clone()));
        }
        if let Some(value) = &self.granularity {
            args.insert("granularity", ExtensionValue::Enum(value.clone()));
        }
        if let Some(value) = &self.cardinality {
            args.insert("cardinality", ExtensionValue::Enum(value.clone()));
        }
        if let Some(value) = &self.delivery {
            args.insert("delivery", ExtensionValue::Enum(value.clone()));
        }
        if let Some(value) = &self.answered_by {
            args.insert("answered_by", value.clone());
        }
        if !self.pruned.is_empty() {
            args.insert("pruned", tuple(&self.pruned));
        }
        Ok(args)
    }
}

fn tuple(names: &[String]) -> ExtensionValue {
    let items: TupleValue = (names.iter().cloned())
        .map(ExtensionValue::String)
        .collect();
    ExtensionValue::Tuple(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_args_refuses_a_negative_compositions() {
        let mut args = ExtensionArgs::default();
        args.insert("compositions", -1_i64);

        let error = args.parse::<Trace>().expect_err("a negative compositions");

        assert!(
            matches!(error, ExtensionError::InvalidArgument(_)),
            "{error:?}"
        );
    }

    #[test]
    fn from_args_refuses_a_compositions_over_u32_range() {
        let mut args = ExtensionArgs::default();
        args.insert("compositions", i64::from(u32::MAX) + 1);

        let error = args
            .parse::<Trace>()
            .expect_err("a compositions over u32::MAX");

        assert!(
            matches!(error, ExtensionError::InvalidArgument(_)),
            "{error:?}"
        );
    }

    #[test]
    fn from_args_refuses_an_unknown_source_role() {
        let mut args = ExtensionArgs::default();
        args.insert("source_role", ExtensionValue::Enum("Bogus".to_owned()));

        let error = args.parse::<Trace>().expect_err("an unknown source_role");

        assert!(
            matches!(error, ExtensionError::InvalidArgument(_)),
            "{error:?}"
        );
    }
}
