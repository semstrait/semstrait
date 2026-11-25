use semstrait_model::expr::{field, physical_col};
use semstrait_model::model::serde::{PhysicalExpr, SemanticExpr, Unresolved};
use semstrait_model::{UnresolvedPhysicalExpr, UnresolvedSemanticExpr};
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Carrier {
    #[serde(with = "Unresolved::<PhysicalExpr>")]
    mapping: UnresolvedPhysicalExpr,
    #[serde(
        rename = "where",
        default,
        skip_serializing_if = "Option::is_none",
        with = "Unresolved::<SemanticExpr>"
    )]
    predicate: Option<UnresolvedSemanticExpr>,
}

#[test]
fn a_bare_and_an_option_field_write_and_read_through_the_attribute() {
    for (carrier, written) in [
        (
            Carrier {
                mapping: physical_col("amount").expect("a name"),
                predicate: field("revenue"),
            },
            "mapping:\n  col: amount\nwhere:\n  field: revenue\n",
        ),
        (
            Carrier {
                mapping: physical_col("amount").expect("a name"),
                predicate: None,
            },
            "mapping:\n  col: amount\n",
        ),
    ] {
        assert_eq!(serde_yaml::to_string(&carrier).expect("written"), written);
        assert_eq!(
            serde_yaml::from_str::<Carrier>(written).expect("read"),
            carrier
        );
    }
}

#[test]
fn an_absent_or_null_option_field_reads_as_none() {
    for yaml in ["mapping: amount", "mapping: amount\nwhere: null"] {
        let carrier: Carrier = serde_yaml::from_str(yaml).expect("read");
        assert_eq!(carrier.predicate, None, "{yaml}");
    }
}

#[test]
fn a_mapping_field_refuses_a_semantic_tag() {
    let refused = serde_yaml::from_str::<Carrier>("mapping: {dim: region}").expect_err("refused");
    let message = "tag `dim` not allowed at physical-mapping site";
    assert!(refused.to_string().contains(message), "{refused}");
}
