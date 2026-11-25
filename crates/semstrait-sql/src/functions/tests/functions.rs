use super::{aggregate_function, fixed};
use crate::datatype::{DataType, TypeFamily};
use crate::expr::{AggregateAllowed, FunctionRef, Literal};
use crate::functions::{
    registry, seal, Additivity, Aggregate, AggregateClauses, AggregateFunction, ArgumentForm,
    Category, ClauseRule, Condition, Constraint, Declaration, ExactType, Function, FunctionCatalog,
    FunctionType, Nullability, OptionDecl, ParameterDecl, ParameterDefault, ParameterType,
    Presence, Registry, ReturnRule, Scalar, ScalarOption, SealFault, Signature, Volatility, Window,
    WindowClauses,
};
use crate::DocumentDialect;
use semstrait_common::diagnostic::{Diagnose, Severity};
use semstrait_common::{Dialect, Release};
use std::collections::HashSet;

const NO_PARAMETERS: Signature = Signature {
    parameters: &[],
    variadic: None,
    constraints: &[],
};

const fn scalar(name: &'static str) -> Declaration<Scalar> {
    Declaration {
        name,
        aliases: &[],
        signature: NO_PARAMETERS,
        returns: ReturnRule::Fixed(ExactType::Type(&DataType::String)),
        nullability: Nullability::Mirror,
        volatility: Volatility::Immutable,
        specific: Scalar {
            category: Category::String,
            options: &[],
        },
    }
}

const fn aggregate(name: &'static str) -> Declaration<Aggregate> {
    Declaration {
        name,
        aliases: &[],
        signature: NO_PARAMETERS,
        returns: ReturnRule::Fixed(ExactType::Type(&DataType::BigInt)),
        nullability: Nullability::Never,
        volatility: Volatility::Immutable,
        specific: Aggregate {
            options: &[],
            clauses: AggregateClauses {
                distinct: ClauseRule::Refused,
                filter: ClauseRule::Allowed,
                order_by: ClauseRule::Refused,
                within_group: ClauseRule::Refused,
                within_group_key: ParameterType::Any,
                nulls: ClauseRule::Refused,
            },
        },
    }
}

const fn additive(canonical: Declaration<Aggregate>) -> AggregateFunction {
    AggregateFunction {
        declarations: Function {
            canonical,
            dialects: &[],
            releases: &[],
        },
        additivity: Additivity::Additive,
        distinct_additivity: Additivity::NonAdditive,
    }
}

const fn window(name: &'static str) -> Declaration<Window> {
    Declaration {
        name,
        aliases: &[],
        signature: NO_PARAMETERS,
        returns: ReturnRule::Fixed(ExactType::Type(&DataType::BigInt)),
        nullability: Nullability::Never,
        volatility: Volatility::Immutable,
        specific: Window {
            options: &[],
            clauses: WindowClauses {
                nulls: ClauseRule::Refused,
                partition_by: ClauseRule::Allowed,
                order_by: ClauseRule::Allowed,
                window_frame: ClauseRule::NoEffect,
            },
        },
    }
}

const NO_FUNCTIONS: FunctionCatalog = FunctionCatalog {
    scalar: &[],
    aggregate: &[],
    window: &[],
};

static UPPER: Function<Scalar> = Function {
    canonical: scalar("upper"),
    dialects: &[],
    releases: &[],
};

static NULLIF: Function<Scalar> = Function {
    canonical: Declaration {
        aliases: &["null_if"],
        ..scalar("nullif")
    },
    dialects: &[],
    releases: &[],
};

static SIGN: Function<Scalar> = Function {
    canonical: scalar("sign"),
    dialects: &[(Dialect::DataFusion, Some(scalar("signum")))],
    releases: &[],
};

static REGEXP_EXTRACT: Function<Scalar> = Function {
    canonical: scalar("regexp_extract"),
    dialects: &[(Dialect::DataFusion, None)],
    releases: &[],
};

static COUNT: AggregateFunction = additive(aggregate("count"));

static ROW_NUMBER: Function<Window> = Function {
    canonical: window("row_number"),
    dialects: &[],
    releases: &[],
};

static WELL_FORMED: FunctionCatalog = FunctionCatalog {
    scalar: &[(Category::String, &[&UPPER, &NULLIF, &SIGN, &REGEXP_EXTRACT])],
    aggregate: &[&COUNT],
    window: &[&ROW_NUMBER],
};

fn sealed() -> Registry {
    seal(&WELL_FORMED).unwrap_or_else(|faults| panic!("the set seals: {faults:?}"))
}

fn faults(catalog: &FunctionCatalog) -> Vec<SealFault> {
    seal(catalog).err().unwrap_or_default()
}

macro_rules! one_scalar {
    ($function:expr) => {{
        static FUNCTION: Function<Scalar> = $function;
        static CATALOG: FunctionCatalog = FunctionCatalog {
            scalar: &[(Category::String, &[&FUNCTION])],
            ..NO_FUNCTIONS
        };
        &CATALOG
    }};
    ($name:literal { $($part:ident: $value:expr),* $(,)? }) => {
        one_scalar!(Function {
            canonical: Declaration {
                $($part: $value,)*
                ..scalar($name)
            },
            dialects: &[],
            releases: &[],
        })
    };
}

const fn number(name: &'static str, presence: Presence) -> ParameterDecl {
    ParameterDecl {
        name,
        ty: ParameterType::Family(&TypeFamily::Number),
        form: ArgumentForm::Value,
        presence,
    }
}

const fn parameters(parameters: &'static [ParameterDecl]) -> Signature {
    Signature {
        parameters,
        ..NO_PARAMETERS
    }
}

const VALUE: ParameterDecl = number("value", Presence::Required);

const fn exact(name: &'static str, ty: ExactType) -> ParameterDecl {
    ParameterDecl {
        ty: ParameterType::Exact(ty),
        ..number(name, Presence::Required)
    }
}

const fn for_week(name: &'static str, argument: usize) -> ParameterDecl {
    number(
        name,
        Presence::When {
            condition: Condition::Value {
                argument,
                value: "week",
            },
            default: None,
        },
    )
}

const fn null_handling(
    values: &'static [&'static str],
    canonical: &'static str,
) -> OptionDecl<ScalarOption> {
    OptionDecl {
        name: ScalarOption::NullHandling,
        values,
        canonical,
    }
}

#[test]
fn the_registry_seals_over_the_catalog_once() {
    assert!(std::ptr::eq(registry(), registry()));
}

#[test]
fn get_finds_a_function_by_its_name() {
    assert_eq!(sealed().get("upper"), Some(FunctionRef::Scalar(&UPPER)));
    assert_eq!(sealed().get("lower"), None);
}

#[test]
fn get_finds_a_function_by_an_alias() {
    assert_eq!(sealed().get("null_if"), Some(FunctionRef::Scalar(&NULLIF)));
}

#[test]
fn get_takes_a_name_as_text_in_any_case() {
    assert_eq!(sealed().get("UPPER"), Some(FunctionRef::Scalar(&UPPER)));
    assert_eq!(sealed().get("Null_If"), Some(FunctionRef::Scalar(&NULLIF)));
    for nothing in ["", "9upper", "upper-case", "über", " upper"] {
        assert_eq!(sealed().get(nothing), None, "{nothing:?}");
    }
}

#[test]
fn every_name_and_alias_of_the_catalog_is_a_lower_case_identifier() {
    fn names<F: FunctionType>(function: &'static Function<F>) -> Vec<&'static str> {
        let canonical = &function.canonical;
        std::iter::once(canonical.name)
            .chain(canonical.aliases.iter().copied())
            .collect()
    }
    let identifier = |name: &str| {
        name.starts_with(|first: char| first.is_ascii_lowercase() || first == '_')
            && (name.chars()).all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    };
    for function in registry().all() {
        let names = match function {
            FunctionRef::Scalar(function) => names(function),
            FunctionRef::Aggregate(function, _) => names(&function.declarations),
            FunctionRef::Window(function) => names(function),
        };
        for name in names {
            assert!(identifier(name), "{name:?}");
        }
    }
}

#[test]
fn all_gives_every_function_in_its_catalog_order() {
    assert_eq!(
        sealed().all().collect::<Vec<_>>(),
        [
            FunctionRef::Scalar(&UPPER),
            FunctionRef::Scalar(&NULLIF),
            FunctionRef::Scalar(&SIGN),
            FunctionRef::Scalar(&REGEXP_EXTRACT),
            FunctionRef::Aggregate(&COUNT, AggregateAllowed),
            FunctionRef::Window(&ROW_NUMBER),
        ]
    );
}

#[test]
fn the_lookup_in_canonical_finds_a_canonical_name_and_no_alias() {
    let registry = sealed();
    let find = |name| registry.find(DocumentDialect::Canonical, name);
    assert_eq!(find("nullif"), Some(FunctionRef::Scalar(&NULLIF)));
    assert_eq!(find("sign"), Some(FunctionRef::Scalar(&SIGN)));
    assert_eq!(
        find("count"),
        Some(FunctionRef::Aggregate(&COUNT, AggregateAllowed))
    );
    assert_eq!(find("null_if"), None);
    assert_eq!(find("signum"), None);
}

#[test]
fn the_lookup_in_a_dialect_finds_its_name_or_the_canonical_name_where_it_has_none() {
    let registry = sealed();
    let find = |name| registry.find(DocumentDialect::Dialect(Dialect::DataFusion), name);
    assert_eq!(find("signum"), Some(FunctionRef::Scalar(&SIGN)));
    assert_eq!(find("upper"), Some(FunctionRef::Scalar(&UPPER)));
    assert_eq!(find("row_number"), Some(FunctionRef::Window(&ROW_NUMBER)));
    assert_eq!(find("sign"), None);
    assert_eq!(find("null_if"), None);
}

#[test]
fn the_lookup_in_a_dialect_finds_nothing_where_its_entry_is_none() {
    let registry = sealed();
    let find = |dialect| registry.find(dialect, "regexp_extract");
    assert_eq!(find(DocumentDialect::Dialect(Dialect::DataFusion)), None);
    assert_eq!(
        find(DocumentDialect::Canonical),
        Some(FunctionRef::Scalar(&REGEXP_EXTRACT))
    );
}

#[test]
fn on_gives_the_dialect_declaration_the_canonical_one_or_none() {
    assert_eq!(
        SIGN.on(Dialect::DataFusion)
            .map(|declaration| declaration.name),
        Some("signum")
    );
    assert_eq!(SIGN.on(Dialect::Substrait), Some(&SIGN.canonical));
    assert_eq!(UPPER.on(Dialect::DataFusion), Some(&UPPER.canonical));
    assert_eq!(REGEXP_EXTRACT.on(Dialect::DataFusion), None);
}

#[test]
fn a_static_declaration_names_an_array_type() {
    const STRING: ExactType = ExactType::Type(&DataType::String);
    static ARRAY_ELEMENT: Function<Scalar> = Function {
        canonical: Declaration {
            signature: parameters(&[
                exact("array", ExactType::Array(&DataType::String)),
                exact("index", ExactType::Type(&DataType::Integer)),
            ]),
            specific: Scalar {
                category: Category::Array,
                options: &[],
            },
            ..scalar("array_element")
        },
        dialects: &[],
        releases: &[],
    };
    static REGEXP_MATCH: Function<Scalar> = Function {
        canonical: Declaration {
            signature: parameters(&[exact("value", STRING), exact("pattern", STRING)]),
            returns: ReturnRule::Fixed(ExactType::Array(&DataType::String)),
            nullability: Nullability::Always,
            specific: Scalar {
                category: Category::Pattern,
                options: &[],
            },
            ..scalar("regexp_match")
        },
        dialects: &[],
        releases: &[],
    };
    static CATALOG: FunctionCatalog = FunctionCatalog {
        scalar: &[
            (Category::Pattern, &[&REGEXP_MATCH]),
            (Category::Array, &[&ARRAY_ELEMENT]),
        ],
        ..NO_FUNCTIONS
    };
    assert_eq!(faults(&CATALOG), []);
}

#[test]
fn the_seal_refuses_two_functions_that_share_a_name() {
    static UPPER_AGGREGATE: AggregateFunction = additive(aggregate("upper"));
    static CATALOG: FunctionCatalog = FunctionCatalog {
        scalar: &[(Category::String, &[&UPPER])],
        aggregate: &[&UPPER_AGGREGATE],
        window: &[],
    };
    assert_eq!(faults(&CATALOG), [SealFault::SharedName { name: "upper" }]);
}

#[test]
fn the_seal_refuses_a_name_equal_to_an_alias() {
    static NULL_IF: Function<Scalar> = Function {
        canonical: scalar("null_if"),
        dialects: &[],
        releases: &[],
    };
    static CATALOG: FunctionCatalog = FunctionCatalog {
        scalar: &[(Category::String, &[&NULLIF, &NULL_IF])],
        ..NO_FUNCTIONS
    };
    assert_eq!(
        faults(&CATALOG),
        [SealFault::SharedName { name: "null_if" }]
    );
}

#[test]
fn the_seal_refuses_two_functions_that_share_a_name_in_a_dialect() {
    static X: Function<Scalar> = Function {
        canonical: scalar("x"),
        dialects: &[],
        releases: &[],
    };
    static NO_X: Function<Scalar> = Function {
        canonical: scalar("x"),
        dialects: &[(Dialect::DataFusion, None)],
        releases: &[],
    };
    static Y: Function<Scalar> = Function {
        canonical: scalar("y"),
        dialects: &[(Dialect::DataFusion, Some(scalar("x")))],
        releases: &[],
    };
    static Z: Function<Scalar> = Function {
        canonical: scalar("z"),
        dialects: &[(Dialect::DataFusion, Some(scalar("x")))],
        releases: &[],
    };
    static SHARED: [FunctionCatalog; 3] = [
        FunctionCatalog {
            scalar: &[(Category::String, &[&X, &Y])],
            ..NO_FUNCTIONS
        },
        FunctionCatalog {
            scalar: &[(Category::String, &[&Y, &X])],
            ..NO_FUNCTIONS
        },
        FunctionCatalog {
            scalar: &[(Category::String, &[&Y, &Z])],
            ..NO_FUNCTIONS
        },
    ];
    static NOT_THERE: FunctionCatalog = FunctionCatalog {
        scalar: &[(Category::String, &[&NO_X, &Y])],
        ..NO_FUNCTIONS
    };
    for catalog in &SHARED {
        assert_eq!(
            faults(catalog),
            [SealFault::SharedDialectName {
                dialect: Dialect::DataFusion,
                name: "x"
            }]
        );
    }
    assert_eq!(faults(&NOT_THERE), []);
}

#[test]
fn the_seal_refuses_a_name_or_an_alias_equal_to_an_operations_name() {
    let cases = [
        (one_scalar!("mod" {}), "mod"),
        (one_scalar!("within" { aliases: &["between"] }), "between"),
    ];
    for (catalog, name) in cases {
        assert_eq!(faults(catalog), [SealFault::OperationName { name }]);
    }
}

#[test]
fn coalesce_and_nullif_are_no_tags_of_the_tree() {
    for catalog in [one_scalar!("coalesce" {}), one_scalar!("nullif" {})] {
        assert_eq!(faults(catalog), []);
    }
}

#[test]
fn the_seal_refuses_a_category_that_is_not_its_files() {
    let catalog = one_scalar!("round" {
        specific: Scalar { category: Category::Math, options: &[] },
    });
    assert_eq!(
        faults(catalog),
        [SealFault::CategoryNotItsFile {
            function: "round",
            category: Category::Math,
            file: Category::String
        }]
    );
}

#[test]
fn the_seal_refuses_an_option_value_outside_its_option() {
    let catalog = one_scalar!("concat" {
        specific: Scalar {
            category: Category::String,
            options: &[null_handling(&["ACCEPT_NULLS", "SKIP_NULLS"], "ACCEPT_NULLS")],
        },
    });
    assert_eq!(
        faults(catalog),
        [SealFault::OptionValueOutsideOption {
            function: "concat",
            option: "null_handling",
            value: "SKIP_NULLS"
        }]
    );
}

#[test]
fn the_seal_refuses_a_canonical_value_not_among_the_values() {
    let catalog = one_scalar!("concat" {
        specific: Scalar {
            category: Category::String,
            options: &[null_handling(&["IGNORE_NULLS"], "ACCEPT_NULLS")],
        },
    });
    assert_eq!(
        faults(catalog),
        [SealFault::CanonicalValueNotAmongValues {
            function: "concat",
            option: "null_handling"
        }]
    );
}

#[test]
fn the_seal_refuses_an_optional_parameter_before_a_required_one() {
    let catalog = one_scalar!("substring" {
        signature: parameters(&[
            VALUE,
            number("length", Presence::Defaulted(ParameterDefault::Integer(1))),
            number("start", Presence::Required),
        ]),
    });
    assert_eq!(
        faults(catalog),
        [SealFault::OptionalBeforeRequired {
            function: "substring",
            parameter: "start"
        }]
    );
}

#[test]
fn the_seal_refuses_an_optional_parameter_that_a_later_parameter_follows_with_no_default() {
    let catalog = one_scalar!("regexp_instr" {
        signature: parameters(&[
            VALUE,
            number("start", Presence::Optional),
            number("occurrence", Presence::Optional),
        ]),
    });
    assert_eq!(
        faults(catalog),
        [SealFault::OptionalWithNoDefault {
            function: "regexp_instr",
            parameter: "start"
        }]
    );
}

#[test]
fn the_seal_refuses_a_default_that_is_not_a_literal_of_its_parameters_type() {
    let catalog = one_scalar!("regexp_instr" {
        signature: parameters(&[
            VALUE,
            number("start", Presence::Defaulted(ParameterDefault::String("1"))),
            number("occurrence", Presence::Optional),
        ]),
    });
    assert_eq!(
        faults(catalog),
        [SealFault::DefaultNotOfItsType {
            function: "regexp_instr",
            parameter: "start"
        }]
    );
}

#[test]
fn the_seal_refuses_a_condition_that_names_a_later_argument() {
    let catalog = one_scalar!("date_trunc" {
        signature: parameters(&[
            number("grain", Presence::Required),
            VALUE,
            for_week("timezone", 3),
            for_week("week_start", 0),
        ]),
    });
    assert_eq!(
        faults(catalog),
        [SealFault::ConditionNamesLaterArgument {
            function: "date_trunc",
            parameter: "timezone"
        }]
    );
}

#[test]
fn the_seal_refuses_an_argument_index_out_of_range() {
    let catalogs = [
        one_scalar!("abs" {
            signature: parameters(&[VALUE]),
            returns: ReturnRule::SameAs(1),
        }),
        one_scalar!("abs" {
            signature: parameters(&[VALUE]),
            nullability: Nullability::MirrorArgs(&[0, 1]),
        }),
        one_scalar!("abs" {
            signature: parameters(&[VALUE]),
            nullability: Nullability::ByOption(&[
                ("ACCEPT_NULLS", Nullability::MirrorArgs(&[1])),
            ]),
        }),
        one_scalar!("abs" {
            signature: Signature {
                constraints: &[Constraint::EnumRefused {
                    enumeration: 1,
                    values: &["hour"],
                    argument: 0,
                    ty: ParameterType::Exact(ExactType::Type(&DataType::Date)),
                }],
                ..parameters(&[VALUE])
            },
        }),
    ];
    for catalog in catalogs {
        assert_eq!(
            faults(catalog),
            [SealFault::ArgumentIndexOutOfRange {
                function: "abs",
                index: 1
            }]
        );
    }
}

#[test]
fn the_seal_refuses_an_unbound_type_variable() {
    let catalog = one_scalar!("abs" {
        signature: parameters(&[VALUE]),
        returns: ReturnRule::Var,
    });
    assert_eq!(
        faults(catalog),
        [SealFault::UnboundTypeVariable { function: "abs" }]
    );
}

#[test]
fn the_seal_refuses_a_scalar_with_an_aggregates_nullability() {
    let catalogs = [
        one_scalar!("upper" { nullability: Nullability::EmptyGroupNull }),
        one_scalar!("upper" {
            nullability: Nullability::ByOption(&[("ACCEPT_NULLS", Nullability::EmptyGroupNull)]),
        }),
    ];
    for catalog in catalogs {
        assert_eq!(
            faults(catalog),
            [SealFault::AggregateNullability { function: "upper" }]
        );
    }
}

#[test]
fn the_seal_refuses_an_aggregates_nullability_only_on_a_scalar() {
    static SUM: AggregateFunction = additive(Declaration {
        nullability: Nullability::EmptyGroupNull,
        ..aggregate("sum")
    });
    static FIRST_VALUE: Function<Window> = Function {
        canonical: Declaration {
            nullability: Nullability::EmptyGroupNull,
            ..window("first_value")
        },
        dialects: &[],
        releases: &[],
    };
    static CATALOG: FunctionCatalog = FunctionCatalog {
        scalar: &[],
        aggregate: &[&SUM],
        window: &[&FIRST_VALUE],
    };
    assert_eq!(faults(&CATALOG), []);
}

#[test]
fn the_seal_refuses_two_dialect_declarations_for_one_dialect() {
    let catalog = one_scalar!(Function {
        canonical: scalar("sign"),
        dialects: &[
            (Dialect::DataFusion, Some(scalar("signum"))),
            (Dialect::DataFusion, None),
        ],
        releases: &[],
    });
    assert_eq!(
        faults(catalog),
        [SealFault::TwoDialectDeclarations {
            function: "sign",
            dialect: Dialect::DataFusion
        }]
    );
}

#[test]
fn the_seal_refuses_aliases_on_a_dialect_declaration() {
    let catalog = one_scalar!(Function {
        canonical: scalar("sign"),
        dialects: &[(
            Dialect::DataFusion,
            Some(Declaration {
                aliases: &["sgn"],
                ..scalar("signum")
            })
        )],
        releases: &[],
    });
    assert_eq!(
        faults(catalog),
        [SealFault::DialectAliases {
            function: "sign",
            dialect: Dialect::DataFusion
        }]
    );
}

#[test]
fn the_seal_refuses_aliases_on_a_release_declaration() {
    let catalog = one_scalar!(Function {
        canonical: scalar("sign"),
        dialects: &[],
        releases: &[(
            Dialect::DataFusion,
            Release::DATAFUSION_OLDEST..=Release(53),
            Some(Declaration {
                aliases: &["sgn"],
                ..scalar("signum")
            })
        )],
    });
    assert_eq!(
        faults(catalog),
        [SealFault::DialectAliases {
            function: "sign",
            dialect: Dialect::DataFusion
        }]
    );
}

#[test]
fn a_seal_fault_is_an_error_with_a_message() {
    let fault = SealFault::DialectAliases {
        function: "sign",
        dialect: Dialect::DataFusion,
    };
    assert_eq!(
        fault.message(),
        "`sign`: the DataFusion declaration has aliases"
    );
    assert_eq!(fault.severity_default(), Severity::Error);
}

#[test]
fn the_seal_refuses_a_dialect_declaration_equal_to_the_canonical_one() {
    let cases = [
        (
            one_scalar!(Function {
                canonical: scalar("upper"),
                dialects: &[(Dialect::DataFusion, Some(scalar("upper")))],
                releases: &[],
            }),
            "upper",
        ),
        (
            one_scalar!(Function {
                canonical: Declaration {
                    aliases: &["null_if"],
                    ..scalar("nullif")
                },
                dialects: &[(Dialect::DataFusion, Some(scalar("nullif")))],
                releases: &[],
            }),
            "nullif",
        ),
    ];
    for (catalog, function) in cases {
        assert_eq!(
            faults(catalog),
            [SealFault::DialectEqualsCanonical {
                function,
                dialect: Dialect::DataFusion
            }]
        );
    }
}

#[test]
fn the_seal_checks_a_dialect_declaration_as_it_checks_the_canonical_one() {
    let catalog = one_scalar!(Function {
        canonical: scalar("sign"),
        dialects: &[(
            Dialect::DataFusion,
            Some(Declaration {
                returns: ReturnRule::SameAs(0),
                ..scalar("signum")
            })
        )],
        releases: &[],
    });
    assert_eq!(
        faults(catalog),
        [SealFault::ArgumentIndexOutOfRange {
            function: "sign",
            index: 0
        }]
    );
}

#[test]
fn the_seal_checks_a_release_declaration_as_it_checks_the_canonical_one() {
    let catalog = one_scalar!(Function {
        canonical: scalar("sign"),
        dialects: &[],
        releases: &[(
            Dialect::DataFusion,
            Release::DATAFUSION_OLDEST..=Release(53),
            Some(Declaration {
                returns: ReturnRule::SameAs(0),
                ..scalar("signum")
            })
        )],
    });
    assert_eq!(
        faults(catalog),
        [SealFault::ArgumentIndexOutOfRange {
            function: "sign",
            index: 0
        }]
    );
}

#[test]
fn the_seal_refuses_two_release_entries_of_one_dialect_that_overlap() {
    let catalog = one_scalar!(Function {
        canonical: scalar("sign"),
        dialects: &[],
        releases: &[
            (
                Dialect::DataFusion,
                Release(50)..=Release(53),
                Some(scalar("signum")),
            ),
            (Dialect::DataFusion, Release(52)..=Release(54), None),
        ],
    });
    assert_eq!(
        faults(catalog),
        [SealFault::OverlappingReleases {
            function: "sign",
            dialect: Dialect::DataFusion
        }]
    );
}

#[test]
fn the_seal_refuses_a_release_entry_that_holds_the_reference_release() {
    let catalog = one_scalar!(Function {
        canonical: scalar("sign"),
        dialects: &[],
        releases: &[(
            Dialect::DataFusion,
            Release(53)..=Release::DATAFUSION_REFERENCE,
            Some(scalar("signum")),
        )],
    });
    assert_eq!(
        faults(catalog),
        [SealFault::ReleaseHoldsReference {
            function: "sign",
            dialect: Dialect::DataFusion
        }]
    );
}

fn declared_parameters() -> Vec<(&'static str, Vec<&'static ParameterDecl>)> {
    fn of<F: FunctionType>(
        function: &'static Function<F>,
    ) -> Vec<(&'static str, Vec<&'static ParameterDecl>)> {
        let dialects =
            (function.dialects.iter()).filter_map(|(_, declaration)| declaration.as_ref());
        std::iter::once(&function.canonical)
            .chain(dialects)
            .map(|Declaration { signature, .. }| {
                let parameters = signature.parameters.iter().chain(&signature.variadic);
                (function.canonical.name, parameters.collect())
            })
            .collect()
    }
    registry()
        .all()
        .flat_map(|function| match function {
            FunctionRef::Scalar(function) => of(function),
            FunctionRef::Aggregate(function, _) => of(&function.declarations),
            FunctionRef::Window(function) => of(function),
        })
        .collect()
}

#[test]
fn every_optional_parameter_that_a_later_parameter_follows_declares_a_default() {
    let optional = |parameter: &&ParameterDecl| {
        matches!(
            parameter.presence,
            Presence::Optional | Presence::Defaulted(_)
        )
    };
    let undeclared: Vec<_> = (declared_parameters().into_iter())
        .flat_map(|(function, parameters)| {
            let followed = parameters.into_iter().rev().skip(1).filter(optional);
            (followed.filter(|parameter| parameter.default().is_none()))
                .map(move |parameter| (function, parameter.name))
        })
        .collect();
    assert!(undeclared.is_empty(), "no default: {undeclared:?}");
}

#[test]
fn each_declared_default_is_a_literal_of_its_parameters_type() {
    let mut declared = Vec::new();
    for (function, parameters) in declared_parameters() {
        for parameter in parameters {
            let Some(default) = parameter.default() else {
                continue;
            };
            assert!(
                parameter.ty.takes(&default.data_type()),
                "`{function}`'s `{}`: {default:?}",
                parameter.name
            );
            declared.push((function, parameter.name, default));
        }
    }
    let one = || Literal::Integer(1);
    let text = |value: &str| Literal::String(value.to_owned());
    assert_eq!(
        declared,
        [
            ("regexp_instr", "start", one()),
            ("regexp_instr", "occurrence", one()),
            ("regexp_instr", "flags", text("")),
            ("date_trunc", "timezone", text("UTC")),
            ("date_trunc", "week_start", text("monday")),
            ("lag", "offset", one()),
            ("lag", "offset", one()),
            ("lead", "offset", one()),
            ("lead", "offset", one()),
        ]
    );
}

const fn clauses(distinct: ClauseRule) -> AggregateClauses {
    AggregateClauses {
        distinct,
        filter: ClauseRule::Allowed,
        order_by: ClauseRule::NoEffect,
        within_group: ClauseRule::Refused,
        within_group_key: ParameterType::Any,
        nulls: ClauseRule::Refused,
    }
}

const FRACTION: ParameterDecl = ParameterDecl {
    form: ArgumentForm::Constant,
    ..exact("fraction", ExactType::Type(&DataType::Double))
};

fn aggregate_functions() -> impl Iterator<Item = &'static AggregateFunction> {
    registry().all().filter_map(|function| match function {
        FunctionRef::Aggregate(function, AggregateAllowed) => Some(function),
        _ => None,
    })
}

#[test]
fn the_function_catalog_declares_rule_17s_aggregate_functions() {
    use Additivity::{Additive, NonAdditive};
    use ClauseRule::{Allowed, NoEffect, Refused, Required};
    use Nullability::{Always, EmptyGroupNull, Never};

    struct Row {
        functions: &'static [(&'static str, &'static [&'static str])],
        parameters: &'static [ParameterDecl],
        returns: ReturnRule,
        nullability: Nullability,
        clauses: AggregateClauses,
        additivity: [Additivity; 2],
    }
    const ROW: Row = Row {
        functions: &[],
        parameters: &[VALUE],
        returns: fixed(&DataType::Double),
        nullability: EmptyGroupNull,
        clauses: clauses(Allowed),
        additivity: [NonAdditive; 2],
    };
    const T: ParameterDecl = ParameterDecl {
        ty: ParameterType::Var(None),
        ..VALUE
    };
    const Y_X: &[ParameterDecl] = &[
        number("y", Presence::Required),
        number("x", Presence::Required),
    ];
    const STRING: ParameterDecl = ParameterDecl {
        ty: ParameterType::Family(&TypeFamily::String),
        ..VALUE
    };
    const ROWS: &[Row] = &[
        Row {
            functions: &[("count", &[])],
            parameters: &[ParameterDecl {
                ty: ParameterType::Any,
                ..number("value", Presence::Optional)
            }],
            returns: fixed(&DataType::BigInt),
            nullability: Never,
            additivity: [Additive, NonAdditive],
            ..ROW
        },
        Row {
            functions: &[("sum", &[])],
            returns: ReturnRule::SumOf,
            additivity: [Additive, NonAdditive],
            ..ROW
        },
        Row {
            functions: &[("avg", &[])],
            returns: ReturnRule::AvgOf,
            ..ROW
        },
        Row {
            functions: &[("min", &[]), ("max", &[]), ("any_value", &[])],
            parameters: &[T],
            returns: ReturnRule::Var,
            clauses: clauses(NoEffect),
            additivity: [Additive; 2],
            ..ROW
        },
        Row {
            functions: &[("bool_and", &["every"]), ("bool_or", &["any", "some"])],
            parameters: &[ParameterDecl {
                ty: ParameterType::Family(&TypeFamily::Boolean),
                ..VALUE
            }],
            returns: fixed(&DataType::Boolean),
            clauses: clauses(NoEffect),
            additivity: [Additive; 2],
            ..ROW
        },
        Row {
            functions: &[("stddev_samp", &[]), ("var_samp", &[])],
            nullability: Always,
            ..ROW
        },
        Row {
            functions: &[("stddev_pop", &[]), ("var_pop", &[])],
            ..ROW
        },
        Row {
            functions: &[
                ("covar_samp", &[]),
                ("corr", &[]),
                ("regr_slope", &[]),
                ("regr_intercept", &[]),
                ("regr_r2", &[]),
            ],
            parameters: Y_X,
            nullability: Always,
            clauses: clauses(Refused),
            ..ROW
        },
        Row {
            functions: &[
                ("covar_pop", &[]),
                ("regr_avgx", &[]),
                ("regr_avgy", &[]),
                ("regr_sxx", &[]),
                ("regr_syy", &[]),
                ("regr_sxy", &[]),
            ],
            parameters: Y_X,
            clauses: clauses(Refused),
            ..ROW
        },
        Row {
            functions: &[("regr_count", &[])],
            parameters: Y_X,
            returns: fixed(&DataType::BigInt),
            nullability: Never,
            clauses: clauses(Refused),
            additivity: [Additive; 2],
        },
        Row {
            functions: &[("percentile_cont", &[])],
            parameters: &[FRACTION],
            clauses: AggregateClauses {
                order_by: Refused,
                within_group: Required,
                within_group_key: ParameterType::Family(&TypeFamily::Number),
                ..clauses(Refused)
            },
            ..ROW
        },
        Row {
            functions: &[("median", &[])],
            returns: ReturnRule::AvgOf,
            ..ROW
        },
        Row {
            functions: &[("string_agg", &[])],
            parameters: &[
                STRING,
                ParameterDecl {
                    name: "separator",
                    form: ArgumentForm::Constant,
                    ..STRING
                },
            ],
            returns: fixed(&DataType::String),
            clauses: AggregateClauses {
                order_by: Allowed,
                ..clauses(Allowed)
            },
            ..ROW
        },
        Row {
            functions: &[("approx_count_distinct", &[])],
            parameters: &[ParameterDecl {
                ty: ParameterType::Any,
                ..VALUE
            }],
            returns: fixed(&DataType::BigInt),
            nullability: Never,
            clauses: clauses(Refused),
            ..ROW
        },
    ];
    for row in ROWS {
        for &(name, aliases) in row.functions {
            let function = aggregate_function(name);
            let canonical = &function.declarations.canonical;
            assert_eq!(
                (
                    canonical.name,
                    canonical.aliases,
                    &canonical.signature,
                    &canonical.returns,
                    &canonical.nullability,
                    canonical.volatility,
                    canonical.specific.options,
                    &canonical.specific.clauses,
                    [function.additivity, function.distinct_additivity],
                ),
                (
                    name,
                    aliases,
                    &parameters(row.parameters),
                    &row.returns,
                    &row.nullability,
                    Volatility::Immutable,
                    &[][..],
                    &row.clauses,
                    row.additivity,
                ),
                "{name}"
            );
            for alias in aliases {
                assert_eq!(aggregate_function(alias), function, "{alias}");
            }
        }
    }
    let mut table = ROWS
        .iter()
        .flat_map(|row| row.functions.iter().map(|&(name, _)| name))
        .collect::<Vec<_>>();
    let mut declared = aggregate_functions()
        .map(|function| function.declarations.canonical.name)
        .collect::<Vec<_>>();
    table.sort_unstable();
    declared.sort_unstable();
    assert_eq!(declared, table);
    assert_eq!(declared.len(), 28);
}

fn differences(
    canonical: &Declaration<Aggregate>,
    dialect: &Declaration<Aggregate>,
) -> Vec<&'static str> {
    let (ours, theirs) = (&canonical.specific.clauses, &dialect.specific.clauses);
    [
        ("name", canonical.name != dialect.name),
        ("signature", canonical.signature != dialect.signature),
        ("returns", canonical.returns != dialect.returns),
        ("nullability", canonical.nullability != dialect.nullability),
        ("volatility", canonical.volatility != dialect.volatility),
        (
            "options",
            canonical.specific.options != dialect.specific.options,
        ),
        ("distinct", ours.distinct != theirs.distinct),
        ("filter", ours.filter != theirs.filter),
        ("order_by", ours.order_by != theirs.order_by),
        ("within_group", ours.within_group != theirs.within_group),
        (
            "within_group_key",
            ours.within_group_key != theirs.within_group_key,
        ),
        ("nulls", ours.nulls != theirs.nulls),
    ]
    .into_iter()
    .filter_map(|(part, differs)| differs.then_some(part))
    .collect()
}

fn on_datafusion(
    function: &str,
) -> (
    &'static Declaration<Aggregate>,
    &'static Declaration<Aggregate>,
) {
    let declarations = &aggregate_function(function).declarations;
    let datafusion = declarations
        .on(Dialect::DataFusion)
        .unwrap_or_else(|| panic!("DataFusion has {function}"));
    (&declarations.canonical, datafusion)
}

#[test]
fn on_datafusion_gives_rule_23s_aggregate_declarations() {
    for function in ["stddev_samp", "stddev_pop"] {
        let (canonical, datafusion) = on_datafusion(function);
        assert_eq!(datafusion.specific.clauses.distinct, ClauseRule::Refused);
        assert_eq!(
            differences(canonical, datafusion),
            ["distinct"],
            "{function}"
        );
    }

    let (canonical, datafusion) = on_datafusion("percentile_cont");
    assert_eq!(datafusion.signature, parameters(&[VALUE, FRACTION]));
    let clauses = &datafusion.specific.clauses;
    assert_eq!(
        (clauses.order_by, clauses.within_group),
        (ClauseRule::Allowed, ClauseRule::Refused)
    );
    assert_eq!(
        differences(canonical, datafusion),
        ["signature", "order_by", "within_group"]
    );

    let (canonical, datafusion) = on_datafusion("approx_count_distinct");
    assert_eq!(datafusion.name, "approx_distinct");
    assert_eq!(differences(canonical, datafusion), ["name"]);
}

#[test]
fn rule_23s_rows_are_every_dialect_entry_of_the_catalog() {
    fn entries<F: FunctionType>(
        function: &'static Function<F>,
    ) -> impl Iterator<Item = (&'static str, Dialect, Option<&'static str>)> {
        (function.dialects.iter()).map(|(dialect, declaration)| {
            let name = declaration.as_ref().map(|declaration| declaration.name);
            (function.canonical.name, *dialect, name)
        })
    }
    let catalog: HashSet<_> = registry()
        .all()
        .flat_map(|function| match function {
            FunctionRef::Scalar(function) => entries(function).collect::<Vec<_>>(),
            FunctionRef::Aggregate(function, _) => entries(&function.declarations).collect(),
            FunctionRef::Window(function) => entries(function).collect(),
        })
        .collect();
    let declared = [
        ("position", "strpos"),
        ("concat", "concat"),
        ("regexp_replace", "regexp_replace"),
        ("sign", "signum"),
        ("extract", "date_part"),
        ("date_trunc", "date_trunc"),
        ("greatest", "greatest"),
        ("least", "least"),
        ("stddev_samp", "stddev_samp"),
        ("stddev_pop", "stddev_pop"),
        ("percentile_cont", "percentile_cont"),
        ("approx_count_distinct", "approx_distinct"),
        ("lag", "lag"),
        ("lead", "lead"),
        ("first_value", "first_value"),
        ("last_value", "last_value"),
        ("nth_value", "nth_value"),
    ]
    .map(|(function, name)| (function, Dialect::DataFusion, Some(name)));
    let none = ["regexp_extract", "width_bucket", "date_diff"]
        .map(|function| (function, Dialect::DataFusion, None));
    assert_eq!(catalog, declared.into_iter().chain(none).collect());
}

#[test]
fn the_lookup_of_approx_distinct_in_datafusion_finds_approx_count_distinct() {
    let datafusion = DocumentDialect::Dialect(Dialect::DataFusion);
    let approx_count_distinct = registry().get("approx_count_distinct");
    assert!(approx_count_distinct.is_some());
    assert_eq!(
        registry().find(datafusion, "approx_distinct"),
        approx_count_distinct
    );
    assert_eq!(registry().find(datafusion, "approx_count_distinct"), None);
    assert_eq!(
        registry().find(DocumentDialect::Canonical, "approx_distinct"),
        None
    );
}

#[test]
fn stddev_variance_and_std_dev_name_no_function() {
    for retired in ["stddev", "variance", "std_dev"] {
        assert_eq!(registry().get(retired), None, "{retired}");
    }
}
