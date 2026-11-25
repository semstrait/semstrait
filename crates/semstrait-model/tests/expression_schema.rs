#![cfg(feature = "schemars")]

use semstrait_model::schema::semantic_model_json_schema;
use semstrait_sql::expr::{FunctionRef, OperationKind};
use semstrait_sql::functions::{registry, ArgumentForm, ClauseRule};
use std::iter;

const TREE_TAGS: [&str; 5] = ["lit", "col", "case", "if", "cast"];
const SEMANTIC_TAGS: [&str; 4] = ["field", "dim", "measure", "metric"];

#[test]
fn each_name_and_alias_that_parse_reads_is_a_branch_and_no_other_declared_name_is() {
    let names =
        |name, aliases: &'static [&'static str]| iter::once(name).chain(aliases.iter().copied());
    let mut branches: Vec<&str> = OperationKind::all()
        .map(|kind| &kind.operation().canonical)
        .flat_map(|operation| names(operation.name, operation.aliases))
        .chain(TREE_TAGS)
        .collect();
    let mut barred = Vec::new();
    for function in registry().all() {
        let (bound, name, aliases) = match function {
            FunctionRef::Scalar(f) => (false, f.canonical.name, f.canonical.aliases),
            FunctionRef::Window(f) => {
                let c = &f.canonical.specific.clauses;
                let rules = [c.nulls, c.partition_by, c.order_by, c.window_frame];
                (
                    rules.contains(&ClauseRule::Required),
                    f.canonical.name,
                    f.canonical.aliases,
                )
            }
            FunctionRef::Aggregate(f, _) => {
                let canonical = &f.declarations.canonical;
                (true, canonical.name, canonical.aliases)
            }
            _ => unreachable!("a function is scalar, aggregate or window"),
        };
        if bound { &mut barred } else { &mut branches }.extend(names(name, aliases));
    }
    let schema = semantic_model_json_schema();
    let tags = |tree: &str| {
        let mut tags: Vec<&str> = schema["$defs"][tree]["oneOf"]
            .as_array()
            .expect("a oneOf of branches")
            .iter()
            .filter_map(|branch| branch["required"][0].as_str())
            .collect();
        tags.sort_unstable();
        tags
    };
    branches.sort_unstable();
    assert_eq!(tags("UnresolvedPhysicalExpr"), branches);
    branches.extend(SEMANTIC_TAGS);
    branches.sort_unstable();
    assert_eq!(tags("UnresolvedSemanticExpr"), branches);
    for name in ["sum", "every", "rank", "lag", "lead"] {
        assert!(barred.contains(&name), "`{name}` gets no branch");
    }
}

#[test]
fn each_enum_parameter_s_declared_values_appear_in_its_function_s_branch() {
    let schema = semantic_model_json_schema();
    let branches = schema["$defs"]["UnresolvedPhysicalExpr"]["oneOf"]
        .as_array()
        .expect("a oneOf of branches");
    for function in registry().all() {
        let (name, signature) = function.canonical();
        for parameter in signature.parameters {
            let ArgumentForm::Enum(values) = parameter.form else {
                continue;
            };
            let branch = (branches.iter())
                .find(|branch| branch["required"][0] == name)
                .expect("a branch")
                .to_string();
            for value in values {
                assert!(
                    branch.contains(&format!("\"{value}\"")),
                    "`{name}`'s `{}` offers `{value}`",
                    parameter.name
                );
            }
        }
    }
}

const SEMANTIC: usize = 0;
const MAPPING: usize = 1;

const POSITIONS: [(usize, &str); 13] = [
    (SEMANTIC, "dimensions: [{name: d, data_type: string, type: categorical, expr: EXPR}]"),
    (SEMANTIC, "measures: [{name: m, data_type: integer, agg: sum, expr: EXPR}]"),
    (SEMANTIC, "measures: [{name: m, data_type: integer, agg: sum, filter: EXPR}]"),
    (SEMANTIC, "metrics: [{name: k, data_type: integer, expr: EXPR}]"),
    (SEMANTIC, "metrics: [{name: k, data_type: integer, expr: m, filter: EXPR}]"),
    (SEMANTIC, "datasets: [{name: o, dimensions: [{ref: d, expr: EXPR}]}]"),
    (SEMANTIC, "datasets: [{name: o, measures: [{ref: m, expr: EXPR}]}]"),
    (SEMANTIC, "datasets: [{name: o, measures: [{ref: m, filter: EXPR}]}]"),
    (SEMANTIC, "datasets: [{name: o, metrics: [{ref: k, expr: EXPR}]}]"),
    (SEMANTIC, "datasets: [{name: o, metrics: [{ref: k, filter: EXPR}]}]"),
    (SEMANTIC, "datasets: [{name: o, filters: [{name: f, expr: EXPR}]}]"),
    (SEMANTIC, "relationships: [{name: r, from: o, to: c, on: [d], cardinality: many_to_one, filter: EXPR}]"),
    (MAPPING, "datasets: [{name: o, extras: {semantic_mapping: {revenue: {expr: EXPR}}}}]"),
];

const SHAPES: &[(&str, [bool; 2])] = &[
    ("amount_cents", [true, true]),
    ("\"'DE'\"", [true, true]),
    ("42", [true, true]),
    ("true", [true, true]),
    ("null", [true, true]),
    ("\"+5\"", [true, true]),
    ("\"a / b\"", [false, false]),
    ("\"\"", [false, false]),
    ("{upper: \"a / b\"}", [true, true]),
    ("{col: amount_cents}", [true, true]),
    ("{field: revenue}", [true, false]),
    ("{dim: country}", [true, false]),
    ("{measure: revenue}", [true, false]),
    ("{metric: margin}", [true, false]),
    ("{frobnicate: [amount_cents]}", [false, false]),
    ("{sum: [amount_cents]}", [false, false]),
    ("{multiply: [amount_cents, 0.01]}", [true, true]),
    ("{null_if: [amount_cents, 0]}", [true, true]),
    ("{nullif: [amount_cents, 0]}", [true, true]),
    ("{UPPER: amount_cents}", [true, true]),
    ("{Null_If: [amount_cents, 0]}", [true, true]),
    ("{Row_Number: []}", [true, true]),
    ("{Multiply: [amount_cents, 0.01]}", [false, false]),
    ("{row_number: []}", [true, true]),
    ("{row_number: amount_cents}", [false, false]),
    ("{rank: []}", [false, false]),
    ("{col: a, lit: 1}", [false, false]),
    ("{}", [false, false]),
    ("{eq: [{dim: country}, \"'DE'\"]}", [true, false]),
    ("{eq: [\"a / b\", 1]}", [true, true]),
    ("{multiply: [amount_cents]}", [false, false]),
    ("{multiply: [amount_cents, 2, 3]}", [false, false]),
    ("{multiply: {left: amount_cents, right: 2}}", [true, true]),
    ("{multiply: {left: amount_cents}}", [false, false]),
    ("{multiply: {left: amount_cents, right: 2, scale: 3}}", [false, false]),
    ("{multiply: {a: 1, b: 2}}", [false, false]),
    ("{multiply: amount_cents}", [false, false]),
    ("{negate: amount_cents}", [true, true]),
    ("{negate: []}", [false, false]),
    ("{negate: [amount_cents, 2]}", [false, false]),
    ("{not: {eq: [amount_cents, 2]}}", [true, true]),
    ("{not: [amount_cents, 2]}", [false, false]),
    ("{is_null: amount_cents}", [true, true]),
    ("{is_null: [amount_cents, 2]}", [false, false]),
    ("{is_null: []}", [false, false]),
    ("{like: {value: country, pattern: \"'D%'\"}}", [true, true]),
    ("{like: {value: country}}", [false, false]),
    ("{like: {value: country, pattern: \"'D%'\", escape: \"'!'\"}}", [false, false]),
    ("{like: [country, \"'D%'\"]}", [false, false]),
    ("{between: {value: amount_cents, low: 1, high: 2}}", [true, true]),
    ("{between: {value: amount_cents, low: 1}}", [false, false]),
    ("{between: {value: amount_cents, low: 1, high: 2, step: 3}}", [false, false]),
    ("{in: {value: amount_cents, list: [1, 2]}}", [true, true]),
    ("{in: {value: amount_cents, list: []}}", [false, false]),
    ("{in: {value: amount_cents}}", [false, false]),
    ("{in: {value: amount_cents, list: [1], also: 2}}", [false, false]),
    ("{in: [amount_cents, 1, 2]}", [false, false]),
    ("{lit: 5}", [true, true]),
    ("{lit: acme}", [true, true]),
    ("{lit: null}", [true, true]),
    ("{lit: [1, 2]}", [false, false]),
    ("{lit: {bigint: 5}}", [true, true]),
    ("{lit: {string: acme}}", [true, true]),
    ("{lit: {bigint: null}}", [true, true]),
    ("{lit: {long: 5}}", [false, false]),
    ("{lit: {BIGINT: 5}}", [false, false]),
    ("{lit: {frobnicate: 5}}", [false, false]),
    ("{lit: {bigint: 5, integer: 6}}", [false, false]),
    ("{lit: {binary: [1, 2]}}", [true, true]),
    ("{lit: {array: [1, 2]}}", [false, false]),
    ("{lit: {array: {element: bigint, values: [1, 2]}}}", [true, true]),
    ("{lit: {array: {element: {decimal: {precision: 18, scale: 2}}, values: [\"1.5\"]}}}", [true, true]),
    ("{lit: {array: {element: \"array<bigint>\", values: [[1]]}}}", [true, true]),
    ("{lit: {array: {element: bigint}}}", [false, false]),
    ("{lit: {array: {element: frobnicate, values: [1]}}}", [false, false]),
    ("{lit: {array: {element: bigint, values: [1], extra: 2}}}", [false, false]),
    ("{lit: {array: {element: bigint, values: null}}}", [true, true]),
    ("{lit: {array: {element: bigint, values: {a: 1}}}}", [false, false]),
    ("{lit: {null: bigint}}", [true, true]),
    ("{lit: {\"null\": long}}", [true, true]),
    ("{lit: {\"null\": {varchar: {length: 10}}}}", [true, true]),
    ("{lit: {\"null\": varchar}}", [false, false]),
    ("{lit: {\"null\": {varchar: {}}}}", [false, false]),
    ("{lit: {decimal: {value: \"1.5\", precision: 18, scale: 4}}}", [true, true]),
    ("{lit: {decimal: {value: \"1.5\"}}}", [true, true]),
    ("{lit: {decimal: {precision: 18}}}", [false, false]),
    ("{lit: {bigint: {value: 5, precision: 3}}}", [false, false]),
    ("{lit: {decimal: {value: \"1.5\", digits: 3}}}", [false, false]),
    ("{lit: {binary: {value: [1, 2], length: 2}}}", [true, true]),
    ("{lit: {binary: {value: [1, 2], precision: 2}}}", [false, false]),
    ("{lit: {timestamp: {value: \"2026-01-01 00:00:00\", precision: 6}}}", [true, true]),
    ("{lit: {timestamp: {value: \"2026-01-01 00:00:00\", precision: 5}}}", [false, false]),
    ("{lit: {decimal: {value: \"1\", precision: 99}}}", [false, false]),
    ("{lit: {decimal: {value: \"1\", precision: 0}}}", [false, false]),
    ("{lit: {varchar: {value: \"ab\", length: 0}}}", [false, false]),
    ("{lit: {decimal: {value: {a: 1}}}}", [false, false]),
    ("{col: [a, b]}", [false, false]),
    ("{col: \"\"}", [false, false]),
    ("{col: 5}", [false, false]),
    ("{field: {name: revenue}}", [false, false]),
    ("{field: \"\"}", [false, false]),
    ("{dim: {name: country, accessor: first}}", [true, false]),
    ("{dim: {name: country, accessor: {lag: 2}}}", [true, false]),
    ("{dim: {name: country, accessor: {first: 1}}}", [true, false]),
    ("{dim: {name: country, accessor: previous}}", [false, false]),
    ("{dim: {name: country, accessor: {lag: -1}}}", [false, false]),
    ("{dim: {accessor: first}}", [false, false]),
    ("{dim: {name: country, extra: 1}}", [false, false]),
    ("{dim: [country]}", [false, false]),
    ("{measure: {name: revenue, accessor: delta}}", [true, false]),
    ("{measure: {name: revenue, accessor: {lead: 1}}}", [true, false]),
    ("{measure: {name: revenue, accessor: first}}", [false, false]),
    ("{measure: {name: revenue, accessor: {lead: 4294967296}}}", [false, false]),
    ("{metric: {name: margin, accessor: percent_change}}", [true, false]),
    ("{metric: {name: margin, accessor: {lag: 1, lead: 2}}}", [false, false]),
    ("{metric: {name: \"\"}}", [false, false]),
    ("{case: {whens: [{when: {eq: [amount_cents, 1]}, then: 2}], else: 3}}", [true, true]),
    ("{case: {whens: [{when: true, then: 2}]}}", [true, true]),
    ("{case: {whens: []}}", [false, false]),
    ("{case: {whens: [{when: true}]}}", [false, false]),
    ("{case: {whens: [{when: true, then: 2, else: 3}]}}", [false, false]),
    ("{case: {else: 3}}", [false, false]),
    ("{case: {whens: [{when: true, then: 2}], default: 3}}", [false, false]),
    ("{case: [true, 2]}", [false, false]),
    ("{if: {cond: true, then: 1, else: 2}}", [true, true]),
    ("{if: {cond: true, then: 1}}", [true, true]),
    ("{if: {cond: true}}", [false, false]),
    ("{if: {cond: true, then: 1, otherwise: 2}}", [false, false]),
    ("{cast: {input: amount_cents, target: bigint}}", [true, true]),
    ("{cast: {input: amount_cents, target: {decimal: {precision: 18, scale: 2}}, on_failure: \"null\"}}", [true, true]),
    ("{cast: {input: amount_cents, target: bigint, on_failure: error}}", [true, true]),
    ("{cast: {input: amount_cents, target: bigint, on_failure: null}}", [false, false]),
    ("{cast: {input: amount_cents, target: bigint, on_failure: skip}}", [false, false]),
    ("{cast: {input: amount_cents}}", [false, false]),
    ("{cast: {input: amount_cents, target: varchar}}", [false, false]),
    ("{cast: {input: amount_cents, target: bigint, mode: x}}", [false, false]),
    ("{upper: [amount_cents, amount_cents]}", [false, false]),
    ("{UPPER: [amount_cents, amount_cents]}", [false, false]),
    ("{round: amount_cents}", [true, true]),
    ("{round: []}", [false, false]),
    ("{round: [amount_cents]}", [true, true]),
    ("{round: [amount_cents, 2]}", [true, true]),
    ("{round: [amount_cents, 2, 3]}", [false, false]),
    ("{round: {value: amount_cents}}", [true, true]),
    ("{round: {value: amount_cents, scale: 2}}", [true, true]),
    (
        "{round: {value: amount_cents, scale: 2, extra: 1}}",
        [false, false],
    ),
    ("{round: {scale: 2}}", [false, false]),
    ("{round: {}}", [false, false]),
    ("{greatest: amount_cents}", [true, true]),
    ("{greatest: []}", [false, false]),
    ("{concat_ws: \"'-'\"}", [false, false]),
    ("{concat_ws: [\"'-'\"]}", [false, false]),
    ("{concat_ws: [\"'-'\", a, b, c]}", [true, true]),
    ("{concat_ws: {separator: \"'-'\"}}", [false, false]),
    ("{null_if: {left: amount_cents, right: 0}}", [true, true]),
    ("{null_if: {left: amount_cents, rigth: 0}}", [false, false]),
    ("{round: [amount_cents, {lit: 2}]}", [true, true]),
    ("{round: [amount_cents, \"'2'\"]}", [true, true]),
    ("{round: [amount_cents, {col: s}]}", [false, false]),
    (
        "{round: {value: amount_cents, scale: {upper: s}}}",
        [false, false],
    ),
    ("{date_trunc: [day, order_ts]}", [true, true]),
    ("{date_trunc: [\"'day'\", order_ts]}", [true, true]),
    (
        "{date_trunc: {granularity: {lit: day}, value: order_ts}}",
        [true, true],
    ),
    (
        "{date_trunc: [{lit: {string: day}}, order_ts]}",
        [true, true],
    ),
    (
        "{date_trunc: [{lit: {string: {value: day}}}, order_ts]}",
        [true, true],
    ),
    (
        "{date_trunc: [{lit: {varchar: day}}, order_ts]}",
        [false, false],
    ),
    ("{date_trunc: [dya, order_ts]}", [false, false]),
    ("{date_trunc: [\"'dya'\", order_ts]}", [false, false]),
    ("{date_trunc: [{lit: \"'day'\"}, order_ts]}", [false, false]),
    ("{date_trunc: [{col: grain}, order_ts]}", [false, false]),
    ("{first_value: [amount_cents]}", [true, true]),
    (
        "{first_value: {value: amount_cents, order_by: [d]}}",
        [false, false],
    ),
    ("{row_number: {partition_by: [d]}}", [false, false]),
];

const DIFFERENCES: &[(&str, [bool; 2], [bool; 2], &str)] = &[
    (
        "{eq: [countyr, \"'DE'\"]}",
        [true, true],
        [true, true],
        "compile: a name that matches no declaration",
    ),
    (
        "{eq: [{upper: [42]}, \"'DE'\"]}",
        [true, true],
        [true, true],
        "compile: an argument type",
    ),
    (
        "{date_trunc: {granularity: day, value: order_ts, timezone: \"'UTC'\"}}",
        [true, true],
        [true, true],
        "compile: a conditioned parameter's condition on a type",
    ),
    (
        "{date_trunc: {granularity: day, value: order_ts, week_start: monday}}",
        [false, false],
        [true, true],
        "parse: a conditioned parameter's condition on a value",
    ),
    (
        "{date_trunc: [day, order_ts, \"'UTC'\", monday]}",
        [false, false],
        [true, true],
        "parse: a conditioned parameter's condition on a value",
    ),
    (
        "{regexp_extract: [v, {lit: \"'(a)(b)'\"}, {lit: 3}]}",
        [false, false],
        [true, true],
        "parse: a `regexp_extract` group that its pattern lacks",
    ),
    (
        "{cast: {input: v, target: \"decimal(10,12)\"}}",
        [false, false],
        [true, true],
        "parse: a decimal whose scale is above its precision",
    ),
    (
        "{cast: {input: v, target: \"array<array<bigint>\"}}",
        [false, false],
        [true, true],
        "parse: unbalanced `array<…>` layers",
    ),
    (
        "{lit: {tinyint: 300}}",
        [false, false],
        [true, true],
        "parse: a literal value that its type does not hold",
    ),
    (
        "{lit: {date: \"2026-02-30\"}}",
        [false, false],
        [true, true],
        "parse: a literal value that its type does not hold",
    ),
    (
        "{lit: {bigint: [1, 2]}}",
        [false, false],
        [true, true],
        "parse: a literal value that its type does not hold",
    ),
    (
        "{lit: {decimal: {value: \"1\", precision: 10, scale: 12}}}",
        [false, false],
        [true, true],
        "parse: a decimal whose scale is above its precision",
    ),
    (
        "{upper: \"\\\"\\\"\"}",
        [false, false],
        [true, true],
        "parse: an empty quoted identifier",
    ),
    (
        "\".x\"",
        [false, false],
        [true, true],
        "parse: an accessor without a name",
    ),
    (
        "{eq: [\"x.\", 1]}",
        [false, false],
        [true, true],
        "parse: a trailing dot",
    ),
    (
        "{upper: \"a.b.c\"}",
        [false, false],
        [true, true],
        "parse: an accessor that holds a dot",
    ),
    (
        "country.first",
        [false, false],
        [true, true],
        "parse: an accessor outside a dim:, measure: or metric: body",
    ),
    (
        "{lit: {varchar: {value: ab, length: 2.0}}}",
        [false, false],
        [true, true],
        "parse: a whole number written as a float",
    ),
    (
        "{dim: {name: country, accessor: {lag: 2.0}}}",
        [false, false],
        [true, false],
        "parse: a whole number written as a float",
    ),
];

#[test]
fn at_each_expression_position_the_schema_gives_parse_s_verdict_but_for_the_listed_differences() {
    let schema = semantic_model_json_schema();
    let validator = jsonschema::validator_for(&schema).expect("the schema compiles");
    let rows = SHAPES
        .iter()
        .map(|&(shape, parse)| (shape, parse, parse, ""));
    let differences = DIFFERENCES.iter().copied();
    for (shape, parse, schema, why) in rows.chain(differences) {
        for (tree, position) in POSITIONS {
            let document = format!(
                "semantic_model:\n  name: parity\n  {}",
                position.replace("EXPR", shape)
            );
            let instance: serde_json::Value =
                serde_yaml::from_str(&document).expect("a YAML document");
            let verdicts = (
                semstrait_model::parse(&document).is_ok(),
                validator.is_valid(&instance),
            );
            assert_eq!(
                verdicts,
                (parse[tree], schema[tree]),
                "(parse, schema) of {document} {why}"
            );
        }
    }
}
