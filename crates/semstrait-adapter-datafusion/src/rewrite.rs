use semstrait_ir::PhysicalExpr;
use semstrait_sql::expr::{
    Call, Expr, ExprOperationExt, FunctionRef, Literal, OptionSetting, ScalarCall,
};
use semstrait_sql::functions::{registry, OptionName, ScalarOption};

pub(crate) fn expression(expr: &PhysicalExpr) -> Option<PhysicalExpr> {
    let Expr::Function(Call::Scalar(offered)) = expr else {
        return None;
    };
    let null_handling = ScalarOption::NullHandling.name();
    match offered.function().canonical.name {
        "concat" if !(offered.options().iter()).any(|option| option.name == null_handling) => {
            let any_null = (offered.args().iter().cloned())
                .map(ExprOperationExt::is_null)
                .reduce(|left, right| left | right)?;
            Some(Expr::Case {
                whens: vec![(!any_null, skipping(offered.args().to_vec())?)],
                else_: None,
            })
        }
        "regexp_extract" => {
            let [value, pattern, group @ Expr::Literal(Literal::Integer(n))] = offered.args()
            else {
                return None;
            };
            let n = u32::try_from(*n).ok()?;
            let text = |text: &str| Expr::Literal(Literal::String(text.to_owned()));
            let searched = vec![value.clone(), pattern.clone()];
            let like = call("regexp_like", searched.clone(), Vec::new())?;
            let matched = if n == 0 {
                like
            } else {
                let one = || Expr::Literal(Literal::Integer(1));
                let args = [searched, vec![one(), one(), text(""), group.clone()]].concat();
                let position = call("regexp_instr", args, Vec::new())?;
                like & position.gt(Expr::Literal(Literal::BigInt(0)))
            };
            let whole = skipping(vec![text("^(?s:.*?)("), pattern.clone(), text(")(?s:.*)$")])?;
            let replacement = text(&format!("\\{}", u64::from(n) + 1));
            let args = vec![value.clone(), whole, replacement];
            let extracted = call("regexp_replace", args, Vec::new())?;
            Some(Expr::Case {
                whens: vec![(matched, extracted)],
                else_: None,
            })
        }
        _ => None,
    }
}

fn skipping(args: Vec<PhysicalExpr>) -> Option<PhysicalExpr> {
    let ignore = OptionSetting {
        name: ScalarOption::NullHandling.name().to_owned(),
        value: "IGNORE_NULLS".to_owned(),
    };
    call("concat", args, vec![ignore])
}

fn call(
    function: &str,
    args: Vec<PhysicalExpr>,
    options: Vec<OptionSetting>,
) -> Option<PhysicalExpr> {
    let Some(FunctionRef::Scalar(function)) = registry().get(function) else {
        return None;
    };
    let call = ScalarCall::builder(function, args).options(options).build();
    Some(Expr::Function(Call::Scalar(call.ok()?)))
}
