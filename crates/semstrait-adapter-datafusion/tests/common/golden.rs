use std::error::Error;
use std::future::Future;

use semstrait_common::Release;

use super::case::{split, Case};

pub const GENERATOR: &str = "generate_goldens";

pub const SWITCH: &str = "SEMSTRAIT_GOLDENS";

#[derive(Debug)]
pub struct Golden {
    file: String,
    key: String,
    field: &'static str,
    rendered: String,
}

impl Golden {
    pub fn case(case: &str, field: &'static str, rendered: String) -> Result<Self, Box<dyn Error>> {
        let (file, key) = split(case)?;
        Ok(Self {
            file: file.to_owned(),
            key: key.to_owned(),
            field,
            rendered,
        })
    }

    pub fn assert_committed(&self) {
        if super::rules().release() != Release::DATAFUSION_REFERENCE {
            return;
        }
        let Self {
            file,
            key,
            field,
            rendered,
        } = self;
        let id = format!("{file}/{key}");
        let case = Case::read(&id).unwrap_or_else(|error| panic!("{error}"));
        let named = format!("`{file}.yaml`'s `{key}`, field `{field}`");
        let committed = case
            .field(field)
            .unwrap_or_else(|| panic!("no {named}: write it with `{}`", command()));
        assert_eq!(rendered, committed, "{}", mismatch(&named));
    }
}

fn mismatch(named: &str) -> String {
    format!(
        "{named} is not what its gate renders: a change an ADR rules regenerates it with `{}`, \
         and its pull request names each class of change; any other change is a defect",
        command()
    )
}

fn command() -> String {
    format!(
        "{SWITCH}=1 cargo test -p {} --test {} {GENERATOR}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_CRATE_NAME")
    )
}

pub fn asked(mut args: impl Iterator<Item = String>, switch: bool) -> bool {
    switch && args.any(|arg| arg == GENERATOR)
}

pub async fn generate(gates: impl Future<Output = Vec<Golden>>) {
    let switch = std::env::var_os(SWITCH).is_some();
    if !asked(std::env::args().skip(1), switch) {
        return;
    }
    if super::rules().release() != Release::DATAFUSION_REFERENCE {
        return;
    }
    for golden in gates.await {
        let Golden {
            file,
            key,
            field,
            rendered,
        } = golden;
        let path = super::fixture(&format!("{file}.yaml"));
        let text = std::fs::read_to_string(&path).expect("read the problem file");
        let spliced = splice(&text, &key, field, &rendered)
            .unwrap_or_else(|error| panic!("{file}/{key}: {error}"));
        std::fs::write(&path, spliced).expect("write the problem file");
    }
}

fn indicator(text: &str) -> Result<&'static str, Box<dyn Error>> {
    if text.contains('\r') {
        return Err("a `\\r` in the rendered text".into());
    }
    let first = text.split('\n').next().unwrap_or("");
    if first.starts_with(' ') || first.starts_with('\t') {
        return Err("the rendered text's first line starts with whitespace".into());
    }
    if text.ends_with("\n\n") {
        return Err("the rendered text ends with two or more newlines".into());
    }
    Ok(if text.ends_with('\n') { "|" } else { "|-" })
}

fn content_lines(text: &str) -> Vec<String> {
    let body = text.strip_suffix('\n').unwrap_or(text);
    body.split('\n')
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("    {line}")
            }
        })
        .collect()
}

fn field_lines(field: &str, text: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut lines = vec![format!("  {field}: {}", indicator(text)?)];
    lines.extend(content_lines(text));
    Ok(lines)
}

fn splice(text: &str, key: &str, field: &str, rendered: &str) -> Result<String, Box<dyn Error>> {
    let mut lines: Vec<String> = text.split('\n').map(str::to_owned).collect();
    let marker = format!("{key}:");
    let start = (lines.iter())
        .position(|line| *line == marker)
        .ok_or_else(|| format!("no block `{marker}`"))?;
    let end = ((start + 1)..lines.len())
        .find(|&i| lines[i].chars().next().is_some_and(|c| !c.is_whitespace()))
        .unwrap_or(lines.len());
    let prefix = format!("{field}:");
    let is_header = |line: &str| {
        line.strip_prefix("  ")
            .is_some_and(|rest| !rest.starts_with(' ') && rest.starts_with(prefix.as_str()))
    };
    let replacement = field_lines(field, rendered)?;
    if let Some(header) = ((start + 1)..end).find(|&i| is_header(&lines[i])) {
        let raw_end = ((header + 1)..end)
            .find(|&i| !(lines[i].is_empty() || lines[i].starts_with("    ")))
            .unwrap_or(end);
        let mut field_end = raw_end;
        while field_end > header + 1 && lines[field_end - 1].is_empty() {
            field_end -= 1;
        }
        lines.splice(header..field_end, replacement);
    } else {
        let last = ((start..end).rev())
            .find(|&i| !lines[i].trim().is_empty())
            .ok_or("an empty block")?;
        lines.splice(last + 1..last + 1, replacement);
    }
    Ok(lines.join("\n"))
}
