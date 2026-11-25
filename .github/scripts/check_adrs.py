#!/usr/bin/env python3
"""Repository-level ADR checks that `adrs doctor` does not make.

Run from the repo root after `adrs doctor`. Reads `adrs export json` and the ADR files.
Prints `path: ERROR code: message` and exits 1 on any error.

  legacy-header   an ADR still uses the old `Status:` / `Source:` header
  status          the status is not one `adrs doctor` accepts (ADR007); doctor only warns
  number          the filename number differs from the frontmatter number
  title           the H1 differs from the frontmatter title
  link-target     a link points to an ADR that does not exist
  link-inverse    a link has no inverse link on its target (amends <-> amended-by, ...)
  link-duplicate  the same link appears twice
  superseded      status `superseded` without a `superseded-by` link, or the reverse
  callout         the Amended/Superseded callout under the H1 does not list exactly the
                  ADRs that the `amended-by` / `superseded-by` links name
  callout-duplicate  the callout names the same ADR in two bullets, or in two leads
  cite-missing    the body cites `ADR NNNN`, or a list such as `ADRs NNNN, NNNN and NNNN`,
                  and one of the numbers has no ADR
  cite-path       a `[ADR NNNN](file)` link points to a file other than ADR NNNN's
"""
import json
import re
import subprocess
import sys
from pathlib import Path

INVERSE = {
    "amends": "amended-by",
    "amended-by": "amends",
    "supersedes": "superseded-by",
    "superseded-by": "supersedes",
    "relates-to": "relates-to",
}

# The values `adrs doctor` (0.12.1, rule ADR007) accepts, compared without regard to case.
VALID_STATUSES = {"proposed", "accepted", "deprecated", "superseded", "rejected", "draft"}

errors = []


def error(path, code, msg):
    errors.append(f"{path}: ERROR {code}: {msg}")


# `ADR 0067`, `ADRs 0067 and 0068`, `ADRs 0062, 0063 and 0072`; a list may wrap a line.
CITATION = re.compile(r"\bADRs? (\d{4}(?:(?:,\s*|,?\s+(?:and|or)\s+)\d{4})*)")
# `[ADR 0074](0074-slug.md)`: the callout's form of a citation.
ADR_LINK = re.compile(r"\[ADR (\d{4})\]\(([^)\s]+)\)")


def callout_targets(body):
    """ADR numbers named by the blockquote directly under the H1, split by kind.

    Lists, not sets: a number named twice must reach the duplicate check.
    """
    m = re.search(r"^# .+\n\s*\n((?:>.*\n)+)", body, re.M)
    if not m:
        return [], []
    quote = m.group(1)
    superseded = re.findall(r"\*\*Superseded by \[ADR (\d{4})\]", quote)
    amended = []
    if "**Amended.**" in quote:
        amended = re.findall(r"^>\s*[-*]\s*\[ADR (\d{4})\]", quote, re.M)
    return [int(n) for n in amended], [int(n) for n in superseded]


def main():
    out = subprocess.run(["adrs", "export", "json"], capture_output=True, text=True, check=True).stdout
    adrs = {a["number"]: a for a in json.loads(out)["adrs"]}

    for num, a in sorted(adrs.items()):
        path = Path(a["path"])
        rel = path.relative_to(Path.cwd()) if path.is_absolute() else path
        text = path.read_text()
        fm = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
        body = text[fm.end():] if fm else text

        if re.match(r"\A---\nStatus:", text) or (fm and re.search(r"^legacy-status:", fm.group(1), re.M)):
            error(rel, "legacy-header", "convert the old Status:/Source: header to frontmatter")
        if (a.get("status") or "").lower() not in VALID_STATUSES:
            error(rel, "status", f"status {a.get('status')!r} is not one of {', '.join(sorted(VALID_STATUSES))}")
        if not path.name.startswith(f"{num:04d}-"):
            error(rel, "number", f"frontmatter number {num} does not match the filename")
        h1 = re.search(r"^# (.+)$", body, re.M)
        if not h1 or h1.group(1).strip() != a["title"].strip():
            error(rel, "title", "the H1 must equal the frontmatter title")

        links = {}
        pairs = [(l["type"], l["target"]) for l in a.get("links") or []]
        for dup in sorted({p for p in pairs if pairs.count(p) > 1}):
            error(rel, "link-duplicate", f"{dup[0]} {dup[1]} appears more than once")
        for link in a.get("links") or []:
            links.setdefault(link["type"], set()).add(link["target"])
            tgt = adrs.get(link["target"])
            if not tgt:
                error(rel, "link-target", f"{link['type']} {link['target']}, which does not exist")
                continue
            inv = INVERSE.get(link["type"])
            if inv and not any(l["type"] == inv and l["target"] == num for l in tgt.get("links") or []):
                error(rel, "link-inverse", f"{link['type']} {link['target']}, but ADR {link['target']} has no {inv} {num}")

        superseded_by = links.get("superseded-by", set())
        if (a["status"].lower() == "superseded") != bool(superseded_by):
            error(rel, "superseded", "status superseded and a superseded-by link go together")

        amended, superseded = callout_targets(body)
        for kind, named in (("amended-by", amended), ("superseded-by", superseded)):
            for dup in sorted({n for n in named if named.count(n) > 1}):
                error(rel, "callout-duplicate", f"callout names {kind} ADR {dup:04d} more than once; write one per link")
        if set(amended) != links.get("amended-by", set()):
            error(rel, "callout", f"callout lists amended-by {sorted(set(amended))}, links say {sorted(links.get('amended-by', set()))}")
        if set(superseded) != superseded_by:
            error(rel, "callout", f"callout lists superseded-by {sorted(set(superseded))}, links say {sorted(superseded_by)}")

        cited = {n for m in CITATION.finditer(body) for n in re.findall(r"\d{4}", m.group(1))}
        for n in sorted(cited):
            if int(n) not in adrs:
                error(rel, "cite-missing", f"cites ADR {n}, which is not on this branch; cite an unmerged ADR by its ticket")

        for m in ADR_LINK.finditer(body):
            tgt = adrs.get(int(m.group(1)))
            if tgt and m.group(2).split("#")[0] != Path(tgt["path"]).name:
                error(rel, "cite-path", f"[ADR {m.group(1)}] links to {m.group(2)}; link to {Path(tgt['path']).name}")

    for e in errors:
        print(e)
    print(f"check_adrs: {len(adrs)} ADR(s); {len(errors)} error(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
