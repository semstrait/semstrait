---
id: 18-ai-context
status: ratified
version: 2.1
impl: implemented
---

# AI Context

This leaf owns the `AiContext` struct — LLM/agent-facing hint surface (synonyms, description, examples, unit). It attaches to `SemanticModel`, top-level DataKinds, and individual Semantics.

## 8. `AiContext`

`AiContext` is the LLM / agent-facing hint surface. It attaches to root-level SemanticModel, top-level data kinds, and individual Semantics — never to structural scaffolding (Nested forms, Extras blocks, Relationships themselves).

```rust
#[non_exhaustive]
pub struct AiContext {
    /// Synonyms the LLM may use to refer to the annotated entity.
    /// Each key is a logical alias; its value is the canonical form the
    /// model exposes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub synonyms: BTreeMap<String, Vec<String>>,

    /// A plain-language description the LLM may surface to the user.
    /// Duplicates the carrier's `description:` with more narrative freedom.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Example queries or phrasings the LLM may emit.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,

    /// Unit of measurement for numeric Semantics (e.g. "usd", "percent",
    /// "events_per_minute").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
}
```

YAML:

```yaml
ai_context:
  synonyms:
    order: [purchase, txn]
    revenue: [sales, income]
  description: "Orders placed by end customers on the retail site."
  examples:
    - "total revenue for Q1 2025"
    - "orders by region last week"
  unit: usd
```

The four fields are the closed v1 roster. The enum / struct is `#[non_exhaustive]`, so adding a field (e.g. `tags`, `capability_hints`) is MINOR.

## Testing Strategy

- synonyms/description/examples/unit fields; YAML round-trip; attachment sites (SemanticModel/DataKind/Semantics); v1 roster closed but #[non_exhaustive].
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
