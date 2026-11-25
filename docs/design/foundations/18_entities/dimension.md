---
id: 18-dimension
status: ratified
version: 2.1
impl: implemented
---

# Dimension

This leaf owns the `Dimension` struct and `DimensionType` roster. See [semantics-pools](semantics-pools.md) for root-pool reference grammar.

## 4. `Dimension`

```rust
#[non_exhaustive]
pub struct Dimension {
    /// Named-entity identity (`32 §2.3`). Authored optionally; parse generates
    /// a UUIDv7 when omitted. Storage key in the model's `dimensions` map and
    /// in a DataKind's `SemanticInterface`.
    pub id: Option<EntityId>,

    pub name: SemanticsName,

    /// Mandatory at declaration; immutable from the root-pool declaration.
    pub data_type: DataType,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ai_context: Option<AiContext>,

    /// The Dimension category — drives planner and adapter behavior.
    #[serde(rename = "type")]
    pub dim_type: DimensionType,

    /// Optional derivation expression. `None` means the Dimension is
    /// bound directly from `semantic_mapping`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expr: Option<UnresolvedSemanticExpr>,
}
```

### 4.1 `DimensionType` roster

```rust
#[non_exhaustive]
pub enum DimensionType {
    Temporal(TemporalDimensionBody),
    Categorical,
    Binary,
    Geo,
    Bucketed(BucketedDimensionBody),
    Metadata(MetadataDimensionBody),
}
```

All six variants stay in the v1 roster. Per-variant body shapes are:

```rust
#[non_exhaustive]
pub struct TemporalDimensionBody {
    /// The set of grains at which this Dimension can be rolled up.
    /// Authors list only the grains the backing source actually supports.
    pub grains: Vec<Grain>,
}

#[non_exhaustive]
pub struct BucketedDimensionBody {
    /// Bucket boundaries. Each bucket spans [lower_inclusive, upper_exclusive).
    pub buckets: Vec<BucketSpec>,
}

#[non_exhaustive]
pub struct BucketSpec {
    pub name: String,
    pub lower: Option<BucketBound>,
    pub upper: Option<BucketBound>,
}

#[non_exhaustive]
pub enum BucketBound {
    Int(i64),
    Float(f64),
    Decimal(String),           // preserved as string for lossless round-trip
    Date(String),              // ISO-8601 date
    Timestamp(String),         // ISO-8601 timestamp
}

#[non_exhaustive]
pub struct MetadataDimensionBody {
    /// Where to extract the value from each PhysicalSource — path token,
    /// partition column, or S3 object metadata field. Details in `15 §8`.
    pub source: MetadataSource,
}
```

`TemporalDimensionBody.grains` is the Dimension-level rollup axis; it can be empty (the Dimension is declared as a Timestamp but not rollable), in which case only the source grain on `extras.temporal.grain:` applies. Sub-shape polish (`BucketBound` roster extension, `MetadataSource` full grammar) is tracked as a v1-minor item.

### 4.2 YAML

```yaml
dimensions:
  - name: order_ts
    data_type: timestamp
    type:
      temporal:
        grains: [minute, hour, day, week, month]

  - name: country
    data_type: string
    type: categorical

  - name: is_active
    data_type: boolean
    type: binary

  - name: order_region
    data_type: geo
    type: geo

  - name: order_value_bucket
    data_type: string
    type:
      bucketed:
        buckets:
          - { name: low,    upper: { int: 100 } }
          - { name: mid,    lower: { int: 100 }, upper: { int: 1000 } }
          - { name: high,   lower: { int: 1000 } }

  - name: source_partition
    data_type: string
    type:
      metadata:
        source:
          partition: year
```

## Testing Strategy

- SR-E-10 (data_type required); DimensionType roster round-trip; per-variant body shapes (Temporal/Bucketed/Metadata); expr optional vs binding; EntityId identity; YAML variant serialization.
- **Shared harness:** [`../../testing/50_behavioral_contract_gates.md`](../../testing/50_behavioral_contract_gates.md).
