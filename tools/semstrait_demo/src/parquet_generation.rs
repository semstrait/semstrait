// Parquet generation enabled
use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringArray, TimestampMicrosecondArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use chrono::{DateTime, Utc};
use std::fs::File;
use std::sync::Arc;

/// Generate demo Parquet fixtures for two tableGroups with grain columns and realistic data
pub fn generate_fixtures(temp_dir: &tempfile::TempDir) -> anyhow::Result<(String, String)> {
    let adwords_path = temp_dir.path().join("adwords_campaigns.parquet");
    let facebook_path = temp_dir.path().join("facebook_campaigns.parquet");

    println!("  📊 Generating Parquet fixtures...");

    // Generate AdWords data (3 campaigns)
    generate_adwords_data(&adwords_path)?;
    println!("    ✅ Generated AdWords data: {} rows", 3);

    // Generate Facebook data (2 campaigns)
    generate_facebook_data(&facebook_path)?;
    println!("    ✅ Generated Facebook data: {} rows", 2);

    Ok((
        adwords_path.to_string_lossy().to_string(),
        facebook_path.to_string_lossy().to_string(),
    ))
}

/// Generate AdWords campaign data with grain columns
fn generate_adwords_data(path: &std::path::Path) -> anyhow::Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int64, false),
        Field::new("event_time_utc", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("day", DataType::Utf8, false),
        Field::new("account_id", DataType::Int64, false),
        Field::new("campaign_id", DataType::Int64, false),
        Field::new("ad_id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, false),  // For distinct count testing
        Field::new("cost", DataType::Float64, true),  // Nullable for testing
        Field::new("impressions", DataType::Int64, false),
    ]));

    // Realistic AdWords campaign data
    let base_time = DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")?.timestamp_micros();

    let row_ids = Int64Array::from(vec![1, 2, 3]);
    let event_times = TimestampMicrosecondArray::from(vec![
        base_time,
        base_time + 6_000_000,     // +6 seconds
        base_time + 12_000_000,    // +12 seconds
    ]);
    let days = StringArray::from(vec!["2024-01-01", "2024-01-01", "2024-01-01"]);
    let account_ids = Int64Array::from(vec![1001, 1001, 1001]);
    let campaign_ids = Int64Array::from(vec![2001, 2002, 2003]);
    let ad_ids = Int64Array::from(vec![3001, 3002, 3003]);
    let user_ids = Int64Array::from(vec![1001, 1002, 1001]); // Intentional duplicate for distinct count testing
    let costs = Float64Array::from(vec![Some(150.25), Some(275.50), Some(200.75)]); // Note: intentionally no NULLs for cost
    // Total cost will be 626.50
    let impressions = Int64Array::from(vec![15000, 25000, 20000]);

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(row_ids),
        Arc::new(event_times),
        Arc::new(days),
        Arc::new(account_ids),
        Arc::new(campaign_ids),
        Arc::new(ad_ids),
        Arc::new(user_ids),
        Arc::new(costs),
        Arc::new(impressions),
    ])?;

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Generate Facebook campaign data with grain columns
fn generate_facebook_data(path: &std::path::Path) -> anyhow::Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int64, false),
        Field::new("event_time_utc", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("day", DataType::Utf8, false),
        Field::new("account_id", DataType::Int64, false),
        Field::new("campaign_id", DataType::Int64, false),
        Field::new("ad_id", DataType::Int64, false),
        Field::new("user_id", DataType::Int64, false),  // For distinct count testing
        Field::new("spend", DataType::Float64, true),  // Nullable for testing
        Field::new("impressions", DataType::Int64, false),
    ]));

    // Realistic Facebook campaign data
    let base_time = DateTime::parse_from_rfc3339("2024-01-01T00:03:00Z")?.timestamp_micros();

    let row_ids = Int64Array::from(vec![4, 5]);
    let event_times = TimestampMicrosecondArray::from(vec![
        base_time,
        base_time + 6_000_000,     // +6 seconds
    ]);
    let days = StringArray::from(vec!["2024-01-01", "2024-01-01"]);
    let account_ids = Int64Array::from(vec![1002, 1002]);
    let campaign_ids = Int64Array::from(vec![2004, 2005]);
    let ad_ids = Int64Array::from(vec![3004, 3005]);
    let user_ids = Int64Array::from(vec![1003, 1001]); // One duplicate with AdWords for cross-platform distinct testing
    let spends = Float64Array::from(vec![Some(125.00), Some(135.50)]); // Note: intentionally different from semantic expectation
    let impressions = Int64Array::from(vec![30000, 15000]);

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(row_ids),
        Arc::new(event_times),
        Arc::new(days),
        Arc::new(account_ids),
        Arc::new(campaign_ids),
        Arc::new(ad_ids),
        Arc::new(user_ids),
        Arc::new(spends),
        Arc::new(impressions),
    ])?;

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}