import os
import pandas as pd
import numpy as np
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages io.delta:delta-spark_2.12:3.2.0 pyspark-shell'
from pyspark.sql import SparkSession
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric
import json
from datetime import datetime

spark = SparkSession.builder \
    .appName("NissanDrift") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.format("delta").load("data/gold/features").toPandas()
spark.stop()

drop_cols = ['vin', 'timestamp', 'event_ts', 'date', 'processed_at',
             'needs_service', 'check_engine_light', 'is_anomaly', 'target',
             'engine_overheat', 'low_battery', 'low_brake_pad',
             'low_oil_pressure', 'is_high_rpm', 'is_low_fuel',
             'service_flag_count_50', 'service_flag_count_100',
             'overheat_count_50', 'low_battery_count_50', 'low_brake_count_50']
feature_cols = [c for c in df.columns if c not in drop_cols]

df = df[feature_cols].fillna(0)

# Split into reference (first 70%) and current (last 30%)
# In production: reference = training data, current = last week's live data
split = int(len(df) * 0.7)
reference = df.iloc[:split]
current   = df.iloc[split:]

print(f"Reference: {len(reference)} rows | Current: {len(current)} rows")

# Run Evidently drift report
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
])

report.run(reference_data=reference, current_data=current)

# Save HTML report
os.makedirs("data/drift_reports", exist_ok=True)
report_path = f"data/drift_reports/drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
report.save_html(report_path)

# Get drift summary
result = report.as_dict()
drift_summary = result['metrics'][0]['result']
n_drifted = drift_summary.get('number_of_drifted_columns', 0)
n_total   = drift_summary.get('number_of_columns', len(feature_cols))
drift_pct  = n_drifted / n_total * 100

print(f"\nDrift Report:")
print(f"Drifted columns: {n_drifted}/{n_total} ({drift_pct:.1f}%)")
print(f"Dataset drift detected: {drift_summary.get('dataset_drift', False)}")
print(f"Report saved: {report_path}")

if drift_pct > 15:
    print(f"\n⚠️  ALERT: {drift_pct:.1f}% feature drift detected — retraining recommended!")
else:
    print(f"\n✅ Drift within acceptable range ({drift_pct:.1f}%)")