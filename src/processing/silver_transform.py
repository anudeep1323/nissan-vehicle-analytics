import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages io.delta:delta-spark_2.12:3.2.0 pyspark-shell'

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import DeltaTable

spark = SparkSession.builder \
    .appName("NissanSilver") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Read bronze
df = spark.read.format("delta").load("data/bronze/vehicle_telemetry")

# Clean + type cast + add derived columns
df_silver = df \
    .withColumn("event_ts", to_timestamp(col("timestamp"))) \
    .withColumn("date", to_date(col("event_ts"))) \
    .withColumn("hour", hour(col("event_ts"))) \
    .withColumn("is_anomaly", col("check_engine_light").cast(BooleanType())) \
    .withColumn("engine_overheat", col("engine_temp_c") > 250)\
    .withColumn("low_battery", col("battery_voltage") < 11.0)\
    .withColumn("low_brake_pad", col("brake_pad_mm") < 2.0)\
    .withColumn("low_oil_pressure", col("oil_pressure_psi") < 15.0)\
    .withColumn("needs_service", 
        (col("engine_overheat") | col("low_battery") | 
         col("low_brake_pad") | col("low_oil_pressure"))) \
    .dropDuplicates(["vin", "timestamp"]) \
    .drop("kafka_timestamp", "ingested_at") \
    .withColumn("processed_at", current_timestamp())

# Write to Silver with Z-ordering on vin
df_silver.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("data/silver/vehicle_telemetry")

# Z-ordering for query performance (matches resume claim)
from delta.tables import DeltaTable
DeltaTable.forPath(spark, "data/silver/vehicle_telemetry").optimize().executeZOrderBy("vin", "event_ts")

print(f"Silver rows: {df_silver.count()}")
df_silver.select("vin", "event_ts", "needs_service", "engine_overheat", "low_battery").show(5)
spark.stop()