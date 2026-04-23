import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages io.delta:delta-spark_2.12:3.2.0 pyspark-shell'

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("NissanFeatures") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.format("delta").load("data/silver/vehicle_telemetry")

# Window specs for rolling features
w1  = Window.partitionBy("vin").orderBy("event_ts").rowsBetween(-10, 0)
w2  = Window.partitionBy("vin").orderBy("event_ts").rowsBetween(-50, 0)
w3  = Window.partitionBy("vin").orderBy("event_ts").rowsBetween(-100, 0)
w4  = Window.partitionBy("vin").orderBy("event_ts").rowsBetween(-200, 0)
w5  = Window.partitionBy("vin").orderBy("event_ts").rowsBetween(-500, 0)
wlag = Window.partitionBy("vin").orderBy("event_ts")

df_features = df \
    .withColumn("avg_engine_temp_10",   avg("engine_temp_c").over(w1)) \
    .withColumn("avg_engine_temp_50",   avg("engine_temp_c").over(w2)) \
    .withColumn("avg_engine_temp_100",  avg("engine_temp_c").over(w3)) \
    .withColumn("avg_engine_temp_200",  avg("engine_temp_c").over(w4)) \
    .withColumn("avg_engine_temp_500",  avg("engine_temp_c").over(w5)) \
    .withColumn("avg_battery_10",       avg("battery_voltage").over(w1)) \
    .withColumn("avg_battery_50",       avg("battery_voltage").over(w2)) \
    .withColumn("avg_battery_100",      avg("battery_voltage").over(w3)) \
    .withColumn("avg_oil_pressure_10",  avg("oil_pressure_psi").over(w1)) \
    .withColumn("avg_oil_pressure_50",  avg("oil_pressure_psi").over(w2)) \
    .withColumn("avg_oil_pressure_100", avg("oil_pressure_psi").over(w3)) \
    .withColumn("avg_brake_pad_10",     avg("brake_pad_mm").over(w1)) \
    .withColumn("avg_brake_pad_50",     avg("brake_pad_mm").over(w2)) \
    .withColumn("avg_brake_pad_100",    avg("brake_pad_mm").over(w3)) \
    .withColumn("avg_rpm_10",           avg("rpm").over(w1)) \
    .withColumn("avg_rpm_50",           avg("rpm").over(w2)) \
    .withColumn("avg_rpm_100",          avg("rpm").over(w3)) \
    .withColumn("std_engine_temp_50",   stddev("engine_temp_c").over(w2)) \
    .withColumn("std_battery_50",       stddev("battery_voltage").over(w2)) \
    .withColumn("std_rpm_50",           stddev("rpm").over(w2)) \
    .withColumn("max_engine_temp_50",   max("engine_temp_c").over(w2)) \
    .withColumn("min_battery_50",       min("battery_voltage").over(w2)) \
    .withColumn("min_oil_pressure_50",  min("oil_pressure_psi").over(w2)) \
    .withColumn("min_brake_pad_50",     min("brake_pad_mm").over(w2)) \
    .withColumn("lag_engine_temp_1",    lag("engine_temp_c", 1).over(wlag)) \
    .withColumn("lag_engine_temp_3",    lag("engine_temp_c", 3).over(wlag)) \
    .withColumn("lag_battery_1",        lag("battery_voltage", 1).over(wlag)) \
    .withColumn("lag_oil_pressure_1",   lag("oil_pressure_psi", 1).over(wlag)) \
    .withColumn("lag_brake_pad_1",      lag("brake_pad_mm", 1).over(wlag)) \
    .withColumn("engine_temp_delta",    col("engine_temp_c") - col("lag_engine_temp_1")) \
    .withColumn("battery_delta",        col("battery_voltage") - col("lag_battery_1")) \
    .withColumn("service_flag_count_50",sum(col("needs_service").cast("int")).over(w2)) \
    .withColumn("service_flag_count_100",sum(col("needs_service").cast("int")).over(w3)) \
    .withColumn("overheat_count_50",    sum(col("engine_overheat").cast("int")).over(w2)) \
    .withColumn("low_battery_count_50", sum(col("low_battery").cast("int")).over(w2)) \
    .withColumn("low_brake_count_50",   sum(col("low_brake_pad").cast("int")).over(w2)) \
    .withColumn("mileage_bucket_num",
        when(col("mileage") < 30000, 0)
        .when(col("mileage") < 80000, 1)
        .otherwise(2)) \
    .withColumn("engine_temp_zscore",
        (col("engine_temp_c") - col("avg_engine_temp_50")) / (col("std_engine_temp_50") + 0.001)) \
    .withColumn("battery_zscore",
        (col("battery_voltage") - col("avg_battery_10")) / (col("std_battery_50") + 0.001)) \
    .withColumn("hour_of_day",          col("hour")) \
    .withColumn("is_high_rpm",          col("rpm") > 4500) \
    .withColumn("is_low_fuel",          col("fuel_level_pct") < 10) \
    .withColumn("target",               col("needs_service").cast("int"))

df_features = df_features.fillna(0)

df_features.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("data/gold/features")

print(f"Features written: {df_features.count()} rows")
print(f"Feature columns: {len(df_features.columns)}")
spark.stop()