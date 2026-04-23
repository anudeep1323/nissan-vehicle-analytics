import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,io.delta:delta-spark_2.12:3.2.0 pyspark-shell'

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("NissanBronzeStream") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.shuffle.partitions", "3") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("vin", StringType()),
    StructField("timestamp", StringType()),
    StructField("speed_mph", DoubleType()),
    StructField("engine_temp_c", DoubleType()),
    StructField("battery_voltage", DoubleType()),
    StructField("oil_pressure_psi", DoubleType()),
    StructField("rpm", DoubleType()),
    StructField("fuel_level_pct", DoubleType()),
    StructField("tire_pressure_fl", DoubleType()),
    StructField("tire_pressure_fr", DoubleType()),
    StructField("tire_pressure_rl", DoubleType()),
    StructField("tire_pressure_rr", DoubleType()),
    StructField("brake_pad_mm", DoubleType()),
    StructField("transmission_temp_c", DoubleType()),
    StructField("check_engine_light", BooleanType()),
    StructField("mileage", IntegerType())
])

df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "vehicle-telemetry") \
    .option("startingOffsets", "latest") \
    .load()

df_parsed = df_raw.select(
    from_json(col("value").cast("string"), schema).alias("data"),
    col("timestamp").alias("kafka_timestamp")
).select("data.*", "kafka_timestamp") \
 .withColumn("ingested_at", current_timestamp())

query = df_parsed.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "data/bronze/_checkpoints") \
    .option("mergeSchema", "true") \
    .start("data/bronze/vehicle_telemetry")

print("Bronze stream running — writing to data/bronze/vehicle_telemetry")
query.awaitTermination()