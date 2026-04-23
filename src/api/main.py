import time
import mlflow
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sqlalchemy as sa
from datetime import datetime
import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages io.delta:delta-spark_2.12:3.2.0 pyspark-shell'
from pyspark.sql import SparkSession

app = FastAPI(title="Nissan Predictive Maintenance API")

# Load model from MLflow registry on startup
MODEL = None
SPARK = None
FEATURE_COLS = None

@app.on_event("startup")
async def load_model():
    global MODEL, SPARK, FEATURE_COLS

    mlflow.set_tracking_uri("mlruns")
    MODEL = mlflow.xgboost.load_model("models:/nissan_maintenance_model/5")

    SPARK = SparkSession.builder \
        .appName("NissanAPI") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    SPARK.sparkContext.setLogLevel("ERROR")

    # Load feature columns
    df = SPARK.read.format("delta").load("data/gold/features").limit(1).toPandas()
    drop_cols = ['vin', 'timestamp', 'event_ts', 'date', 'processed_at',
                 'needs_service', 'check_engine_light', 'is_anomaly', 'target',
                 'engine_overheat', 'low_battery', 'low_brake_pad',
                 'low_oil_pressure', 'is_high_rpm', 'is_low_fuel',
                 'service_flag_count_50', 'service_flag_count_100',
                 'overheat_count_50', 'low_battery_count_50', 'low_brake_count_50']
    FEATURE_COLS = [c for c in df.columns if c not in drop_cols]
    print(f"Model loaded. Features: {len(FEATURE_COLS)}")

# DB for prediction logging
engine = sa.create_engine(
    "postgresql://nissan:nissan123@localhost:5432/nissan_analytics"
)

def init_db():
    with engine.connect() as conn:
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                vin VARCHAR(20),
                breakdown_probability FLOAT,
                prediction INT,
                latency_ms FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()

init_db()

class PredictRequest(BaseModel):
    vin: str

class PredictResponse(BaseModel):
    vin: str
    breakdown_probability: float
    prediction: int
    risk_level: str
    latency_ms: float

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.time()

    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get latest features for this VIN
    df = SPARK.read.format("delta").load("data/gold/features")
    vin_df = df.filter(df.vin == request.vin) \
               .orderBy("event_ts", ascending=False) \
               .limit(1).toPandas()

    if vin_df.empty:
        raise HTTPException(status_code=404, detail=f"VIN {request.vin} not found")

    X = vin_df[FEATURE_COLS].fillna(0)
    prob = float(MODEL.predict_proba(X)[0][1])
    pred = int(prob > 0.5)
    latency = (time.time() - start) * 1000

    risk = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"

    # Log to PostgreSQL
    with engine.connect() as conn:
        conn.execute(sa.text("""
            INSERT INTO predictions (vin, breakdown_probability, prediction, latency_ms)
            VALUES (:vin, :prob, :pred, :latency)
        """), {"vin": request.vin, "prob": prob, "pred": pred, "latency": latency})
        conn.commit()

    return PredictResponse(
        vin=request.vin,
        breakdown_probability=round(prob, 4),
        prediction=pred,
        risk_level=risk,
        latency_ms=round(latency, 2)
    )

@app.get("/predictions/history/{vin}")
def prediction_history(vin: str, limit: int = 10):
    with engine.connect() as conn:
        result = conn.execute(sa.text("""
            SELECT vin, breakdown_probability, prediction, latency_ms, created_at
            FROM predictions WHERE vin = :vin
            ORDER BY created_at DESC LIMIT :limit
        """), {"vin": vin, "limit": limit})
        rows = [dict(r._mapping) for r in result]
    return {"vin": vin, "history": rows}

@app.get("/stats")
def stats():
    with engine.connect() as conn:
        result = conn.execute(sa.text("""
            SELECT 
                COUNT(*) as total_predictions,
                AVG(latency_ms) as avg_latency_ms,
                AVG(breakdown_probability) as avg_risk_score,
                SUM(prediction) as total_high_risk
            FROM predictions
        """))
        row = dict(result.fetchone()._mapping)
    return row