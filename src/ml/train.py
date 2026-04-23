import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages io.delta:delta-spark_2.12:3.2.0 pyspark-shell'
from pyspark.sql import SparkSession

# Load features
spark = SparkSession.builder \
    .appName("NissanTrain") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.format("delta").load("data/gold/features").toPandas()
spark.stop()

# Feature columns
drop_cols = ['vin', 'timestamp', 'event_ts', 'date', 'processed_at',
             'needs_service', 'check_engine_light', 'is_anomaly', 'target',
             'engine_overheat', 'low_battery', 'low_brake_pad',
             'low_oil_pressure', 'is_high_rpm', 'is_low_fuel',
             'service_flag_count_50', 'service_flag_count_100',
             'overheat_count_50', 'low_battery_count_50', 'low_brake_count_50']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].fillna(0)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_cols)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# MLflow setup
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("nissan_predictive_maintenance")

# Optuna hyperparameter tuning — Bayesian optimization (50 trials)
def objective(trial):
    params = {
        'max_depth':        trial.suggest_int('max_depth', 3, 10),
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
        'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma':            trial.suggest_float('gamma', 0, 5),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

print("Running Optuna hyperparameter tuning (50 trials)...")
study = optuna.create_study(direction='maximize')
optuna.logging.set_verbosity(optuna.logging.WARNING)
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_params.update({'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': 42})
print(f"Best params: {best_params}")

# Train final model with best params + log to MLflow
with mlflow.start_run(run_name="xgboost_best"):
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, probs)

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc",      auc)
    mlflow.log_metric("train_size", len(X_train))
    mlflow.log_metric("test_size",  len(X_test))
    mlflow.xgboost.log_model(model, "model",
        registered_model_name="nissan_maintenance_model")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(classification_report(y_test, preds))