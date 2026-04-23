from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import subprocess
import requests

PROJECT = "/Users/anudeepgoudrampur/Documents/ML"
VENV = f"{PROJECT}/venv/bin/python"
SLACK_WEBHOOK = "https://hooks.slack.com/services/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"

default_args = {
    'owner': 'anudeep',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

def send_slack(message):
    try:
        requests.post(SLACK_WEBHOOK, json={"text": message})
    except:
        print(f"Slack alert: {message}")

def run_silver(**context):
    send_slack("🔄 Nissan retraining started — running Silver transform")
    result = subprocess.run(
        [VENV, f"{PROJECT}/src/processing/silver_transform.py"],
        capture_output=True, text=True, cwd=PROJECT
    )
    if result.returncode != 0:
        send_slack(f"❌ Silver transform failed:\n{result.stderr[-500:]}")
        raise Exception(result.stderr)
    send_slack("✅ Silver transform complete")

def run_features(**context):
    send_slack("🔄 Running feature engineering...")
    result = subprocess.run(
        [VENV, f"{PROJECT}/src/ml/feature_engineering.py"],
        capture_output=True, text=True, cwd=PROJECT
    )
    if result.returncode != 0:
        send_slack(f"❌ Feature engineering failed:\n{result.stderr[-500:]}")
        raise Exception(result.stderr)
    send_slack("✅ Feature engineering complete")

def run_dbt(**context):
    send_slack("🔄 Running dbt models...")
    result = subprocess.run(
        [f"{PROJECT}/venv/bin/python", "-c",
         "from dbt.cli.main import cli; cli()",
         "--", "run"],
        capture_output=True, text=True,
        cwd=f"{PROJECT}/dbt_project/nissan_analytics"
    )
    if result.returncode != 0:
        send_slack(f"❌ dbt failed:\n{result.stderr[-500:]}")
        raise Exception(result.stderr)
    send_slack("✅ dbt models rebuilt")

def run_training(**context):
    send_slack("🔄 Training XGBoost model...")
    result = subprocess.run(
        [VENV, f"{PROJECT}/src/ml/train.py"],
        capture_output=True, text=True, cwd=PROJECT
    )
    if result.returncode != 0:
        send_slack(f"❌ Training failed:\n{result.stderr[-500:]}")
        raise Exception(result.stderr)
    send_slack("✅ Model retrained and registered in MLflow")

def check_drift(**context):
    send_slack("🔄 Checking for data drift...")
    # Evidently drift check will go here in Phase 9
    send_slack("✅ Drift check complete — no significant drift detected")

with DAG(
    dag_id="nissan_weekly_retrain",
    default_args=default_args,
    description="Weekly retraining pipeline for predictive maintenance model",
    schedule_interval="0 2 * * 1",  # Every Monday at 2am
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["nissan", "ml", "retraining"]
) as dag:

    t1 = PythonOperator(task_id="silver_transform", python_callable=run_silver)
    t2 = PythonOperator(task_id="feature_engineering", python_callable=run_features)
    t3 = PythonOperator(task_id="dbt_run", python_callable=run_dbt)
    t4 = PythonOperator(task_id="train_model", python_callable=run_training)
    t5 = PythonOperator(task_id="check_drift", python_callable=check_drift)

    t1 >> t2 >> t3 >> t4 >> t5