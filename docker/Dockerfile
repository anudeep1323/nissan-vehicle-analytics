FROM eclipse-temurin:17-jre-jammy

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/opt/java/openjdk
ENV PATH=$JAVA_HOME/bin:$PATH

RUN pip3 install --no-cache-dir \
    pyspark==3.5.0 \
    delta-spark==3.2.0 \
    xgboost==2.0.3 \
    mlflow==2.10.0 \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    pandas==2.1.4 \
    numpy==1.26.3 \
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.25 \
    python-dotenv==1.0.0 \
    pyarrow==14.0.1

COPY src/ ./src/
COPY mlruns/ ./mlruns/
COPY data/gold/features/ ./data/gold/features/

EXPOSE 8002

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8002"]