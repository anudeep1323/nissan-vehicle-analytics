{{ config(materialized='table') }}

WITH vehicle_base AS (
    SELECT
        vin,
        MIN(event_ts)                                    AS first_seen,
        MAX(event_ts)                                    AS last_seen,
        MAX(mileage)                                     AS current_mileage,
        CASE
            WHEN MAX(mileage) < 30000  THEN 'new'
            WHEN MAX(mileage) < 80000  THEN 'mid'
            ELSE 'high'
        END                                              AS mileage_bucket,
        SUM(CASE WHEN needs_service THEN 1 ELSE 0 END)  AS total_service_flags,
        COUNT(*)                                         AS total_events,
        TRUE                                             AS is_current,
        CURRENT_TIMESTAMP                                AS valid_from,
        CAST('9999-12-31' AS TIMESTAMP)                  AS valid_to
    FROM read_parquet('/Users/anudeepgoudrampur/Documents/ML/data/silver/vehicle_telemetry/*.parquet')
    GROUP BY vin
)

SELECT
    {{ dbt_utils.generate_surrogate_key(['vin', 'valid_from']) }} AS vehicle_sk,
    vin,
    first_seen,
    last_seen,
    current_mileage,
    mileage_bucket,
    total_service_flags,
    total_events,
    is_current,
    valid_from,
    valid_to
FROM vehicle_base
