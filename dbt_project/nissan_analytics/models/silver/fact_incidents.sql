{{ config(materialized='table') }}

SELECT
    {{ dbt_utils.generate_surrogate_key(['vin', 'event_ts']) }} AS incident_sk,
    vin,
    event_ts,
    date,
    engine_overheat,
    low_battery,
    low_brake_pad,
    low_oil_pressure,
    engine_temp_c,
    battery_voltage,
    brake_pad_mm,
    oil_pressure_psi,
    mileage,
    processed_at
FROM read_parquet('/Users/anudeepgoudrampur/Documents/ML/data/silver/vehicle_telemetry/*.parquet')
WHERE needs_service = TRUE