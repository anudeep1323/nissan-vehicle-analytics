{{ config(materialized='table') }}

SELECT
    {{ dbt_utils.generate_surrogate_key(['vin', 'event_ts']) }} AS event_sk,
    vin,
    event_ts,
    date,
    hour,
    speed_mph,
    engine_temp_c,
    battery_voltage,
    oil_pressure_psi,
    rpm,
    fuel_level_pct,
    brake_pad_mm,
    transmission_temp_c,
    mileage,
    is_anomaly,
    engine_overheat,
    low_battery,
    low_brake_pad,
    low_oil_pressure,
    needs_service,
    processed_at
FROM read_parquet('/Users/anudeepgoudrampur/Documents/ML/data/silver/vehicle_telemetry/*.parquet')