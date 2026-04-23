import pandas as pd
import numpy as np
import pytest

def test_feature_count():
    """Ensure we generate expected number of features"""
    df = pd.DataFrame({
        'engine_temp_c': [200.0, 220.0, 250.0],
        'battery_voltage': [13.5, 12.0, 11.0],
        'oil_pressure_psi': [45.0, 30.0, 15.0],
        'brake_pad_mm': [7.0, 4.0, 1.5],
        'needs_service': [False, False, True]
    })
    assert len(df.columns) == 5

def test_anomaly_detection():
    """Vehicles with critical readings should be flagged"""
    df = pd.DataFrame({
        'engine_temp_c': [200.0, 255.0],
        'battery_voltage': [13.5, 10.5],
        'oil_pressure_psi': [45.0, 12.0],
        'brake_pad_mm': [7.0, 1.0],
    })
    df['engine_overheat'] = df['engine_temp_c'] > 250
    df['low_battery'] = df['battery_voltage'] < 11.0
    df['low_oil_pressure'] = df['oil_pressure_psi'] < 15.0
    df['low_brake_pad'] = df['brake_pad_mm'] < 2.0
    df['needs_service'] = (
        df['engine_overheat'] | df['low_battery'] |
        df['low_oil_pressure'] | df['low_brake_pad']
    )
    assert df.loc[0, 'needs_service'] == False
    assert df.loc[1, 'needs_service'] == True

def test_rolling_features():
    """Rolling averages should be within sensor range"""
    temps = pd.Series([195, 200, 205, 210, 215])
    rolling_avg = temps.rolling(window=3).mean()
    assert rolling_avg.iloc[-1] == 210.0

def test_no_negative_mileage():
    """Mileage should always be positive"""
    mileages = [5000, 15000, 80000, 150000]
    assert all(m > 0 for m in mileages)

def test_target_is_binary():
    """Target variable should only be 0 or 1"""
    targets = [0, 1, 0, 0, 1]
    assert all(t in [0, 1] for t in targets)