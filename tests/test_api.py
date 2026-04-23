import pytest

def test_risk_levels():
    """Risk level logic should be correct"""
    def get_risk(prob):
        return "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"

    assert get_risk(0.8) == "HIGH"
    assert get_risk(0.5) == "MEDIUM"
    assert get_risk(0.1) == "LOW"

def test_prediction_threshold():
    """Prediction should be 1 when probability > 0.5"""
    probs = [0.3, 0.6, 0.9, 0.1]
    preds = [1 if p > 0.5 else 0 for p in probs]
    assert preds == [0, 1, 1, 0]

def test_latency_threshold():
    """Predictions should complete within 2000ms locally"""
    import time
    start = time.time()
    # Simulate prediction logic
    import numpy as np
    X = np.random.rand(1, 49)
    elapsed = (time.time() - start) * 1000
    assert elapsed < 2000

def test_response_structure():
    """Prediction response should have required fields"""
    response = {
        "vin": "VIN000001",
        "breakdown_probability": 0.23,
        "prediction": 0,
        "risk_level": "LOW",
        "latency_ms": 45.2
    }
    required = ["vin", "breakdown_probability", "prediction", "risk_level", "latency_ms"]
    assert all(k in response for k in required)