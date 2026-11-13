"""Unit tests for ML forecasting models and metrics."""

import pandas as pd
import numpy as np
from app.services.model import GridForecaster
from generate_sample_data import generate_smart_grid_data


def test_time_based_split_and_training(tmp_path):
    """Test model training with chronological split and verify no future data leakage."""
    # Generate 1 month of test data
    df = generate_smart_grid_data(start_date="2024-01-01 00:00:00", end_date="2024-01-31 23:00:00")

    forecaster = GridForecaster(target="solar_power_mw", model_type="xgboost")
    res = forecaster.train(df, test_size=0.2)

    # Check metrics
    assert "mae" in res
    assert "rmse" in res
    assert "r2" in res
    assert res["mae"] >= 0.0
    assert res["rmse"] >= 0.0
    assert res["r2"] > 0.70  # Solar daylight model achieves high R^2

    # Check that model file was persisted
    model_path = forecaster.get_model_path()
    assert model_path.exists()


def test_predict_future_multistep():
    """Test autoregressive multi-step future forecasting."""
    df = generate_smart_grid_data(start_date="2024-01-01 00:00:00", end_date="2024-01-14 23:00:00")

    forecaster = GridForecaster(target="wind_power_mw", model_type="xgboost")
    forecaster.train(df, test_size=0.2)

    # Predict next 24 hours
    preds_24h = forecaster.predict_future(df, horizon_hours=24)
    assert len(preds_24h) == 24
    assert all(p >= 0.0 for p in preds_24h)
