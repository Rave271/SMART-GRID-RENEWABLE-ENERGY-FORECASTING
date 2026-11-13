"""Unit tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
from app.services.features import engineer_features, get_season


def test_get_season():
    assert get_season(1) == 1   # Winter
    assert get_season(4) == 2   # Spring
    assert get_season(7) == 3   # Summer
    assert get_season(10) == 4  # Fall


def test_engineer_features_generation():
    timestamps = pd.date_range("2024-06-01 00:00:00", periods=50, freq="h")
    df = pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": np.random.uniform(15, 30, 50),
        "humidity": np.random.uniform(40, 80, 50),
        "wind_speed": np.random.uniform(2, 12, 50),
        "solar_irradiance": np.random.uniform(0, 900, 50),
        "solar_power_mw": np.random.uniform(0, 80, 50),
        "wind_power_mw": np.random.uniform(0, 100, 50),
        "electricity_demand_mw": np.random.uniform(300, 500, 50),
    })

    featured_df, feature_cols = engineer_features(df, target_col="solar_power_mw", include_lags=True)

    # Check temporal features
    assert "hour" in featured_df.columns
    assert "day_of_week" in featured_df.columns
    assert "month" in featured_df.columns
    assert "is_weekend" in featured_df.columns

    # Check cyclical encodings
    assert "sin_hour" in featured_df.columns
    assert "cos_hour" in featured_df.columns
    assert featured_df["sin_hour"].between(-1.0, 1.0).all()
    assert featured_df["cos_hour"].between(-1.0, 1.0).all()

    # Check domain interactions
    assert "wind_speed_cubed" in featured_df.columns
    assert "solar_effective_irr" in featured_df.columns
    assert "cooling_degree_hours" in featured_df.columns

    # Check lag & rolling features
    assert "solar_power_mw_lag_1" in featured_df.columns
    assert "solar_power_mw_rolling_mean_6h" in featured_df.columns

    # Check that feature_cols is non-empty and contains no NaNs
    assert len(feature_cols) > 10
    assert featured_df[feature_cols].isnull().sum().sum() == 0
