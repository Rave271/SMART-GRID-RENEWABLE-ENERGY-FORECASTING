"""Feature engineering service for Smart Grid energy forecasting."""

from typing import List, Tuple
import numpy as np
import pandas as pd


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    include_lags: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare model features through the single shared feature-engineering path."""
    return engineer_features(df, target_col=target_col, include_lags=include_lags)


def get_season(month: int) -> int:
    """1: Winter, 2: Spring, 3: Summer, 4: Fall"""
    if month in (12, 1, 2):
        return 1
    elif month in (3, 4, 5):
        return 2
    elif month in (6, 7, 8):
        return 3
    else:
        return 4


def engineer_features(
    df: pd.DataFrame,
    target_col: str = "solar_power_mw",
    include_lags: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Generate calendar, cyclical, domain physical, lag, and rolling features.
    Returns: (featured_df, list_of_feature_column_names)
    """
    df = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["dt"] = pd.to_datetime(df["timestamp"])
    else:
        df["dt"] = df["timestamp"]

    # 1. Calendar Features
    df["hour"] = df["dt"].dt.hour
    if hasattr(df["dt"].dt, "day_of_week"):
        df["day_of_week"] = df["dt"].dt.day_of_week
    else:
        df["day_of_week"] = df["dt"].dt.dayofweek
    df["month"] = df["dt"].dt.month
    if hasattr(df["dt"].dt, "day_of_year"):
        df["day_of_year"] = df["dt"].dt.day_of_year
    else:
        df["day_of_year"] = df["dt"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(float)
    df["season"] = df["month"].apply(get_season)

    # 2. Cyclical Encodings (sine and cosine transforms)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    # 3. Domain Physical Interactions
    # Wind power cubic relationship
    df["wind_speed_cubed"] = df["wind_speed"] ** 3
    # Solar temperature derating interaction
    temp_derating = 1.0 - np.maximum(0.0, (df["temperature"] - 25.0) * 0.004)
    df["solar_effective_irr"] = df["solar_irradiance"] * temp_derating
    # HVAC cooling and heating demand drivers
    df["cooling_degree_hours"] = np.maximum(0.0, df["temperature"] - 24.0)
    df["heating_degree_hours"] = np.maximum(0.0, 15.0 - df["temperature"])

    # 4. Lag & Rolling Features for the target
    if include_lags and target_col in df.columns:
        target_series = df[target_col]
        # Short and diurnal lags
        df[f"{target_col}_lag_1"] = target_series.shift(1)
        df[f"{target_col}_lag_2"] = target_series.shift(2)
        df[f"{target_col}_lag_24"] = target_series.shift(24)
        df[f"{target_col}_lag_48"] = target_series.shift(48)

        # Rolling statistics (excluding current point to avoid leakage)
        df[f"{target_col}_rolling_mean_6h"] = (
            target_series.shift(1).rolling(window=6, min_periods=1).mean()
        )
        df[f"{target_col}_rolling_std_6h"] = (
            target_series.shift(1).rolling(window=6, min_periods=1).std().fillna(0)
        )
        df[f"{target_col}_rolling_mean_24h"] = (
            target_series.shift(1).rolling(window=24, min_periods=1).mean()
        )
        df[f"{target_col}_rolling_std_24h"] = (
            target_series.shift(1).rolling(window=24, min_periods=1).std().fillna(0)
        )

        # Forward-fill / back-fill initial lag NaN rows
        lag_cols = [
            c for c in df.columns if c.startswith(f"{target_col}_lag_") or c.startswith(f"{target_col}_rolling_")
        ]
        df[lag_cols] = df[lag_cols].bfill().ffill()

    # Drop temporary datetime column
    df = df.drop(columns=["dt"])

    # List feature columns (all except timestamp, raw targets, unless lag feature)
    raw_targets = ["solar_power_mw", "wind_power_mw", "electricity_demand_mw"]
    feature_cols = [
        c for c in df.columns if c != "timestamp" and (c not in raw_targets or c.endswith("_lag_") or c.endswith("_rolling_"))
    ]

    return df, feature_cols
