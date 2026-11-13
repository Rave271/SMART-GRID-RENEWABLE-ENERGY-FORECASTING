"""Unit tests for data preprocessing and validation pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.services.preprocessing import (
    map_columns_to_canonical,
    validate_csv,
    preprocess_energy_data,
    CANONICAL_COLUMNS,
)


def test_map_columns_to_canonical():
    """Verify alias mapping correctly resolves Kaggle/external column formats."""
    raw_df = pd.DataFrame({
        "Date_Time": ["2024-01-01 00:00:00"],
        "temp": [15.2],
        "rh": [60.0],
        "windspeed": [5.5],
        "radiation": [0.0],
        "solar_generation": [0.0],
        "wind_generation": [12.0],
        "load": [350.0],
    })

    mapped_df, col_map, missing = map_columns_to_canonical(raw_df)
    assert len(missing) == 0, f"Expected 0 missing columns, got: {missing}"
    for col in CANONICAL_COLUMNS:
        assert col in mapped_df.columns


def test_validate_csv_success(tmp_path):
    """Test validation with valid smart grid CSV."""
    csv_file = tmp_path / "valid.csv"
    df = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
        "temperature": [10.0, 11.0],
        "humidity": [60.0, 65.0],
        "wind_speed": [4.0, 5.0],
        "solar_irradiance": [0.0, 0.0],
        "solar_power_mw": [0.0, 0.0],
        "wind_power_mw": [15.0, 20.0],
        "electricity_demand_mw": [300.0, 310.0],
    })
    df.to_csv(csv_file, index=False)

    is_valid, msg, details = validate_csv(csv_file)
    assert is_valid is True
    assert details["row_count"] == 2
    assert "preview" in details


def test_validate_csv_missing_columns(tmp_path):
    """Test validation failure when mandatory energy columns are absent."""
    csv_file = tmp_path / "invalid.csv"
    df = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00"],
        "temperature": [10.0],
    })
    df.to_csv(csv_file, index=False)

    is_valid, msg, details = validate_csv(csv_file)
    assert is_valid is False
    assert "missing" in msg.lower()


def test_preprocess_energy_data_pipeline(tmp_path):
    """Test interpolation of missing values and physical bounds clipping."""
    raw_file = tmp_path / "dirty.csv"
    out_file = tmp_path / "clean.csv"

    # Dataset with missing value and an extreme outlier
    df = pd.DataFrame({
        "timestamp": ["2024-01-01 00:00:00", "2024-01-01 01:00:00", "2024-01-01 02:00:00"],
        "temperature": [15.0, np.nan, 85.0],  # 85 C is an outlier to be clipped
        "humidity": [50.0, 55.0, 60.0],
        "wind_speed": [5.0, 6.0, 7.0],
        "solar_irradiance": [0.0, 0.0, 0.0],
        "solar_power_mw": [0.0, 0.0, 0.0],
        "wind_power_mw": [20.0, 25.0, 30.0],
        "electricity_demand_mw": [400.0, 410.0, 420.0],
    })
    df.to_csv(raw_file, index=False)

    clean_df, stats = preprocess_energy_data(raw_file, out_file)
    assert len(clean_df) == 3
    assert clean_df["temperature"].isnull().sum() == 0
    assert clean_df["temperature"].max() <= 60.0  # Clipped by physical boundary
    assert stats["missing_imputed"] >= 1
    assert stats["outliers_clipped"] >= 1
