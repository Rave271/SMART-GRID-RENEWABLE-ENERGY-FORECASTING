"""Data preprocessing and validation pipeline for Smart Grid forecasting."""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd

# Canonical column names required for model pipeline
CANONICAL_COLUMNS = [
    "timestamp",
    "temperature",
    "humidity",
    "wind_speed",
    "solar_irradiance",
    "solar_power_mw",
    "wind_power_mw",
    "electricity_demand_mw",
]

# Alias dictionary to support various dataset conventions (e.g. Kaggle datasets)
COLUMN_ALIASES = {
    "timestamp": ["timestamp", "datetime", "date_time", "time", "date"],
    "temperature": ["temperature", "temp", "ambient_temperature", "air_temp", "t_ambient"],
    "humidity": ["humidity", "rel_humidity", "rh", "relative_humidity"],
    "wind_speed": ["wind_speed", "windspeed", "ws", "wind_velocity", "wind"],
    "solar_irradiance": ["solar_irradiance", "irradiance", "ghi", "dni", "radiation", "global_horizontal_irradiance"],
    "solar_power_mw": ["solar_power_mw", "solar_power", "solar_generation", "solar_mw", "solar"],
    "wind_power_mw": ["wind_power_mw", "wind_power", "wind_generation", "wind_mw", "wind"],
    "electricity_demand_mw": ["electricity_demand_mw", "demand_mw", "electricity_demand", "load_mw", "load", "demand"],
}

# Physical boundary constraints for sanity checking and outlier clipping
PHYSICAL_BOUNDS = {
    "temperature": (-30.0, 60.0),
    "humidity": (0.0, 100.0),
    "wind_speed": (0.0, 45.0),
    "solar_irradiance": (0.0, 1400.0),
    "solar_power_mw": (0.0, 5000.0),
    "wind_power_mw": (0.0, 5000.0),
    "electricity_demand_mw": (10.0, 20000.0),
}


def map_columns_to_canonical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Map arbitrary column headers to canonical names using the alias dictionary.
    Returns: (renamed_df, mapping_used, missing_canonical_columns)
    """
    col_map = {}
    df_lower = {str(c).strip().lower(): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        found = False
        for alias in aliases:
            if alias.lower() in df_lower:
                col_map[df_lower[alias.lower()]] = canonical
                found = True
                break

    mapped_df = df.rename(columns=col_map)
    missing = [c for c in CANONICAL_COLUMNS if c not in mapped_df.columns]
    return mapped_df, col_map, missing


def validate_csv(filepath: Path, max_bytes: int = 50 * 1024 * 1024) -> Tuple[bool, str, Dict[str, Any]]:
    """Validate uploaded CSV file size, encoding, and column requirements."""
    if not filepath.exists():
        return False, f"File does not exist: {filepath}", {}

    file_size = filepath.stat().st_size
    if file_size > max_bytes:
        return False, f"File size ({file_size / (1024*1024):.2f} MB) exceeds maximum allowed limit ({max_bytes / (1024*1024):.0f} MB).", {}

    # Attempt reading with utf-8, fallback to latin-1
    try:
        sample_df = pd.read_csv(filepath, nrows=10, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            sample_df = pd.read_csv(filepath, nrows=10, encoding="latin-1")
        except Exception as e:
            return False, f"Encoding error reading CSV file: {str(e)}", {}
    except Exception as e:
        return False, f"Invalid CSV file format: {str(e)}", {}

    mapped_sample, col_map, missing = map_columns_to_canonical(sample_df)
    if missing:
        return False, f"Dataset is missing required canonical columns (or recognizable aliases): {', '.join(missing)}", {
            "missing_columns": missing,
            "detected_columns": list(sample_df.columns),
        }

    # Full read for row count & summary preview
    try:
        full_df = pd.read_csv(filepath, encoding="utf-8" if "sample_df" in locals() else "latin-1")
    except Exception:
        full_df = pd.read_csv(filepath, encoding="latin-1")

    mapped_full, _, _ = map_columns_to_canonical(full_df)

    missing_counts = mapped_full[CANONICAL_COLUMNS].isnull().sum().to_dict()
    preview = mapped_full[CANONICAL_COLUMNS].head(5).to_dict(orient="records")

    return True, "CSV validation successful.", {
        "row_count": len(mapped_full),
        "column_count": len(mapped_full.columns),
        "columns": list(mapped_full.columns),
        "missing_values": missing_counts,
        "preview": preview,
    }


def preprocess_energy_data(
    input_path: Path,
    output_path: Optional[Path] = None,
    resample_freq: str = "1h",
    clip_outliers: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Comprehensive preprocessing:
    - Column mapping
    - Datetime parsing & sorting
    - Deduplication
    - Resampling to regular frequency
    - Missing value interpolation (forward/backward fill)
    - Physical constraint clipping / outlier management
    """
    # Read file
    try:
        df = pd.read_csv(input_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="latin-1")

    orig_rows = len(df)
    mapped_df, _, missing = map_columns_to_canonical(df)
    if missing:
        raise ValueError(f"Missing required columns for preprocessing: {missing}")

    work_df = mapped_df[CANONICAL_COLUMNS].copy()

    # Datetime parsing
    work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], errors="coerce")
    work_df = work_df.dropna(subset=["timestamp"])
    work_df = work_df.sort_values(by="timestamp").reset_index(drop=True)

    # Deduplicate timestamps by averaging duplicates
    if work_df["timestamp"].duplicated().any():
        work_df = work_df.groupby("timestamp", as_index=False).mean()

    # Set datetime index for resampling
    work_df = work_df.set_index("timestamp")

    # Regular hourly resampling
    if resample_freq:
        work_df = work_df.resample(resample_freq).mean()

    # Count missing before imputation
    missing_before = work_df.isnull().sum().sum()

    # Linear interpolation for weather variables + ffill/bfill for edges
    work_df = work_df.interpolate(method="time").bfill().ffill()

    # Solar irradiance is strictly 0 at night (hour 20 to 5)
    hours = work_df.index.hour
    night_mask = (hours >= 20) | (hours < 6)
    work_df.loc[night_mask, "solar_irradiance"] = 0.0
    work_df.loc[night_mask, "solar_power_mw"] = 0.0

    # Physical boundary outlier clipping
    outliers_clipped_count = 0
    if clip_outliers:
        for col, (min_val, max_val) in PHYSICAL_BOUNDS.items():
            if col in work_df.columns:
                clipped = np.clip(work_df[col].values, min_val, max_val)
                diff = np.sum(work_df[col].values != clipped)
                outliers_clipped_count += int(diff)
                work_df[col] = clipped

    work_df = work_df.reset_index()

    # Format timestamp back to string for consistency
    work_df["timestamp"] = work_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save processed dataset
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_df.to_csv(output_path, index=False)

    stats = {
        "original_rows": orig_rows,
        "processed_rows": len(work_df),
        "missing_imputed": int(missing_before),
        "outliers_clipped": int(outliers_clipped_count),
        "start_time": work_df["timestamp"].iloc[0],
        "end_time": work_df["timestamp"].iloc[-1],
        "columns": list(work_df.columns),
    }

    return work_df, stats
