"""Routes for data preprocessing and ML model training."""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends
import pandas as pd

from app.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR, DEFAULT_SAMPLE_PATH, require_api_key, rate_limit_train
from app.services.preprocessing import preprocess_energy_data
from app.services.model import GridForecaster, train_all_models, TARGETS, MODEL_TYPES
from app.db.database import record_trained_model
from app.schemas import PreprocessRequest, PreprocessResponse, TrainRequest, TrainResponse, ModelMetric

router = APIRouter(prefix="/api", tags=["Preprocessing & Training"], dependencies=[Depends(require_api_key), Depends(rate_limit_train)])


def resolve_active_dataset(filepath: Optional[str] = None) -> Path:
    """Find requested dataset or default to latest raw/sample file."""
    if filepath:
        requested_path = Path(filepath).expanduser().resolve()
        approved_roots = [RAW_DATA_DIR.resolve(), SAMPLE_DATA_DIR.resolve()]
        if not any(requested_path.is_relative_to(root) for root in approved_roots):
            raise HTTPException(status_code=400, detail="Dataset path must be inside the approved data directories.")
        if requested_path.is_file() and requested_path.suffix.lower() == ".csv":
            return requested_path
        raise HTTPException(status_code=404, detail="Requested dataset was not found or is not a CSV file.")

    # Check for uploaded raw files
    raw_files = sorted(RAW_DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if raw_files:
        return raw_files[0]

    # Fallback to sample dataset
    if DEFAULT_SAMPLE_PATH.exists():
        return DEFAULT_SAMPLE_PATH

    raise HTTPException(status_code=404, detail="No dataset available. Please upload a CSV or generate sample data.")


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_dataset(req: Optional[PreprocessRequest] = None):
    """Clean dataset: parse datetimes, deduplicate, resample to hourly,
    interpolate missing values, clip physical outliers, and prepare for ML training.
    """
    raw_path = resolve_active_dataset(req.filepath if req else None)
    output_path = PROCESSED_DATA_DIR / "processed_grid_data.csv"

    try:
        clean_df, stats = preprocess_energy_data(
            input_path=raw_path,
            output_path=output_path,
            resample_freq=req.resample_freq if req else "1h",
            clip_outliers=req.clip_outliers if req else True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    return PreprocessResponse(
        status="success",
        message="Dataset successfully cleaned and resampled.",
        original_rows=stats["original_rows"],
        processed_rows=stats["processed_rows"],
        missing_imputed=stats["missing_imputed"],
        outliers_clipped=stats["outliers_clipped"],
        columns=stats["columns"],
        sample_preview=clean_df.head(5).to_dict(orient="records"),
    )


@router.post("/train", response_model=TrainResponse)
async def train_model(
    req: Optional[TrainRequest] = None,
    model_type: Optional[str] = Query(None, description="xgboost or mlp"),
    target: Optional[str] = Query(None, description="solar_power_mw, wind_power_mw, electricity_demand_mw, or all"),
):
    """Train ML forecasting model(s) (XGBoost or MLP) using strict time-based split
    (no data leakage) and persist trained models to disk.
    """
    query_model_type = model_type if isinstance(model_type, str) else None
    query_target = target if isinstance(target, str) else None
    m_type = (query_model_type or (req.model_type if req else "xgboost")).lower()
    t_name = query_target or (req.target if req else "all")
    test_size = req.test_size if req else 0.2
    if m_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid model_type '{m_type}'. Must be one of {MODEL_TYPES}.")

    # Load preprocessed data (or run preprocess on raw if not yet done)
    processed_path = PROCESSED_DATA_DIR / "processed_grid_data.csv"
    if not processed_path.exists():
        raw_path = resolve_active_dataset()
        preprocess_energy_data(raw_path, processed_path)

    df = pd.read_csv(processed_path)

    results = []
    targets_to_train = TARGETS if t_name == "all" else [t_name]

    for t in targets_to_train:
        if t not in TARGETS:
            raise HTTPException(status_code=400, detail=f"Invalid target '{t}'. Must be one of {TARGETS} or 'all'.")

        forecaster = GridForecaster(target=t, model_type=m_type)
        res = forecaster.train(df, test_size=test_size)
        forecaster.metadata["dataset_version"] = f"{processed_path.stat().st_size}:{processed_path.stat().st_mtime_ns}"
        forecaster.save()

        # Record in DB
        record_trained_model(
            model_name=f"{t}_{m_type}",
            target=t,
            model_type=m_type,
            mae=res["mae"],
            rmse=res["rmse"],
            r2=res["r2"],
            features=forecaster.feature_names,
            model_path=res["model_path"],
        )

        results.append(
            ModelMetric(
                target=t,
                model_type=m_type,
                mae=res["mae"],
                rmse=res["rmse"],
                r2=res["r2"],
                train_samples=res["train_samples"],
                test_samples=res["test_samples"],
                feature_importance=res["feature_importance"],
            )
        )

    return TrainResponse(
        status="success",
        message=f"Successfully trained {len(results)} model(s) with {m_type.upper()}.",
        results=results,
    )
