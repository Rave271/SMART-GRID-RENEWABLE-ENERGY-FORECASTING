"""Route for generating future renewable energy and demand forecasts."""

import re
from typing import Optional, List, Dict, Any
from pathlib import Path
from fastapi import APIRouter, Query, HTTPException, Depends
import pandas as pd
import numpy as np

from app.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, DEFAULT_SAMPLE_PATH, require_api_key, rate_limit_predict
from app.services.model import GridForecaster, TARGETS, MODEL_TYPES
from app.services.preprocessing import preprocess_energy_data
from app.services.weather_client import fetch_weather, align_weather_forecast
from app.db.database import record_prediction
from app.schemas import PredictResponse

router = APIRouter(prefix="/api", tags=["Forecasting"], dependencies=[Depends(require_api_key), Depends(rate_limit_predict)])


def get_or_create_processed_data() -> pd.DataFrame:
    """Load processed dataset or run preprocessing on raw/sample data."""
    processed_path = PROCESSED_DATA_DIR / "processed_grid_data.csv"
    if not processed_path.exists():
        raw_files = sorted(RAW_DATA_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        raw_path = raw_files[0] if raw_files else DEFAULT_SAMPLE_PATH
        if not raw_path.exists():
            raise HTTPException(status_code=404, detail="No historical dataset found to base forecast upon.")
        preprocess_energy_data(raw_path, processed_path)
    return pd.read_csv(processed_path)


@router.get("/predict", response_model=PredictResponse)
async def get_predictions(
    horizon: str = Query("24h", description="Forecast horizon: '24h' or '7d'"),
    target: str = Query("all", description="Target: solar_power_mw, wind_power_mw, electricity_demand_mw, or all"),
    model_type: str = Query("xgboost", description="Model architecture: 'xgboost' or 'mlp'"),
    city: Optional[str] = Query(None, description="City name to incorporate live weather forecast context"),
):
    """Predict future renewable generation and demand for 24 hours or 7 days."""
    # Parse horizon hours
    horizon_value = horizon if isinstance(horizon, str) else "24h"
    model_type_value = model_type if isinstance(model_type, str) else "xgboost"
    city_value = city if isinstance(city, str) else None
    horizon_clean = horizon_value.lower().strip()
    if horizon_clean == "24h":
        hours = 24
    elif horizon_clean in ("7d", "168h"):
        hours = 168
    elif re.fullmatch(r"[1-9]\d*h", horizon_clean):
        hours = int(horizon_clean[:-1])
        if hours > 168:
            raise HTTPException(status_code=400, detail="Forecast horizon cannot exceed 168 hours.")
    else:
        raise HTTPException(status_code=400, detail="Invalid horizon. Use 24h, 7d, or an hourly value up to 168h.")
    m_type = model_type_value.lower()
    if m_type not in MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid model_type '{m_type}'. Must be one of {MODEL_TYPES}.")
    target_value = target if isinstance(target, str) else "all"
    if target_value != "all" and target_value not in TARGETS:
        raise HTTPException(status_code=400, detail=f"Invalid target '{target_value}'. Must be one of {TARGETS} or 'all'.")

    df = get_or_create_processed_data()

    # Optional weather context
    future_weather_df = None
    if city_value:
        weather_res = fetch_weather(city=city_value)
        if weather_res.get("forecast"):
            last_timestamp = pd.to_datetime(df["timestamp"].iloc[-1])
            future_weather_df = align_weather_forecast(
                weather_res["forecast"],
                start_timestamp=last_timestamp + pd.Timedelta(hours=1),
                hours=hours,
            )
            if future_weather_df.empty:
                future_weather_df = None

    active_targets = TARGETS if target_value == "all" else [target_value]
    for t in active_targets:
        fc = GridForecaster(target=t, model_type=m_type)
        if not fc.get_model_path().exists():
            raise HTTPException(
                status_code=409,
                detail=f"Model for {t} ({m_type}) is not trained. Train it with POST /api/train first.",
            )

    # Perform multi-step forecasting
    pred_solar = []
    pred_wind = []
    pred_demand = []

    forecasts = {}
    for t in active_targets:
        forecaster = GridForecaster(target=t, model_type=m_type)
        forecasts[t] = forecaster.predict_future(df, horizon_hours=hours, future_weather=future_weather_df)
    pred_solar = forecasts.get("solar_power_mw", [0.0] * hours)
    pred_wind = forecasts.get("wind_power_mw", [0.0] * hours)
    pred_demand = forecasts.get("electricity_demand_mw", [0.0] * hours)

    # Generate future timestamps
    last_ts = pd.to_datetime(df["timestamp"].iloc[-1])
    future_dates = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=hours, freq="h")
    timestamps = [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in future_dates]

    # Combine into hourly prediction records
    predictions = []
    total_solar = 0.0
    total_wind = 0.0
    total_gen = 0.0
    total_demand = 0.0
    max_surplus = 0.0
    max_deficit = 0.0
    peak_surplus_time = None
    peak_deficit_time = None

    for i in range(hours):
        s = pred_solar[i]
        w = pred_wind[i]
        g = round(s + w, 2)
        d = pred_demand[i]
        bal = round(g - d, 2)
        ts = timestamps[i]

        total_solar += s
        total_wind += w
        total_gen += g
        total_demand += d

        if bal > max_surplus:
            max_surplus = bal
            peak_surplus_time = ts

        if bal < -max_deficit:
            max_deficit = abs(bal)
            peak_deficit_time = ts

        prediction = {
            "timestamp": ts,
        }
        if target_value in ("all", "solar_power_mw"):
            prediction["predicted_solar"] = s
        if target_value in ("all", "wind_power_mw"):
            prediction["predicted_wind"] = w
        if target_value in ("all", "electricity_demand_mw"):
            prediction["predicted_demand"] = d
        if target_value == "all":
            prediction.update({"predicted_generation": g, "surplus_deficit": bal})
        predictions.append(prediction)

    summary = {
        "horizon_hours": hours,
        "total_solar_mwh": round(total_solar, 1),
        "total_wind_mwh": round(total_wind, 1),
        "total_generation_mwh": round(total_gen, 1),
        "total_demand_mwh": round(total_demand, 1),
        "net_balance_mwh": round(total_gen - total_demand, 1),
        "avg_renewable_penetration_pct": round((total_gen / total_demand * 100.0) if total_demand > 0 else 0.0, 1),
        "max_surplus_mw": round(max_surplus, 1),
        "max_deficit_mw": round(max_deficit, 1),
        "peak_surplus_time": peak_surplus_time,
        "peak_deficit_time": peak_deficit_time,
    }

    # Record in DB
    record_prediction(target=target_value, horizon=f"{hours}h", model_type=m_type, summary=summary)

    return PredictResponse(
        status="success",
        horizon=f"{hours}h",
        model_type=m_type,
        timestamps=timestamps,
        predictions=predictions,
        summary=summary,
    )
