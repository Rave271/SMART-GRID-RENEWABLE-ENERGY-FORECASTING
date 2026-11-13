"""Pydantic schemas for request and response validation."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DatasetUploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    filepath: str
    row_count: int
    column_count: int
    columns: List[str]
    missing_values: Dict[str, int]
    preview: List[Dict[str, Any]]


class PreprocessRequest(BaseModel):
    filepath: Optional[str] = None
    resample_freq: str = "1h"
    clip_outliers: bool = True


class PreprocessResponse(BaseModel):
    status: str
    message: str
    original_rows: int
    processed_rows: int
    missing_imputed: int
    outliers_clipped: int
    columns: List[str]
    sample_preview: List[Dict[str, Any]]


class TrainRequest(BaseModel):
    model_type: str = Field(default="xgboost", description="xgboost or mlp")
    target: str = Field(default="all", description="solar_power_mw, wind_power_mw, electricity_demand_mw, or all")
    test_size: float = Field(default=0.2, ge=0.05, le=0.4)


class ModelMetric(BaseModel):
    target: str
    model_type: str
    mae: float
    rmse: float
    r2: float
    train_samples: int
    test_samples: int
    feature_importance: Optional[Dict[str, float]] = None


class TrainResponse(BaseModel):
    status: str
    message: str
    results: List[ModelMetric]


class PredictResponse(BaseModel):
    status: str
    horizon: str
    model_type: str
    timestamps: List[str]
    predictions: List[Dict[str, Any]]
    summary: Dict[str, Any]


class WeatherItem(BaseModel):
    timestamp: str
    temperature: float
    humidity: float
    wind_speed: float
    solar_irradiance: float
    weather_desc: str


class WeatherResponse(BaseModel):
    city: str
    source: str
    is_live_key: bool
    current: WeatherItem
    forecast: List[WeatherItem]


class RecommendationItem(BaseModel):
    id: int
    hour_offset: int
    timestamp: str
    category: str  # surplus, deficit, normal
    severity: str  # info, warning, danger, success
    title: str
    description: str
    action: str


class RecommendationResponse(BaseModel):
    horizon: str
    peak_surplus_time: Optional[str] = None
    peak_deficit_time: Optional[str] = None
    max_surplus_mw: float = 0.0
    max_deficit_mw: float = 0.0
    total_renewable_energy_mwh: float = 0.0
    total_demand_mwh: float = 0.0
    avg_renewable_penetration_pct: float = 0.0
    recommendations: List[RecommendationItem]
