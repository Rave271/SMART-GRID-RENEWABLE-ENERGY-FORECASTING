"""Routes for energy utilization recommendations, metrics, and health."""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException, Depends
from app.routes.predict import get_predictions
from app.config import require_api_key, rate_limit_predict
from app.services.recommender import generate_recommendations
from app.db.database import get_all_models_metrics
from app.schemas import RecommendationResponse

router = APIRouter(prefix="/api", tags=["Recommendations & Metrics"])


@router.get(
    "/recommendations",
    response_model=RecommendationResponse,
    dependencies=[Depends(require_api_key), Depends(rate_limit_predict)],
)
async def get_grid_recommendations(
    horizon: str = Query("24h", description="Forecast horizon: '24h' or '7d'"),
    model_type: str = Query("xgboost", description="Model architecture: 'xgboost' or 'mlp'"),
    city: Optional[str] = Query(None, description="City for live weather context"),
):
    """Generate rule-based dispatch, battery storage, and load shifting recommendations based on forecast."""
    horizon_value = horizon if isinstance(horizon, str) else "24h"
    model_type_value = model_type if isinstance(model_type, str) else "xgboost"
    city_value = city if isinstance(city, str) else None
    pred_res = await get_predictions(horizon=horizon_value, target="all", model_type=model_type_value, city=city_value)
    recom_data = generate_recommendations(pred_res.predictions)
    return RecommendationResponse(**recom_data)


@router.get("/metrics", dependencies=[Depends(require_api_key)])
async def get_metrics():
    """Fetch latest model performance metrics (MAE, RMSE, R²) for all trained models."""
    metrics_list = get_all_models_metrics()
    return {
        "status": "success",
        "count": len(metrics_list),
        "models": metrics_list,
    }


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Smart Grid Renewable Energy Forecasting API",
        "version": "1.0.0",
    }
