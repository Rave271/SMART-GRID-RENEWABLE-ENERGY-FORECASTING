"""Integration and API route tests for Smart Grid endpoints."""

import asyncio
from pathlib import Path
from fastapi import UploadFile
from fastapi.datastructures import Headers

from app.routes.recommendations import health_check, get_grid_recommendations, get_metrics
from app.routes.weather import get_weather
from app.routes.train import preprocess_dataset, train_model
from app.routes.predict import get_predictions
from app.routes.reports import export_report
from app.schemas import TrainRequest


def test_health_endpoint():
    res = asyncio.run(health_check())
    assert res["status"] == "healthy"
    assert "version" in res


def test_weather_endpoint():
    res = asyncio.run(get_weather(city="London"))
    assert res.city.lower() == "london"
    assert res.current.temperature is not None
    assert len(res.forecast) > 0


def test_preprocess_endpoint():
    res = asyncio.run(preprocess_dataset())
    assert res.status == "success"
    assert res.processed_rows > 0
    assert len(res.columns) == 8


def test_train_endpoint():
    req = TrainRequest(model_type="xgboost", target="solar_power_mw", test_size=0.2)
    res = asyncio.run(train_model(req=req))
    assert res.status == "success"
    assert len(res.results) == 1
    assert res.results[0].mae >= 0.0
    assert res.results[0].r2 > 0.80


def test_predict_endpoint_24h():
    res = asyncio.run(get_predictions(horizon="24h", target="all", model_type="xgboost"))
    assert res.status == "success"
    assert res.horizon == "24h"
    assert len(res.predictions) == 24
    assert res.summary["total_generation_mwh"] > 0


def test_predict_endpoint_7d():
    res = asyncio.run(get_predictions(horizon="7d", target="all", model_type="xgboost"))
    assert res.status == "success"
    assert res.horizon == "168h"
    assert len(res.predictions) == 168


def test_recommendations_endpoint():
    res = asyncio.run(get_grid_recommendations(horizon="24h", model_type="xgboost"))
    assert res.horizon == "24h"
    assert len(res.recommendations) > 0
    assert res.total_renewable_energy_mwh > 0


def test_metrics_endpoint():
    res = asyncio.run(get_metrics())
    assert res["status"] == "success"
    assert res["count"] > 0
    assert len(res["models"]) > 0


def test_export_report_csv():
    res = asyncio.run(export_report(format="csv", horizon="24h", model_type="xgboost"))
    assert res.media_type == "text/csv"
    assert Path(res.path).exists()


def test_export_report_pdf():
    res = asyncio.run(export_report(format="pdf", horizon="24h", model_type="xgboost"))
    assert res.media_type == "application/pdf"
    assert Path(res.path).exists()
