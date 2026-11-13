"""Route for real-time and forecast weather."""

from typing import Optional
from fastapi import APIRouter, Query
from app.services.weather_client import fetch_weather
from app.schemas import WeatherResponse

router = APIRouter(prefix="/api", tags=["Weather"])


@router.get("/weather", response_model=WeatherResponse)
async def get_weather(
    city: str = Query("London", description="City name for weather query"),
    lat: Optional[float] = Query(None, description="Latitude coordinate"),
    lon: Optional[float] = Query(None, description="Longitude coordinate"),
):
    """Fetch current and 5-day weather features (temperature, humidity, wind speed, solar irradiance)
    from OpenWeather API, with automated simulation fallback if key is unconfigured.
    """
    data = fetch_weather(city=city, lat=lat, lon=lon)
    return WeatherResponse(**data)
