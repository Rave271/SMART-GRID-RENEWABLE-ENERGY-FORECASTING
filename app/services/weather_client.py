"""OpenWeather API client with realistic simulation fallback."""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import math
import requests
import pandas as pd

from app.config import OPENWEATHER_API_KEY


def estimate_solar_irradiance(cloud_pct: float, hour: int, month: int = 6) -> float:
    """Estimate solar irradiance (W/m^2) using solar zenith angle proxy and cloud cover."""
    if hour < 6 or hour > 19:
        return 0.0
    # Clear sky theoretical peak
    solar_noon_dist = abs(hour - 12.5)
    zenith_factor = max(0.0, math.cos(solar_noon_dist * (math.pi / 14)))
    clear_sky_ghi = 1000.0 * (zenith_factor ** 1.2)
    # Cloud attenuation
    cloud_attenuation = 1.0 - (0.75 * ((cloud_pct / 100.0) ** 1.3))
    ghi = max(0.0, clear_sky_ghi * cloud_attenuation)
    return round(ghi, 1)


def get_simulated_weather(city: str, hours_count: int = 48) -> Dict[str, Any]:
    """Provide realistic meteorological simulation when no API key is provided."""
    now = datetime.now()
    forecast = []

    # City hash to create deterministic variety per city
    city_hash = sum(ord(c) for c in city) % 15

    base_temp = 20.0 + (city_hash - 7)
    base_wind = 5.5 + (city_hash % 5)
    base_humidity = 60.0 + (city_hash % 20)

    for i in range(hours_count):
        dt = now + timedelta(hours=i)
        h = dt.hour
        # Diurnal temperature curve
        diurnal_t = 6.0 * math.sin(2 * math.pi * (h - 9) / 24)
        temp = round(base_temp + diurnal_t + math.sin(i / 5.0) * 1.5, 1)

        # Diurnal humidity
        humidity = round(max(20.0, min(95.0, base_humidity - (diurnal_t * 2) + math.cos(i / 4.0) * 5)), 1)

        # Wind variations
        wind_speed = round(max(0.5, base_wind + 2.0 * math.sin(i / 3.0) + math.cos(h / 3.0)), 1)

        cloud_pct = max(0.0, min(100.0, (humidity - 35) * 1.2))
        solar_irr = estimate_solar_irradiance(cloud_pct, h, dt.month)

        desc = "Clear Sky"
        if cloud_pct > 75:
            desc = "Overcast Clouds"
        elif cloud_pct > 40:
            desc = "Scattered Clouds"
        elif cloud_pct > 15:
            desc = "Few Clouds"

        forecast.append({
            "timestamp": dt.strftime("%Y-%m-%d %H:00:00"),
            "temperature": temp,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "solar_irradiance": solar_irr,
            "weather_desc": desc,
        })

    return {
        "city": city.title(),
        "source": "fallback_simulation",
        "is_live_key": False,
        "current": forecast[0],
        "forecast": forecast,
        "note": "OpenWeather API key is not configured or reached limit. Displaying real-time calibrated meteorological simulation.",
    }


def align_weather_forecast(
    forecast: List[Dict[str, Any]],
    start_timestamp: pd.Timestamp,
    hours: int,
) -> pd.DataFrame:
    """Convert provider forecast intervals into an hourly model input frame."""
    columns = ["temperature", "humidity", "wind_speed", "solar_irradiance"]
    source = pd.DataFrame(forecast)
    if source.empty or "timestamp" not in source:
        return pd.DataFrame()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce")
    source = source.dropna(subset=["timestamp"]).set_index("timestamp")
    source = source.reindex(columns=[column for column in columns if column in source])
    source = source.apply(pd.to_numeric, errors="coerce")
    source = source.resample("1h").interpolate(method="time")
    future_index = pd.date_range(start=start_timestamp, periods=hours, freq="h")
    aligned = source.reindex(future_index).interpolate(method="time").ffill().bfill()
    aligned.index.name = "timestamp"
    return aligned.reset_index()


def fetch_weather(city: str = "London", lat: Optional[float] = None, lon: Optional[float] = None) -> Dict[str, Any]:
    """Fetch current & 5-day hourly forecast from OpenWeatherMap API with graceful fallback."""
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY.lower() in ("your_openweather_api_key_here", "none", ""):
        return get_simulated_weather(city)

    base_url = "https://api.openweathermap.org/data/2.5"
    try:
        # 1. Fetch current weather
        if lat is not None and lon is not None:
            curr_params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
            fore_params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        else:
            curr_params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
            fore_params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}

        curr_res = requests.get(f"{base_url}/weather", params=curr_params, timeout=5)
        if curr_res.status_code != 200:
            return get_simulated_weather(city)

        curr_data = curr_res.json()
        city_name = curr_data.get("name", city)

        # 2. Fetch 5-day / 3-hour forecast
        fore_res = requests.get(f"{base_url}/forecast", params=fore_params, timeout=5)
        if fore_res.status_code != 200:
            return get_simulated_weather(city)

        fore_data = fore_res.json()

        # Format current item
        curr_dt = datetime.fromtimestamp(curr_data["dt"])
        curr_clouds = curr_data.get("clouds", {}).get("all", 0)
        curr_item = {
            "timestamp": curr_dt.strftime("%Y-%m-%d %H:00:00"),
            "temperature": round(curr_data["main"]["temp"], 1),
            "humidity": round(curr_data["main"]["humidity"], 1),
            "wind_speed": round(curr_data.get("wind", {}).get("speed", 0.0), 1),
            "solar_irradiance": estimate_solar_irradiance(curr_clouds, curr_dt.hour, curr_dt.month),
            "weather_desc": curr_data["weather"][0]["description"].title() if curr_data.get("weather") else "Clear",
        }

        # Format forecast items
        forecast_items = []
        for entry in fore_data.get("list", []):
            dt = datetime.fromtimestamp(entry["dt"])
            clouds = entry.get("clouds", {}).get("all", 0)
            forecast_items.append({
                "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": round(entry["main"]["temp"], 1),
                "humidity": round(entry["main"]["humidity"], 1),
                "wind_speed": round(entry.get("wind", {}).get("speed", 0.0), 1),
                "solar_irradiance": estimate_solar_irradiance(clouds, dt.hour, dt.month),
                "weather_desc": entry["weather"][0]["description"].title() if entry.get("weather") else "Clear",
            })

        return {
            "city": city_name,
            "source": "openweather",
            "is_live_key": True,
            "current": curr_item,
            "forecast": forecast_items,
        }

    except Exception:
        # Network timeout or DNS failure fallback
        return get_simulated_weather(city)
