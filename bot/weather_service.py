"""
Weather Service: geocodes city names and fetches live weather data from Open-Meteo.
Returns None (never raises) on any HTTP error, timeout, or empty geocoding results.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

import httpx

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Fields always requested from Open-Meteo (Requirements 2.4)
WEATHER_FIELDS = [
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
    "precipitation_probability",
    "uv_index",
    "weather_code",
]

REQUEST_TIMEOUT = 10.0  # seconds


@dataclass
class WeatherData:
    temperature: float          # °C  (temperature_2m)
    wind_speed: float           # km/h (wind_speed_10m)
    precipitation: float        # mm  (precipitation)
    precipitation_probability: int  # 0-100 (precipitation_probability)
    uv_index: float             # (uv_index)
    weather_code: int           # WMO code (weather_code)
    fetched_at: str             # ISO 8601 timestamp


def geocode(city: str) -> Optional[Tuple[float, float]]:
    """
    Resolve a city name to (latitude, longitude) using Open-Meteo geocoding.

    Returns the first candidate when multiple results are returned (Requirement 2.6).
    Returns None if the city cannot be resolved or any error occurs (Requirement 2.2).
    """
    try:
        response = httpx.get(
            GEOCODING_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results")
        if not results:
            return None
        first = results[0]
        return (float(first["latitude"]), float(first["longitude"]))
    except Exception:
        return None


def fetch_weather(lat: float, lon: float) -> Optional[WeatherData]:
    """
    Fetch the current weather forecast for the given coordinates from Open-Meteo.

    Always requests: temperature_2m, wind_speed_10m, precipitation,
    precipitation_probability, uv_index, weather_code (Requirement 2.4).

    Returns None on any HTTP error or timeout (Requirement 2.3).
    """
    try:
        response = httpx.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": ",".join(WEATHER_FIELDS),
                "timezone": "auto",
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        return WeatherData(
            temperature=float(current["temperature_2m"]),
            wind_speed=float(current["wind_speed_10m"]),
            precipitation=float(current["precipitation"]),
            precipitation_probability=int(current["precipitation_probability"]),
            uv_index=float(current["uv_index"]),
            weather_code=int(current["weather_code"]),
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception:
        return None
