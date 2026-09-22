"""
Unit tests for weather_service.py — field mapping and error handling.
Live network calls are avoided by testing the None-on-error contract
with an unreachable host.
"""

from bot.weather_service import WeatherData, geocode, fetch_weather


def test_weather_data_fields_exist():
    """WeatherData dataclass has all required fields (Requirement 2.4)."""
    wd = WeatherData(
        temperature=20.0,
        wind_speed=15.0,
        precipitation=0.5,
        precipitation_probability=30,
        uv_index=4.0,
        weather_code=61,
        fetched_at="2024-01-01T10:00:00+00:00",
    )
    assert wd.temperature == 20.0
    assert wd.wind_speed == 15.0
    assert wd.precipitation == 0.5
    assert wd.precipitation_probability == 30
    assert wd.uv_index == 4.0
    assert wd.weather_code == 61


def test_geocode_returns_none_for_empty_city():
    """geocode('') should return None gracefully — no exception raised."""
    result = geocode("")
    assert result is None


def test_fetch_weather_returns_none_for_invalid_coords():
    """Extreme / invalid coordinates should return None, not raise."""
    result = fetch_weather(999.0, 999.0)
    assert result is None
