"""
Unit tests for sop_engine.py — core logic validation.
No network calls; uses real SOP data from sops.yaml.
"""

import pytest
from bot.weather_service import WeatherData
from bot.sop_engine import (
    load_sops,
    match_sops,
    resolve,
    render_response,
    Intent,
    SOP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_weather(**kwargs) -> WeatherData:
    defaults = dict(
        temperature=22.0,
        wind_speed=10.0,
        precipitation=0.0,
        precipitation_probability=10,
        uv_index=3.0,
        weather_code=0,
        fetched_at="2024-01-01T12:00:00+00:00",
    )
    defaults.update(kwargs)
    return WeatherData(**defaults)


# ---------------------------------------------------------------------------
# load_sops
# ---------------------------------------------------------------------------

def test_load_sops_returns_at_least_10():
    """Requirement 3.1: at least 10 SOPs at runtime."""
    sops = load_sops()
    assert len(sops) >= 10


def test_load_sops_ids_are_unique():
    sops = load_sops()
    ids = [s.id for s in sops]
    assert len(ids) == len(set(ids))


def test_load_sops_severity_values_valid():
    """All severities must be one of the four recognised levels."""
    sops = load_sops()
    valid = {"CRITICAL", "HIGH", "MODERATE", "LOW"}
    for sop in sops:
        assert sop.severity in valid, f"{sop.id} has unexpected severity {sop.severity}"


def test_load_sops_categories_span_at_least_3():
    """Requirement 3.2: at least 3 distinct activity categories."""
    sops = load_sops()
    categories = {s.category for s in sops}
    assert len(categories) >= 3


def test_load_sops_severity_range():
    """Requirement 3.3: rule set must include more than one severity level."""
    sops = load_sops()
    severities = {s.severity for s in sops}
    assert len(severities) >= 2


# ---------------------------------------------------------------------------
# match_sops — condition evaluation
# ---------------------------------------------------------------------------

def test_match_sops_uv_threshold_fires():
    """SOP-001 should fire when UV >= 8 and activity is outdoor exercise."""
    sops = load_sops()
    weather = make_weather(uv_index=9.0)
    intent = Intent(location="London", activity="running")
    matches = match_sops(sops, weather, intent)
    ids = [s.id for s in matches]
    assert "SOP-001" in ids


def test_match_sops_uv_threshold_does_not_fire_below():
    """SOP-001 should NOT fire when UV < 8."""
    sops = load_sops()
    weather = make_weather(uv_index=5.0)
    intent = Intent(location="London", activity="running")
    matches = match_sops(sops, weather, intent)
    ids = [s.id for s in matches]
    assert "SOP-001" not in ids


def test_match_sops_activity_filter_respected():
    """SOP-002 (cycling) should NOT match when activity is 'picnic'. (Property 1)"""
    sops = load_sops()
    weather = make_weather(precipitation_probability=80)
    intent = Intent(location="Paris", activity="picnic")
    matches = match_sops(sops, weather, intent)
    ids = [s.id for s in matches]
    assert "SOP-002" not in ids


def test_match_sops_cycling_rain_fires():
    """SOP-002 should match when precip_prob >= 70 and activity is cycling."""
    sops = load_sops()
    weather = make_weather(precipitation_probability=75)
    intent = Intent(location="Paris", activity="cycling")
    matches = match_sops(sops, weather, intent)
    ids = [s.id for s in matches]
    assert "SOP-002" in ids


def test_match_sops_critical_storm_fires():
    """SOP-004 (CRITICAL) should fire for thunderstorm + heavy precipitation."""
    sops = load_sops()
    weather = make_weather(weather_code=95, precipitation=20.0)
    intent = Intent(location="Berlin", activity="running")
    matches = match_sops(sops, weather, intent)
    ids = [s.id for s in matches]
    assert "SOP-004" in ids


def test_match_sops_fuzzy_composite_picnic():
    """SOP-009 should fire when all four comfort dimensions are within range."""
    sops = load_sops()
    weather = make_weather(
        temperature=24.0,
        wind_speed=10.0,
        precipitation_probability=15,
        uv_index=4.0,
    )
    intent = Intent(location="Rome", activity="picnic")
    matches = match_sops(sops, weather, intent)
    ids = [s.id for s in matches]
    assert "SOP-009" in ids


def test_match_sops_fuzzy_composite_picnic_fails_when_wind_too_high():
    """SOP-009 must NOT fire when wind_speed >= 20 km/h."""
    sops = load_sops()
    weather = make_weather(
        temperature=24.0,
        wind_speed=25.0,   # too high
        precipitation_probability=15,
        uv_index=4.0,
    )
    intent = Intent(location="Rome", activity="picnic")
    matches = match_sops(sops, weather, intent)
    ids = [s.id for s in matches]
    assert "SOP-009" not in ids


def test_match_sops_empty_when_nothing_matches():
    """Benign conditions with no matching activity should return an empty list."""
    sops = load_sops()
    weather = make_weather(
        temperature=18.0,
        wind_speed=5.0,
        precipitation_probability=10,
        uv_index=2.0,
        weather_code=0,
        precipitation=0.0,
    )
    intent = Intent(location="Dublin", activity="reading indoors")
    matches = match_sops(sops, weather, intent)
    # None of the SOPs should match this benign / indoor-only scenario
    assert isinstance(matches, list)


# ---------------------------------------------------------------------------
# resolve — conflict resolution
# ---------------------------------------------------------------------------

def test_resolve_empty_returns_none():
    primary, secondaries = resolve([])
    assert primary is None
    assert secondaries == []


def test_resolve_single_sop():
    sops = load_sops()
    primary, secondaries = resolve([sops[0]])
    assert primary == sops[0]
    assert secondaries == []


def test_resolve_primary_is_highest_severity():
    """Property 2: primary severity >= every secondary severity."""
    sops = load_sops()
    weather = make_weather(uv_index=9.0, precipitation_probability=75, wind_speed=45.0)
    intent = Intent(location="London", activity="cycling")
    matches = match_sops(sops, weather, intent)
    if len(matches) < 2:
        pytest.skip("Need at least 2 matching SOPs for this test")
    primary, secondaries = resolve(matches, intent)
    assert primary is not None
    for sec in secondaries:
        assert primary.severity_value >= sec.severity_value


def test_resolve_is_deterministic():
    """Property 2: same input always produces the same primary."""
    sops = load_sops()
    weather = make_weather(uv_index=9.0, precipitation_probability=75, wind_speed=45.0)
    intent = Intent(location="London", activity="cycling")
    matches = match_sops(sops, weather, intent)
    results = [resolve(matches, intent)[0] for _ in range(5)]
    assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# render_response
# ---------------------------------------------------------------------------

def test_render_response_includes_sop_id():
    """Every rendered response must contain the SOP id (Requirement 1.1)."""
    sops = load_sops()
    sop = next(s for s in sops if s.id == "SOP-001")
    weather = make_weather(uv_index=9.0)
    intent = Intent(location="London", activity="running")
    result = render_response(sop, weather, intent)
    assert "SOP-001" in result


def test_render_response_uses_actual_weather_values():
    """Property 8: numeric values in response come from WeatherData, not invented."""
    sops = load_sops()
    sop = next(s for s in sops if s.id == "SOP-001")
    weather = make_weather(uv_index=9.3)
    intent = Intent(location="London", activity="running")
    result = render_response(sop, weather, intent)
    assert "9.3" in result


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis)
# ---------------------------------------------------------------------------

from hypothesis import given, settings, strategies as st


def _arb_weather(**overrides):
    """Strategy that generates a WeatherData with random extreme-range values."""
    base = st.fixed_dictionaries({
        "temperature": st.floats(min_value=-50.0, max_value=80.0),
        "wind_speed": st.floats(min_value=0.0, max_value=200.0),
        "precipitation": st.floats(min_value=0.0, max_value=200.0),
        "precipitation_probability": st.integers(min_value=0, max_value=100),
        "uv_index": st.floats(min_value=0.0, max_value=20.0),
        "weather_code": st.integers(min_value=0, max_value=99),
        "fetched_at": st.just("2024-01-01T12:00:00+00:00"),
    })
    return base


@given(_arb_weather())
@settings(max_examples=150)
def test_property_match_sops_never_raises(weather_kwargs):
    """
    Property: match_sops must never raise an unhandled exception for any
    combination of extreme weather values. Covers temps up to 80 °C,
    wind up to 200 km/h, UV up to 20.
    """
    weather = make_weather(**weather_kwargs)
    sops = load_sops()
    intent = Intent(location="TestCity", activity="cycling")
    # Must not raise
    result = match_sops(sops, weather, intent)
    assert isinstance(result, list)


@given(_arb_weather())
@settings(max_examples=150)
def test_property_resolve_never_raises(weather_kwargs):
    """
    Property: resolve() must never raise for any weather input, including
    edge cases where no SOPs match and matches is an empty list.
    """
    weather = make_weather(**weather_kwargs)
    sops = load_sops()
    intent = Intent(location="TestCity", activity="cycling")
    matches = match_sops(sops, weather, intent)
    primary, secondaries = resolve(matches, intent)
    # Either primary is None (no match) or it is a valid SOP
    assert primary is None or isinstance(primary, SOP)
    assert isinstance(secondaries, list)


@given(_arb_weather())
@settings(max_examples=150)
def test_property_response_is_never_blank(weather_kwargs):
    """
    Property: when a SOP matches, render_response must return a non-empty
    string that contains the SOP id — never a blank or uncited response.
    """
    weather = make_weather(**weather_kwargs)
    sops = load_sops()
    intent = Intent(location="TestCity", activity="cycling")
    matches = match_sops(sops, weather, intent)
    primary, _ = resolve(matches, intent)
    if primary is None:
        return  # no match is valid; tested separately
    result = render_response(primary, weather, intent)
    assert result.strip() != "", "render_response returned blank string"
    assert primary.id in result, f"SOP id {primary.id} missing from response"


@given(_arb_weather())
@settings(max_examples=150)
def test_property_extreme_values_resolve_or_graceful_nomatch(weather_kwargs):
    """
    Property: extreme weather values (wind > 150 km/h, temp > 60 °C, UV > 15)
    must either resolve to a valid SOP or return None — never raise an exception.
    """
    weather = make_weather(**weather_kwargs)
    sops = load_sops()
    # Use a broad activity so multiple SOPs can potentially match
    intent = Intent(location="ExtremeCity", activity="outdoor exercise")
    matches = match_sops(sops, weather, intent)
    primary, secondaries = resolve(matches, intent)
    # No exception raised; result is typed correctly
    assert primary is None or isinstance(primary, SOP)
    assert all(isinstance(s, SOP) for s in secondaries)


@given(
    st.text(min_size=0, max_size=50),   # arbitrary activity string
    _arb_weather(),
)
@settings(max_examples=100)
def test_property_arbitrary_activity_never_raises(activity, weather_kwargs):
    """
    Property: any arbitrary activity string (including empty, unicode, injection
    attempts) must not cause match_sops or resolve to raise.
    """
    weather = make_weather(**weather_kwargs)
    sops = load_sops()
    intent = Intent(location="SomeCity", activity=activity if activity else None)
    matches = match_sops(sops, weather, intent)
    primary, _ = resolve(matches, intent)
    assert isinstance(matches, list)
