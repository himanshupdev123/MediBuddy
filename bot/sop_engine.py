"""
SOP Engine: loads, matches, and resolves Standard Operating Procedures.

Public API
----------
load_sops(path)           -> list[SOP]
match_sops(sops, weather, intent) -> list[SOP]
resolve(matches)          -> tuple[SOP | None, list[SOP]]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from bot.weather_service import WeatherData

# ---------------------------------------------------------------------------
# WMO weather-code label lookup (used in response templates)
# ---------------------------------------------------------------------------
_WMO_LABELS: dict[int, str] = {
    45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}

# Severity order used for comparisons
_SEVERITY_ORDER: dict[str, int] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MODERATE": 2,
    "LOW": 1,
}

DEFAULT_SOP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "sops.yaml"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Intent:
    """Extracted intent from the user's message."""
    location: str
    activity: str | None = None
    timeframe: str | None = None   # "now", "morning", "evening", "today", etc.


@dataclass
class SOP:
    id: str
    title: str
    category: str
    severity: str                   # CRITICAL | HIGH | MODERATE | LOW
    description: str
    conditions: dict[str, Any]
    activities: list[str] = field(default_factory=list)
    response_template: str = ""

    @property
    def severity_value(self) -> int:
        return _SEVERITY_ORDER.get(self.severity.upper(), 0)


# ---------------------------------------------------------------------------
# load_sops
# ---------------------------------------------------------------------------

def load_sops(path: str = DEFAULT_SOP_PATH) -> list[SOP]:
    """
    Read SOPs from a YAML file on *every* call so that edits take effect
    without restarting the application (hot-reload, Requirement 4.2 / 4.4).
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw: list[dict] = yaml.safe_load(fh) or []

    sops: list[SOP] = []
    for item in raw:
        sops.append(SOP(
            id=str(item["id"]),
            title=str(item.get("title", "")),
            category=str(item.get("category", "")),
            severity=str(item.get("severity", "LOW")).upper(),
            description=str(item.get("description", "")),
            conditions=dict(item.get("conditions") or {}),
            activities=list(item.get("activities") or []),
            response_template=str(item.get("response_template", "")),
        ))
    return sops


# ---------------------------------------------------------------------------
# Condition evaluation helpers
# ---------------------------------------------------------------------------

def _activity_matches(sop: SOP, intent: Intent) -> bool:
    """
    Returns True if:
    - the SOP has no activity filter (applies to all), OR
    - the user's activity string contains one of the SOP's activity keywords.
    (Property 1: SOP match is activity-aware — Requirements 3.1, 3.3)
    """
    if not sop.activities:
        return True
    if not intent.activity:
        return False
    activity_lower = intent.activity.lower()
    return any(kw.lower() in activity_lower for kw in sop.activities)


def _eval_fuzzy_composite(sub_conditions: dict[str, Any], weather: WeatherData) -> bool:
    """
    Fuzzy composite matching for SOP-009-style SOPs.
    All sub-conditions must be satisfied simultaneously (Requirement 3.4).
    Supported keys: temperature_range, precip_prob_lt, wind_speed_lt, uv_index_lt,
                    temperature_gte, temperature_lt, uv_index_gte, precip_prob_gte,
                    wind_speed_gt, wind_speed_gte.
    """
    for key, value in sub_conditions.items():
        if key == "temperature_range":
            lo, hi = value[0], value[1]
            if not (lo <= weather.temperature <= hi):
                return False
        elif key == "precip_prob_lt":
            if not (weather.precipitation_probability < value):
                return False
        elif key == "precip_prob_gte":
            if not (weather.precipitation_probability >= value):
                return False
        elif key == "wind_speed_lt":
            if not (weather.wind_speed < value):
                return False
        elif key == "wind_speed_gt":
            if not (weather.wind_speed > value):
                return False
        elif key == "wind_speed_gte":
            if not (weather.wind_speed >= value):
                return False
        elif key == "uv_index_lt":
            if not (weather.uv_index < value):
                return False
        elif key == "uv_index_gte":
            if not (weather.uv_index >= value):
                return False
        elif key == "temperature_gte":
            if not (weather.temperature >= value):
                return False
        elif key == "temperature_lt":
            if not (weather.temperature < value):
                return False
    return True


def _eval_conditions(conditions: dict[str, Any], weather: WeatherData) -> bool:
    """
    Evaluate a top-level conditions dict against WeatherData.
    Multiple top-level keys are AND-ed together.
    """
    for key, value in conditions.items():
        if key == "uv_index_gte":
            if not (weather.uv_index >= value):
                return False
        elif key == "uv_index_lt":
            if not (weather.uv_index < value):
                return False
        elif key == "precip_prob_gte":
            if not (weather.precipitation_probability >= value):
                return False
        elif key == "precip_prob_lt":
            if not (weather.precipitation_probability < value):
                return False
        elif key == "wind_speed_gt":
            if not (weather.wind_speed > value):
                return False
        elif key == "wind_speed_gte":
            if not (weather.wind_speed >= value):
                return False
        elif key == "wind_speed_lt":
            if not (weather.wind_speed < value):
                return False
        elif key == "temperature_gte":
            if not (weather.temperature >= value):
                return False
        elif key == "temperature_lt":
            if not (weather.temperature < value):
                return False
        elif key == "temperature_range":
            lo, hi = value[0], value[1]
            if not (lo <= weather.temperature <= hi):
                return False
        elif key == "weather_code_in":
            if weather.weather_code not in value:
                return False
        elif key == "precipitation_gte":
            if not (weather.precipitation >= value):
                return False
        elif key == "fuzzy_composite":
            if not _eval_fuzzy_composite(value, weather):
                return False
        # Unknown condition keys are silently ignored (forward-compat)
    return True


# ---------------------------------------------------------------------------
# match_sops
# ---------------------------------------------------------------------------

def match_sops(
    sops: list[SOP],
    weather: WeatherData,
    intent: Intent,
) -> list[SOP]:
    """
    Return the subset of SOPs whose conditions are satisfied for the given
    weather snapshot and user intent, sorted by severity descending.

    Activity filter is applied: a SOP with a non-empty activities list only
    matches if the user's activity contains one of those keywords (Property 1).
    """
    matched: list[SOP] = []
    for sop in sops:
        if not _activity_matches(sop, intent):
            continue
        if _eval_conditions(sop.conditions, weather):
            matched.append(sop)

    # Sort by severity descending, then id ascending (stable)
    matched.sort(key=lambda s: (-s.severity_value, s.id))
    return matched


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------

def resolve(matches: list[SOP], intent: Intent | None = None) -> tuple[SOP | None, list[SOP]]:
    """
    Apply conflict-resolution strategy and return (primary, secondaries).

    Strategy (Requirement 3.5 / 3.6 / 3.7):
    1. Primary = highest severity SOP.
    2. Tie-break 1: SOP whose activities list better matches the user's activity
       (more specific = activities list is non-empty and intent.activity appears).
    3. Tie-break 2: lower SOP id (lexicographic on "SOP-NNN").

    (Property 2: resolve() is deterministic and severity-ordered)
    """
    if not matches:
        return None, []

    def sort_key(sop: SOP) -> tuple:
        # Higher severity → lower sort value (we want descending)
        sev = -sop.severity_value
        # More specific activity match → lower sort value (preferred)
        activity_specificity = 0 if (sop.activities and intent and intent.activity) else 1
        return (sev, activity_specificity, sop.id)

    sorted_matches = sorted(matches, key=sort_key)
    primary = sorted_matches[0]
    secondaries = sorted_matches[1:]
    return primary, secondaries


# ---------------------------------------------------------------------------
# Response template rendering helper
# ---------------------------------------------------------------------------

def render_response(sop: SOP, weather: WeatherData, intent: Intent) -> str:
    """
    Fill response_template placeholders with WeatherData values and intent fields.
    Only WeatherData fields and intent fields are substituted — no LLM involvement.
    (Property 8: Response template uses only WeatherData fields — Requirement 10.1)
    """
    weather_code_label = _WMO_LABELS.get(weather.weather_code, str(weather.weather_code))
    return sop.response_template.format(
        temperature=weather.temperature,
        wind_speed=weather.wind_speed,
        precipitation=weather.precipitation,
        precipitation_probability=weather.precipitation_probability,
        uv_index=weather.uv_index,
        weather_code=weather.weather_code,
        weather_code_label=weather_code_label,
        activity=intent.activity or "outdoor activity",
        location=intent.location,
        timeframe=intent.timeframe or "now",
    ).strip()
