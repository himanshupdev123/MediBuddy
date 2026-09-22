"""
LangGraph node implementations for the Weather Advisory Bot.

Nodes
-----
parse_intent      – LLM extracts location/activity/timeframe; merges with session context
fetch_weather     – geocode + live weather fetch
match_sop         – load SOPs, match, resolve
generate_response – fill SOP template with WeatherData values
handle_failure    – hard-coded failure messages, no LLM
"""

from __future__ import annotations

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from bot.state import AgentState
from bot import weather_service, sop_engine
from bot.sop_engine import Intent


# ---------------------------------------------------------------------------
# LLM instance (lazy — only parse_intent uses it)
# ---------------------------------------------------------------------------

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("LLM_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Failure reason constants
# ---------------------------------------------------------------------------

FAILURE_NO_LOCATION = "no_location"
FAILURE_GEOCODING = "geocoding_failed"
FAILURE_WEATHER_API = "weather_api_failed"
FAILURE_NO_SOP_MATCH = "no_sop_match"
FAILURE_MALFORMED_SOP = "malformed_sop"


# ---------------------------------------------------------------------------
# Node: parse_intent
# Requirements: 5.1, 6.1, 6.2
# ---------------------------------------------------------------------------

_PARSE_SYSTEM_PROMPT = """You are an intent extraction assistant.
Extract the following fields from the user's message. Return ONLY valid JSON with these keys:
  "location"  : string or null — city or place name the user is asking about
  "activity"  : string or null — what the user wants to do outdoors (e.g. "cycling", "picnic")
  "timeframe" : string or null — when (e.g. "now", "this morning", "tomorrow", "evening")

Rules:
- If the user's message is just a city or place name (e.g. "bangalore", "London"), treat it as the location.
- If a field is not mentioned, return null for that field.
- Do not add any extra keys.
- Do not include markdown fences or any text outside the JSON object.
"""


def parse_intent(state: AgentState) -> AgentState:
    """
    Use the LLM to extract location, activity, and timeframe from the current
    user_input.  Merges with any prior context: only overwrite a field if the
    LLM returns a non-null value for it (carry-forward for follow-ups).
    Sets failure_reason=FAILURE_NO_LOCATION if no location can be determined.
    """
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    llm = _get_llm()

    # Build message list: system prompt + conversation history + current input
    prompt_messages = [SystemMessage(content=_PARSE_SYSTEM_PROMPT)]
    # Include recent history so the LLM can resolve references like "what about evening?"
    for msg in messages[-6:]:  # last 3 turns (human + assistant pairs)
        prompt_messages.append(msg)
    prompt_messages.append(HumanMessage(content=user_input))

    try:
        response = llm.invoke(prompt_messages)
        extracted = json.loads(response.content)
    except Exception:
        extracted = {}

    # Carry forward prior context; update only what changed
    location = extracted.get("location") or state.get("location")
    activity = extracted.get("activity") or state.get("activity")
    timeframe = extracted.get("timeframe") or state.get("timeframe")

    if not location:
        return {
            **state,
            "location": None,
            "activity": activity,
            "timeframe": timeframe,
            "failure_reason": FAILURE_NO_LOCATION,
        }

    return {
        **state,
        "location": location,
        "activity": activity,
        "timeframe": timeframe,
        "failure_reason": None,
    }


# ---------------------------------------------------------------------------
# Node: fetch_weather
# Requirements: 2.1, 2.2, 2.3, 5.1
# ---------------------------------------------------------------------------

def fetch_weather(state: AgentState) -> AgentState:
    """
    Geocode state.location then fetch live weather.
    Populates state.lat, state.lon, state.weather.
    Sets failure_reason on any failure so the graph routes to handle_failure.
    """
    location = state.get("location", "")

    coords = weather_service.geocode(location)
    if coords is None:
        return {
            **state,
            "lat": None,
            "lon": None,
            "weather": None,
            "failure_reason": FAILURE_GEOCODING,
        }

    lat, lon = coords
    weather = weather_service.fetch_weather(lat, lon)
    if weather is None:
        return {
            **state,
            "lat": lat,
            "lon": lon,
            "weather": None,
            "failure_reason": FAILURE_WEATHER_API,
        }

    return {
        **state,
        "lat": lat,
        "lon": lon,
        "weather": weather,
        "failure_reason": None,
    }


# ---------------------------------------------------------------------------
# Node: match_sop
# Requirements: 3.5, 4.2, 5.1
# ---------------------------------------------------------------------------

def match_sop(state: AgentState) -> AgentState:
    """
    Hot-reload SOPs, evaluate each against current weather + intent, apply
    conflict-resolution strategy.
    Sets failure_reason=FAILURE_NO_SOP_MATCH if nothing matches.
    """
    weather = state.get("weather")
    location = state.get("location", "")
    activity = state.get("activity")
    timeframe = state.get("timeframe")

    intent = Intent(location=location or "", activity=activity, timeframe=timeframe)

    try:
        sops = sop_engine.load_sops()
    except Exception:
        return {
            **state,
            "matched_sops": [],
            "primary_sop": None,
            "secondary_sops": [],
            "failure_reason": FAILURE_MALFORMED_SOP,
        }

    matched = sop_engine.match_sops(sops, weather, intent)
    primary, secondaries = sop_engine.resolve(matched, intent)

    if primary is None:
        return {
            **state,
            "matched_sops": matched,
            "primary_sop": None,
            "secondary_sops": [],
            "failure_reason": FAILURE_NO_SOP_MATCH,
        }

    return {
        **state,
        "matched_sops": matched,
        "primary_sop": primary,
        "secondary_sops": secondaries,
        "failure_reason": None,
    }


# ---------------------------------------------------------------------------
# Node: generate_response
# Requirements: 1.1, 2.5, 3.6, 10.1
# ---------------------------------------------------------------------------

def generate_response(state: AgentState) -> AgentState:
    """
    Compose the final response from the primary SOP's template, substituting
    only WeatherData values (no LLM, no hallucinated numbers).
    Appends secondary SOP notes when present.
    SOP id is always included in the output (via response_template convention).
    """
    primary: sop_engine.SOP = state["primary_sop"]
    weather: weather_service.WeatherData = state["weather"]
    secondaries: list[sop_engine.SOP] = state.get("secondary_sops", [])

    location = state.get("location", "")
    activity = state.get("activity")
    timeframe = state.get("timeframe")
    intent = Intent(location=location or "", activity=activity, timeframe=timeframe)

    response = sop_engine.render_response(primary, weather, intent)

    if secondaries:
        notes = "\n\nAdditional advisories:"
        for sop in secondaries:
            notes += f"\n- {sop.title} ({sop.id}): {sop.description.strip()}"
        response += notes

    return {**state, "response": response, "failure_reason": None}


# ---------------------------------------------------------------------------
# Node: handle_failure
# Requirements: 2.2, 2.3, 8.1, 8.2, 8.4
# ---------------------------------------------------------------------------

# Hard-coded failure messages — LLM is never invoked here (Requirement 8.1–8.4)
_FAILURE_MESSAGES: dict[str, str] = {
    FAILURE_NO_LOCATION: (
        "I need a location to check the weather. "
        "Which city or place are you asking about?"
    ),
    FAILURE_GEOCODING: (
        "I couldn't find weather data for that location. "
        "Please check the city name and try again."
    ),
    FAILURE_WEATHER_API: (
        "The weather service is currently unavailable. "
        "Please try again shortly."
    ),
    FAILURE_NO_SOP_MATCH: (
        "I don't have any policy guidance covering that scenario. "
        "Please consult a local authority or official source."
    ),
    FAILURE_MALFORMED_SOP: (
        "There's a configuration issue with the policy file. "
        "Please contact support."
    ),
}

_DEFAULT_FAILURE_MESSAGE = (
    "Something went wrong and I'm unable to provide an advisory right now. "
    "Please try again."
)


def handle_failure(state: AgentState) -> AgentState:
    """
    Produce an honest, hard-coded failure response based on failure_reason.
    Never calls the LLM.
    """
    reason = state.get("failure_reason") or ""
    message = _FAILURE_MESSAGES.get(reason, _DEFAULT_FAILURE_MESSAGE)
    return {**state, "response": message}
