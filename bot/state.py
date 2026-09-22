"""
AgentState: the shared typed state passed between all LangGraph nodes.
Requirements: 5.4
"""

from __future__ import annotations

from typing import Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from bot.weather_service import WeatherData
from bot.sop_engine import SOP


class AgentState(TypedDict, total=False):
    # Conversation history (full thread)
    messages: list[BaseMessage]

    # Current turn's raw user input
    user_input: str

    # Extracted intent fields
    location: Optional[str]
    activity: Optional[str]
    timeframe: Optional[str]

    # Weather layer
    lat: Optional[float]
    lon: Optional[float]
    weather: Optional[WeatherData]

    # SOP layer
    matched_sops: list[SOP]
    primary_sop: Optional[SOP]
    secondary_sops: list[SOP]

    # Output
    response: str
    failure_reason: Optional[str]
