"""
LangGraph StateGraph assembly for the Weather Advisory Bot.

Graph topology (Requirements 5.1, 5.2, 5.3):

    parse_intent
        ├── [failure_reason set] → handle_failure → END
        └── [ok]                → fetch_weather
                                      ├── [failure_reason set] → handle_failure → END
                                      └── [ok]                → match_sop
                                                                    ├── [failure_reason set] → handle_failure → END
                                                                    └── [ok]                → generate_response → END

Session memory (Requirements 6.1, 6.3):
    The compiled graph uses an in-memory MemorySaver checkpointer so that
    conversation history is retained across turns within the same session.
    Each session is identified by a `thread_id` passed via the LangGraph
    config dict: {"configurable": {"thread_id": "<session-id>"}}.
    Different thread_ids get fully isolated history; a new session simply
    uses a new thread_id.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from bot.state import AgentState
from bot.nodes import (
    parse_intent,
    fetch_weather,
    match_sop,
    generate_response,
    handle_failure,
)


# ---------------------------------------------------------------------------
# Conditional edge routing helpers
# ---------------------------------------------------------------------------

def _route_after_parse(state: AgentState) -> str:
    """Route to handle_failure if location could not be extracted."""
    return "handle_failure" if state.get("failure_reason") else "fetch_weather"


def _route_after_fetch(state: AgentState) -> str:
    """Route to handle_failure if geocoding or weather API failed."""
    return "handle_failure" if state.get("failure_reason") else "match_sop"


def _route_after_match(state: AgentState) -> str:
    """Route to handle_failure if no SOP matched or SOP file is malformed."""
    return "handle_failure" if state.get("failure_reason") else "generate_response"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """
    Construct and compile the LangGraph StateGraph.

    Parameters
    ----------
    checkpointer:
        A LangGraph checkpointer instance.  Defaults to a fresh MemorySaver
        so that session history is retained across turns within a thread.
        Pass ``None`` explicitly only in tests that do not need persistence.

    Returns a compiled graph ready for invocation.  To use session memory,
    pass a config dict with a thread_id:

        graph.invoke(state, config={"configurable": {"thread_id": "abc123"}})

    Each unique thread_id gets its own isolated conversation history
    (Requirements 6.1, 6.3).  Starting a new session means using a new
    thread_id — prior history is never carried across sessions (Requirement 6.3).
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("parse_intent", parse_intent)
    builder.add_node("fetch_weather", fetch_weather)
    builder.add_node("match_sop", match_sop)
    builder.add_node("generate_response", generate_response)
    builder.add_node("handle_failure", handle_failure)

    # Entry point
    builder.set_entry_point("parse_intent")

    # Conditional edges (Requirements 5.2, 5.3)
    builder.add_conditional_edges(
        "parse_intent",
        _route_after_parse,
        {"fetch_weather": "fetch_weather", "handle_failure": "handle_failure"},
    )
    builder.add_conditional_edges(
        "fetch_weather",
        _route_after_fetch,
        {"match_sop": "match_sop", "handle_failure": "handle_failure"},
    )
    builder.add_conditional_edges(
        "match_sop",
        _route_after_match,
        {"generate_response": "generate_response", "handle_failure": "handle_failure"},
    )

    # Terminal edges
    builder.add_edge("generate_response", END)
    builder.add_edge("handle_failure", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Module-level compiled graph (singleton for app use)
# Uses MemorySaver so all sessions share one in-process store, isolated by
# thread_id.  Requirements 6.1, 6.3.
# ---------------------------------------------------------------------------

graph = build_graph()
