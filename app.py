"""
Streamlit chat frontend for the Weather Advisory Bot.

Run with:
    streamlit run app.py

Requirements: 7.1, 7.2, 7.3
"""

import uuid
import streamlit as st
from bot.graph import graph

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Weather Advisory Bot",
    page_icon="🌤️",
    layout="centered",
)

st.title("🌤️ Weather Advisory Bot")
st.caption("Ask about outdoor activity safety for any location.")

# ---------------------------------------------------------------------------
# Session state initialisation (Requirements 7.1, 7.3)
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    # Each browser session gets its own isolated LangGraph thread
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # List of {"role": "user"|"assistant", "content": str}
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Render conversation history (Requirement 7.2)
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------------------------------------------------------
# Chat input and graph invocation (Requirements 7.1, 7.2)
# ---------------------------------------------------------------------------

user_input = st.chat_input("Ask about weather safety (e.g. 'Is it safe to cycle in London today?')")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Invoke the LangGraph agent with a loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Checking weather and policies…"):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = graph.invoke({"user_input": user_input}, config=config)
            bot_response = result.get("response", "Sorry, something went wrong.")

        st.write(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
