"""
Streamlit chat frontend for the Weather Advisory Bot.

Run with:
    streamlit run app.py
"""

import streamlit as st
from bot.graph import graph
from langchain_core.messages import HumanMessage, AIMessage

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
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------------------------------------------------------
# Chat input and graph invocation
# ---------------------------------------------------------------------------

user_input = st.chat_input("Ask about weather safety (e.g. 'Is it safe to cycle in London today?')")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking weather and policies…"):
            lc_messages = []
            for m in st.session_state.messages[:-1]:
                if m["role"] == "user":
                    lc_messages.append(HumanMessage(content=m["content"]))
                else:
                    lc_messages.append(AIMessage(content=m["content"]))

            result = graph.invoke(
                {"user_input": user_input, "messages": lc_messages},
            )
            bot_response = result.get("response", "Sorry, something went wrong.")

        st.write(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
