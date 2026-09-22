# Weather Advisory Bot

A LangGraph-powered conversational agent that answers outdoor activity safety questions using live [Open-Meteo](https://open-meteo.com/) weather data and a human-authored Standard Operating Procedures (SOP) policy file. Every response is grounded in a real weather reading and traceable to a specific SOP id, or the bot explicitly states that no policy covers the question.

---

## Why sops.yaml with Hot-Reloading

Policy lives in `sops.yaml`, completely decoupled from the control code. The SOP engine calls `load_sops()` on every request, so operations, clinical, or safety teams can add or modify rules — including adding a live 11th SOP during a review — without triggering a redeployment or touching Python. The YAML schema is intentionally narrow (10 supported condition keys, documented in the file header) so non-engineers can author rules safely.

## State and Branching Architecture

The graph has five nodes with explicit failure routing at every step:

```
parse_intent
    ├── [no location extracted] → handle_failure → END
    └── [ok]                   → fetch_weather
                                     ├── [geocoding/API failure] → handle_failure → END
                                     └── [ok]                    → match_sop
                                                                       ├── [no match / bad YAML] → handle_failure → END
                                                                       └── [ok]                  → generate_response → END
```

`handle_failure` never calls the LLM — every failure message is a hard-coded string. This keeps failure responses fast, deterministic, and honest when the weather API is unreachable or a city name can't be resolved.

Session memory is provided by a LangGraph `MemorySaver` checkpointer keyed on `thread_id`. Each browser tab gets a fresh UUID, so follow-up questions like "what about this evening?" resolve correctly without cross-session leakage.

## Data Integrity and Anti-Hallucination

Numeric weather values (temperature, wind speed, precipitation, UV index, weather code) are fetched from Open-Meteo and stored as typed fields on `WeatherData`. The `generate_response` node fills the primary SOP's `response_template` using only those fields — there is no LLM call in this node. The model never has a chance to invent a temperature or fabricate a probability. If Open-Meteo is unreachable, the graph routes to `handle_failure` and tells the user honestly rather than generating a plausible-sounding but ungrounded advisory.

## Latency vs. Reliability Trade-off

SOP conditions are evaluated deterministically in Python (`sop_engine.py`) rather than asking an LLM to interpret the raw YAML on every turn. This eliminates schema drift (the LLM cannot misread a YAML key), saves tokens, and makes the matching fully unit-testable. The only LLM call in the happy path is `parse_intent`, which extracts location, activity, and timeframe and returns plain JSON — a task that is fast and cheap with `gpt-4o-mini`.

---

## Prerequisites

- Python 3.11+
- An OpenAI-compatible API key (used only for `parse_intent`)

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# edit .env and set LLM_API_KEY=sk-...
```

Weather data comes from the public Open-Meteo API — no key required.

## Running the Bot

```bash
streamlit run app.py
```

Example queries:

- "Is it safe to go cycling in London today?"
- "Should elderly people go outside in Dubai right now?"
- "What about this evening?" *(follow-up — the bot remembers the session)*

## Running the Tests

```bash
# Unit + property-based tests
pytest tests/

# Evaluation suite (8 graded scenarios)
python eval/eval_suite.py
```

Property-based tests use `hypothesis` and exercise the SOP engine with generated extreme values (temperatures up to 80 °C, wind speeds up to 200 km/h, UV up to 20). A full `pytest tests/` run completes in a few seconds.

## Deployment (Streamlit Community Cloud)

1. Push this repository to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app" → select `app.py`.
3. Under "Advanced settings → Secrets", add:
   ```
   LLM_API_KEY = "sk-..."
   ```
4. Deploy. The live URL is available within ~2 minutes.

> The app reads `LLM_API_KEY` via `os.getenv`. Streamlit Cloud injects secrets as environment variables, so no code changes are needed.

## Extending the SOP Policy

Add a new SOP to `sops.yaml` — no Python changes, no restart needed:

```yaml
- id: "SOP-011"
  title: "My New SOP"
  category: "outdoor_exercise"
  severity: "MODERATE"
  description: >
    Brief description of when this fires.
  conditions:
    uv_index_gte: 7
  activities: ["hiking", "walking"]
  response_template: >
    UV index is {uv_index}. Advise caution for {activity}. (SOP-011)
```

Supported condition keys: `uv_index_gte`, `uv_index_lt`, `precip_prob_gte`, `precip_prob_lt`, `wind_speed_gt`, `wind_speed_lt`, `wind_speed_gte`, `temperature_gte`, `temperature_lt`, `temperature_range`, `weather_code_in`, `precipitation_gte`, `fuzzy_composite`.

Conflict resolution when multiple SOPs match: highest severity wins → activity specificity → lower SOP id.

## Project Structure

```
app.py                     Streamlit frontend
bot/
  graph.py                 LangGraph StateGraph (5 nodes, failure routing)
  nodes.py                 Node implementations
  state.py                 AgentState TypedDict
  sop_engine.py            SOP loader, matcher, resolver, renderer
  weather_service.py       Open-Meteo geocoding and forecast
sops.yaml                  Human-authored SOP policy (10 SOPs, hot-reloaded)
eval/
  eval_suite.py            8 graded evaluation scenarios
tests/
  test_sop_engine.py       Unit + property-based tests for SOP engine
  test_weather_service.py  Unit tests for weather service
requirements.txt
.env.example
```

## Trade-offs and Future Work

**Trade-offs made:**

- In-memory `MemorySaver` means session history is lost on server restart. A persistent checkpointer (Redis, SQLite) would fix this but adds ops overhead.
- `parse_intent` is the only LLM call; it does not validate location against a known city list, so typos route to the geocoding failure path rather than a clarifying prompt.
- SOP matching is single-turn per request — the engine does not accumulate weather readings over time for trend analysis.

**Future work:**

- Add a vector store for semantic SOP retrieval when the rule count exceeds ~50.
- Support forecast-based queries ("tomorrow morning") by querying the hourly Open-Meteo endpoint rather than current conditions.
- Persistent session storage so users can resume conversations across tabs or days.
- SOP authoring UI so non-engineers can add rules without editing raw YAML.
