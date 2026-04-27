# WISMO AI Assistant

> *Where Is My Order?* — A production-grade RAG-based AI system for DHL logistics tracking.

---

## What This System Does

DHL receives approximately 50,000 "Where Is My Order?" calls per day. This assistant answers shipment tracking questions automatically and accurately — using a hybrid retrieval system that pulls structured data from a logistics database, grounds every response in real tracking records, and applies output guardrails to prevent hallucinated delivery dates or locations. The result is a conversational AI that reduces inbound support volume while remaining factual, GDPR-aware, and safe to deploy.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        USER                             │
│              (Gradio UI / API call)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               LANGGRAPH ReAct AGENT                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Tools:                                          │   │
│  │  • query_tracking_db(tracking_id)                │   │
│  │  • get_delivery_estimate(tracking_id)            │   │
│  │  • find_similar_cases(issue_description)         │   │
│  └──────────────────────────────────────────────────┘   │
│  ConversationBufferMemory  |  not_found fallback node   │
└──────────────┬───────────────────────┬──────────────────┘
               │                       │
               ▼                       ▼
┌──────────────────────┐  ┌────────────────────────────┐
│   SQLite (structured)│  │   ChromaDB (semantic)      │
│   200 shipment rows  │  │   200 status descriptions  │
│   Parameterised SQL  │  │   all-MiniLM-L6-v2 embeds  │
└──────────────────────┘  └────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│                    GUARDRAILS                           │
│  OutputValidator: date/location hallucination checks    │
│  PIIFilter: redact names/emails/phones before logging   │
└──────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│                     RESPONSE                            │
│         Grounded, validated, PII-safe answer            │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Ollama (llama3.2, configurable) |
| Agent orchestration | LangGraph (ReAct) + LangChain LCEL |
| Structured retrieval | SQLite + parameterised queries |
| Semantic retrieval | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) |
| Guardrails | Custom `OutputValidator` + `PIIFilter` (regex + spaCy NER) |
| UI | Gradio 5 |
| Testing | pytest + unittest.mock |
| Config | python-dotenv |
| Data generation | Faker |

---

## Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running

```bash
# 1. Clone the repo
git clone <repo-url>
cd wismo-ai-assistant

# 2. Create and activate virtualenv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy NER model
python -m spacy download en_core_web_sm

# 5. Configure environment
cp .env.example .env
# Edit .env — set MODEL_NAME to a model you have in Ollama

# 6. Start Ollama and pull a model
ollama serve &
ollama pull llama3.2          # or llama3.2:3b for faster CPU inference

# 7. Generate synthetic tracking data
python data/simulate_tracking_db.py

# 8. Run tests
pytest tests/ -v

# 9. Launch the UI
python app/gradio_ui.py
# → http://localhost:7860
```

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | LLM backend (currently only `ollama`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `MODEL_NAME` | `llama3.2` | Model name in Ollama |
| `DB_PATH` | `data/tracking.db` | SQLite database path |
| `CHROMA_PATH` | `data/chroma_db` | ChromaDB persistence path |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## Example Interaction

**Tab 1 — Track My Parcel:**
```
Input:  JD960013389083863794

AI Summary:
  Your package has been successfully delivered in Copenhagen.
  It was delivered on 2026-04-21, matching the estimated delivery date.

Raw Data (expandable):
  Tracking Id: JD960013389083863794
  Status Code: DELIVERED
  Current Location: Copenhagen
  Destination City: Copenhagen
  Estimated Delivery: 2026-04-21
  Actual Delivery: 2026-04-21
  ...

🛡️ Guardrail Status: ✅ PASSED
```

**Tab 2 — Ask WISMO (multi-turn):**
```
User:      Where is my package JD123456789012345678?
Assistant: Your package is currently in transit at the Munich hub and is
           expected to arrive in Berlin by 2026-04-29.

User:      What if it doesn't arrive by then?
Assistant: If your package doesn't arrive by the estimated date, I recommend
           contacting DHL customer support. The current status shows IN_TRANSIT
           with no delays flagged.
```

---

## Architectural Decisions

### Why hybrid retrieval (SQL + vector)?

SQL gives exact, deterministic lookups when the user provides a tracking ID — zero ambiguity, millisecond latency. ChromaDB handles the fuzzy case: "my package seems stuck at customs" maps semantically to similar historical cases without requiring an exact ID. The two layers complement each other; neither alone is sufficient.

### Why guardrails on output, not input?

Input guardrails (blocking certain queries) create a cat-and-mouse game with adversarial users and risk blocking legitimate requests. Output guardrails are orthogonal to input phrasing: they validate the *claim* the LLM makes against *what the database actually says*. A hallucinated date is wrong regardless of how the user phrased their question.

### Why LangGraph over a simple chain?

A simple LCEL chain is stateless and single-step. LangGraph enables:
- **Tool-calling loops** — the agent can call `get_delivery_estimate` *after* `query_tracking_db` within a single turn.
- **Conditional routing** — if no data is found, the graph routes to a scripted `not_found` node rather than letting the LLM improvise.
- **Memory across turns** — conversation history is threaded through the state graph, enabling coherent follow-up questions.
- **Observability** — each node is a discrete, testable unit.

### Why Ollama?

Zero API cost during development, full data locality (no customer data leaves the machine), and easy model swapping. The `_build_llm()` factory in both `wismo_chain.py` and `wismo_agent.py` is the single extension point to add other providers.

---

## Evaluation Results

Run `python evaluation/run_evals.py` after setup to populate this section.

| Metric | Value |
|--------|-------|
| Answer Faithfulness | _%  |
| Answer Relevance | _% |
| Not-Found Handling | _% |
| Avg Latency | _ ms |
| p95 Latency | _ ms |
| Guardrail Trigger Rate | _% |

---

## Project Structure

```
wismo-ai-assistant/
├── data/
│   └── simulate_tracking_db.py   # Generates SQLite + ChromaDB from Faker
├── retrieval/
│   ├── structured_retriever.py   # SQL lookup by tracking_id
│   └── semantic_retriever.py     # ChromaDB semantic search
├── agent/
│   ├── wismo_chain.py            # LCEL chain (single-turn)
│   └── wismo_agent.py            # LangGraph ReAct agent (multi-turn)
├── guardrails/
│   ├── output_validator.py       # Hallucination detection
│   └── pii_filter.py             # PII redaction for logs/display
├── evaluation/
│   ├── eval_dataset.json         # 50 synthetic Q&A test cases
│   └── run_evals.py              # Eval runner + metrics
├── tests/
│   ├── test_retrieval.py
│   ├── test_guardrails.py
│   └── test_agent.py
├── app/
│   └── gradio_ui.py              # Two-tab Gradio demo (port 7860)
├── notebooks/
│   └── red_teaming.ipynb         # 10 adversarial prompt tests
├── .env.example
├── requirements.txt
├── PLAYBOOK.md                   # Operational guide for engineers
└── README.md
```
