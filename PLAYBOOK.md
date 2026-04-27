# WISMO AI Assistant — Engineering Playbook

Operational guide for engineers maintaining or extending the system.

---

## 1. How to Add a New Data Source

**Use case:** You want to add a second database (e.g. DHL Express international feed) or a CSV of historical delays.

### Steps

1. **Create the ingestor** in `data/`:
   ```python
   # data/ingest_express_db.py
   def ingest(source_path: str, db_path: str) -> None:
       ...
   ```

2. **Map your schema** to the canonical `shipments` table columns. If new columns are needed, add an `ALTER TABLE` migration in `data/simulate_tracking_db.py` and update `CREATE_TABLE_SQL`.

3. **Update `retrieval/structured_retriever.py`** if the new source requires a JOIN or a separate query. Add a new function (e.g. `get_express_tracking_by_id`) rather than modifying the existing one.

4. **Re-index ChromaDB** by running:
   ```bash
   python data/simulate_tracking_db.py   # or your new ingestor
   ```
   The `create_chroma_index()` function drops and rebuilds the collection each run.

5. **Add eval cases** to `evaluation/eval_dataset.json` covering the new source's status codes.

---

## 2. How to Update the Prompt Template

The chain prompt lives in `agent/wismo_chain.py` as `SYSTEM_PROMPT` and `USER_PROMPT`.

### Steps

1. **Edit the constants** directly:
   ```python
   SYSTEM_PROMPT = """
   You are the DHL WISMO Assistant...
   [your changes here]
   """
   ```

2. **Key constraints to preserve:**
   - Keep rule 2: "NEVER invent, estimate, or guess delivery dates, locations, or statuses."
   - Keep the `{tracking_data}` placeholder — the chain injects the DB record here.
   - Keep `{query}` in `USER_PROMPT`.

3. **Test with** `python agent/wismo_chain.py` (uses first DB record).

4. **Run the eval suite** to confirm relevance doesn't drop:
   ```bash
   python evaluation/run_evals.py
   ```

The agent system prompt lives in `agent/wismo_agent.py` as `AGENT_SYSTEM`. Same process applies.

---

## 3. How to Add a New Guardrail Check

The validator is in `guardrails/output_validator.py` — `OutputValidator.validate()`.

### Steps

1. **Write a helper function** that checks for the new condition:
   ```python
   def _check_status_code_match(response: str, data: dict) -> list[str]:
       violations = []
       status = data.get("status_code", "")
       # e.g. response says "delivered" but status is IN_TRANSIT
       if "delivered" in response.lower() and status != "DELIVERED":
           violations.append("Response claims delivery but status is not DELIVERED.")
       return violations
   ```

2. **Call it inside `validate()`** and extend the violations list:
   ```python
   violations.extend(_check_status_code_match(response, retrieved_data))
   ```

3. **Add a test** in `tests/test_guardrails.py`:
   ```python
   def test_status_mismatch_is_caught(validator, sample_tracking_data):
       response = "Your package has been delivered."
       # sample_tracking_data has status IN_TRANSIT
       result = validator.validate(response, sample_tracking_data)
       assert result.is_valid is False
   ```

4. Run `pytest tests/test_guardrails.py` to confirm.

---

## 4. How to Extend the Eval Dataset

The dataset lives in `evaluation/eval_dataset.json` as a JSON array of 50 objects.

### Adding cases

1. Increment the `id` counter from 51.
2. Follow the schema:
   ```json
   {
     "id": 51,
     "query": "Your test question",
     "tracking_id": "JD012345678901234567",
     "expected_status": "IN_TRANSIT",
     "expected_location": "Munich",
     "expected_answer_contains": ["Munich", "in transit"]
   }
   ```
3. Use `null` for `tracking_id` and `"NOT_FOUND"` for `expected_status` when testing invalid IDs.
4. Re-run `python evaluation/run_evals.py` and check that `answer_relevance_pct` doesn't regress.

### Valid `expected_status` values
`IN_TRANSIT` | `OUT_FOR_DELIVERY` | `DELIVERED` | `DELAYED` | `EXCEPTION` | `NOT_FOUND` | `AMBIGUOUS`

---

## 5. Common Failure Modes and Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `RuntimeError: Database not found` | DB not generated | `python data/simulate_tracking_db.py` |
| `RuntimeError: ChromaDB not found` | Vector index not built | Same as above — both are built together |
| `ollama: connection refused` | Ollama not running | `ollama serve` in a separate terminal |
| `model not found` error from Ollama | Model not pulled | `ollama pull llama3.2` |
| All eval cases return `confidence: not_found` | Wrong `DB_PATH` in `.env` | Set `DB_PATH=data/tracking.db` (relative to project root) |
| Guardrail triggers on every response | Response mentions a city not in the DB record | Normal — the LLM may have added context. The guardrail is working. |
| spaCy `OSError: [E050]` | Model not downloaded | `python -m spacy download en_core_web_sm` |
| `ImportError: No module named 'langchain_ollama'` | Missing dep | `pip install langchain-ollama` |
| High latency (>10s per query) | Ollama running on CPU | Use a quantized model: `ollama pull llama3.2:3b` |
| Gradio port 7860 in use | Another process | Change port: `demo.launch(server_port=7861)` |

---

## 6. Environment Setup Quick Reference

```bash
# 1. Clone and enter project
cd wismo-ai-assistant

# 2. Create venv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install deps
pip install -r requirements.txt

# 4. Download spaCy model (for PII filter)
python -m spacy download en_core_web_sm

# 5. Copy and fill .env
cp .env.example .env
# Set MODEL_NAME to whatever you have pulled in Ollama

# 6. Start Ollama
ollama serve &
ollama pull llama3.2

# 7. Generate data
python data/simulate_tracking_db.py

# 8. Run tests
pytest tests/ -v

# 9. Run evals
python evaluation/run_evals.py

# 10. Launch UI
python app/gradio_ui.py
```
