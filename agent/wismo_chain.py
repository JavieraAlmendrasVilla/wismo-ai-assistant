"""
wismo_chain.py
--------------
LangChain LCEL chain for single-turn WISMO lookups.
Flow: tracking_id → structured_retriever → prompt → LLM → output parser
"""

import os
import logging
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel

from retrieval.structured_retriever import get_tracking_by_id

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class WISMOResponse(BaseModel):
    """Structured output from the WISMO chain."""
    answer: str
    tracking_data: dict
    confidence: str  # "high" | "low" | "not_found"


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm():
    """Instantiate the configured LLM. Defaults to Ollama."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("MODEL_NAME", "llama3.2")

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model, base_url=base_url, temperature=0)

    raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'. Set LLM_PROVIDER=ollama in .env")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the DHL WISMO Assistant — a helpful, factual logistics support agent.

STRICT RULES:
1. Only answer based on the tracking data provided below.
2. NEVER invent, estimate, or guess delivery dates, locations, or statuses.
3. If the data does not contain the answer, say exactly: "I don't have that information in the current tracking data."
4. Do not mention internal systems, APIs, or databases.
5. Keep responses concise (2–4 sentences).

Tracking Data:
{tracking_data}
"""

USER_PROMPT = "Customer query: {query}"

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
])


# ---------------------------------------------------------------------------
# Chain components
# ---------------------------------------------------------------------------

def _retrieve(inputs: dict) -> dict:
    """Fetch tracking data and attach it to the chain inputs."""
    tracking_id: str = inputs.get("tracking_id", "").strip()
    data = get_tracking_by_id(tracking_id) if tracking_id else None

    if data:
        # Format data as readable text for the prompt
        tracking_text = "\n".join(
            f"  {k.replace('_', ' ').title()}: {v}"
            for k, v in data.items()
            if v is not None
        )
        confidence = "high"
    else:
        tracking_text = "No tracking record found for the provided ID."
        confidence = "not_found"
        data = {}

    return {
        "tracking_id": tracking_id,
        "query": inputs.get("query", f"What is the status of tracking ID {tracking_id}?"),
        "tracking_data": tracking_text,
        "_raw_data": data,
        "_confidence": confidence,
    }


def _format_response(inputs: dict, llm_answer: str) -> WISMOResponse:
    """Wrap the LLM output in a WISMOResponse."""
    return WISMOResponse(
        answer=llm_answer.strip(),
        tracking_data=inputs.get("_raw_data", {}),
        confidence=inputs.get("_confidence", "low"),
    )


# ---------------------------------------------------------------------------
# Public chain builder
# ---------------------------------------------------------------------------

def build_wismo_chain():
    """
    Build and return the WISMO LCEL chain.

    Usage:
        chain = build_wismo_chain()
        result: WISMOResponse = chain.invoke({"tracking_id": "JD...", "query": "..."})
    """
    llm = _build_llm()
    str_parser = StrOutputParser()

    def run_chain(inputs: dict) -> WISMOResponse:
        enriched = _retrieve(inputs)
        prompt_value = PROMPT.format_messages(
            tracking_data=enriched["tracking_data"],
            query=enriched["query"],
        )
        raw_answer = str_parser.invoke(llm.invoke(prompt_value))
        return _format_response(enriched, raw_answer)

    return RunnableLambda(run_chain)


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_chain = None


def get_chain():
    """Return a cached WISMO chain instance."""
    global _chain
    if _chain is None:
        _chain = build_wismo_chain()
    return _chain


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from retrieval.structured_retriever import list_all_tracking_ids

    logging.basicConfig(level="INFO")
    ids = list_all_tracking_ids()
    tid = ids[0] if ids else "UNKNOWN"
    print(f"Testing chain with tracking_id={tid}")
    chain = build_wismo_chain()
    result = chain.invoke({"tracking_id": tid, "query": "Where is my package?"})
    print("\nAnswer:", result.answer)
    print("Confidence:", result.confidence)
    print("Data:", result.tracking_data)
