"""
test_agent.py
-------------
Unit tests for wismo_chain and wismo_agent.
LLM calls are mocked so no real Ollama/API is required.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage

from agent.wismo_chain import WISMOResponse, build_wismo_chain, PROMPT
from agent.wismo_agent import query_tracking_db, get_delivery_estimate, find_similar_cases


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tracking_data() -> dict:
    return {
        "tracking_id": "JD012345678901234567",
        "sender_name": "Hans Mueller",
        "recipient_name": "Jane Doe",
        "origin_city": "Hamburg",
        "destination_city": "Berlin",
        "current_location": "Munich",
        "status_code": "IN_TRANSIT",
        "estimated_delivery": "2024-12-20",
        "actual_delivery": None,
        "last_update": "2024-12-18T14:30:00",
        "status_description": "Package arrived at Munich hub.",
    }


@pytest.fixture
def delivered_tracking_data() -> dict:
    return {
        "tracking_id": "JD999000111222333444",
        "sender_name": "Sender Name",
        "recipient_name": "Recipient Name",
        "origin_city": "Frankfurt",
        "destination_city": "Cologne",
        "current_location": "Cologne",
        "status_code": "DELIVERED",
        "estimated_delivery": "2024-12-15",
        "actual_delivery": "2024-12-15",
        "last_update": "2024-12-15T11:00:00",
        "status_description": "Package successfully delivered.",
    }


# ---------------------------------------------------------------------------
# WISMO Chain tests
# ---------------------------------------------------------------------------

class TestWISMOChain:
    """Tests for agent.wismo_chain."""

    def test_chain_returns_structured_response(self, sample_tracking_data):
        """Chain returns a WISMOResponse with the correct shape."""
        mock_llm_response = AIMessage(content="Your package is in Munich, heading to Berlin.")

        with patch("agent.wismo_chain.get_tracking_by_id", return_value=sample_tracking_data), \
             patch("agent.wismo_chain._build_llm") as mock_build_llm:

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_llm_response
            mock_build_llm.return_value = mock_llm

            chain = build_wismo_chain()
            result = chain.invoke({
                "tracking_id": "JD012345678901234567",
                "query": "Where is my package?",
            })

        assert isinstance(result, WISMOResponse)
        assert result.confidence == "high"
        assert isinstance(result.tracking_data, dict)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0

    def test_chain_handles_missing_tracking_id(self):
        """Chain returns not_found confidence when tracking ID doesn't exist."""
        mock_llm_response = AIMessage(content="I don't have information for that tracking ID.")

        with patch("agent.wismo_chain.get_tracking_by_id", return_value=None), \
             patch("agent.wismo_chain._build_llm") as mock_build_llm:

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_llm_response
            mock_build_llm.return_value = mock_llm

            chain = build_wismo_chain()
            result = chain.invoke({
                "tracking_id": "NONEXISTENT123",
                "query": "Where is my order?",
            })

        assert isinstance(result, WISMOResponse)
        assert result.confidence == "not_found"
        assert result.tracking_data == {}

    def test_chain_prompt_contains_grounding_instruction(self):
        """System prompt must contain the hallucination-prevention instruction."""
        # Access the prompt template messages
        messages = PROMPT.messages
        system_message = next(
            (m for m in messages if hasattr(m, "prompt") and "STRICT RULES" in m.prompt.template),
            None,
        )
        assert system_message is not None, "System prompt must contain 'STRICT RULES'"

    def test_chain_uses_query_fallback_when_not_provided(self, sample_tracking_data):
        """Chain generates a default query if none is passed."""
        mock_llm_response = AIMessage(content="Your package is in transit.")

        with patch("agent.wismo_chain.get_tracking_by_id", return_value=sample_tracking_data), \
             patch("agent.wismo_chain._build_llm") as mock_build_llm:

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_llm_response
            mock_build_llm.return_value = mock_llm

            chain = build_wismo_chain()
            # No "query" key — chain should fall back gracefully
            result = chain.invoke({"tracking_id": "JD012345678901234567"})

        assert isinstance(result, WISMOResponse)


# ---------------------------------------------------------------------------
# Agent Tools tests (no LLM needed — tools call real retrievers)
# ---------------------------------------------------------------------------

class TestAgentTools:
    """Tests for the LangGraph agent's tool functions."""

    def test_tool_query_tracking_db_found(self, sample_tracking_data):
        """query_tracking_db returns a formatted string when data exists."""
        with patch("agent.wismo_agent.get_tracking_by_id", return_value=sample_tracking_data):
            result = query_tracking_db.invoke({"tracking_id": "JD012345678901234567"})

        assert isinstance(result, str)
        assert "IN_TRANSIT" in result
        assert "Munich" in result

    def test_tool_query_tracking_db_not_found(self):
        """query_tracking_db returns a helpful not-found message."""
        with patch("agent.wismo_agent.get_tracking_by_id", return_value=None):
            result = query_tracking_db.invoke({"tracking_id": "FAKE999"})

        assert isinstance(result, str)
        assert "not found" in result.lower() or "no shipment" in result.lower()

    def test_tool_get_delivery_estimate_delivered(self, delivered_tracking_data):
        """get_delivery_estimate returns actual delivery date for DELIVERED packages."""
        with patch("agent.wismo_agent.get_tracking_by_id", return_value=delivered_tracking_data):
            result = get_delivery_estimate.invoke({"tracking_id": "JD999000111222333444"})

        assert "delivered" in result.lower()
        assert "2024-12-15" in result

    def test_tool_get_delivery_estimate_in_transit(self, sample_tracking_data):
        """get_delivery_estimate returns estimated date for in-transit packages."""
        with patch("agent.wismo_agent.get_tracking_by_id", return_value=sample_tracking_data):
            result = get_delivery_estimate.invoke({"tracking_id": "JD012345678901234567"})

        assert "2024-12-20" in result or "Estimated" in result

    def test_tool_find_similar_cases_returns_string(self):
        """find_similar_cases returns a string (even if empty results)."""
        mock_results = [
            {
                "tracking_id": "JD111",
                "status_code": "DELAYED",
                "current_location": "Frankfurt",
                "status_description": "Package delayed at Frankfurt hub.",
                "distance": 0.15,
            }
        ]
        with patch("agent.wismo_agent.semantic_search", return_value=mock_results):
            result = find_similar_cases.invoke({"issue_description": "package delayed"})

        assert isinstance(result, str)
        assert "DELAYED" in result

    def test_tool_find_similar_cases_no_results(self):
        """find_similar_cases handles empty semantic search gracefully."""
        with patch("agent.wismo_agent.semantic_search", return_value=[]):
            result = find_similar_cases.invoke({"issue_description": "something obscure"})

        assert isinstance(result, str)
        assert "no similar" in result.lower()
