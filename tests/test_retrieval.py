"""
test_retrieval.py
-----------------
Unit tests for the structured and semantic retrieval layers.
"""

import pytest
import sqlite3
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure the project root is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.structured_retriever import get_tracking_by_id, list_all_tracking_ids
from retrieval.semantic_retriever import semantic_search


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_db_available() -> bool:
    """Return True if the tracking DB has been generated."""
    db_path = Path(os.getenv("DB_PATH", "data/tracking.db"))
    return db_path.exists()


@pytest.fixture(scope="module")
def first_tracking_id(real_db_available) -> str | None:
    """Return the first tracking_id from the DB, or None if DB unavailable."""
    if not real_db_available:
        return None
    ids = list_all_tracking_ids()
    return ids[0] if ids else None


# ---------------------------------------------------------------------------
# Structured retriever tests
# ---------------------------------------------------------------------------

class TestStructuredRetriever:
    """Tests for retrieval.structured_retriever.get_tracking_by_id."""

    def test_get_existing_tracking_id(self, real_db_available, first_tracking_id):
        """Happy path: a valid tracking ID returns a populated dict."""
        if not real_db_available or not first_tracking_id:
            pytest.skip("Tracking DB not generated. Run data/simulate_tracking_db.py first.")

        result = get_tracking_by_id(first_tracking_id)

        assert result is not None, "Expected a dict, got None"
        assert isinstance(result, dict)

        required_keys = {
            "tracking_id", "sender_name", "recipient_name", "origin_city",
            "destination_city", "current_location", "status_code",
            "estimated_delivery", "last_update", "status_description",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )
        assert result["tracking_id"] == first_tracking_id

    def test_get_nonexistent_tracking_id(self, real_db_available):
        """Non-existent tracking ID returns None gracefully (no exception)."""
        if not real_db_available:
            pytest.skip("Tracking DB not generated.")

        result = get_tracking_by_id("DOES_NOT_EXIST_999")
        assert result is None

    def test_get_empty_tracking_id_returns_none(self, real_db_available):
        """Empty string tracking ID returns None without crashing."""
        if not real_db_available:
            pytest.skip("Tracking DB not generated.")

        assert get_tracking_by_id("") is None
        assert get_tracking_by_id("   ") is None

    def test_get_tracking_id_strips_whitespace(self, real_db_available, first_tracking_id):
        """Tracking IDs with surrounding whitespace are handled correctly."""
        if not real_db_available or not first_tracking_id:
            pytest.skip("Tracking DB not generated.")

        result = get_tracking_by_id(f"  {first_tracking_id}  ")
        assert result is not None
        assert result["tracking_id"] == first_tracking_id

    def test_db_not_found_raises_runtime_error(self, tmp_path):
        """RuntimeError raised when DB file does not exist."""
        with patch.dict(os.environ, {"DB_PATH": str(tmp_path / "nonexistent.db")}):
            # Re-import to pick up patched env
            import importlib
            import retrieval.structured_retriever as sr
            original = sr.DB_PATH
            sr.DB_PATH = tmp_path / "nonexistent.db"
            try:
                with pytest.raises(RuntimeError, match="Database not found"):
                    sr.get_tracking_by_id("SOMETRACKINGID")
            finally:
                sr.DB_PATH = original


# ---------------------------------------------------------------------------
# Semantic retriever tests
# ---------------------------------------------------------------------------

class TestSemanticRetriever:
    """Tests for retrieval.semantic_retriever.semantic_search."""

    def test_semantic_search_returns_results(self, real_db_available):
        """A valid query returns a non-empty list of dicts."""
        if not real_db_available:
            pytest.skip("Tracking DB / ChromaDB not generated.")

        chroma_path = Path(os.getenv("CHROMA_PATH", "data/chroma_db"))
        if not chroma_path.exists():
            pytest.skip("ChromaDB index not generated.")

        results = semantic_search("package delayed at hub", top_k=3)

        assert isinstance(results, list)
        assert len(results) > 0, "Expected at least one result"
        assert "tracking_id" in results[0]
        assert "status_description" in results[0]
        assert "distance" in results[0]

    def test_semantic_search_empty_query_returns_empty_list(self):
        """Empty query returns empty list without crashing."""
        result = semantic_search("")
        assert result == []

        result = semantic_search("   ")
        assert result == []

    def test_semantic_search_respects_top_k(self, real_db_available):
        """top_k parameter limits number of results returned."""
        if not real_db_available:
            pytest.skip("Tracking DB / ChromaDB not generated.")

        chroma_path = Path(os.getenv("CHROMA_PATH", "data/chroma_db"))
        if not chroma_path.exists():
            pytest.skip("ChromaDB index not generated.")

        results_1 = semantic_search("package delivered", top_k=1)
        results_3 = semantic_search("package delivered", top_k=3)

        assert len(results_1) <= 1
        assert len(results_3) <= 3

    def test_semantic_search_chroma_unavailable_returns_empty(self, tmp_path, monkeypatch):
        """Gracefully returns [] when ChromaDB is not available."""
        import retrieval.semantic_retriever as sr
        monkeypatch.setattr(sr, "_collection", None)
        monkeypatch.setattr(sr, "CHROMA_PATH", tmp_path / "nonexistent_chroma")

        results = semantic_search("some query")
        assert results == []
