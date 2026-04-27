"""
semantic_retriever.py
---------------------
ChromaDB semantic search over shipment status_descriptions using
sentence-transformers (all-MiniLM-L6-v2).
"""

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", ROOT / "data" / "chroma_db"))
COLLECTION_NAME = "shipment_descriptions"

# Module-level cache so we only load the model once per process
_collection = None


def _get_collection():
    """Lazy-load and cache the ChromaDB collection."""
    global _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as exc:
        raise RuntimeError(
            "chromadb and sentence-transformers are required. "
            "Install via: pip install chromadb sentence-transformers"
        ) from exc

    if not CHROMA_PATH.exists():
        raise RuntimeError(
            f"ChromaDB not found at {CHROMA_PATH}. "
            "Run `python data/simulate_tracking_db.py` first."
        )

    embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    _collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )
    logger.debug("ChromaDB collection loaded: %s", COLLECTION_NAME)
    return _collection


def semantic_search(query: str, top_k: int = 3) -> list[dict]:
    """
    Search shipment status descriptions semantically.

    Args:
        query: Natural language query (e.g. "package stuck at customs").
        top_k: Number of top results to return.

    Returns:
        List of dicts, each containing metadata fields plus
        'status_description' (the matched document) and 'distance' score.
        Returns an empty list if the query is empty or the index is unavailable.
    """
    if not query or not query.strip():
        logger.warning("semantic_search called with empty query")
        return []

    try:
        collection = _get_collection()
    except RuntimeError as exc:
        logger.error("Cannot access ChromaDB: %s", exc)
        return []

    results = collection.query(
        query_texts=[query.strip()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output: list[dict] = []
    if not results["ids"] or not results["ids"][0]:
        return output

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append(
            {
                **meta,
                "status_description": doc,
                "distance": round(dist, 4),
            }
        )

    logger.debug("semantic_search('%s') returned %d results", query, len(output))
    return output


if __name__ == "__main__":
    import sys
    logging.basicConfig(level="DEBUG")
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "package delayed at hub"
    hits = semantic_search(q)
    for h in hits:
        print(h)
