"""
structured_retriever.py
-----------------------
SQL-based lookup of DHL shipment records by tracking_id.
"""

import os
import sqlite3
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "tracking.db"))


def get_tracking_by_id(tracking_id: str) -> Optional[dict]:
    """
    Look up a shipment record by its tracking ID.

    Args:
        tracking_id: The DHL tracking ID string (e.g. "JD123456789012345678").

    Returns:
        A dict with all shipment fields, or None if not found.

    Raises:
        RuntimeError: If the database file cannot be opened.
    """
    if not tracking_id or not tracking_id.strip():
        logger.warning("get_tracking_by_id called with empty tracking_id")
        return None

    tracking_id = tracking_id.strip()

    if not DB_PATH.exists():
        raise RuntimeError(
            f"Database not found at {DB_PATH}. "
            "Run `python data/simulate_tracking_db.py` first."
        )

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM shipments WHERE tracking_id = ?",
                (tracking_id,),
            )
            row = cursor.fetchone()

        if row is None:
            logger.info("No record found for tracking_id=%s", tracking_id)
            return None

        result = dict(row)
        logger.debug("Retrieved record for tracking_id=%s", tracking_id)
        return result

    except sqlite3.Error as exc:
        logger.error("SQLite error during lookup of %s: %s", tracking_id, exc)
        raise RuntimeError(f"Database error: {exc}") from exc


def list_all_tracking_ids() -> list[str]:
    """
    Return all tracking IDs in the database (for testing and eval generation).

    Returns:
        List of tracking ID strings.
    """
    if not DB_PATH.exists():
        raise RuntimeError(f"Database not found at {DB_PATH}.")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT tracking_id FROM shipments")
        return [row[0] for row in cursor.fetchall()]


if __name__ == "__main__":
    import sys
    logging.basicConfig(level="DEBUG")
    tid = sys.argv[1] if len(sys.argv) > 1 else None
    if tid:
        print(get_tracking_by_id(tid))
    else:
        ids = list_all_tracking_ids()
        print(f"Total records: {len(ids)}")
        print("First 5:", ids[:5])
