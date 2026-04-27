"""
simulate_tracking_db.py
-----------------------
Generates 200 realistic DHL shipment records into SQLite and indexes
their status_descriptions into ChromaDB for semantic search.
"""

import os
import sys
import sqlite3
import random
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from faker import Faker

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "tracking.db"))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", ROOT / "data" / "chroma_db"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
fake = Faker("en_US")
Faker.seed(42)
random.seed(42)

STATUS_CODES = ["IN_TRANSIT", "OUT_FOR_DELIVERY", "DELIVERED", "DELAYED", "EXCEPTION"]

STATUS_WEIGHTS = [0.35, 0.15, 0.30, 0.12, 0.08]

CITIES = [
    "Munich", "Berlin", "Hamburg", "Frankfurt", "Cologne",
    "Stuttgart", "Dusseldorf", "Leipzig", "Nuremberg", "Dresden",
    "Bremen", "Hanover", "Dortmund", "Essen", "Bochum",
    "London", "Paris", "Amsterdam", "Brussels", "Vienna",
    "Zurich", "Warsaw", "Prague", "Budapest", "Copenhagen",
]

HUB_CITIES = ["Munich", "Frankfurt", "Berlin", "Hamburg", "Cologne"]

DESCRIPTION_TEMPLATES: dict[str, list[str]] = {
    "IN_TRANSIT": [
        "Package arrived at {city} hub and is being processed for onward dispatch.",
        "Shipment is in transit via {city} sorting facility.",
        "Package scanned at {city} distribution center — moving to next waypoint.",
        "Consignment loaded onto outbound vehicle at {city}.",
    ],
    "OUT_FOR_DELIVERY": [
        "Package is out for delivery with courier in {city} — expected today.",
        "Delivery attempt scheduled for {city}. Courier is en route.",
        "Package loaded onto delivery van in {city}. Estimated delivery this afternoon.",
        "Final-mile delivery underway in {city}.",
    ],
    "DELIVERED": [
        "Package successfully delivered to recipient in {city}.",
        "Shipment delivered and signed for in {city}.",
        "Delivery confirmed in {city} — package handed to recipient.",
        "Parcel delivered to address in {city}. Proof of delivery captured.",
    ],
    "DELAYED": [
        "Package delayed at {city} hub due to high volume. Revised ETA updated.",
        "Shipment held at {city} facility — weather conditions causing delays.",
        "Delay reported at {city} sorting center; customs processing taking longer than expected.",
        "Package missed connecting flight at {city}. New estimated delivery date set.",
    ],
    "EXCEPTION": [
        "Delivery exception in {city}: address not found. Customer contact required.",
        "Exception raised at {city} — package damaged during transit. Claim process initiated.",
        "Failed delivery attempt in {city}: recipient unavailable. Re-delivery scheduled.",
        "Exception: incorrect label detected at {city} facility. Manual review in progress.",
    ],
}


def generate_tracking_id() -> str:
    """Generate a realistic DHL-style tracking ID."""
    prefix = random.choice(["JD", "GM", "LX", "RX"])
    digits = "".join([str(random.randint(0, 9)) for _ in range(18)])
    return f"{prefix}{digits}"


def generate_shipment(idx: int) -> dict:
    """Generate a single realistic shipment record."""
    status_code = random.choices(STATUS_CODES, weights=STATUS_WEIGHTS, k=1)[0]

    origin_city = random.choice(CITIES)
    destination_city = random.choice([c for c in CITIES if c != origin_city])
    current_location = random.choice(HUB_CITIES) if status_code in ("IN_TRANSIT", "DELAYED", "EXCEPTION") else destination_city

    # Dates
    created_days_ago = random.randint(1, 30)
    created_date = date.today() - timedelta(days=created_days_ago)
    estimated_delivery = created_date + timedelta(days=random.randint(2, 7))
    actual_delivery: date | None = None
    if status_code == "DELIVERED":
        actual_delivery = estimated_delivery + timedelta(days=random.randint(-1, 3))
    elif status_code == "DELAYED":
        estimated_delivery = estimated_delivery + timedelta(days=random.randint(1, 5))

    last_update = datetime.now() - timedelta(
        hours=random.randint(0, 72),
        minutes=random.randint(0, 59),
    )

    template = random.choice(DESCRIPTION_TEMPLATES[status_code])
    status_description = template.format(city=current_location)

    return {
        "tracking_id": generate_tracking_id(),
        "sender_name": fake.name(),
        "recipient_name": fake.name(),
        "origin_city": origin_city,
        "destination_city": destination_city,
        "current_location": current_location,
        "status_code": status_code,
        "estimated_delivery": estimated_delivery.isoformat(),
        "actual_delivery": actual_delivery.isoformat() if actual_delivery else None,
        "last_update": last_update.isoformat(timespec="seconds"),
        "status_description": status_description,
    }


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS shipments (
    tracking_id         TEXT PRIMARY KEY,
    sender_name         TEXT NOT NULL,
    recipient_name      TEXT NOT NULL,
    origin_city         TEXT NOT NULL,
    destination_city    TEXT NOT NULL,
    current_location    TEXT NOT NULL,
    status_code         TEXT NOT NULL,
    estimated_delivery  DATE NOT NULL,
    actual_delivery     DATE,
    last_update         TIMESTAMP NOT NULL,
    status_description  TEXT NOT NULL
);
"""

INSERT_SQL = """
INSERT OR REPLACE INTO shipments VALUES (
    :tracking_id, :sender_name, :recipient_name, :origin_city,
    :destination_city, :current_location, :status_code, :estimated_delivery,
    :actual_delivery, :last_update, :status_description
)
"""


def create_sqlite_db(shipments: list[dict]) -> None:
    """Write shipment records to SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.executemany(INSERT_SQL, shipments)
        conn.commit()
    logger.info("SQLite DB written to %s (%d records)", DB_PATH, len(shipments))


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

def create_chroma_index(shipments: list[dict]) -> None:
    """Index status_descriptions into ChromaDB for semantic search."""
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        logger.warning("chromadb or sentence-transformers not installed — skipping vector index.")
        return

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Drop and recreate to avoid stale data on re-runs
    try:
        client.delete_collection("shipment_descriptions")
    except Exception:
        pass

    embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.create_collection(
        name="shipment_descriptions",
        embedding_function=embed_fn,
    )

    ids = [s["tracking_id"] for s in shipments]
    documents = [s["status_description"] for s in shipments]
    metadatas = [
        {
            "tracking_id": s["tracking_id"],
            "status_code": s["status_code"],
            "current_location": s["current_location"],
            "destination_city": s["destination_city"],
            "estimated_delivery": s["estimated_delivery"],
        }
        for s in shipments
    ]

    # ChromaDB recommends batches ≤ 5000
    batch_size = 100
    for i in range(0, len(shipments), batch_size):
        collection.add(
            ids=ids[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    logger.info("ChromaDB index written to %s (%d documents)", CHROMA_PATH, len(shipments))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate 200 shipments, persist to SQLite and ChromaDB."""
    logger.info("Generating 200 synthetic DHL shipments…")
    shipments = [generate_shipment(i) for i in range(200)]

    create_sqlite_db(shipments)
    create_chroma_index(shipments)

    # Print a quick sample
    sample = shipments[0]
    logger.info("Sample record: %s", sample)
    print("\nDone. Sample tracking ID:", sample["tracking_id"])


if __name__ == "__main__":
    main()
