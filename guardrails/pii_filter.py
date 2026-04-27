"""
pii_filter.py
-------------
PII detection and redaction for logs and display.

Methods:
  PIIFilter.redact_for_logging(text)  — replaces emails, phones, names with [REDACTED]
  PIIFilter.safe_to_display(data)     — masks recipient_name as "J. D***" style
"""

import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

_PHONE_RE = re.compile(
    r"\b(?:\+?\d[\d\s\-().]{7,14}\d)\b"
)

# Simple pattern for "First Last" — two title-case words
_NAME_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b"
)

# ---------------------------------------------------------------------------
# spaCy NER (optional — gracefully degrades if not installed/downloaded)
# ---------------------------------------------------------------------------

_nlp = None
_spacy_available = False


def _load_spacy() -> bool:
    """Try to load spaCy en_core_web_sm. Returns True if successful."""
    global _nlp, _spacy_available
    if _nlp is not None:
        return _spacy_available
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        _spacy_available = True
        logger.debug("spaCy NER loaded (en_core_web_sm)")
    except Exception as exc:
        logger.warning(
            "spaCy NER unavailable (%s). Falling back to regex-only PII detection.", exc
        )
        _spacy_available = False
    return _spacy_available


def _ner_names(text: str) -> list[str]:
    """Extract PERSON entity spans using spaCy NER."""
    if not _load_spacy() or _nlp is None:
        return []
    doc = _nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PERSON"]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PIIFilter:
    """
    Redacts PII from strings and dicts before logging or display.

    Handles:
      - Email addresses
      - Phone numbers
      - Full names (regex heuristic + spaCy NER when available)
    """

    def redact_for_logging(self, text: str) -> str:
        """
        Replace emails, phone numbers, and detected names with [REDACTED].

        Args:
            text: Any string that may contain PII.

        Returns:
            Redacted string safe for logging.
        """
        if not text:
            return text

        # 1. Emails (most specific — do first to avoid phone regex overlap)
        redacted = _EMAIL_RE.sub("[REDACTED]", text)

        # 2. Phone numbers
        redacted = _PHONE_RE.sub("[REDACTED]", redacted)

        # 3. spaCy NER names (precise)
        ner_names = _ner_names(redacted)
        for name in sorted(ner_names, key=len, reverse=True):  # longest first
            redacted = redacted.replace(name, "[REDACTED]")

        # 4. Regex name fallback (two consecutive Title Case words)
        redacted = _NAME_RE.sub("[REDACTED]", redacted)

        return redacted

    def safe_to_display(self, tracking_data: dict) -> dict:
        """
        Return a copy of tracking_data with recipient_name partially masked.

        Masking format: "John Doe" → "J. D***"
        Other PII fields (sender_name) are also masked.

        Args:
            tracking_data: Raw dict from get_tracking_by_id.

        Returns:
            New dict with name fields masked.
        """
        if not tracking_data:
            return tracking_data

        result = dict(tracking_data)

        for field in ("recipient_name", "sender_name"):
            raw = result.get(field)
            if raw:
                result[field] = self._mask_name(raw)

        return result

    @staticmethod
    def _mask_name(name: str) -> str:
        """
        Mask a full name to show only first initial and last initial + stars.

        Example: "John Doe" → "J. D***"
                 "Anna Maria Schmidt" → "A. S***"
        """
        parts = name.strip().split()
        if len(parts) == 0:
            return "***"
        first_initial = parts[0][0].upper() + "."
        if len(parts) > 1:
            last_initial = parts[-1][0].upper()
            return f"{first_initial} {last_initial}***"
        return f"{first_initial} ***"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_pii_filter: PIIFilter | None = None


def get_pii_filter() -> PIIFilter:
    """Return a cached PIIFilter instance."""
    global _pii_filter
    if _pii_filter is None:
        _pii_filter = PIIFilter()
    return _pii_filter


if __name__ == "__main__":
    logging.basicConfig(level="DEBUG")
    pf = PIIFilter()

    tests = [
        "Contact John Smith at john.smith@example.com or +49 89 1234567.",
        "Package for Maria Garcia arriving tomorrow.",
        "Call us at (800) 555-0199 for support.",
    ]
    for t in tests:
        print(f"Original : {t}")
        print(f"Redacted : {pf.redact_for_logging(t)}\n")

    sample = {
        "tracking_id": "JD123",
        "recipient_name": "John Doe",
        "sender_name": "Anna Schmidt",
        "status_code": "IN_TRANSIT",
    }
    print("Display-safe data:", pf.safe_to_display(sample))
