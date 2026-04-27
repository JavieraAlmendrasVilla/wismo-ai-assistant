"""
output_validator.py
-------------------
Validates LLM responses against retrieved tracking data to catch hallucinations.

Checks:
  1. Any date mentioned in the response exists in retrieved_data.
  2. Any location/city mentioned exists in retrieved_data.
  3. Response length is reasonable (non-empty, ≤ 500 chars).
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of a single response validation."""
    is_valid: bool
    violations: list[str] = field(default_factory=list)
    safe_response: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches ISO dates (2024-03-15), written dates (March 15, 2024), and
# short forms (15/03/2024, 03-15-2024)
_DATE_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}"          # ISO 8601
    r"|\d{2}/\d{2}/\d{4}"            # DD/MM/YYYY
    r"|\d{2}-\d{2}-\d{4}"            # DD-MM-YYYY
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+\d{4}"  # Month DD YYYY
    r")\b",
    re.IGNORECASE,
)

# Cities we know about from the data layer
_KNOWN_CITIES = {
    "Munich", "Berlin", "Hamburg", "Frankfurt", "Cologne",
    "Stuttgart", "Dusseldorf", "Leipzig", "Nuremberg", "Dresden",
    "Bremen", "Hanover", "Dortmund", "Essen", "Bochum",
    "London", "Paris", "Amsterdam", "Brussels", "Vienna",
    "Zurich", "Warsaw", "Prague", "Budapest", "Copenhagen",
}


def _extract_dates_from_data(data: dict) -> set[str]:
    """Pull all date strings out of retrieved_data values."""
    dates: set[str] = set()
    for key in ("estimated_delivery", "actual_delivery", "last_update"):
        val = data.get(key)
        if val:
            # Normalise: keep the date portion only (first 10 chars of ISO timestamp)
            dates.add(str(val)[:10])
    return dates


def _extract_locations_from_data(data: dict) -> set[str]:
    """Pull all known location strings out of retrieved_data."""
    locations: set[str] = set()
    for key in ("origin_city", "destination_city", "current_location"):
        val = data.get(key)
        if val:
            locations.add(val.strip().lower())
    return locations


def _extract_dates_from_response(response: str) -> list[str]:
    """Extract date-like strings from the LLM response."""
    return _DATE_PATTERN.findall(response)


def _extract_locations_from_response(response: str) -> list[str]:
    """Extract city names from the LLM response that appear in the known-cities list."""
    found = []
    for city in _KNOWN_CITIES:
        # whole-word match, case-insensitive
        if re.search(rf"\b{re.escape(city)}\b", response, re.IGNORECASE):
            found.append(city)
    return found


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_SAFE_FALLBACK_TEMPLATE = (
    "Based on the tracking data available:\n"
    "  Status: {status_code}\n"
    "  Current location: {current_location}\n"
    "  Destination: {destination_city}\n"
    "  Estimated delivery: {estimated_delivery}\n"
    "  Last update: {last_update}\n\n"
    "For further assistance, please contact DHL customer support."
)

_SAFE_FALLBACK_NOT_FOUND = (
    "I could not find tracking information for the provided ID. "
    "Please verify the tracking number and try again, or contact DHL support."
)


class OutputValidator:
    """Validates LLM responses for hallucinations and length issues."""

    MAX_RESPONSE_LENGTH = 500
    MIN_RESPONSE_LENGTH = 10

    def validate(self, response: str, retrieved_data: dict) -> ValidationResult:
        """
        Validate the LLM response against the retrieved tracking data.

        Args:
            response: The raw LLM-generated answer string.
            retrieved_data: The dict returned by get_tracking_by_id (may be empty).

        Returns:
            ValidationResult with is_valid flag, violation list, and safe_response.
        """
        violations: list[str] = []

        # --- 1. Length checks ---
        if not response or len(response.strip()) < self.MIN_RESPONSE_LENGTH:
            violations.append("Response is empty or too short.")

        if len(response) > self.MAX_RESPONSE_LENGTH:
            violations.append(
                f"Response exceeds {self.MAX_RESPONSE_LENGTH} chars "
                f"(got {len(response)})."
            )

        # If no tracking data, skip hallucination checks
        if not retrieved_data:
            is_valid = len(violations) == 0
            return ValidationResult(
                is_valid=is_valid,
                violations=violations,
                safe_response=response if is_valid else _SAFE_FALLBACK_NOT_FOUND,
            )

        known_dates = _extract_dates_from_data(retrieved_data)
        known_locations = _extract_locations_from_data(retrieved_data)

        # --- 2. Date hallucination check ---
        mentioned_dates = _extract_dates_from_response(response)
        for d in mentioned_dates:
            # Normalise the response date to ISO (take first 10 chars if longer)
            normalised = d[:10] if len(d) >= 10 else d
            if normalised not in known_dates:
                violations.append(f"Hallucinated date detected: '{d}' not in tracking data.")

        # --- 3. Location hallucination check ---
        mentioned_locations = _extract_locations_from_response(response)
        for loc in mentioned_locations:
            if loc.lower() not in known_locations:
                violations.append(
                    f"Location '{loc}' mentioned in response is not in tracking data."
                )

        is_valid = len(violations) == 0

        if is_valid:
            safe_response = response
        else:
            logger.warning("Guardrail triggered — violations: %s", violations)
            # Build a safe, data-only fallback
            safe_response = _SAFE_FALLBACK_TEMPLATE.format(
                status_code=retrieved_data.get("status_code", "N/A"),
                current_location=retrieved_data.get("current_location", "N/A"),
                destination_city=retrieved_data.get("destination_city", "N/A"),
                estimated_delivery=retrieved_data.get("estimated_delivery", "N/A"),
                last_update=retrieved_data.get("last_update", "N/A"),
            )

        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            safe_response=safe_response,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_validator: OutputValidator | None = None


def get_validator() -> OutputValidator:
    """Return a cached OutputValidator instance."""
    global _validator
    if _validator is None:
        _validator = OutputValidator()
    return _validator


if __name__ == "__main__":
    import json
    logging.basicConfig(level="DEBUG")

    sample_data = {
        "tracking_id": "JD012345678901234567",
        "status_code": "IN_TRANSIT",
        "current_location": "Munich",
        "destination_city": "Berlin",
        "estimated_delivery": "2024-12-20",
        "actual_delivery": None,
        "last_update": "2024-12-18T14:30:00",
    }

    validator = OutputValidator()

    # Good response
    good = "Your package is currently in Munich and expected to arrive in Berlin by 2024-12-20."
    print("Good:", validator.validate(good, sample_data))

    # Hallucinated date
    bad = "Your package will arrive in Berlin by 2025-01-15."
    print("Bad (date):", validator.validate(bad, sample_data))

    # Hallucinated city
    bad2 = "Your package is currently in Paris."
    print("Bad (city):", validator.validate(bad2, sample_data))
