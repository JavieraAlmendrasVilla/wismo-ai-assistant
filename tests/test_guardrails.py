"""
test_guardrails.py
------------------
Unit tests for OutputValidator and PIIFilter guardrails.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrails.output_validator import OutputValidator, ValidationResult
from guardrails.pii_filter import PIIFilter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def validator() -> OutputValidator:
    return OutputValidator()


@pytest.fixture
def pii_filter() -> PIIFilter:
    return PIIFilter()


@pytest.fixture
def sample_tracking_data() -> dict:
    return {
        "tracking_id": "JD012345678901234567",
        "status_code": "IN_TRANSIT",
        "current_location": "Munich",
        "origin_city": "Hamburg",
        "destination_city": "Berlin",
        "estimated_delivery": "2024-12-20",
        "actual_delivery": None,
        "last_update": "2024-12-18T14:30:00",
        "status_description": "Package arrived at Munich hub.",
        "sender_name": "Hans Müller",
        "recipient_name": "Jane Doe",
    }


# ---------------------------------------------------------------------------
# OutputValidator tests
# ---------------------------------------------------------------------------

class TestOutputValidator:
    """Tests for guardrails.output_validator.OutputValidator."""

    def test_valid_response_passes_validation(self, validator, sample_tracking_data):
        """A response that only references real data passes validation."""
        response = (
            "Your package is currently in Munich and is expected to arrive "
            "in Berlin by 2024-12-20."
        )
        result = validator.validate(response, sample_tracking_data)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.violations == []
        assert result.safe_response == response

    def test_hallucinated_date_is_caught(self, validator, sample_tracking_data):
        """A response with a date not in retrieved_data triggers a violation."""
        response = "Your package will arrive by 2025-03-01. Everything looks fine!"
        result = validator.validate(response, sample_tracking_data)

        assert result.is_valid is False
        assert any("2025-03-01" in v or "Hallucinated date" in v for v in result.violations)
        # Safe response should NOT contain the invented date
        assert "2025-03-01" not in result.safe_response

    def test_hallucinated_location_is_caught(self, validator, sample_tracking_data):
        """A response mentioning a city not in retrieved_data triggers a violation."""
        response = "Your package is currently in Paris and heading to Vienna."
        result = validator.validate(response, sample_tracking_data)

        assert result.is_valid is False
        assert len(result.violations) >= 1
        violation_text = " ".join(result.violations)
        assert "Paris" in violation_text or "Vienna" in violation_text

    def test_empty_response_is_invalid(self, validator, sample_tracking_data):
        """Empty response triggers a length violation."""
        result = validator.validate("", sample_tracking_data)

        assert result.is_valid is False
        assert any("empty" in v.lower() or "short" in v.lower() for v in result.violations)

    def test_too_long_response_is_invalid(self, validator, sample_tracking_data):
        """Response exceeding 500 chars triggers a length violation."""
        long_response = "Your package is in Munich. " * 25  # well over 500 chars
        result = validator.validate(long_response, sample_tracking_data)

        assert result.is_valid is False
        assert any("500" in v or "exceeds" in v.lower() for v in result.violations)

    def test_no_tracking_data_allows_response(self, validator):
        """When no tracking data, the response is still returned as-is (not-found flow)."""
        response = "I couldn't find tracking information for this ID."
        result = validator.validate(response, {})

        assert result.safe_response == response

    def test_valid_response_with_actual_delivery(self, validator, sample_tracking_data):
        """Actual delivery date (when present) is also a valid date to mention."""
        data = {**sample_tracking_data, "actual_delivery": "2024-12-19"}
        response = "Your package was delivered on 2024-12-19."
        result = validator.validate(response, data)

        assert result.is_valid is True


# ---------------------------------------------------------------------------
# PIIFilter tests
# ---------------------------------------------------------------------------

class TestPIIFilter:
    """Tests for guardrails.pii_filter.PIIFilter."""

    def test_pii_redaction_removes_email(self, pii_filter):
        """Email addresses are replaced with [REDACTED]."""
        text = "Contact us at support@dhl.com for help."
        redacted = pii_filter.redact_for_logging(text)

        assert "support@dhl.com" not in redacted
        assert "[REDACTED]" in redacted

    def test_pii_redaction_removes_phone(self, pii_filter):
        """Phone numbers are replaced with [REDACTED]."""
        cases = [
            "Call +49 89 12345678 for support.",
            "Reach us at (800) 555-0199.",
            "Phone: 0049-89-1234567",
        ]
        for text in cases:
            redacted = pii_filter.redact_for_logging(text)
            assert "[REDACTED]" in redacted, f"Phone not redacted in: {text!r}"

    def test_pii_redaction_removes_name(self, pii_filter):
        """Full names are replaced with [REDACTED] by regex heuristic."""
        text = "Package is addressed to John Smith in Berlin."
        redacted = pii_filter.redact_for_logging(text)

        # Either the name itself or its initials should not appear
        assert "John Smith" not in redacted

    def test_pii_redaction_empty_string(self, pii_filter):
        """Empty string returns empty string without error."""
        assert pii_filter.redact_for_logging("") == ""

    def test_safe_to_display_masks_recipient_name(self, pii_filter):
        """recipient_name is masked to 'F. L***' format."""
        data = {
            "tracking_id": "JD123",
            "recipient_name": "John Doe",
            "sender_name": "Anna Schmidt",
            "status_code": "IN_TRANSIT",
        }
        safe = pii_filter.safe_to_display(data)

        assert safe["recipient_name"] != "John Doe"
        assert safe["recipient_name"].startswith("J.")
        assert "***" in safe["recipient_name"]

    def test_safe_to_display_masks_sender_name(self, pii_filter):
        """sender_name is also masked."""
        data = {"sender_name": "Anna Schmidt", "recipient_name": "Bob Jones"}
        safe = pii_filter.safe_to_display(data)

        assert safe["sender_name"] != "Anna Schmidt"
        assert "A." in safe["sender_name"]

    def test_safe_to_display_preserves_non_pii_fields(self, pii_filter):
        """Non-name fields are unchanged."""
        data = {
            "tracking_id": "JD999",
            "status_code": "DELIVERED",
            "recipient_name": "Jane Doe",
        }
        safe = pii_filter.safe_to_display(data)

        assert safe["tracking_id"] == "JD999"
        assert safe["status_code"] == "DELIVERED"

    def test_safe_to_display_empty_dict(self, pii_filter):
        """Empty dict is returned unchanged."""
        assert pii_filter.safe_to_display({}) == {}
