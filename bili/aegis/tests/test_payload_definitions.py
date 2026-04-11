"""Tests for all AEGIS payload definition modules.

Validates that each payload library exports a PAYLOADS_BY_ID dict,
every entry has the required fields (payload_id, payload, severity,
injection_type, notes), and payload text is non-empty.
"""

import pytest

from bili.aegis.tests.agent_impersonation.payloads.agent_impersonation_payloads import (
    AGENT_IMPERSONATION_PAYLOADS,
)
from bili.aegis.tests.agent_impersonation.payloads.agent_impersonation_payloads import (
    PAYLOADS_BY_ID as AI_PAYLOADS_BY_ID,
)
from bili.aegis.tests.bias_inheritance.payloads.bias_inheritance_payloads import (
    BIAS_INHERITANCE_PAYLOADS,
)
from bili.aegis.tests.bias_inheritance.payloads.bias_inheritance_payloads import (
    PAYLOADS_BY_ID as BI_PAYLOADS_BY_ID,
)
from bili.aegis.tests.injection.payloads.prompt_injection_payloads import (
    INJECTION_PAYLOADS,
)
from bili.aegis.tests.injection.payloads.prompt_injection_payloads import (
    PAYLOADS_BY_ID as PI_PAYLOADS_BY_ID,
)
from bili.aegis.tests.jailbreak.payloads.jailbreak_payloads import JAILBREAK_PAYLOADS
from bili.aegis.tests.jailbreak.payloads.jailbreak_payloads import (
    PAYLOADS_BY_ID as JB_PAYLOADS_BY_ID,
)
from bili.aegis.tests.memory_poisoning.payloads.memory_poisoning_payloads import (
    MEMORY_POISONING_PAYLOADS,
)
from bili.aegis.tests.memory_poisoning.payloads.memory_poisoning_payloads import (
    PAYLOADS_BY_ID as MP_PAYLOADS_BY_ID,
)

# Required attributes on every payload dataclass instance.
_REQUIRED_FIELDS = {
    "payload_id",
    "payload",
    "severity",
    "injection_type",
    "notes",
}

# All suites bundled for parametrized tests.
_ALL_SUITES = [
    pytest.param(
        AGENT_IMPERSONATION_PAYLOADS,
        AI_PAYLOADS_BY_ID,
        id="agent_impersonation",
    ),
    pytest.param(
        BIAS_INHERITANCE_PAYLOADS,
        BI_PAYLOADS_BY_ID,
        id="bias_inheritance",
    ),
    pytest.param(
        INJECTION_PAYLOADS,
        PI_PAYLOADS_BY_ID,
        id="prompt_injection",
    ),
    pytest.param(
        JAILBREAK_PAYLOADS,
        JB_PAYLOADS_BY_ID,
        id="jailbreak",
    ),
    pytest.param(
        MEMORY_POISONING_PAYLOADS,
        MP_PAYLOADS_BY_ID,
        id="memory_poisoning",
    ),
]


class TestPayloadsByIdDict:
    """PAYLOADS_BY_ID dict must match the payload list."""

    @pytest.mark.parametrize("payloads,_by_id", _ALL_SUITES)
    def test_list_is_not_empty(self, payloads, _by_id):
        """Payload list has at least one entry."""
        assert len(payloads) > 0

    @pytest.mark.parametrize("payloads,by_id", _ALL_SUITES)
    def test_by_id_length_matches_list(self, payloads, by_id):
        """PAYLOADS_BY_ID has same length as the list."""
        assert len(by_id) == len(payloads)

    @pytest.mark.parametrize("payloads,by_id", _ALL_SUITES)
    def test_by_id_keys_match_payload_ids(self, payloads, by_id):
        """Every payload_id from the list appears as a key."""
        ids_from_list = {p.payload_id for p in payloads}
        assert set(by_id.keys()) == ids_from_list

    @pytest.mark.parametrize("payloads,_by_id", _ALL_SUITES)
    def test_no_duplicate_ids(self, payloads, _by_id):
        """No two payloads share the same payload_id."""
        ids = [p.payload_id for p in payloads]
        assert len(ids) == len(set(ids))


class TestPayloadRequiredFields:
    """Every payload must have all required fields populated."""

    @pytest.mark.parametrize("payloads,_by_id", _ALL_SUITES)
    def test_all_required_fields_present(self, payloads, _by_id):
        """Each payload has payload_id, payload, severity, etc."""
        for entry in payloads:
            for field_name in _REQUIRED_FIELDS:
                assert hasattr(
                    entry, field_name
                ), f"{entry.payload_id} missing {field_name}"

    @pytest.mark.parametrize("payloads,_by_id", _ALL_SUITES)
    def test_payload_text_is_non_empty(self, payloads, _by_id):
        """The payload text string is non-empty."""
        for entry in payloads:
            assert entry.payload.strip(), f"{entry.payload_id} has empty payload"

    @pytest.mark.parametrize("payloads,_by_id", _ALL_SUITES)
    def test_payload_id_is_non_empty(self, payloads, _by_id):
        """The payload_id string is non-empty."""
        for entry in payloads:
            assert entry.payload_id.strip()

    @pytest.mark.parametrize("payloads,_by_id", _ALL_SUITES)
    def test_severity_is_valid(self, payloads, _by_id):
        """Severity is one of low, medium, high."""
        valid = {"low", "medium", "high"}
        for entry in payloads:
            assert entry.severity in valid, (
                f"{entry.payload_id} severity " f"'{entry.severity}' not in {valid}"
            )

    @pytest.mark.parametrize("payloads,_by_id", _ALL_SUITES)
    def test_notes_is_non_empty(self, payloads, _by_id):
        """Notes string is non-empty."""
        for entry in payloads:
            assert entry.notes.strip(), f"{entry.payload_id} has empty notes"

    @pytest.mark.parametrize("payloads,_by_id", _ALL_SUITES)
    def test_min_length_matches_payload_len(self, payloads, _by_id):
        """min_length_chars equals len(payload) for each entry."""
        for entry in payloads:
            if hasattr(entry, "min_length_chars"):
                assert entry.min_length_chars == len(entry.payload), (
                    f"{entry.payload_id}: "
                    f"min_length_chars={entry.min_length_chars} "
                    f"but len(payload)={len(entry.payload)}"
                )
