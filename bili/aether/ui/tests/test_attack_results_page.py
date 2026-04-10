"""Tests for bili.aether.ui.attack_results_page -- Attack Results viewer.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.
"""

# pylint: disable=import-outside-toplevel, protected-access, reimported

from streamlit.testing.v1 import AppTest

from bili.aether.ui import attack_results_page as arp_mod


def test_empty_state_shows_info_message():
    """When no attack results exist the page shows an info message."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import attack_results_page as arp
with patch.object(arp, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(arp, "_load_suite_results", return_value=[]):
        arp._render_main("All Suites", [])
"""
    )
    at.run()
    assert not at.exception
    assert "No results found" in " ".join(m.value for m in at.info)


def test_empty_state_single_suite():
    """Empty state for a single suite names the suite."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import attack_results_page as arp
with patch.object(arp, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(arp, "_load_suite_results", return_value=[]):
        arp._render_main("Injection", [])
"""
    )
    at.run()
    assert not at.exception
    assert "Injection" in " ".join(m.value for m in at.info)


def test_main_renders_aegis_heading():
    """The main area renders the AEGIS Attack Results heading."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import attack_results_page as arp
with patch.object(arp, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(arp, "_load_suite_results", return_value=[]):
        arp._render_main("All Suites", [])
"""
    )
    at.run()
    assert not at.exception
    assert "AEGIS Attack Results" in " ".join(m.value for m in at.markdown)


def test_sidebar_renders_heading_and_suite_selector():
    """The sidebar contains the AEGIS heading and a suite selectbox."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.aether.ui import attack_results_page as arp
with st.sidebar:
    with patch.object(arp, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        arp._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    assert "AEGIS" in " ".join(m.value for m in at.sidebar.markdown)
    assert len(at.sidebar.selectbox) >= 1


def test_render_attack_results_page_no_exception():
    """The full render_attack_results_page runs without exception."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
from bili.aether.ui import attack_results_page as arp
with patch.object(arp, "LOGO_PATH") as lp:
    lp.exists.return_value = False
    with patch.object(arp, "_load_suite_results", return_value=[]):
        arp.render_attack_results_page()
"""
    )
    at.run()
    assert not at.exception


def test_normalise_fills_missing_fields():
    """The _normalise helper fills in default values for absent keys."""
    raw = {
        "payload_id": "p1",
        "injection_type": "injection",
        "severity": "high",
        "mas_id": "test",
        "injection_phase": "pre",
        "attack_suite": "injection",
        "execution": {"success": True, "duration_ms": 100, "agent_count": 2},
        "run_metadata": {"stub_mode": True, "timestamp": "2025-01-01"},
    }
    result = arp_mod._normalise(raw)
    assert result["model_id"] is None
    assert result["tier3_score"] is None


def test_normalise_preserves_tier3_score():
    """The _normalise helper preserves a numeric tier3_score."""
    raw = {
        "payload_id": "p1",
        "injection_type": "injection",
        "severity": "high",
        "mas_id": "test",
        "injection_phase": "pre",
        "attack_suite": "injection",
        "execution": {"success": True, "duration_ms": 100, "agent_count": 2},
        "run_metadata": {"stub_mode": False, "timestamp": "t", "tier3_score": 2},
    }
    assert arp_mod._normalise(raw)["tier3_score"] == 2


def test_tier2_tier3_disagree_influenced_but_score_zero():
    """Disagreement when T2 says influenced but T3 says score 0."""
    assert arp_mod._tier2_tier3_disagree(["agent_1"], 0) is True


def test_tier2_tier3_disagree_not_influenced_but_high_score():
    """Disagreement when T2 says not influenced but T3 >= 2."""
    assert arp_mod._tier2_tier3_disagree([], 2) is True


def test_tier2_tier3_no_disagree_when_aligned():
    """No disagreement when T2 and T3 agree."""
    assert arp_mod._tier2_tier3_disagree(["a"], 2) is False
    assert arp_mod._tier2_tier3_disagree([], 0) is False


def test_tier2_tier3_no_disagree_when_tier3_none():
    """No disagreement when T3 score is None."""
    assert arp_mod._tier2_tier3_disagree(["a"], None) is False
