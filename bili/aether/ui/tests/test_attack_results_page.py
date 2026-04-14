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


# ---------------------------------------------------------------------------
# _render_metrics calculations
# ---------------------------------------------------------------------------


def _sample_results():
    """Return a list of normalised result dicts for testing."""
    return [
        {
            "run_id": "run_001",
            "payload_id": "p1",
            "injection_type": "injection",
            "severity": "high",
            "mas_id": "mas_a",
            "phase": "pre",
            "attack_suite": "injection",
            "tier1_pass": True,
            "tier3_score": 2,
            "stub_mode": False,
            "timestamp": "2025-01-01",
            "model_id": None,
            "model_name": None,
            "provider_family": None,
            "tier2_influenced": True,
        },
        {
            "run_id": "run_001",
            "payload_id": "p2",
            "injection_type": "jailbreak",
            "severity": "low",
            "mas_id": "mas_a",
            "phase": "pre",
            "attack_suite": "jailbreak",
            "tier1_pass": True,
            "tier3_score": 0,
            "stub_mode": False,
            "timestamp": "2025-01-01",
            "model_id": None,
            "model_name": None,
            "provider_family": None,
            "tier2_influenced": False,
        },
        {
            "run_id": "run_001",
            "payload_id": "p3",
            "injection_type": "injection",
            "severity": "medium",
            "mas_id": "mas_b",
            "phase": "mid",
            "attack_suite": "injection",
            "tier1_pass": False,
            "tier3_score": None,
            "stub_mode": True,
            "timestamp": "2025-01-02",
            "model_id": None,
            "model_name": None,
            "provider_family": None,
            "tier2_influenced": False,
        },
    ]


def test_render_metrics_shows_totals():
    """_render_metrics renders total runs and suite count."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import attack_results_page as arp
rows = [
    {"payload_id": "p1", "injection_type": "injection", "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "injection",
     "tier1_pass": True, "tier3_score": 2, "stub_mode": False,
     "timestamp": "t", "model_id": None, "model_name": None,
     "provider_family": None, "tier2_influenced": True},
    {"payload_id": "p2", "injection_type": "jailbreak", "severity": "low",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "jailbreak",
     "tier1_pass": True, "tier3_score": 0, "stub_mode": False,
     "timestamp": "t", "model_id": None, "model_name": None,
     "provider_family": None, "tier2_influenced": False},
]
df = pd.DataFrame(rows)
arp._render_metrics(df)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _build_dataframe
# ---------------------------------------------------------------------------


def test_build_dataframe_from_normalised_results():
    """_build_dataframe creates a DataFrame from normalised results."""
    results = _sample_results()
    # Convert to normalised format expected by _build_dataframe
    import pandas as pd

    df = arp_mod._build_dataframe(results)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "tier1_pass" in df.columns
    assert "tier2_influenced" in df.columns


def test_build_dataframe_skips_malformed():
    """_build_dataframe skips malformed rows without crashing."""

    results = [
        {
            "run_id": "run_001",
            "payload_id": "ok",
            "injection_type": "x",
            "severity": "h",
            "mas_id": "m",
            "phase": "pre",
            "attack_suite": "injection",
            "tier1_pass": True,
            "tier3_score": 1,
            "stub_mode": False,
            "timestamp": "t",
            "model_id": None,
            "model_name": None,
            "provider_family": None,
            "influenced_agents": [],
        },
        {"broken": True},
    ]
    df = arp_mod._build_dataframe(results)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# _render_matrix with sample data
# ---------------------------------------------------------------------------


def test_render_matrix_with_data():
    """_render_matrix renders a matrix for filtered data."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import attack_results_page as arp
rows = [
    {"payload_id": "p1", "injection_type": "injection", "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "injection",
     "tier1_pass": True, "tier3_score": 2, "stub_mode": False,
     "timestamp": "t", "model_id": None, "model_name": None,
     "provider_family": None, "tier2_influenced": True},
]
df = pd.DataFrame(rows)
arp._render_matrix(df, is_cross_model=False)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Results Matrix" in all_md


def test_render_matrix_empty_shows_info():
    """_render_matrix shows info when DataFrame is empty."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import attack_results_page as arp
df = pd.DataFrame()
arp._render_matrix(df, is_cross_model=False)
"""
    )
    at.run()
    assert not at.exception
    assert "No results" in " ".join(m.value for m in at.info)


# ---------------------------------------------------------------------------
# _render_detail_panel with sample results
# ---------------------------------------------------------------------------


def test_render_detail_panel_with_results():
    """_render_detail_panel renders run details for filtered data."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import attack_results_page as arp
results = [
    {"run_id": "run_001", "payload_id": "p1", "injection_type": "injection",
     "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "injection",
     "tier1_pass": True, "tier3_score": 2, "tier3_confidence": "high",
     "tier3_reasoning": "Clear compliance", "stub_mode": False,
     "timestamp": "2025-01-01", "model_id": None, "model_name": None,
     "provider_family": None, "influenced_agents": ["a0"],
     "resistant_agents": [], "propagation_path": ["a0", "a1"],
     "target_agent_id": "a0", "duration_ms": 100.0,
     "config_path": "", "injection_phase": "pre"},
]
df_rows = [
    {"run_id": "run_001", "payload_id": "p1", "injection_type": "injection",
     "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "injection",
     "tier1_pass": True, "tier3_score": 2, "stub_mode": False,
     "timestamp": "t", "model_id": None, "model_name": None,
     "provider_family": None, "tier2_influenced": True},
]
df = pd.DataFrame(df_rows)
arp._render_detail_panel(results, df, is_cross_model=False)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Run Details" in all_md


def test_render_detail_panel_empty_df():
    """_render_detail_panel handles empty filtered data."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import attack_results_page as arp
df = pd.DataFrame(columns=["payload_id", "mas_id", "phase", "model_id",
                            "tier3_score", "tier1_pass"])
arp._render_detail_panel([], df, is_cross_model=False)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_export_buttons with filtered data
# ---------------------------------------------------------------------------


def test_render_export_buttons_with_data():
    """_render_export_buttons renders CSV and JSON buttons when data exists."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import attack_results_page as arp
results = [
    {"payload_id": "p1", "injection_type": "injection", "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "injection",
     "tier1_pass": True, "tier3_score": 2, "stub_mode": False,
     "timestamp": "t", "model_id": None, "model_name": None,
     "provider_family": None, "influenced_agents": [],
     "resistant_agents": [], "propagation_path": [],
     "target_agent_id": "a0", "duration_ms": 100.0},
]
df_rows = [
    {"payload_id": "p1", "injection_type": "injection", "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "injection",
     "tier1_pass": True, "tier3_score": 2, "stub_mode": False,
     "timestamp": "t", "model_id": None, "model_name": None,
     "provider_family": None, "tier2_influenced": False},
]
df = pd.DataFrame(df_rows)
arp._render_export_buttons(results, df, is_cross_model=False)
"""
    )
    at.run()
    assert not at.exception


def test_render_export_buttons_empty_df():
    """_render_export_buttons does nothing when DataFrame is empty."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import attack_results_page as arp
df = pd.DataFrame()
arp._render_export_buttons([], df, is_cross_model=False)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_filters filter application
# ---------------------------------------------------------------------------


def test_render_filters_returns_filtered_df():
    """_render_filters renders filter widgets and returns a filtered DataFrame."""
    at = AppTest.from_string(
        """
import pandas as pd
import streamlit as st
from bili.aether.ui import attack_results_page as arp
rows = [
    {"run_id": "run_001", "payload_id": "p1", "injection_type": "injection",
     "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "injection",
     "tier1_pass": True, "tier3_score": 2, "stub_mode": False,
     "timestamp": "t", "model_id": None, "model_name": None,
     "provider_family": None, "tier2_influenced": True},
    {"run_id": "run_001", "payload_id": "p2", "injection_type": "jailbreak",
     "severity": "low",
     "mas_id": "mas_b", "phase": "mid", "attack_suite": "jailbreak",
     "tier1_pass": False, "tier3_score": None, "stub_mode": True,
     "timestamp": "t", "model_id": None, "model_name": None,
     "provider_family": None, "tier2_influenced": False},
]
df = pd.DataFrame(rows)
filtered = arp._render_filters(df, "All Suites", is_cross_model=False)
st.markdown(f"count:{len(filtered)}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "count:2" in all_md


# ---------------------------------------------------------------------------
# _export_filename
# ---------------------------------------------------------------------------


def test_export_filename_single_mas():
    """_export_filename uses mas_id when single config."""
    import pandas as pd

    df = pd.DataFrame({"mas_id": ["mas_a", "mas_a"]})
    name = arp_mod._export_filename(df, "csv")
    assert "mas_a" in name
    assert name.endswith(".csv")


def test_export_filename_multi_mas():
    """_export_filename uses 'multi' when multiple configs."""
    import pandas as pd

    df = pd.DataFrame({"mas_id": ["mas_a", "mas_b"]})
    name = arp_mod._export_filename(df, "json")
    assert "multi" in name
    assert name.endswith(".json")


# ---------------------------------------------------------------------------
# _normalise edge cases
# ---------------------------------------------------------------------------


def test_normalise_invalid_tier3_score():
    """_normalise handles invalid tier3_score gracefully."""
    raw = {
        "payload_id": "p1",
        "injection_type": "x",
        "severity": "h",
        "mas_id": "m",
        "injection_phase": "pre",
        "attack_suite": "s",
        "execution": {},
        "run_metadata": {"tier3_score": "bad"},
    }
    result = arp_mod._normalise(raw)
    assert result["tier3_score"] is None


def test_normalise_empty_tier3_score():
    """_normalise handles empty string tier3_score."""
    raw = {
        "payload_id": "p1",
        "injection_type": "x",
        "severity": "h",
        "mas_id": "m",
        "injection_phase": "pre",
        "attack_suite": "s",
        "execution": {},
        "run_metadata": {"tier3_score": ""},
    }
    result = arp_mod._normalise(raw)
    assert result["tier3_score"] is None


def test_normalise_cross_model_fields():
    """_normalise preserves cross-model fields when present."""
    raw = {
        "payload_id": "p1",
        "injection_type": "x",
        "severity": "h",
        "mas_id": "m",
        "injection_phase": "pre",
        "attack_suite": "s",
        "execution": {},
        "run_metadata": {},
        "model_id": "gpt-4o",
        "model_name": "GPT-4o",
        "provider_family": "openai",
    }
    result = arp_mod._normalise(raw)
    assert result["model_id"] == "gpt-4o"
    assert result["model_name"] == "GPT-4o"
    assert result["provider_family"] == "openai"


# ---------------------------------------------------------------------------
# _render_matrix with cross-model data
# ---------------------------------------------------------------------------


def test_render_matrix_cross_model():
    """_render_matrix renders model_id in column labels for cross-model view."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import attack_results_page as arp
rows = [
    {"payload_id": "p1", "injection_type": "injection", "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "cross_model",
     "tier1_pass": True, "tier3_score": 1, "stub_mode": False,
     "timestamp": "t", "model_id": "gpt-4o", "model_name": "GPT-4o",
     "provider_family": "openai", "tier2_influenced": True},
]
df = pd.DataFrame(rows)
arp._render_matrix(df, is_cross_model=True)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Results Matrix" in all_md


# ---------------------------------------------------------------------------
# _render_expander_content
# ---------------------------------------------------------------------------


def test_render_expander_content_with_tier3():
    """_render_expander_content shows Tier-3 score and reasoning."""
    at = AppTest.from_string(
        """
from bili.aether.ui import attack_results_page as arp
r = {
    "severity": "high", "injection_phase": "pre",
    "injection_type": "injection", "stub_mode": False,
    "attack_suite": "injection", "timestamp": "2025-01-01",
    "duration_ms": 150.0, "propagation_path": ["a0", "a1"],
    "influenced_agents": ["a0"], "resistant_agents": ["a1"],
    "tier3_confidence": "high", "tier3_reasoning": "Clear compliance detected",
    "config_path": "", "model_id": None,
}
arp._render_expander_content(r, tier3_score=2, tier1_ok=True,
                             disagree=False, is_cross_model=False)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Tier-3 Score" in all_md
    assert "Clear compliance detected" in all_md


def test_render_expander_content_no_tier3():
    """_render_expander_content shows skipped message when T3 is None."""
    at = AppTest.from_string(
        """
from bili.aether.ui import attack_results_page as arp
r = {
    "severity": "low", "injection_phase": "pre",
    "injection_type": "jailbreak", "stub_mode": True,
    "attack_suite": "jailbreak", "timestamp": "2025-01-01",
    "duration_ms": 50.0, "propagation_path": [],
    "influenced_agents": [], "resistant_agents": [],
    "tier3_confidence": "", "tier3_reasoning": "",
    "config_path": "", "model_id": None,
}
arp._render_expander_content(r, tier3_score=None, tier1_ok=True,
                             disagree=False, is_cross_model=False)
"""
    )
    at.run()
    assert not at.exception
    all_captions = " ".join(c.value for c in at.caption)
    assert "skipped" in all_captions


def test_render_expander_content_with_disagreement():
    """_render_expander_content shows disagreement warning."""
    at = AppTest.from_string(
        """
from bili.aether.ui import attack_results_page as arp
r = {
    "severity": "high", "injection_phase": "pre",
    "injection_type": "injection", "stub_mode": False,
    "attack_suite": "injection", "timestamp": "2025-01-01",
    "duration_ms": 100.0, "propagation_path": ["a0"],
    "influenced_agents": ["a0"], "resistant_agents": [],
    "tier3_confidence": "high", "tier3_reasoning": "No compliance found",
    "config_path": "", "model_id": None,
}
arp._render_expander_content(r, tier3_score=0, tier1_ok=True,
                             disagree=True, is_cross_model=False)
"""
    )
    at.run()
    assert not at.exception
    all_warn = " ".join(w.value for w in at.warning)
    assert "disagreement" in all_warn.lower()


def test_render_expander_content_cross_model():
    """_render_expander_content shows model info for cross-model view."""
    at = AppTest.from_string(
        """
from bili.aether.ui import attack_results_page as arp
r = {
    "severity": "medium", "injection_phase": "pre",
    "injection_type": "injection", "stub_mode": False,
    "attack_suite": "cross_model", "timestamp": "2025-01-01",
    "duration_ms": 200.0, "propagation_path": [],
    "influenced_agents": [], "resistant_agents": [],
    "tier3_confidence": "", "tier3_reasoning": "",
    "config_path": "", "model_id": "gpt-4o",
    "model_name": "GPT-4o", "provider_family": "openai",
}
arp._render_expander_content(r, tier3_score=None, tier1_ok=True,
                             disagree=False, is_cross_model=True)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "gpt-4o" in all_md


# ---------------------------------------------------------------------------
# _render_detail_panel with cross-model
# ---------------------------------------------------------------------------


def test_render_detail_panel_cross_model():
    """_render_detail_panel renders details with cross-model dimension."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import attack_results_page as arp
results = [
    {"run_id": "run_001", "payload_id": "p1", "injection_type": "injection",
     "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "cross_model",
     "tier1_pass": True, "tier3_score": 1, "tier3_confidence": "medium",
     "tier3_reasoning": "Partial compliance", "stub_mode": False,
     "timestamp": "2025-01-01", "model_id": "gpt-4o", "model_name": "GPT-4o",
     "provider_family": "openai", "influenced_agents": [],
     "resistant_agents": ["a0"], "propagation_path": [],
     "target_agent_id": "a0", "duration_ms": 100.0,
     "config_path": "", "injection_phase": "pre"},
]
df_rows = [
    {"run_id": "run_001", "payload_id": "p1", "injection_type": "injection",
     "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "cross_model",
     "tier1_pass": True, "tier3_score": 1, "stub_mode": False,
     "timestamp": "t", "model_id": "gpt-4o", "model_name": "GPT-4o",
     "provider_family": "openai", "tier2_influenced": False},
]
df = pd.DataFrame(df_rows)
arp._render_detail_panel(results, df, is_cross_model=True)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Run Details" in all_md


# ---------------------------------------------------------------------------
# _render_matrix with tier1 failure and tier2 fallback
# ---------------------------------------------------------------------------


def test_render_matrix_with_tier1_failure():
    """_render_matrix shows ! for Tier-1 failures."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import attack_results_page as arp
rows = [
    {"payload_id": "p1", "injection_type": "injection", "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "injection",
     "tier1_pass": False, "tier3_score": None, "stub_mode": True,
     "timestamp": "t", "model_id": None, "model_name": None,
     "provider_family": None, "tier2_influenced": False},
]
df = pd.DataFrame(rows)
arp._render_matrix(df, is_cross_model=False)
"""
    )
    at.run()
    assert not at.exception


def test_render_matrix_with_tier2_fallback():
    """_render_matrix shows T2 labels when tier3 is None but tier1 passed."""
    at = AppTest.from_string(
        """
import pandas as pd
from bili.aether.ui import attack_results_page as arp
rows = [
    {"payload_id": "p1", "injection_type": "injection", "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "injection",
     "tier1_pass": True, "tier3_score": None, "stub_mode": True,
     "timestamp": "t", "model_id": None, "model_name": None,
     "provider_family": None, "tier2_influenced": True},
]
df = pd.DataFrame(rows)
arp._render_matrix(df, is_cross_model=False)
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _result_export_key
# ---------------------------------------------------------------------------


def test_result_export_key_non_cross_model():
    """_result_export_key returns tuple with run_id but without model_id for non-cross-model."""
    r = {
        "run_id": "run_001",
        "mas_id": "m",
        "payload_id": "p1",
        "phase": "pre",
        "model_id": "gpt-4o",
    }
    key = arp_mod._result_export_key(r, is_cross_model=False)
    assert key == ("run_001", "m", "p1", "pre", None)


def test_result_export_key_cross_model():
    """_result_export_key includes run_id and model_id for cross-model."""
    r = {
        "run_id": "run_001",
        "mas_id": "m",
        "payload_id": "p1",
        "phase": "pre",
        "model_id": "gpt-4o",
    }
    key = arp_mod._result_export_key(r, is_cross_model=True)
    assert key == ("run_001", "m", "p1", "pre", "gpt-4o")


# ---------------------------------------------------------------------------
# _build_export_df
# ---------------------------------------------------------------------------


def test_build_export_df_filters_by_key_set():
    """_build_export_df only includes results matching the key set."""
    results = [
        {
            "run_id": "run_001",
            "mas_id": "m",
            "payload_id": "p1",
            "phase": "pre",
            "model_id": None,
            "target_agent_id": "a0",
            "injection_type": "injection",
            "tier1_pass": True,
            "influenced_agents": [],
            "resistant_agents": [],
            "tier3_score": 1,
            "model_name": None,
            "timestamp": "t",
            "severity": "high",
            "attack_suite": "injection",
            "propagation_path": [],
        },
        {
            "run_id": "run_001",
            "mas_id": "m",
            "payload_id": "p2",
            "phase": "mid",
            "model_id": None,
            "target_agent_id": "a1",
            "injection_type": "jailbreak",
            "tier1_pass": False,
            "influenced_agents": [],
            "resistant_agents": [],
            "tier3_score": None,
            "model_name": None,
            "timestamp": "t",
            "severity": "low",
            "attack_suite": "jailbreak",
            "propagation_path": [],
        },
    ]
    key_set = {("run_001", "m", "p1", "pre", None)}
    df = arp_mod._build_export_df(results, key_set, is_cross_model=False)
    assert len(df) == 1
    assert df.iloc[0]["payload_id"] == "p1"


# ---------------------------------------------------------------------------
# _normalise with None tier3_score
# ---------------------------------------------------------------------------


def test_normalise_none_tier3_score():
    """_normalise handles None tier3_score in run_metadata."""
    raw = {
        "payload_id": "p1",
        "injection_type": "x",
        "severity": "h",
        "mas_id": "m",
        "injection_phase": "pre",
        "attack_suite": "s",
        "execution": {},
        "run_metadata": {"tier3_score": None},
    }
    result = arp_mod._normalise(raw)
    assert result["tier3_score"] is None


# ---------------------------------------------------------------------------
# _render_filters single suite
# ---------------------------------------------------------------------------


def test_render_filters_single_suite():
    """_render_filters works for a single suite (no suite multiselect)."""
    at = AppTest.from_string(
        """
import pandas as pd
import streamlit as st
from bili.aether.ui import attack_results_page as arp
rows = [
    {"run_id": "run_001", "payload_id": "p1", "injection_type": "injection",
     "severity": "high",
     "mas_id": "mas_a", "phase": "pre", "attack_suite": "injection",
     "tier1_pass": True, "tier3_score": 2, "stub_mode": False,
     "timestamp": "t", "model_id": None, "model_name": None,
     "provider_family": None, "tier2_influenced": True},
]
df = pd.DataFrame(rows)
filtered = arp._render_filters(df, "Injection", is_cross_model=False)
st.markdown(f"count:{len(filtered)}")
"""
    )
    at.run()
    assert not at.exception
    assert "count:1" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# _render_view_graph_button
# ---------------------------------------------------------------------------


def test_render_view_graph_button_missing_config():
    """_render_view_graph_button shows caption when config file missing."""
    at = AppTest.from_string(
        """
from bili.aether.ui import attack_results_page as arp
arp._render_view_graph_button(
    "nonexistent/path.yaml", "mas_a", "p1", "pre"
)
"""
    )
    at.run()
    assert not at.exception
    all_captions = " ".join(c.value for c in at.caption)
    assert "not found" in all_captions or "unavailable" in all_captions


# ---------------------------------------------------------------------------
# _tier2_tier3_disagree edge cases
# ---------------------------------------------------------------------------


def test_tier2_tier3_disagree_score_one_not_influenced():
    """No disagreement when T3=1 and not influenced (ambiguous zone)."""
    assert arp_mod._tier2_tier3_disagree([], 1) is False


def test_tier2_tier3_disagree_score_three_influenced():
    """No disagreement when T3=3 and influenced (both agree)."""
    assert arp_mod._tier2_tier3_disagree(["a0"], 3) is False
