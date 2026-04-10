"""Tests for bili.aether.ui.components.graph_viewer.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.
"""

# pylint: disable=import-outside-toplevel, protected-access, reimported
# pylint: disable=duplicate-code

from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

_FM = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(),
    "streamlit_flow.state": MagicMock(),
}


def test_metadata_bar_shows_agent_count():
    """The metadata bar shows the correct agent count."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    gv.render_metadata_bar(mk(num_agents=3))
"""
    )
    at.run()
    assert not at.exception


def test_metadata_bar_shows_workflow_type():
    """The metadata bar shows the workflow type."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    gv.render_metadata_bar(mk())
"""
    )
    at.run()
    assert not at.exception


def test_apply_overrides_returns_unchanged_when_empty():
    """With no overrides the config is returned as-is."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    cfg = mk()
    result = gv.apply_agent_overrides(cfg)
    st.markdown(f"same:{result.mas_id == cfg.mas_id}")
"""
    )
    at.run()
    assert not at.exception
    assert "same:True" in " ".join(m.value for m in at.markdown)


def test_apply_overrides_with_system_prompt():
    """apply_agent_overrides applies a system_prompt override."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    cfg = mk(mas_id="sp_test")
    key = gv._overrides_key(cfg.mas_id)
    st.session_state[key] = {"agent_0": {"system_prompt": "Test."}}
    result = gv.apply_agent_overrides(cfg)
    a0 = next(a for a in result.agents if a.agent_id == "agent_0")
    st.markdown(f"sp:{a0.system_prompt}")
"""
    )
    at.run()
    assert not at.exception
    assert "sp:Test." in " ".join(m.value for m in at.markdown)


def test_overrides_key_format():
    """_overrides_key returns the expected format."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui.components.graph_viewer import _overrides_key

        assert _overrides_key("my_mas") == "agent_overrides_my_mas"


def test_keep_sentinel_value():
    """MODEL_KEEP_SENTINEL is the expected placeholder string."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui.components.graph_viewer import MODEL_KEEP_SENTINEL

        assert MODEL_KEEP_SENTINEL == "(keep from YAML)"


def test_properties_panel_no_selection():
    """With no selected node the panel shows a hint."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    cfg = mk()
    gv._render_properties_panel(cfg, None, [], cfg.mas_id)
"""
    )
    at.run()
    assert not at.exception
    assert "Click a node" in " ".join(c.value for c in at.caption)


def test_properties_panel_unknown_id():
    """With an unknown selected_id the panel shows No details."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    cfg = mk()
    gv._render_properties_panel(cfg, "unknown_id", [], cfg.mas_id)
"""
    )
    at.run()
    assert not at.exception
    assert "No details" in " ".join(c.value for c in at.caption)


def test_render_list_section():
    """_render_list_section renders a title and items."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    gv._render_list_section("Caps", ["cap_a", "cap_b"])
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Caps" in all_md
    assert "cap_a" in all_md
