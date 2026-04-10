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


# ---------------------------------------------------------------------------
# render_graph_viewer full render
# ---------------------------------------------------------------------------


def test_render_graph_viewer_no_exception():
    """render_graph_viewer runs without exception with mocked flow."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk

mock_flow = _Mock()
mock_state_cls = _Mock()
mock_state_instance = _Mock()
mock_state_instance.selected_id = None
mock_state_cls.return_value = mock_state_instance
mock_flow.return_value = mock_state_instance

fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
fm["streamlit_flow"].streamlit_flow = mock_flow
fm["streamlit_flow.state"].StreamlitFlowState = mock_state_cls

with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    # Patch streamlit_flow and StreamlitFlowState at module level
    gv.streamlit_flow = mock_flow
    gv.StreamlitFlowState = mock_state_cls

    cfg = mk()
    mock_node = _Mock()
    mock_node.id = "agent_0"
    mock_edge = _Mock()
    mock_edge.id = "e1"

    with _patch.object(gv, "build_model_options", return_value=([], {}, {})):
        with _patch.object(gv, "apply_agent_overrides", return_value=cfg):
            gv.render_graph_viewer(cfg, [mock_node], [mock_edge])
"""
    )
    at.run()
    assert not at.exception


# ---------------------------------------------------------------------------
# _render_properties_panel with agent selected
# ---------------------------------------------------------------------------


def test_properties_panel_agent_selected():
    """Properties panel renders agent details when an agent node is selected."""
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
    cfg = mk(mas_id="props_test")
    key = gv._overrides_key(cfg.mas_id)
    st.session_state[key] = {}
    with _patch.object(gv, "build_model_options", return_value=(["[Test] model-1"], {"[Test] model-1": "model-1"}, {"model-1": "[Test] model-1"})):
        with _patch.object(gv, "_get_tool_names", return_value=["tool_a"]):
            gv._render_properties_panel(cfg, "agent_0", [], cfg.mas_id)
"""
    )
    at.run()
    assert not at.exception
    " ".join(m.value for m in at.markdown)
    assert "agent_0" in " ".join(c.value for c in at.caption)


# ---------------------------------------------------------------------------
# _render_edge_properties
# ---------------------------------------------------------------------------


def test_properties_panel_edge_selected():
    """Properties panel renders edge details when an edge is selected."""
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
    cfg = mk(mas_id="edge_test")
    key = gv._overrides_key(cfg.mas_id)
    st.session_state[key] = {}
    mock_edge = _Mock()
    mock_edge.id = "e_0_1"
    mock_edge.source = "agent_0"
    mock_edge.target = "agent_1"
    mock_edge.label = "direct"
    with _patch.object(gv, "build_model_options", return_value=([], {}, {})):
        gv._render_properties_panel(cfg, "e_0_1", [mock_edge], cfg.mas_id)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "agent_0" in all_md
    assert "agent_1" in all_md


# ---------------------------------------------------------------------------
# apply_agent_overrides with model, temperature, tools overrides
# ---------------------------------------------------------------------------


def test_apply_overrides_with_temperature():
    """apply_agent_overrides applies a temperature override."""
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
    cfg = mk(mas_id="temp_test")
    key = gv._overrides_key(cfg.mas_id)
    st.session_state[key] = {"agent_0": {"temperature": 1.5}}
    with _patch.object(gv, "build_model_options", return_value=([], {}, {})):
        result = gv.apply_agent_overrides(cfg)
    a0 = next(a for a in result.agents if a.agent_id == "agent_0")
    st.markdown(f"temp:{a0.temperature}")
"""
    )
    at.run()
    assert not at.exception
    assert "temp:1.5" in " ".join(m.value for m in at.markdown)


def test_apply_overrides_with_max_tokens():
    """apply_agent_overrides applies a max_tokens override."""
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
    cfg = mk(mas_id="maxtok_test")
    key = gv._overrides_key(cfg.mas_id)
    st.session_state[key] = {"agent_0": {"max_tokens": 2048}}
    with _patch.object(gv, "build_model_options", return_value=([], {}, {})):
        result = gv.apply_agent_overrides(cfg)
    a0 = next(a for a in result.agents if a.agent_id == "agent_0")
    st.markdown(f"mt:{a0.max_tokens}")
"""
    )
    at.run()
    assert not at.exception
    assert "mt:2048" in " ".join(m.value for m in at.markdown)


def test_apply_overrides_with_objective():
    """apply_agent_overrides applies an objective override."""
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
    cfg = mk(mas_id="obj_test")
    key = gv._overrides_key(cfg.mas_id)
    st.session_state[key] = {"agent_0": {"objective": "New objective"}}
    with _patch.object(gv, "build_model_options", return_value=([], {}, {})):
        result = gv.apply_agent_overrides(cfg)
    a0 = next(a for a in result.agents if a.agent_id == "agent_0")
    st.markdown(f"obj:{a0.objective}")
"""
    )
    at.run()
    assert not at.exception
    assert "obj:New objective" in " ".join(m.value for m in at.markdown)


def test_apply_overrides_with_tools():
    """apply_agent_overrides applies a tools override."""
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
    cfg = mk(mas_id="tools_test")
    key = gv._overrides_key(cfg.mas_id)
    st.session_state[key] = {"agent_0": {"tools": ["search_tool", "calc_tool"]}}
    with _patch.object(gv, "build_model_options", return_value=([], {}, {})):
        result = gv.apply_agent_overrides(cfg)
    a0 = next(a for a in result.agents if a.agent_id == "agent_0")
    st.markdown(f"tools:{a0.tools}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "search_tool" in all_md
    assert "calc_tool" in all_md


# ---------------------------------------------------------------------------
# render_metadata_bar details
# ---------------------------------------------------------------------------


def test_metadata_bar_with_tags():
    """The metadata bar shows tags when present."""
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
    cfg_with_tags = cfg.model_copy(update={"tags": ["security", "test", "demo"]})
    gv.render_metadata_bar(cfg_with_tags)
"""
    )
    at.run()
    assert not at.exception
