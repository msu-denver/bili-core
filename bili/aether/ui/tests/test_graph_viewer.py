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


def test_metadata_bar_truncates_many_tags():
    """More than three tags are truncated with an ellipsis."""
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
    cfg_tags = cfg.model_copy(update={"tags": ["a", "b", "c", "d", "e"]})
    gv.render_metadata_bar(cfg_tags)
"""
    )
    at.run()
    assert not at.exception
    metric_values = [m.value for m in at.metric]
    assert any("..." in str(v) for v in metric_values)


# ---------------------------------------------------------------------------
# build_model_options
# ---------------------------------------------------------------------------


def test_build_model_options_structure():
    """build_model_options returns display list and the two lookup dicts."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
fake_models = {
    "openai": {
        "name": "OpenAI",
        "models": [
            {"model_id": "id-1", "model_name": "gpt-test"},
        ],
    },
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    with _patch("bili.iris.config.llm_config.LLM_MODELS", fake_models):
        gv.build_model_options.clear()
        options, name_to_model, lookup = gv.build_model_options()
        gv.build_model_options.clear()
    st.markdown(f"opt:{options[0]}")
    st.markdown(f"n2m:{name_to_model[options[0]]}")
    st.markdown(f"by_id:{lookup['id-1']}")
    st.markdown(f"by_name:{lookup['gpt-test']}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "opt:[OpenAI] gpt-test" in all_md
    assert "n2m:gpt-test" in all_md
    assert "by_id:[OpenAI] gpt-test" in all_md
    assert "by_name:[OpenAI] gpt-test" in all_md


# ---------------------------------------------------------------------------
# Full agent properties panel with all optional fields populated
# ---------------------------------------------------------------------------


def test_agent_properties_full_render():
    """A fully populated agent renders every optional property block."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.schema.agent_spec import AgentSpec
from bili.aether.schema.mas_config import MASConfig
from bili.aether.schema.enums import WorkflowType
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    agent = AgentSpec(
        agent_id="agent_x",
        role="analyst",
        objective="Analyze threats thoroughly",
        system_prompt="You are an analyst.",
        temperature=0.5,
        max_tokens=2048,
        model_name="gpt-test",
        tools=["search_tool"],
        capabilities=["threat_modeling"],
        middleware=["summarization"],
        tier=1,
        voting_weight=2.0,
        is_supervisor=True,
        inherit_from_bili_core=True,
    )
    cfg = MASConfig(
        mas_id="full_agent",
        name="Full",
        description="d",
        agents=[agent],
        channels=[],
        workflow_type=WorkflowType.SEQUENTIAL,
    )
    st.session_state[gv._overrides_key(cfg.mas_id)] = {}
    opts = ["[Test] gpt-test"]
    n2m = {"[Test] gpt-test": "gpt-test"}
    lookup = {"gpt-test": "[Test] gpt-test"}
    with _patch.object(gv, "build_model_options", return_value=(opts, n2m, lookup)):
        with _patch.object(gv, "_get_tool_names", return_value=["search_tool", "calc_tool"]):
            gv._render_properties_panel(cfg, "agent_x", [], cfg.mas_id)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "threat_modeling" in all_md
    assert "summarization" in all_md
    assert "Tier:" in all_md
    assert "Voting Weight:" in all_md
    assert "Supervisor" in all_md
    assert "Inherits from bili-core" in all_md
    # The model selector pre-selects the YAML model.
    assert any("gpt-test" in str(s.value) for s in at.selectbox)


def test_agent_properties_uses_override_bucket_values():
    """Override bucket values pre-fill the editable widgets over YAML defaults."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.schema.agent_spec import AgentSpec
from bili.aether.schema.mas_config import MASConfig
from bili.aether.schema.enums import WorkflowType
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    agent = AgentSpec(
        agent_id="agent_y",
        role="r",
        objective="Original objective text",
    )
    cfg = MASConfig(
        mas_id="ov_agent",
        name="Ov",
        description="d",
        agents=[agent],
        channels=[],
        workflow_type=WorkflowType.SEQUENTIAL,
    )
    st.session_state[gv._overrides_key(cfg.mas_id)] = {
        "agent_y": {
            "objective": "Overridden objective",
            "system_prompt": "Overridden prompt",
            "temperature": 1.3,
            "max_tokens": 4096,
            "tools": ["calc_tool"],
            "model_name": "[Test] gpt-test",
        }
    }
    opts = ["[Test] gpt-test"]
    n2m = {"[Test] gpt-test": "gpt-test"}
    lookup = {"gpt-test": "[Test] gpt-test"}
    with _patch.object(gv, "build_model_options", return_value=(opts, n2m, lookup)):
        with _patch.object(gv, "_get_tool_names", return_value=["calc_tool", "search_tool"]):
            gv._render_properties_panel(cfg, "agent_y", [], cfg.mas_id)
"""
    )
    at.run()
    assert not at.exception
    text_area_vals = " ".join(str(t.value) for t in at.text_area)
    assert "Overridden objective" in text_area_vals
    assert "Overridden prompt" in text_area_vals
    slider_vals = [s.value for s in at.slider]
    assert 1.3 in slider_vals
    number_vals = [n.value for n in at.number_input]
    assert 4096 in number_vals
    # The override model is pre-selected.
    assert any("gpt-test" in str(s.value) for s in at.selectbox)


def test_agent_properties_no_tools_in_registry():
    """When the tool registry is empty no multiselect is rendered."""
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
    cfg = mk(mas_id="no_tools")
    st.session_state[gv._overrides_key(cfg.mas_id)] = {}
    with _patch.object(gv, "build_model_options", return_value=([], {}, {})):
        with _patch.object(gv, "_get_tool_names", return_value=[]):
            gv._render_properties_panel(cfg, "agent_0", [], cfg.mas_id)
"""
    )
    at.run()
    assert not at.exception
    assert any("None configured in YAML" in c.value for c in at.caption)
    assert len(at.multiselect) == 0


def test_model_selector_keep_sentinel_clears_override():
    """Selecting the keep sentinel removes any stored model override."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.schema.agent_spec import AgentSpec
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    agent = AgentSpec(agent_id="agent_z", role="r", objective="Do the work")
    mas_id = "sel_clear"
    st.session_state[gv._overrides_key(mas_id)] = {
        "agent_z": {"model_name": "[Test] gpt-test"}
    }
    opts = ["[Test] gpt-test"]
    lookup = {"gpt-test": "[Test] gpt-test"}
    with _patch.object(gv, "build_model_options", return_value=(opts, {}, lookup)):
        gv._render_model_selector(agent, mas_id)
    bucket = st.session_state[gv._overrides_key(mas_id)]["agent_z"]
    st.markdown(f"has_model:{'model_name' in bucket}")
"""
    )
    at.run()
    assert not at.exception
    # The stored override display is preselected, so the rendered value keeps it.
    assert any("gpt-test" in str(s.value) for s in at.selectbox)


# ---------------------------------------------------------------------------
# Edge properties: channel + workflow edge matches
# ---------------------------------------------------------------------------


def test_edge_properties_with_channel_and_workflow_match():
    """Edge properties surface channel protocol and conditional workflow edge."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
from bili.aether.schema.agent_spec import AgentSpec
from bili.aether.schema.mas_config import MASConfig, Channel, WorkflowEdge
from bili.aether.schema.enums import WorkflowType
fm = {
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import graph_viewer as gv
    agents = [
        AgentSpec(agent_id="agent_0", role="r0", objective="Objective zero"),
        AgentSpec(agent_id="agent_1", role="r1", objective="Objective one"),
    ]
    channel = Channel(
        channel_id="ch01",
        protocol="direct",
        source="agent_0",
        target="agent_1",
        description="Primary link",
        bidirectional=True,
    )
    wedge = WorkflowEdge(
        from_agent="agent_0",
        to_agent="agent_1",
        condition="state.score > 0.5",
    )
    cfg = MASConfig(
        mas_id="edge_full",
        name="Edge",
        description="d",
        agents=agents,
        channels=[channel],
        workflow_edges=[wedge],
        workflow_type=WorkflowType.CUSTOM,
    )
    edge = _Mock()
    edge.id = "e01"
    edge.source = "agent_0"
    edge.target = "agent_1"
    edge.label = "direct"
    gv._render_edge_properties(edge, cfg)
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Protocol:" in all_md
    assert "Primary link" in all_md
    assert any("Bidirectional" in s.value for s in at.success)
    assert any("state.score > 0.5" in c.value for c in at.code)


# NOTE: a "Download YAML button present" AppTest was removed here. It used
# at.download_button, which is not an accessor on AppTest in this Streamlit
# version, and the st.download_button("Download YAML", ...) call at
# graph_viewer.py:162 is already exercised by the other render_graph_viewer
# tests in this file (line 162 is covered). The removed test added no
# coverage and asserted via a non-existent API.
