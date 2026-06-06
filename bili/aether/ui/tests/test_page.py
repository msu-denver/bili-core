"""Tests for bili.aether.ui.page -- AETHER page rendering.

Covers render_aether_page() including the Visualizer/Chat radio toggle,
sidebar branding, the empty-config info message, and callbacks.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.  All
Streamlit-dependent imports therefore live inside ``AppTest.from_string``
scripts which execute within a proper Streamlit runtime context.
"""

# pylint: disable=import-outside-toplevel, protected-access

from streamlit.testing.v1 import AppTest


def test_visualizer_shows_info_when_no_config():
    """Without a config the visualizer shows an info message."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    with patch.object(pm, "EXAMPLES_DIR") as md:
        md.exists.return_value = True
        md.glob.return_value = []
        with patch.object(pm, "LOGO_PATH") as lp:
            lp.exists.return_value = False
            pm._render_visualizer_main()
"""
    )
    at.run()
    assert not at.exception
    assert any("Select a YAML" in m.value for m in at.info)


def test_intro_renders_aether_heading():
    """The intro section renders the AETHER heading."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    with patch.object(pm, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        pm._render_intro()
"""
    )
    at.run()
    assert not at.exception
    assert "AETHER" in " ".join(m.value for m in at.markdown)


def test_intro_mentions_workflow_patterns():
    """The intro describes the seven workflow patterns."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    with patch.object(pm, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        pm._render_intro()
"""
    )
    at.run()
    assert not at.exception
    assert "Sequential chains" in " ".join(m.value for m in at.markdown)


def test_intro_mentions_github_link():
    """The intro section includes a link to GitHub."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    with patch.object(pm, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        pm._render_intro()
"""
    )
    at.run()
    assert not at.exception
    assert "BiliCore on GitHub" in " ".join(m.value for m in at.markdown)


def test_legend_renders_without_error():
    """The legend expander renders without error."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    pm._render_legend()
"""
    )
    at.run()
    assert not at.exception


def test_sidebar_renders_aether_heading():
    """The sidebar contains the AETHER heading text."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with st.sidebar:
    with patch.dict("sys.modules", fm):
        from bili.aether.ui import page as pm
        with patch.object(pm, "LOGO_PATH") as lp:
            lp.exists.return_value = False
            with patch.object(pm, "EXAMPLES_DIR") as ed:
                ed.exists.return_value = True
                ed.glob.return_value = []
                pm._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    assert "AETHER" in " ".join(m.value for m in at.sidebar.markdown)


def test_sidebar_caption_shows_acronym():
    """The sidebar caption shows the full AETHER acronym."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with st.sidebar:
    with patch.dict("sys.modules", fm):
        from bili.aether.ui import page as pm
        with patch.object(pm, "LOGO_PATH") as lp:
            lp.exists.return_value = False
            with patch.object(pm, "EXAMPLES_DIR") as ed:
                ed.exists.return_value = True
                ed.glob.return_value = []
                pm._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    assert "Evaluation" in " ".join(c.value for c in at.sidebar.caption)


def test_sidebar_has_radio_toggle():
    """The sidebar contains a Visualizer/Chat radio toggle."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with st.sidebar:
    with patch.dict("sys.modules", fm):
        from bili.aether.ui import page as pm
        with patch.object(pm, "LOGO_PATH") as lp:
            lp.exists.return_value = False
            with patch.object(pm, "EXAMPLES_DIR") as ed:
                ed.exists.return_value = True
                ed.glob.return_value = []
                pm._render_sidebar()
"""
    )
    at.run()
    assert not at.exception
    assert len(at.sidebar.radio) >= 1


def test_render_aether_page_no_exception():
    """The full render_aether_page runs without exception."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    with patch.object(pm, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        with patch.object(pm, "EXAMPLES_DIR") as ed:
            ed.exists.return_value = True
            ed.glob.return_value = []
            pm.render_aether_page()
"""
    )
    at.run()
    assert not at.exception


def test_on_send_to_chat_noop_without_config():
    """The send-to-chat callback is a no-op when no config exists."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    pm._on_send_to_chat()
st.markdown(f"no_uploads:{'chat_uploaded_configs' not in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    assert "no_uploads:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# Logo branches (lines 75, 129)
# ---------------------------------------------------------------------------


def test_sidebar_renders_logo_when_present():
    """_render_sidebar calls st.image when the logo file exists."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with st.sidebar:
    with patch.dict("sys.modules", fm):
        from bili.aether.ui import page as pm
        with patch.object(pm, "LOGO_PATH") as lp:
            lp.exists.return_value = True
            lp.__str__ = lambda self: "/fake/logo.png"
            with patch.object(pm, "EXAMPLES_DIR") as ed:
                ed.exists.return_value = True
                ed.glob.return_value = []
                with patch("streamlit.image") as img:
                    pm._render_sidebar()
                    st.markdown(f"image_called:{img.called}")
"""
    )
    at.run()
    assert not at.exception
    assert "image_called:True" in " ".join(m.value for m in at.sidebar.markdown)


def test_intro_renders_logo_when_present():
    """_render_intro calls st.image when the logo file exists."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    with patch.object(pm, "LOGO_PATH") as lp:
        lp.exists.return_value = True
        lp.__str__ = lambda self: "/fake/logo.png"
        with patch("streamlit.image") as img:
            pm._render_intro()
            st.markdown(f"image_called:{img.called}")
"""
    )
    at.run()
    assert not at.exception
    assert "image_called:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# Chat-page branches (lines 64, 93)
# ---------------------------------------------------------------------------


def test_render_aether_page_chat_branch():
    """render_aether_page dispatches to the chat renderer when page is Chat."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    st.session_state["aether_page"] = "Chat"
    with patch.object(pm, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        with patch.object(pm, "render_chat_sidebar_content") as sc:
            with patch.object(pm, "render_chat_main") as cm:
                pm.render_aether_page()
                st.markdown(f"chat_main:{cm.called}")
                st.markdown(f"chat_sidebar:{sc.called}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "chat_main:True" in all_md
    assert "chat_sidebar:True" in all_md


# ---------------------------------------------------------------------------
# Visualizer sidebar: examples dir missing, with yaml files (104-105, 112-123)
# ---------------------------------------------------------------------------


def test_visualizer_sidebar_examples_missing_shows_error():
    """_render_visualizer_sidebar shows an error when EXAMPLES_DIR is absent."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    with patch.object(pm, "EXAMPLES_DIR") as ed:
        ed.exists.return_value = False
        pm._render_visualizer_sidebar()
"""
    )
    at.run()
    assert not at.exception
    assert "not found" in " ".join(e.value for e in at.error)


def test_visualizer_sidebar_with_yaml_files_loads_config(tmp_path):
    """_render_visualizer_sidebar renders a selectbox and loads the chosen config."""
    (tmp_path / "demo_config.yaml").write_text("mas_id: x\n", encoding="utf-8")
    at = AppTest.from_string(
        f"""
from unittest.mock import MagicMock, patch
import streamlit as st
from pathlib import Path
fm = {{
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    with patch.object(pm, "EXAMPLES_DIR", Path({str(tmp_path)!r})):
        with patch.object(pm, "_load_config") as lc:
            pm._render_visualizer_sidebar()
            st.markdown(f"load_called:{{lc.called}}")
"""
    )
    at.run()
    assert not at.exception
    assert len(at.selectbox) >= 1
    assert "load_called:True" in " ".join(m.value for m in at.markdown)


# ---------------------------------------------------------------------------
# Visualizer main with config (lines 222-232)
# ---------------------------------------------------------------------------


def test_visualizer_main_with_config_renders_graph():
    """_render_visualizer_main renders the config name, graph, and legend."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    cfg = mk(mas_id="vis_main_test", name="Vis Main MAS")
    st.session_state["mas_config"] = cfg
    with patch.object(pm, "LOGO_PATH") as lp:
        lp.exists.return_value = False
        with patch.object(pm, "convert_mas_to_graph", return_value=([], [])):
            with patch.object(pm, "render_graph_viewer"):
                with patch.object(pm, "render_metadata_bar"):
                    pm._render_visualizer_main()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Vis Main MAS" in all_md


# ---------------------------------------------------------------------------
# Send-to callbacks with config (lines 246-253, 258-263, 268-273)
# ---------------------------------------------------------------------------


def test_on_send_to_chat_with_config():
    """_on_send_to_chat pushes the config to the chat upload store and switches page."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    cfg = mk(mas_id="send_chat")
    st.session_state["mas_config"] = cfg
    st.session_state["current_yaml_path"] = "my_cfg.yaml"
    with patch.object(pm, "apply_agent_overrides", side_effect=lambda c: c):
        pm._on_send_to_chat()
    st.markdown(f"page:{st.session_state.get('aether_page')}")
    st.markdown(f"autoload:{st.session_state.get('chat_autoload_name')}")
    st.markdown(f"uploaded:{'my_cfg.yaml' in st.session_state.get('chat_uploaded_configs', {})}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "page:Chat" in all_md
    assert "autoload:my_cfg.yaml" in all_md
    assert "uploaded:True" in all_md


def test_on_send_to_baseline_with_config():
    """_on_send_to_baseline pushes the config to the baseline runner state."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    cfg = mk(mas_id="send_baseline")
    st.session_state["mas_config"] = cfg
    st.session_state["current_yaml_path"] = "/p/cfg.yaml"
    with patch.object(pm, "apply_agent_overrides", side_effect=lambda c: c):
        with patch.object(pm, "push_config_to_baseline_state") as push:
            pm._on_send_to_baseline()
            st.markdown(f"pushed:{push.called}")
"""
    )
    at.run()
    assert not at.exception
    assert "pushed:True" in " ".join(m.value for m in at.markdown)


def test_on_send_to_attack_with_config():
    """_on_send_to_attack pushes the config to the attack page state."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    cfg = mk(mas_id="send_attack")
    st.session_state["mas_config"] = cfg
    st.session_state["current_yaml_path"] = "/p/cfg.yaml"
    with patch.object(pm, "apply_agent_overrides", side_effect=lambda c: c):
        with patch.object(pm, "push_config_to_attack_state") as push:
            pm._on_send_to_attack()
            st.markdown(f"pushed:{push.called}")
"""
    )
    at.run()
    assert not at.exception
    assert "pushed:True" in " ".join(m.value for m in at.markdown)


def test_send_callbacks_noop_without_config():
    """The baseline and attack send callbacks are no-ops without a config."""
    at = AppTest.from_string(
        """
from unittest.mock import MagicMock, patch
import streamlit as st
fm = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    st.session_state.pop("mas_config", None)
    with patch.object(pm, "push_config_to_baseline_state") as pb:
        with patch.object(pm, "push_config_to_attack_state") as pa:
            pm._on_send_to_baseline()
            pm._on_send_to_attack()
            st.markdown(f"baseline:{pb.called}")
            st.markdown(f"attack:{pa.called}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "baseline:False" in all_md
    assert "attack:False" in all_md


# ---------------------------------------------------------------------------
# _load_config (lines 278-339)
# ---------------------------------------------------------------------------


def test_load_config_loads_and_renders_summary(tmp_path):
    """_load_config loads a YAML, renders send buttons, and shows the MAS summary."""
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("mas_id: x\n", encoding="utf-8")
    at = AppTest.from_string(
        f"""
from unittest.mock import MagicMock, patch
import streamlit as st
from pathlib import Path
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {{
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    cfg = mk(mas_id="load_cfg_test")
    st.session_state.pop("current_yaml_path", None)
    st.session_state.pop("mas_config", None)
    with patch.object(pm, "load_mas_from_yaml", return_value=cfg):
        pm._load_config(Path({str(yaml_file)!r}))
    st.markdown(f"stored:{{st.session_state.get('mas_config') is not None}}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "stored:True" in all_md
    assert "load_cfg_test" in all_md
    button_labels = [b.label for b in at.button]
    assert any("Send to Chat" in lbl for lbl in button_labels)


def test_load_config_uses_cache_when_path_unchanged(tmp_path):
    """_load_config reuses the cached config when the path is unchanged."""
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("mas_id: x\n", encoding="utf-8")
    at = AppTest.from_string(
        f"""
from unittest.mock import MagicMock, patch
import streamlit as st
from pathlib import Path
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {{
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    cfg = mk(mas_id="cached_cfg")
    st.session_state["current_yaml_path"] = {str(yaml_file)!r}
    st.session_state["mas_config"] = cfg
    with patch.object(pm, "load_mas_from_yaml") as loader:
        pm._load_config(Path({str(yaml_file)!r}))
        st.markdown(f"loader_called:{{loader.called}}")
    st.markdown(f"same:{{st.session_state.get('mas_config') is cfg}}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "loader_called:False" in all_md
    assert "same:True" in all_md


def test_load_config_handles_load_error(tmp_path):
    """_load_config shows an error and clears mas_config when loading fails."""
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("bad", encoding="utf-8")
    at = AppTest.from_string(
        f"""
from unittest.mock import MagicMock, patch
import streamlit as st
from pathlib import Path
fm = {{
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    st.session_state.pop("current_yaml_path", None)
    st.session_state.pop("mas_config", None)
    with patch.object(pm, "load_mas_from_yaml", side_effect=ValueError("bad yaml")):
        pm._load_config(Path({str(yaml_file)!r}))
    st.markdown(f"cleared:{{st.session_state.get('mas_config') is None}}")
"""
    )
    at.run()
    assert not at.exception
    assert "Failed to load" in " ".join(e.value for e in at.error)
    assert "cleared:True" in " ".join(m.value for m in at.markdown)


def test_load_config_clears_stale_widget_state(tmp_path):
    """_load_config clears stale per-config widget keys on a fresh load."""
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("mas_id: x\n", encoding="utf-8")
    at = AppTest.from_string(
        f"""
from unittest.mock import MagicMock, patch
import streamlit as st
from pathlib import Path
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {{
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    cfg = mk(mas_id="stale_clear")
    st.session_state.pop("current_yaml_path", None)
    st.session_state.pop("mas_config", None)
    st.session_state["flow_state_old"] = "stale"
    st.session_state["agent_overrides_old"] = "stale"
    with patch.object(pm, "load_mas_from_yaml", return_value=cfg):
        pm._load_config(Path({str(yaml_file)!r}))
    st.markdown(f"flow_cleared:{{'flow_state_old' not in st.session_state}}")
    st.markdown(f"ovr_cleared:{{'agent_overrides_old' not in st.session_state}}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "flow_cleared:True" in all_md
    assert "ovr_cleared:True" in all_md


def test_load_config_summary_shows_consensus_and_hitl(tmp_path):
    """_load_config renders consensus threshold and the human-in-loop warning."""
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("mas_id: x\n", encoding="utf-8")
    at = AppTest.from_string(
        f"""
from unittest.mock import MagicMock, patch
import streamlit as st
from pathlib import Path
from bili.aether.schema.agent_spec import AgentSpec
from bili.aether.schema.enums import WorkflowType
from bili.aether.schema.mas_config import MASConfig
fm = {{
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(StreamlitFlowNode=MagicMock()),
    "streamlit_flow.state": MagicMock(),
}}
with patch.dict("sys.modules", fm):
    from bili.aether.ui import page as pm
    agents = [
        AgentSpec(agent_id="a0", role="a0", objective="Vote on the proposals"),
        AgentSpec(agent_id="a1", role="a1", objective="Vote on the proposals"),
    ]
    cfg = MASConfig(
        mas_id="consensus_cfg", name="Consensus", description="d",
        agents=agents, channels=[], workflow_type=WorkflowType.CONSENSUS,
        consensus_threshold=0.66, human_in_loop=True,
    )
    st.session_state.pop("current_yaml_path", None)
    st.session_state.pop("mas_config", None)
    with patch.object(pm, "load_mas_from_yaml", return_value=cfg):
        pm._load_config(Path({str(yaml_file)!r}))
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Consensus" in all_md
    assert "Human-in-loop enabled" in " ".join(w.value for w in at.warning)


# ---------------------------------------------------------------------------
# streamlit-flow ImportError fallback (lines 26-31)
# ---------------------------------------------------------------------------


def test_missing_streamlit_flow_shows_error_and_stops():
    """Importing page.py without streamlit-flow shows an install error and stops."""
    at = AppTest.from_string(
        """
import sys
import importlib
from unittest.mock import patch
import streamlit as st

# Drop any cached page module and force the streamlit_flow.elements import to
# fail so the module-level ImportError fallback (st.error + st.stop) runs.
sys.modules.pop("bili.aether.ui.page", None)
fm = {"streamlit_flow": None, "streamlit_flow.elements": None}
with patch.dict("sys.modules", fm):
    try:
        importlib.import_module("bili.aether.ui.page")
    finally:
        # Re-import cleanly so later tests get a working module again.
        sys.modules.pop("bili.aether.ui.page", None)
import bili.aether.ui.page  # noqa: F401  re-establish a healthy module
"""
    )
    at.run()
    assert not at.exception
    assert "streamlit-flow-component" in " ".join(e.value for e in at.error)
