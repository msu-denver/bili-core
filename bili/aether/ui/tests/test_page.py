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
