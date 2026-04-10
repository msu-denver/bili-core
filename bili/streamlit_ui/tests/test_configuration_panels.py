"""Tests for bili.streamlit_ui.ui.configuration_panels.

Covers display_configuration_panels() panel rendering, import/export
functionality, helper functions, and prompt/tool initialization.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.  All
Streamlit-dependent imports therefore live inside ``AppTest.from_function``
callbacks which execute within a proper Streamlit runtime context.
"""

# pylint: disable=import-outside-toplevel, protected-access

from streamlit.testing.v1 import AppTest

# ------------------------------------------------------------------
# Full panel render -- no exception
# ------------------------------------------------------------------


def test_display_configuration_panels_no_exception():
    """display_configuration_panels renders without exception."""

    def _app():
        from bili.streamlit_ui.ui import configuration_panels as cp_mod

        cp_mod.display_configuration_panels()

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception


# ------------------------------------------------------------------
# Configuration Panel heading
# ------------------------------------------------------------------


def test_renders_configuration_panel_heading():
    """The page renders a Configuration Panel heading."""

    def _app():
        from bili.streamlit_ui.ui import configuration_panels as cp_mod

        cp_mod.display_configuration_panels()

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Configuration Panel" in all_md


# ------------------------------------------------------------------
# LLM Configuration expander
# ------------------------------------------------------------------


def test_llm_configuration_section_present():
    """The LLM Configuration section marker is rendered."""

    def _app():
        from bili.streamlit_ui.ui import configuration_panels as cp_mod

        cp_mod.display_configuration_panels()

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "llm-configuration" in all_md


# ------------------------------------------------------------------
# Prompt Customization section
# ------------------------------------------------------------------


def test_prompt_customization_section_present():
    """The Prompt Customization section marker is rendered."""

    def _app():
        from bili.streamlit_ui.ui import configuration_panels as cp_mod

        cp_mod.display_configuration_panels()

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "prompts" in all_md


# ------------------------------------------------------------------
# Tool section
# ------------------------------------------------------------------


def test_tools_section_present():
    """The Tools section marker is rendered."""

    def _app():
        from bili.streamlit_ui.ui import configuration_panels as cp_mod

        cp_mod.display_configuration_panels()

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "tools" in all_md


# ------------------------------------------------------------------
# Import/Export section
# ------------------------------------------------------------------


def test_import_export_section_present():
    """The Import/Export section marker is rendered."""

    def _app():
        from bili.streamlit_ui.ui import configuration_panels as cp_mod

        cp_mod.display_configuration_panels()

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "import-export" in all_md


# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------


def test_initializes_model_type():
    """display_configuration_panels initializes model_type."""

    def _app():
        import streamlit as st

        from bili.streamlit_ui.ui import configuration_panels as cp_mod

        cp_mod.display_configuration_panels()
        st.markdown(f"has_type:{'model_type' in st.session_state}")

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "has_type:True" in all_md


def test_initializes_streaming_toggle():
    """display_configuration_panels initializes streaming_enabled."""

    def _app():
        import streamlit as st

        from bili.streamlit_ui.ui import configuration_panels as cp_mod

        cp_mod.display_configuration_panels()
        st.markdown(f"streaming:{'streaming_enabled' in st.session_state}")

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "streaming:True" in all_md


def test_initializes_persona():
    """display_configuration_panels initializes persona from defaults."""

    def _app():
        import streamlit as st

        from bili.streamlit_ui.ui import configuration_panels as cp_mod

        cp_mod.display_configuration_panels()
        st.markdown(f"has_persona:{'persona' in st.session_state}")

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "has_persona:True" in all_md


def test_initializes_selected_tools():
    """display_configuration_panels initializes selected_tools list."""

    def _app():
        import streamlit as st

        from bili.streamlit_ui.ui import configuration_panels as cp_mod

        cp_mod.display_configuration_panels()
        st.markdown(f"has_tools:{'selected_tools' in st.session_state}")

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "has_tools:True" in all_md


# ------------------------------------------------------------------
# update_selected_tools helper
# ------------------------------------------------------------------


def test_update_selected_tools_adds_tool():
    """update_selected_tools adds an enabled tool to selected list."""

    def _app():
        import streamlit as st

        from bili.streamlit_ui.ui.configuration_panels import update_selected_tools

        st.session_state["selected_tools"] = []
        st.session_state["test_tool_enabled"] = True
        update_selected_tools("test_tool", "test_tool_enabled")
        st.markdown(f"added:{'test_tool' in st.session_state['selected_tools']}")

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "added:True" in all_md


def test_update_selected_tools_removes_tool():
    """update_selected_tools removes a disabled tool."""

    def _app():
        import streamlit as st

        from bili.streamlit_ui.ui.configuration_panels import update_selected_tools

        st.session_state["selected_tools"] = ["test_tool"]
        st.session_state["test_tool_enabled"] = False
        update_selected_tools("test_tool", "test_tool_enabled")
        st.markdown(f"removed:{'test_tool' not in st.session_state['selected_tools']}")

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "removed:True" in all_md


def test_update_selected_tools_no_duplicate():
    """update_selected_tools does not add a tool twice."""

    def _app():
        import streamlit as st

        from bili.streamlit_ui.ui.configuration_panels import update_selected_tools

        st.session_state["selected_tools"] = ["test_tool"]
        st.session_state["test_tool_enabled"] = True
        update_selected_tools("test_tool", "test_tool_enabled")
        st.markdown(f"count:{st.session_state['selected_tools'].count('test_tool')}")

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "count:1" in all_md


# ------------------------------------------------------------------
# update_prompt_state helper
# ------------------------------------------------------------------


def test_update_prompt_state():
    """update_prompt_state synchronizes session state key."""

    def _app():
        import streamlit as st

        from bili.streamlit_ui.ui.configuration_panels import update_prompt_state

        st.session_state["my_prompt"] = "original"
        update_prompt_state("my_prompt")
        st.markdown(f"value:{st.session_state['my_prompt']}")

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "value:original" in all_md


# ------------------------------------------------------------------
# Export button renders
# ------------------------------------------------------------------


def test_export_button_present():
    """The Export Configuration button renders."""

    def _app():
        from bili.streamlit_ui.ui import configuration_panels as cp_mod

        cp_mod.display_configuration_panels()

    at = AppTest.from_function(_app)
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert any("Export" in label for label in labels)
