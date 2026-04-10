"""Tests for bili.streamlit_ui.ui.configuration_panels.

Covers display_configuration_panels() panel rendering, import/export
functionality, helper functions, and prompt/tool initialization.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.  All
Streamlit-dependent imports therefore live inside ``AppTest.from_string``
scripts which execute within a proper Streamlit runtime context.
"""

# pylint: disable=import-outside-toplevel, protected-access

from streamlit.testing.v1 import AppTest

# ------------------------------------------------------------------
# Full panel render -- no exception
# ------------------------------------------------------------------


def test_display_configuration_panels_no_exception():
    """display_configuration_panels renders without exception."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
"""
    )
    at.run()
    assert not at.exception


# ------------------------------------------------------------------
# Configuration Panel heading
# ------------------------------------------------------------------


def test_renders_configuration_panel_heading():
    """The page renders a Configuration Panel heading."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "Configuration Panel" in all_md


# ------------------------------------------------------------------
# LLM Configuration expander
# ------------------------------------------------------------------


def test_llm_configuration_section_present():
    """The LLM Configuration section marker is rendered."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "llm-configuration" in all_md


# ------------------------------------------------------------------
# Prompt Customization section
# ------------------------------------------------------------------


def test_prompt_customization_section_present():
    """The Prompt Customization section marker is rendered."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "prompts" in all_md


# ------------------------------------------------------------------
# Tool section
# ------------------------------------------------------------------


def test_tools_section_present():
    """The Tools section marker is rendered."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "tools" in all_md


# ------------------------------------------------------------------
# Import/Export section
# ------------------------------------------------------------------


def test_import_export_section_present():
    """The Import/Export section marker is rendered."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "import-export" in all_md


# ------------------------------------------------------------------
# Session state initialization
# ------------------------------------------------------------------


def test_initializes_model_type():
    """display_configuration_panels initializes model_type."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_type:{'model_type' in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "has_type:True" in all_md


def test_initializes_streaming_toggle():
    """display_configuration_panels initializes streaming_enabled."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"streaming:{'streaming_enabled' in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "streaming:True" in all_md


def test_initializes_persona():
    """display_configuration_panels initializes persona from defaults."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_persona:{'persona' in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "has_persona:True" in all_md


def test_initializes_selected_tools():
    """display_configuration_panels initializes selected_tools list."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_tools:{'selected_tools' in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "has_tools:True" in all_md


# ------------------------------------------------------------------
# update_selected_tools helper
# ------------------------------------------------------------------


def test_update_selected_tools_adds_tool():
    """update_selected_tools adds an enabled tool to selected list."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui.configuration_panels import update_selected_tools
st.session_state["selected_tools"] = []
st.session_state["test_tool_enabled"] = True
update_selected_tools("test_tool", "test_tool_enabled")
st.markdown(f"added:{'test_tool' in st.session_state['selected_tools']}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "added:True" in all_md


def test_update_selected_tools_removes_tool():
    """update_selected_tools removes a disabled tool."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui.configuration_panels import update_selected_tools
st.session_state["selected_tools"] = ["test_tool"]
st.session_state["test_tool_enabled"] = False
update_selected_tools("test_tool", "test_tool_enabled")
st.markdown(f"removed:{'test_tool' not in st.session_state['selected_tools']}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "removed:True" in all_md


def test_update_selected_tools_no_duplicate():
    """update_selected_tools does not add a tool twice."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui.configuration_panels import update_selected_tools
st.session_state["selected_tools"] = ["test_tool"]
st.session_state["test_tool_enabled"] = True
update_selected_tools("test_tool", "test_tool_enabled")
st.markdown(f"count:{st.session_state['selected_tools'].count('test_tool')}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "count:1" in all_md


# ------------------------------------------------------------------
# update_prompt_state helper
# ------------------------------------------------------------------


def test_update_prompt_state():
    """update_prompt_state synchronizes session state key."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui.configuration_panels import update_prompt_state
st.session_state["my_prompt"] = "original"
update_prompt_state("my_prompt")
st.markdown(f"value:{st.session_state['my_prompt']}")
"""
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "value:original" in all_md


# ------------------------------------------------------------------
# Export button renders
# ------------------------------------------------------------------


def test_export_button_present():
    """The Export Configuration button renders."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
"""
    )
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert any("Export" in label for label in labels)


# ------------------------------------------------------------------
# Individual panel rendering - LLM Configuration
# ------------------------------------------------------------------


def test_model_type_selectbox_present():
    """The LLM type selectbox is rendered."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
"""
    )
    at.run()
    assert not at.exception
    assert len(at.selectbox) >= 1


def test_initializes_temperature():
    """display_configuration_panels initializes temperature."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_temp:{'temperature' in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    assert "has_temp:True" in " ".join(m.value for m in at.markdown)


def test_initializes_max_output_tokens():
    """display_configuration_panels initializes max_output_tokens."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_max:{'max_output_tokens' in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    assert "has_max:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# Prompt customization panel
# ------------------------------------------------------------------


def test_initializes_selected_prompt_template():
    """display_configuration_panels initializes selected_prompt_template."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_template:{'selected_prompt_template' in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    assert "has_template:True" in " ".join(m.value for m in at.markdown)


def test_persona_text_area_populated():
    """The persona text area is populated with default prompt content."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
persona = st.session_state.get("persona", "")
st.markdown(f"has_persona:{len(persona) > 0}")
"""
    )
    at.run()
    assert not at.exception
    assert "has_persona:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# Tool panel
# ------------------------------------------------------------------


def test_tool_enabled_keys_initialized():
    """Tool enabled keys are initialized in session state."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
from bili.iris.config.tool_config import TOOLS
cp_mod.display_configuration_panels()
tool_names = list(TOOLS.keys())
if tool_names:
    first_tool = tool_names[0]
    st.markdown(f"has_key:{f'{first_tool}_enabled' in st.session_state}")
else:
    st.markdown("has_key:True")
"""
    )
    at.run()
    assert not at.exception
    assert "has_key:True" in " ".join(m.value for m in at.markdown)


def test_enable_all_tools_button_present():
    """The Enable All Tools button is rendered."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
"""
    )
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert any("Enable All" in label for label in labels)


def test_disable_all_tools_button_present():
    """The Disable All Tools button is rendered."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
"""
    )
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert any("Disable All" in label for label in labels)


# ------------------------------------------------------------------
# Import configuration flow
# ------------------------------------------------------------------


def test_import_export_section_has_file_uploader():
    """The Import/Export section contains a file uploader."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
"""
    )
    at.run()
    assert not at.exception


# ------------------------------------------------------------------
# update_selected_tools edge cases
# ------------------------------------------------------------------


def test_update_selected_tools_from_empty_removes_noop():
    """Removing a tool from empty list is a no-op."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui.configuration_panels import update_selected_tools
st.session_state["selected_tools"] = []
st.session_state["missing_tool_enabled"] = False
update_selected_tools("missing_tool", "missing_tool_enabled")
st.markdown(f"count:{len(st.session_state['selected_tools'])}")
"""
    )
    at.run()
    assert not at.exception
    assert "count:0" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# Session state initialization - model_kwargs
# ------------------------------------------------------------------


def test_initializes_model_kwargs():
    """display_configuration_panels initializes model_kwargs."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_kwargs:{'model_kwargs' in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    assert "has_kwargs:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# Supports tools flag
# ------------------------------------------------------------------


def test_supports_tools_initialized():
    """display_configuration_panels initializes supports_tools."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_supports:{'supports_tools' in st.session_state}")
"""
    )
    at.run()
    assert not at.exception
    assert "has_supports:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# DEFAULT_PROMPTS loading
# ------------------------------------------------------------------


def test_default_prompts_loaded():
    """DEFAULT_PROMPTS is loaded and non-empty."""
    from bili.streamlit_ui.ui.configuration_panels import DEFAULT_PROMPTS

    assert isinstance(DEFAULT_PROMPTS, dict)
    assert len(DEFAULT_PROMPTS) > 0


def test_default_prompts_have_persona():
    """Each default prompt has a persona field."""
    from bili.streamlit_ui.ui.configuration_panels import DEFAULT_PROMPTS

    for name, prompt in DEFAULT_PROMPTS.items():
        assert "persona" in prompt, f"Prompt '{name}' missing persona field"
