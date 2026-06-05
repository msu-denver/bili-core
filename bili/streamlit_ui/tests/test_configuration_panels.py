"""Tests for bili.streamlit_ui.ui.configuration_panels.

Covers display_configuration_panels() panel rendering, import/export
functionality, helper functions, and prompt/tool initialization.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.  All
Streamlit-dependent imports therefore live inside ``AppTest.from_string``
scripts which execute within a proper Streamlit runtime context.
"""

# pylint: disable=import-outside-toplevel, protected-access

import pytest
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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
""",
        default_timeout=15,
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


# ------------------------------------------------------------------
# Individual panel rendering details
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# update_selected_tools — multiple operations
# ------------------------------------------------------------------


def test_update_selected_tools_add_multiple():
    """update_selected_tools handles adding multiple tools."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui.configuration_panels import (
    update_selected_tools,
)
st.session_state["selected_tools"] = ["tool_a"]
st.session_state["tool_b_enabled"] = True
update_selected_tools("tool_b", "tool_b_enabled")
st.markdown(f"count:{len(st.session_state['selected_tools'])}")
st.markdown(f"has_b:{'tool_b' in st.session_state['selected_tools']}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "count:2" in all_md
    assert "has_b:True" in all_md


def test_update_selected_tools_add_then_remove():
    """Adding then removing a tool leaves list unchanged."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui.configuration_panels import (
    update_selected_tools,
)
st.session_state["selected_tools"] = []
st.session_state["tool_x_enabled"] = True
update_selected_tools("tool_x", "tool_x_enabled")
st.session_state["tool_x_enabled"] = False
update_selected_tools("tool_x", "tool_x_enabled")
st.markdown(f"count:{len(st.session_state['selected_tools'])}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert "count:0" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# update_prompt_state — additional cases
# ------------------------------------------------------------------


def test_update_prompt_state_existing_key():
    """update_prompt_state syncs an existing session state key."""
    at = AppTest.from_string(
        """
import streamlit as st
st.session_state["test_prompt"] = "original value"
from bili.streamlit_ui.ui.configuration_panels import (
    update_prompt_state,
)
update_prompt_state("test_prompt")
st.markdown("done:True")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert "done:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# DEFAULT_PROMPTS structure validation
# ------------------------------------------------------------------


def test_default_prompts_persona_is_string():
    """Each prompt's persona is a non-empty string."""
    from bili.streamlit_ui.ui.configuration_panels import DEFAULT_PROMPTS

    for name, prompt in DEFAULT_PROMPTS.items():
        assert isinstance(prompt["persona"], str), f"'{name}' persona not a string"
        assert len(prompt["persona"]) > 0, f"'{name}' persona is empty"


def test_default_prompts_keys_are_strings():
    """All DEFAULT_PROMPTS keys are strings."""
    from bili.streamlit_ui.ui.configuration_panels import DEFAULT_PROMPTS

    for key in DEFAULT_PROMPTS:
        assert isinstance(key, str)


# ------------------------------------------------------------------
# LLM_MODELS integration
# ------------------------------------------------------------------


def test_llm_models_contains_entries():
    """LLM_MODELS has at least one provider."""
    from bili.iris.config.llm_config import LLM_MODELS

    assert len(LLM_MODELS) > 0


def test_llm_models_providers_have_models():
    """Each LLM_MODELS provider has at least one model."""
    from bili.iris.config.llm_config import LLM_MODELS

    for key, info in LLM_MODELS.items():
        assert "models" in info, f"Provider '{key}' missing models"
        assert len(info["models"]) > 0, f"Provider '{key}' has no models"


# ------------------------------------------------------------------
# TOOLS integration
# ------------------------------------------------------------------


def test_tools_config_non_empty():
    """TOOLS config is a non-empty dict."""
    from bili.iris.config.tool_config import TOOLS

    assert isinstance(TOOLS, dict)
    assert len(TOOLS) > 0


def test_tools_have_default_prompt():
    """Each tool has a default_prompt configuration key."""
    from bili.iris.config.tool_config import TOOLS

    for name, config in TOOLS.items():
        assert "default_prompt" in config, f"Tool '{name}' missing 'default_prompt'"


# ------------------------------------------------------------------
# Configuration panel state initialization - model_id
# ------------------------------------------------------------------


def test_initializes_model_id():
    """display_configuration_panels initializes model_id."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_id:{'model_id' in st.session_state}")
""",
        default_timeout=10,
    )
    at.run()
    assert not at.exception
    assert "has_id:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# Configuration panel state - supports_structured_output
# ------------------------------------------------------------------


def test_initializes_supports_structured_output():
    """display_configuration_panels initializes supports_structured_output."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_struct:{'supports_structured_output' in st.session_state}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert "has_struct:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# Multiple model selectboxes
# ------------------------------------------------------------------


def test_model_name_selectbox_present():
    """The model name selectbox is rendered."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    # Should have at least 2 selectboxes (model type + model name)
    assert len(at.selectbox) >= 2


# ------------------------------------------------------------------
# Configuration panel - top_p, top_k, seed defaults
# ------------------------------------------------------------------


def test_initializes_top_p_default():
    """display_configuration_panels initializes top_p."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_top_p:{'top_p' in st.session_state}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert "has_top_p:True" in " ".join(m.value for m in at.markdown)


def test_initializes_top_k_default():
    """display_configuration_panels initializes top_k."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_top_k:{'top_k' in st.session_state}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert "has_top_k:True" in " ".join(m.value for m in at.markdown)


def test_initializes_seed_value_default():
    """display_configuration_panels initializes seed_value."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_seed:{'seed_value' in st.session_state}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert "has_seed:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# update_selected_tools - preserves other tools
# ------------------------------------------------------------------


def test_update_selected_tools_preserves_others():
    """Adding a tool preserves existing tools in the list."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui.configuration_panels import update_selected_tools
st.session_state["selected_tools"] = ["tool_a", "tool_b"]
st.session_state["tool_c_enabled"] = True
update_selected_tools("tool_c", "tool_c_enabled")
st.markdown(f"count:{len(st.session_state['selected_tools'])}")
st.markdown(f"has_a:{'tool_a' in st.session_state['selected_tools']}")
st.markdown(f"has_c:{'tool_c' in st.session_state['selected_tools']}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "count:3" in all_md
    assert "has_a:True" in all_md
    assert "has_c:True" in all_md


# ------------------------------------------------------------------
# LLM_MODELS structure validation
# ------------------------------------------------------------------


def test_llm_models_have_name_and_description():
    """Each LLM_MODELS provider has name and description."""
    from bili.iris.config.llm_config import LLM_MODELS

    for key, info in LLM_MODELS.items():
        assert "name" in info, f"Provider '{key}' missing name"
        assert "description" in info, f"Provider '{key}' missing description"


def test_llm_models_have_model_help():
    """Each LLM_MODELS provider has model_help."""
    from bili.iris.config.llm_config import LLM_MODELS

    for key, info in LLM_MODELS.items():
        assert "model_help" in info, f"Provider '{key}' missing model_help"


# ------------------------------------------------------------------
# TOOLS structure validation
# ------------------------------------------------------------------


def test_tools_have_default_prompt_field():
    """Each tool has a default_prompt field with content."""
    from bili.iris.config.tool_config import TOOLS

    for name, tool_config in TOOLS.items():
        prompt = tool_config.get("default_prompt", "")
        assert len(prompt) > 0, f"Tool '{name}' has empty default_prompt"


# ------------------------------------------------------------------
# Configuration panel renders checkbox widgets
# ------------------------------------------------------------------


def test_renders_checkboxes():
    """display_configuration_panels renders checkbox widgets."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert len(at.checkbox) >= 1


# ------------------------------------------------------------------
# LLM Configuration panel details
# ------------------------------------------------------------------


def test_streaming_checkbox_present():
    """The streaming responses checkbox is rendered."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    labels = [cb.label for cb in at.checkbox]
    assert any("streaming" in label.lower() for label in labels)


def test_model_type_selectbox_options_populated():
    """The LLM type selectbox has at least one option."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
model_type = st.session_state.get("model_type", "")
st.markdown(f"has_model_type:{len(model_type) > 0}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert "has_model_type:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# Prompt Customization panel
# ------------------------------------------------------------------


def test_prompt_description_rendered():
    """The prompt description is rendered after template selection."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
desc = st.session_state.get("prompt_description", "")
st.markdown(f"has_desc:{len(desc) > 0 if desc else False}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception


def test_persona_text_area_rendered():
    """The persona text area widget is rendered."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert len(at.text_area) >= 1


# ------------------------------------------------------------------
# Tool Configuration panel details
# ------------------------------------------------------------------


def test_tool_prompt_text_areas_rendered():
    """Each tool has a prompt text area rendered."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
from bili.iris.config.tool_config import TOOLS
cp_mod.display_configuration_panels()
tool_prompt_count = sum(
    1 for tool in TOOLS if f"{tool}_prompt" in st.session_state
)
st.markdown(f"prompts:{tool_prompt_count}")
st.markdown(f"total_tools:{len(TOOLS)}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "prompts:" in all_md


# ------------------------------------------------------------------
# Export configuration - button click
# ------------------------------------------------------------------


def test_export_button_click_renders_download():
    """Clicking Export Configuration renders a download button."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    export_buttons = [b for b in at.button if "Export" in b.label]
    if export_buttons:
        export_buttons[0].click()
        at.run()
        assert not at.exception


# ------------------------------------------------------------------
# Number inputs rendered
# ------------------------------------------------------------------


def test_number_inputs_rendered():
    """display_configuration_panels renders number input widgets."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert len(at.number_input) >= 1


# ------------------------------------------------------------------
# Initialization of thinking_budget
# ------------------------------------------------------------------


def test_initializes_thinking_budget():
    """display_configuration_panels initializes thinking_budget."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_thinking:{'thinking_budget' in st.session_state}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert "has_thinking:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# Initialization of max_retries
# ------------------------------------------------------------------


def test_initializes_max_retries():
    """display_configuration_panels initializes max_retries."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
st.markdown(f"has_retries:{'max_retries' in st.session_state}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert "has_retries:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# Prompt template selectbox present
# ------------------------------------------------------------------


def test_prompt_template_selectbox_present():
    """The prompt template selectbox is rendered."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    assert len(at.selectbox) >= 3


# ------------------------------------------------------------------
# update_selected_tools — no selected_tools in state
# ------------------------------------------------------------------


def test_update_selected_tools_creates_list():
    """update_selected_tools creates selected_tools if missing."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui.configuration_panels import update_selected_tools
st.session_state.pop("selected_tools", None)
st.session_state["new_tool_enabled"] = True
update_selected_tools("new_tool", "new_tool_enabled")
st.markdown(f"created:{isinstance(st.session_state.get('selected_tools'), list)}")
st.markdown(f"has:{('new_tool' in st.session_state.get('selected_tools', []))}")
""",
        default_timeout=15,
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "created:True" in all_md
    assert "has:True" in all_md


# ------------------------------------------------------------------
# TOOLS structure - description field
# ------------------------------------------------------------------


def test_tools_have_description():
    """Each tool has a description field."""
    from bili.iris.config.tool_config import TOOLS

    for name, config in TOOLS.items():
        assert "description" in config, f"Tool '{name}' missing 'description'"


# ------------------------------------------------------------------
# LLM model entries have required fields
# ------------------------------------------------------------------


def test_llm_model_entries_have_model_id():
    """Each model entry has a model_id."""
    from bili.iris.config.llm_config import LLM_MODELS

    for key, info in LLM_MODELS.items():
        for model in info["models"]:
            assert (
                "model_id" in model
            ), f"Provider '{key}' model '{model.get('model_name')}' missing model_id"


def test_llm_model_entries_have_model_name():
    """Each model entry has a model_name."""
    from bili.iris.config.llm_config import LLM_MODELS

    for key, info in LLM_MODELS.items():
        for model in info["models"]:
            assert "model_name" in model, f"Provider '{key}' model missing model_name"


# ------------------------------------------------------------------
# Model-capability dependent branches
# ------------------------------------------------------------------


def test_gemini_structured_output_block_renders():
    """Selecting a Gemini model renders the structured-output configuration."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_name"] = "Gemini 2.5 Pro"
cp_mod.display_configuration_panels()
st.markdown(f"struct:{st.session_state.get('supports_structured_output')}")
st.markdown(f"mime:{'response_mime_type' in st.session_state}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "struct:True" in all_md
    assert "mime:True" in all_md


def test_gemini_json_mime_renders_schema_editor():
    """Choosing the JSON MIME type renders the custom schema editor and validation."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_name"] = "Gemini 2.5 Pro"
st.session_state["response_mime_type"] = "application/json"
st.session_state["custom_response_schema"] = '{"type": "string"}'
st.session_state["schema_preset"] = "Custom"
cp_mod.display_configuration_panels()
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    # A valid JSON schema shows a success message.
    assert any("Valid JSON schema" in s.value for s in at.success)


def test_gemini_json_invalid_schema_shows_error():
    """An invalid custom JSON schema renders an error message."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_name"] = "Gemini 2.5 Pro"
st.session_state["response_mime_type"] = "application/json"
st.session_state["custom_response_schema"] = "{not valid json"
st.session_state["schema_preset"] = "Custom"
cp_mod.display_configuration_panels()
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert any("Invalid JSON" in e.value for e in at.error)


def test_gemini_text_mime_clears_schema_state():
    """Switching back to text MIME type clears the custom schema session keys."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_name"] = "Gemini 2.5 Pro"
st.session_state["response_mime_type"] = "text/plain"
st.session_state["custom_response_schema"] = '{"type": "string"}'
st.session_state["schema_preset"] = "Custom"
cp_mod.display_configuration_panels()
st.markdown(f"schema_gone:{'custom_response_schema' not in st.session_state}")
st.markdown(f"preset_gone:{'schema_preset' not in st.session_state}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "schema_gone:True" in all_md
    assert "preset_gone:True" in all_md


def test_gemini_thinking_budget_block_renders():
    """A Gemini model renders the thinking-budget control and initializes it."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_name"] = "Gemini 2.5 Pro"
cp_mod.display_configuration_panels()
st.markdown(f"thinking:{st.session_state.get('thinking_budget') is not None}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "thinking:True" in " ".join(m.value for m in at.markdown)


def test_openai_max_retries_block_renders():
    """An OpenAI model renders the max-retries control and initializes it."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_openai"
st.session_state["model_name"] = "OpenAI GPT-4o Omni"
cp_mod.display_configuration_panels()
st.markdown(f"retries:{st.session_state.get('max_retries') is not None}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "retries:True" in " ".join(m.value for m in at.markdown)


def test_model_lacking_top_k_sets_none():
    """A model without top_k support leaves top_k as None."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_aws_bedrock"
st.session_state["model_name"] = "AI21 Jamba 1.5 Large"
cp_mod.display_configuration_panels()
st.markdown(f"top_k:{st.session_state.get('top_k')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "top_k:None" in " ".join(m.value for m in at.markdown)


def test_model_lacking_temperature_sets_none():
    """A model without temperature support leaves temperature as None."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_azure_openai"
st.session_state["model_name"] = "Azure OpenAI o1-mini"
cp_mod.display_configuration_panels()
st.markdown(f"temp:{st.session_state.get('temperature')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "temp:None" in " ".join(m.value for m in at.markdown)


def test_model_lacking_seed_sets_none():
    """A model without seed support leaves seed_value as None."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_aws_bedrock"
st.session_state["model_name"] = "DeepSeek-R1"
cp_mod.display_configuration_panels()
st.markdown(f"seed:{st.session_state.get('seed_value')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "seed:None" in " ".join(m.value for m in at.markdown)


def test_model_lacking_max_output_tokens_sets_none():
    """A model without max-output-token support leaves max_output_tokens as None."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_azure_openai"
st.session_state["model_name"] = "Azure OpenAI o3"
cp_mod.display_configuration_panels()
st.markdown(f"max_out:{st.session_state.get('max_output_tokens')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "max_out:None" in " ".join(m.value for m in at.markdown)


def test_unsupported_tools_model_shows_warning():
    """A model that does not support tools renders a warning and clears the flag."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_aws_bedrock"
st.session_state["model_name"] = "Amazon Titan Text G1 - Premier"
cp_mod.display_configuration_panels()
st.markdown(f"supports:{st.session_state.get('supports_tools')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert any("does not support tools" in w.value for w in at.warning)
    assert "supports:False" in " ".join(m.value for m in at.markdown)


def test_custom_model_path_renders_text_input():
    """A model with custom_model_path renders the custom model path input."""
    at = AppTest.from_string(
        """
import os
os.environ["ENV"] = "development"
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "local_llamacpp"
st.session_state["model_name"] = "LlamaCpp Local (In Memory) Model"
cp_mod.display_configuration_panels()
st.markdown(f"has_id:{'model_id' in st.session_state}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "has_id:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# Enable / Disable all tools button clicks
# ------------------------------------------------------------------


def test_enable_all_tools_button_click_enables_tools():
    """Clicking Enable All Tools enables non-local tools in session state."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    enable_buttons = [b for b in at.button if "Enable All" in b.label]
    assert enable_buttons
    enable_buttons[0].click()
    at.run()
    assert not at.exception


def test_disable_all_tools_button_click_disables_tools():
    """Clicking Disable All Tools disables non-local tools in session state."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    disable_buttons = [b for b in at.button if "Disable All" in b.label]
    assert disable_buttons
    disable_buttons[0].click()
    at.run()
    assert not at.exception


# ------------------------------------------------------------------
# Import configuration flow
# ------------------------------------------------------------------


def test_import_configuration_applies_uploaded_values():
    """Uploading a JSON config writes its keys into session state."""
    at = AppTest.from_string(
        """
import io, json
from unittest.mock import patch
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod

class _Upload(io.BytesIO):
    pass

upload = _Upload(json.dumps({"persona": "imported-persona"}).encode("utf-8"))
st.session_state.pop("config_imported", None)
# Patch rerun to a no-op so the import branch (which calls st.rerun) does
# not trigger an AppTest re-run loop; we assert directly on the state it set.
with patch.object(cp_mod.st, "file_uploader", return_value=upload):
    with patch.object(cp_mod.st, "rerun"):
        cp_mod.display_configuration_panels()
st.markdown(f"persona:{st.session_state.get('persona')}")
st.markdown(f"flag:{st.session_state.get('config_imported')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    all_md = " ".join(m.value for m in at.markdown)
    assert "persona:imported-persona" in all_md
    assert "flag:True" in all_md


def test_import_configuration_already_imported_shows_success():
    """When config was already imported the success message is shown."""
    at = AppTest.from_string(
        """
import io, json
from unittest.mock import patch
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod

class _Upload(io.BytesIO):
    pass

upload = _Upload(json.dumps({"persona": "x"}).encode("utf-8"))
st.session_state["config_imported"] = True
with patch.object(cp_mod.st, "file_uploader", return_value=upload):
    cp_mod.display_configuration_panels()
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert any("imported successfully" in s.value for s in at.success)


# ------------------------------------------------------------------
# reset_model_name on-change callback
# ------------------------------------------------------------------


def test_model_type_change_resets_model_name():
    """Changing the LLM type selectbox clears the previously chosen model name."""
    at = AppTest.from_string(
        """
from bili.streamlit_ui.ui import configuration_panels as cp_mod
cp_mod.display_configuration_panels()
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    type_selectbox = at.selectbox[0]
    options = type_selectbox.options
    # Choose a different LLM type option to fire the reset_model_name callback.
    different = next(o for o in options if o != type_selectbox.value)
    type_selectbox.set_value(different)
    at.run()
    assert not at.exception


# ------------------------------------------------------------------
# Boolean tool parameter widget
# ------------------------------------------------------------------


def test_bool_tool_param_renders_checkbox():
    """A tool with a boolean parameter renders a checkbox for that parameter."""
    at = AppTest.from_string(
        """
from unittest.mock import patch
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod

bool_tools = {
    "bool_param_tool": {
        "enabled": True,
        "description": "A tool with a boolean parameter",
        "default_prompt": "Use the bool tool",
        "params": {
            "flag": {
                "type": "bool",
                "default": True,
                "description": "A boolean flag",
            }
        },
    }
}
with patch.object(cp_mod, "TOOLS", bool_tools):
    cp_mod.display_configuration_panels()
st.markdown(f"flag_key:{'bool_param_tool_flag' in st.session_state}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    assert "flag_key:True" in " ".join(m.value for m in at.markdown)


# ------------------------------------------------------------------
# Schema preset reset/clear buttons
# ------------------------------------------------------------------


def test_reset_schema_button_present_for_json():
    """The schema reset and clear buttons render in JSON MIME mode."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_name"] = "Gemini 2.5 Pro"
st.session_state["response_mime_type"] = "application/json"
cp_mod.display_configuration_panels()
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert any("Reset to Default Schema" in l for l in labels)
    assert any("Clear Schema" in l for l in labels)


# ------------------------------------------------------------------
# Temperature clamping and schema callback / button branches
# ------------------------------------------------------------------


def test_temperature_above_max_is_clamped():
    """A stored temperature above the model maximum is clamped down."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
# Nova Pro supports temperature; seed an out-of-range value to force clamping.
st.session_state["model_type"] = "remote_aws_bedrock"
st.session_state["model_name"] = "Amazon Nova Pro"
st.session_state["temperature"] = 9999.0
cp_mod.display_configuration_panels()
st.markdown(f"temp:{st.session_state.get('temperature')}")
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    # The clamped temperature must be a finite value well below the seeded 9999.
    all_md = " ".join(m.value for m in at.markdown)
    assert "temp:9999" not in all_md


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: Reset to Default Schema handler assigns st.session_state["
        "'schema_preset'] (configuration_panels.py line 571), but schema_preset "
        "is a widget key, so Streamlit raises StreamlitAPIException on the "
        "click rerun. The button crashes the panel instead of resetting it."
    ),
)
def test_reset_schema_button_click_sets_object_preset():
    """Clicking Reset to Default Schema should set the object-response preset."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_name"] = "Gemini 2.5 Pro"
st.session_state["response_mime_type"] = "application/json"
cp_mod.display_configuration_panels()
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    reset_buttons = [b for b in at.button if "Reset to Default Schema" in b.label]
    assert reset_buttons
    reset_buttons[0].click()
    at.run()
    assert not at.exception
    assert at.session_state["schema_preset"] == "Object Response"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "BUG: Clear Schema handler assigns st.session_state['schema_preset'] "
        "(configuration_panels.py line 576), but schema_preset is a widget key, "
        "so Streamlit raises StreamlitAPIException on the click rerun. The "
        "button crashes the panel instead of clearing the schema."
    ),
)
def test_clear_schema_button_click_empties_schema():
    """Clicking Clear Schema should empty the custom schema and reset the preset."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_name"] = "Gemini 2.5 Pro"
st.session_state["response_mime_type"] = "application/json"
cp_mod.display_configuration_panels()
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    clear_buttons = [b for b in at.button if "Clear Schema" in b.label]
    assert clear_buttons
    clear_buttons[0].click()
    at.run()
    assert not at.exception
    assert at.session_state["custom_response_schema"] == "{}"
    assert at.session_state["schema_preset"] == "Custom"


def test_schema_preset_change_updates_schema():
    """Selecting a non-custom schema preset updates the custom schema text."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_name"] = "Gemini 2.5 Pro"
st.session_state["response_mime_type"] = "application/json"
cp_mod.display_configuration_panels()
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    preset_boxes = [s for s in at.selectbox if s.label == "Schema Preset"]
    assert preset_boxes
    preset_boxes[0].set_value("Array Response")
    at.run()
    assert not at.exception
    assert "array" in at.session_state["custom_response_schema"]


def test_schema_textarea_edit_switches_to_custom():
    """Editing the schema text area switches the preset back to Custom."""
    at = AppTest.from_string(
        """
import streamlit as st
from bili.streamlit_ui.ui import configuration_panels as cp_mod
st.session_state["model_type"] = "remote_google_vertex"
st.session_state["model_name"] = "Gemini 2.5 Pro"
st.session_state["response_mime_type"] = "application/json"
st.session_state["schema_preset"] = "Object Response"
cp_mod.display_configuration_panels()
""",
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    schema_areas = [t for t in at.text_area if "JSON Schema" in (t.label or "")]
    assert schema_areas
    schema_areas[0].set_value('{"type": "number"}')
    at.run()
    assert not at.exception
    # The on_change callback syncs the edited text into custom_response_schema.
    assert at.session_state["custom_response_schema"] == '{"type": "number"}'
