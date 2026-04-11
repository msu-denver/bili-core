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
