"""
Module: configuration_panels

This module provides interactive configuration panels for a chatbot system
within a Streamlit application. It includes functions to display and manage
configuration settings, import/export configurations, and customize prompts
and tools.

Functions:
    - display_configuration_panels():
      Displays interactive configuration panels for the chatbot system and
      handles import/export functionality for configuration settings.
    - update_selected_tools(tool, tool_enabled_key):
      Updates the list of selected tools in the session state based on the
      given tool's enabled status.
    - update_prompt_state(prompt_key):
      Updates the Streamlit session state for a given key to its current value.

Dependencies:
    - json: Provides JSON handling capabilities.
    - os: Provides a way of using operating system dependent functionality.
    - pathlib.Path: Provides an object-oriented filesystem path representation.
    - streamlit: Provides the Streamlit library for building web applications.
    - bili.config.llm_config: Imports LLM_MODELS for language model configurations.
    - bili.config.tool_config: Imports TOOLS for tool configurations.
    - bili.utils.file_utils: Imports load_from_json for loading JSON files.

Usage:
    This module is intended to be used within a Streamlit application to manage
    chatbot system configurations, including LLM settings, tool support, and
    prompt customization.

Example:
    from bili.streamlit.ui.configuration_panels import display_configuration_panels

    # Display the configuration panels
    display_configuration_panels()
"""

import json
import os
from pathlib import Path

import streamlit as st

from bili.config.llm_config import LLM_MODELS
from bili.config.tool_config import TOOLS
from bili.utils.file_utils import load_from_json

# Load default prompts from JSON that will be used in the chatbot system
# This populated the dropdown for selecting a prompt template in the LLM configuration panel
DEFAULT_PROMPTS = load_from_json(
    os.getenv(
        "DEFAULT_PROMPT_PATH",
        Path(__file__)
        .parent.parent.parent.joinpath("prompts", "default_prompts.json")
        .as_posix(),
    ),
    "templates",
)


def display_configuration_panels():
    """
    Display interactive configuration panels for the chatbot system and handle import/export
    functionality for configuration settings.

    This function renders a user interface using `streamlit` to allow for configuring
    parameters such as LLM type, model settings, token generation parameters, and more.
    Users can import configuration from a JSON file to set these parameters or export
    the current configuration setup as a JSON file for reuse or sharing. Interactive
    widgets are used wherever necessary for a dynamic and user-friendly setup.

    The LLM Configuration Panel dynamically adapts based on selected model type and its
    capabilities, for instance, toggling the visibility or values of specific controls such
    as `top_p`, `top_k`, and `temperature`.

    Users may also reset, adjust, or fine-tune the attributes or strategies governing the
    chatbot system from the same interface. Components of the layout such as seed value,
    tool support, and memory strategies are also covered.

    :param st: Streamlit object for rendering the interface.
    :type st: streamlit
    :return: None
    """
    st.markdown(
        """<h2>Configuration Panel</h2>
        <p>Configure the LLM, tools, prompts, and memory strategy for the chatbot system.</p>""",
        unsafe_allow_html=True,
    )

    with st.expander("Import/Export Configuration"):
        uploaded_file = st.file_uploader(
            "Import Configuration from JSON",
            type="json",
            on_change=lambda: st.session_state.update({"config_imported": False}),
        )
        if uploaded_file is not None:
            if not st.session_state.get("config_imported", False):
                imported_configuration = json.loads(uploaded_file.read())
                for key, value in imported_configuration.items():
                    st.session_state[key] = value
                st.session_state["config_imported"] = True
                st.rerun()
            elif "config_imported" in st.session_state:
                st.success("Configuration imported successfully!")

        if st.button("Export Configuration", use_container_width=True):
            exported_configuration = {
                "model_type": st.session_state["model_type"],
                "model_name": st.session_state.get("model_name"),
                "model_id": st.session_state.get("model_id"),
                "model_kwargs": st.session_state.get("model_kwargs"),
                "max_output_tokens": st.session_state.get("max_output_tokens"),
                "max_output_tokens_max": st.session_state.get("max_output_tokens_max"),
                "max_input_tokens": st.session_state.get("max_input_tokens"),
                "temperature": st.session_state["temperature"],
                "top_p": st.session_state.get("top_p"),
                "top_k": st.session_state.get("top_k"),
                "seed_value": st.session_state.get("seed_value"),
                "max_retries": st.session_state.get("max_retries"),
                "selected_prompt_template": st.session_state[
                    "selected_prompt_template"
                ],
                "persona": st.session_state.get("persona", ""),
                "supports_tools": st.session_state.get("supports_tools", True),
                "selected_tools": st.session_state.get("selected_tools", []),
                "tool_prompts": {
                    tool: st.session_state.get(f"{tool}_prompt") for tool in TOOLS
                },
                "memory_strategy": st.session_state.get("memory_strategy"),
                "memory_limit_type": st.session_state.get("memory_limit_type"),
                "memory_limit_value": st.session_state.get("memory_limit_value"),
                "memory_limit_trim_value": st.session_state.get(
                    "memory_limit_trim_value"
                ),
            }
            st.download_button(
                label="Download Configuration as JSON",
                data=json.dumps(exported_configuration),
                file_name="chatbot_configuration.json",
                mime="application/json",
                use_container_width=True,
            )

    def reset_model_name():
        if "model_name" in st.session_state:
            del st.session_state["model_name"]

    # ---- LLM Configuration Panel ----
    with st.expander("LLM Configuration"):
        model_type_options = [
            (key, value["name"], value["description"])
            for key, value in LLM_MODELS.items()
            if os.getenv("ENV") == "development"
            or not value.get("name", "").startswith("Local")
        ]

        if "model_type" not in st.session_state:
            st.session_state["model_type"] = model_type_options[0][0]

        model_type_display = [
            f"{name}: {description}" for _, name, description in model_type_options
        ]
        selected_model_type = st.selectbox(
            "Choose LLM Type",
            model_type_display,
            index=[key for key, _, _ in model_type_options].index(
                st.session_state["model_type"]
            ),
            key="model_type_display",
            on_change=reset_model_name,
        )

        st.session_state["model_type"] = model_type_options[
            model_type_display.index(selected_model_type)
        ][0]

        model_name_options = LLM_MODELS[st.session_state["model_type"]]["models"]
        model_names = [m["model_name"] for m in model_name_options]
        if st.session_state.get("model_name") not in model_names:
            st.session_state["model_name"] = model_names[0]

        selected_model_name = st.selectbox(
            "Choose LLM Model",
            model_names,
            index=(
                model_names.index(st.session_state["model_name"])
                if st.session_state["model_name"] in model_names
                else 0
            ),
            key="model_name",
            help=LLM_MODELS[st.session_state["model_type"]]["model_help"],
        )

        selected_model = next(
            (m for m in model_name_options if m["model_name"] == selected_model_name),
            None,
        )
        st.session_state["model_kwargs"] = selected_model.get("kwargs", {})

        # If model supports custom_model_path, add a text input for the path
        if selected_model.get("custom_model_path", False):
            st.session_state["model_id"] = selected_model.get("model_id")
            st.session_state["model_id"] = st.text_input(
                "Custom Model Path",
                value=st.session_state.get("model_id", ""),
                help="Path to the custom model file.",
            )
        else:
            st.session_state["model_id"] = selected_model.get("model_id")

        # insert horizontal rule
        st.markdown("---")

        # Check if the model supports top_p
        if selected_model.get("supports_top_p", False):
            st.session_state["top_p"] = st.number_input(
                "Top-p (Nucleus Sampling), Maximum Value: 1.0 (Optional) ",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("top_p", None),
                step=0.05,
                key="top_p_setting",
                help="Controls the cumulative probability threshold for token sampling.",
            )
            st.button(
                "Clear Top-p",
                on_click=lambda: st.session_state.update({"top_p": None}),
                help="Clear the top-p value.",
                use_container_width=True,
            )
            # Create tooltip explaining what top_p is
            st.markdown(
                """<a href='https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/"""
                """content-generation-parameters#top-p'>What is top-p?</a>""",
                unsafe_allow_html=True,
            )

            # insert horizontal rule
            st.markdown("---")
        else:
            st.session_state["top_p"] = None

        # Check if the model supports top_k
        if selected_model.get("supports_top_k", False):
            top_k_max = selected_model.get("top_k_max")
            st.session_state["top_k"] = st.number_input(
                f"Top-k, Maximum Value: {top_k_max} (Optional)",
                min_value=1,
                max_value=top_k_max,
                value=st.session_state.get("top_k", None),
                step=1,
                key="top_k_setting",
                help="Restricts token sampling to the top-k most likely tokens at each step.",
            )
            st.button(
                "Clear Top-k",
                on_click=lambda: st.session_state.update({"top_k": None}),
                help="Clear the top-k value.",
                use_container_width=True,
            )
            # Create tooltip explaining what top_k is
            st.markdown(
                """<a href='https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/"""
                """content-generation-parameters#top-k'>What is top-k?</a>""",
                unsafe_allow_html=True,
            )
        else:
            st.session_state["top_k"] = None

        # Check if the model supports temperature
        if selected_model.get("supports_temperature", False):
            temperature_max = float(selected_model.get("temperature_max", "1.0"))
            current_temperature = st.session_state.get("temperature") or 0.0
            if current_temperature > temperature_max:
                st.session_state["temperature"] = temperature_max
            st.session_state["temperature"] = st.number_input(
                f"LLM Temperature, Maximum Value: {temperature_max}",
                0.0,
                temperature_max,
                st.session_state.get("temperature") or 0.7,
                step=0.1,
                key="temperature_setting",
                help="Controls the randomness of token sampling.",
            )
            # Create tooltip explaining what temperature is
            st.markdown(
                """<a href='https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/"""
                """content-generation-parameters#temperature'>What is temperature?</a>""",
                unsafe_allow_html=True,
            )

            # insert horizontal rule
            st.markdown("---")
        else:
            st.session_state["temperature"] = None

        # If the model supports seed
        if selected_model.get("supports_seed"):
            st.session_state["seed_value"] = st.number_input(
                "Seed Value (Optional)",
                min_value=0,
                value=st.session_state.get("seed_value") or None,
                key="seed_value_setting",
                help="A seed value allows you to generate (approximately) "
                "the same output for the same input.",
            )
            col1, col2 = st.columns(2)
            with col1:
                st.button(
                    "Random Seed",
                    on_click=lambda: st.session_state.update(
                        {"seed_value_setting": int.from_bytes(os.urandom(4), "big")}
                    ),
                    use_container_width=True,
                )
            with col2:
                st.button(
                    "Clear Seed",
                    on_click=lambda: st.session_state.update(
                        {"seed_value_setting": None}
                    ),
                    use_container_width=True,
                )
            # Create tooltip explaining what seed is
            st.markdown(
                """<a href='https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/"""
                """content-generation-parameters#seed'>What is a seed?</a>""",
                unsafe_allow_html=True,
            )

            # insert horizontal rule
            st.markdown("---")
        else:
            st.session_state["seed_value"] = None

        if selected_model["supports_max_output_tokens"]:
            st.session_state["max_output_tokens_max"] = selected_model.get(
                "max_output_tokens", None
            )
            st.session_state["max_input_tokens"] = selected_model.get(
                "max_input_tokens", None
            )
            input_description = (
                f"Max Tokens: Maximum number of tokens to generate in the response."
                f" The maximum value allowed is "
                f"{st.session_state['max_output_tokens_max']}."
            )

            st.session_state["max_output_tokens"] = st.number_input(
                input_description,
                min_value=1,
                max_value=st.session_state["max_output_tokens_max"],
                value=st.session_state.get("max_output_tokens") or 1500,
                key="max_output_tokens_setting",
                help="Controls the maximum number of tokens to generate in the response.",
            )

            # Create tooltip explaining what max tokens is
            st.markdown(
                """<a href='https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/"""
                """content-generation-parameters#max-output-tokens'>What does max tokens do?</a>""",
                unsafe_allow_html=True,
            )
        else:
            st.session_state["max_output_tokens"] = None

        if selected_model.get("supports_max_retries", False):
            max_retries_max = selected_model.get("max_retries_max", 10)
            max_retries_default = selected_model.get("max_retries_default", 3)

            st.session_state["max_retries"] = st.number_input(
                f"Max Retries (API Retry Attempts), Maximum Value: {max_retries_max}",
                min_value=0,
                max_value=max_retries_max,
                value=st.session_state.get("max_retries", max_retries_default) or 3,
                step=1,
                key="max_retries_setting",
                help="Maximum number of retry attempts for failed API calls."
                     "Set to 0 to disable retries.",
            )

            # Add clear button
            st.button(
                "Reset to Default",
                on_click=lambda: st.session_state.update(
                    {"max_retries_setting": max_retries_default}
                    ),
                help=f"Reset max retries to default value ({max_retries_default})",
                use_container_width=True,
            )

            # Add help link
            st.markdown(
                """<a href='https://platform.openai.com/docs/guides/error-codes'>"""
                """Learn about OpenAI error handling and retries</a>""",
                unsafe_allow_html=True,
            )

            # insert horizontal rule
            st.markdown("---")
        else:
            st.session_state["max_retries"] = None

    # ---- Prompt Customization ----
    with st.expander("Prompt Customization"):
        if "selected_prompt_template" not in st.session_state:
            st.session_state["selected_prompt_template"] = list(DEFAULT_PROMPTS.keys())[
                0
            ]

        selected_template = st.selectbox(
            "Select Prompt Template",
            options=list(DEFAULT_PROMPTS.keys()),
            index=list(DEFAULT_PROMPTS.keys()).index(
                st.session_state["selected_prompt_template"]
            ),
            key="selected_prompt_template",
            on_change=lambda: st.session_state.update(
                {
                    "persona": DEFAULT_PROMPTS[
                        st.session_state["selected_prompt_template"]
                    ]["persona"],
                    "prompt_description": DEFAULT_PROMPTS[
                        st.session_state["selected_prompt_template"]
                    ].get("description", "No description provided."),
                }
            ),
        )

        if "persona" not in st.session_state:
            st.session_state["persona"] = DEFAULT_PROMPTS[selected_template]["persona"]
            st.session_state["prompt_description"] = DEFAULT_PROMPTS[
                selected_template
            ].get("description", "")

        st.markdown(
            f"**Prompt Description:**\n\n"
            f"{st.session_state.get('prompt_description','No description provided.')}\n"
        )

        st.markdown("**System Prefix (LangGraph)**:")
        st.text_area(
            "System Prefix",
            value=st.session_state["persona"],
            key="persona",
        )

    # ---- Tool Selection ----
    for tool, details in TOOLS.items():
        tool_enabled_key = f"{tool}_enabled"
        prompt_key = f"{tool}_prompt"
        if tool_enabled_key not in st.session_state:
            st.session_state[tool_enabled_key] = details["enabled"]
        if prompt_key not in st.session_state:
            st.session_state[prompt_key] = details.get("default_prompt", "")

    with st.expander("Tool Selection and Customization"):

        selected_tools = []
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Enable All Tools", use_container_width=True):
                for tool, details in TOOLS.items():
                    # if tool name starts with local and not in development, skip the tool
                    if tool.startswith("local") and os.getenv("ENV") != "development":
                        continue
                    st.session_state[f"{tool}_enabled"] = True
        with col2:
            if st.button("Disable All Tools", use_container_width=True):
                for tool, details in TOOLS.items():
                    # if tool name starts with local and not in development, skip the tool
                    if tool.startswith("local") and os.getenv("ENV") != "development":
                        continue
                    st.session_state[f"{tool}_enabled"] = False

        for tool, details in TOOLS.items():
            # if tool name starts with local and not in development, skip the tool
            if tool.startswith("local") and os.getenv("ENV") != "development":
                continue
            tool_enabled_key = f"{tool}_enabled"
            prompt_key = f"{tool}_prompt"
            previous_state = st.session_state[tool_enabled_key]
            current_state = st.checkbox(
                f"Enable {tool}",
                value=previous_state,
                key=tool_enabled_key,
                on_change=update_selected_tools,
                args=(tool, tool_enabled_key),
            )
            if current_state:
                selected_tools.append(tool)

            st.text_area(
                f"{details['description']}",
                value=st.session_state[prompt_key],
                key=prompt_key,
                disabled=not st.session_state[tool_enabled_key],
                on_change=update_prompt_state,
                args=(prompt_key,),
            )
            if "params" in details:
                st.markdown(f"**Parameters for {tool}:**")
                for param_key, param_details in details["params"].items():
                    param_state_key = f"{tool}_{param_key}"
                    st.session_state[param_state_key] = st.session_state.get(
                        param_state_key, param_details["default"]
                    )

                    if "choices" in param_details:
                        st.selectbox(
                            param_details["description"],
                            options=param_details["choices"],
                            key=param_state_key,
                            disabled=not st.session_state[tool_enabled_key],
                        )
                    else:
                        data_type = param_details.get("type", "str")
                        if data_type == "int":
                            st.number_input(
                                param_details["description"],
                                value=st.session_state[param_state_key],
                                key=param_state_key,
                                step=1,
                                disabled=not st.session_state[tool_enabled_key],
                            )
                        elif data_type == "float":
                            st.number_input(
                                param_details["description"],
                                value=st.session_state[param_state_key],
                                key=param_state_key,
                                step=0.01,
                                disabled=not st.session_state[tool_enabled_key],
                            )
                        elif data_type == "bool":
                            st.checkbox(
                                param_details["description"],
                                value=st.session_state[param_state_key],
                                key=param_state_key,
                                disabled=not st.session_state[tool_enabled_key],
                            )
                        else:
                            st.text_input(
                                param_details["description"],
                                value=st.session_state[param_state_key],
                                key=param_state_key,
                                disabled=not st.session_state[tool_enabled_key],
                            )

        st.session_state["selected_tools"] = selected_tools

    if selected_model.get("supports_tools", True):
        st.session_state["supports_tools"] = True
    else:
        st.warning(
            "NOTE: The current selected model does not support tools. "
            "Tools will not be used during the conversation."
        )
        st.session_state["supports_tools"] = False


def update_selected_tools(tool, tool_enabled_key):
    """
    Updates the list of selected tools in the session state based on the given tool's
    enabled status. If the tool is enabled, it will be added to the selected tools list,
    and if it is disabled, it will be removed.

    The function interacts with the Streamlit session state, ensuring that the updated
    list of selected tools is always synchronized with the current tool's enabled state.

    :param tool: The name or identifier of the tool to be added/removed from the
        selected tools list.
    :param tool_enabled_key: A key in the session state indicating whether the tool is
        currently enabled.
    :return: None
    """
    selected_tools = st.session_state.get("selected_tools", [])
    if st.session_state[tool_enabled_key]:
        if tool not in selected_tools:
            selected_tools.append(tool)
    else:
        if tool in selected_tools:
            selected_tools.remove(tool)
    st.session_state["selected_tools"] = selected_tools


def update_prompt_state(prompt_key):
    """
    Updates the Streamlit session state for a given key to its current value.
    This function directly modifies the session state to ensure that the
    session state variable corresponding to the provided key is synchronized
    with its own value. It assumes that the key exists in the session state
    and its value can be updated accordingly.

    :param prompt_key: The key in the session state dictionary whose value
                       should be updated.
    :type prompt_key: str
    :return: None
    """
    st.session_state[prompt_key] = st.session_state[prompt_key]
