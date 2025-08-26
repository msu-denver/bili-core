"""
Module: chat_interface

This module provides the main interface for interacting with a chatbot within a
Streamlit application. It includes functions to run the application page, manage
chat history, display model configurations, and handle user authentication.

Functions:
    - run_app_page(checkpointer=None):
      Executes the primary application page for configuring and interacting with
      a chatbot.
    - display_state_management_management():
      Manages display and configuration for chat history management using
      Streamlit components.
    - display_state_management(question_form):
      Handles the visualization and management of the current state of an ongoing
      conversation.
    - display_model_configuration():
      Displays detailed configuration information of the currently loaded
      language model, chat history, and tools in the session state.
    - load_system_components(checkpointer):
      Load and initialize all necessary components for a conversational system,
      including the model, tools, memory strategy, and conversation chain.

Dependencies:
    - streamlit: Provides the Streamlit library for building web applications.
    - langchain_core.messages: Imports HumanMessage and AIMessage for message
      handling.
    - langgraph.checkpoint.memory: Imports MemorySaver for in-memory checkpointing.
    - langgraph.checkpoint.serde.jsonplus: Imports JsonPlusSerializer for JSON
      serialization.
    - bili.loaders.langchain_loader: Imports load_langgraph_agent for loading
      language graph agents.
    - bili.loaders.llm_loader: Imports load_model for loading language models.
    - bili.checkpointers.checkpointer_functions: Imports get_state_config for
      retrieving state configuration.
    - bili.streamlit.query.streamlit_query_handler: Imports process_query for
      processing user queries.
    - bili.config.tool_config: Imports TOOLS for tool configuration.
    - bili.utils.langgraph_utils: Imports format_message_with_citations and
      clear_state for message formatting and state clearing.
    - bili.streamlit.ui.configuration_panels: Imports display_configuration_panels
      for displaying configuration panels.
    - bili.streamlit.ui.auth_ui: Imports is_authenticated, display_login_signup,
      and st.session_state.auth_manager for user authentication.
    - bili.streamlit.utils.state_management: Imports enable_form and disable_form
      for form state management.

Usage:
    This module is intended to be used within a Streamlit application to manage
    chatbot interactions, including configuration, authentication, and state
    management.

Example:
    from bili.streamlit.ui.chat_interface import run_app_page

    # Run the application page
    run_app_page()
"""
import json
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from bili.config.tool_config import TOOLS
from bili.loaders.langchain_loader import DEFAULT_GRAPH_DEFINITION, build_agent_graph
from bili.loaders.llm_loader import load_model
from bili.loaders.tools_loader import initialize_tools
from bili.streamlit_ui.query.streamlit_query_handler import process_query
from bili.streamlit_ui.ui.auth_ui import display_login_signup, is_authenticated
from bili.streamlit_ui.ui.configuration_panels import display_configuration_panels
from bili.streamlit_ui.utils.state_management import (
    disable_form,
    enable_form,
    get_state_config,
)
from bili.utils.langgraph_utils import State, clear_state, format_message_with_citations


def run_app_page(checkpointer=None):
    """
    Executes the primary application page for configuring and interacting with
    a chatbot. This method orchestrates authentication, user inputs, loading
    of system configurations, and interaction through a conversation sandbox.

    :param checkpointer: Checkpoint handler used to manage and restore system
        state and parameters, optional.
    :return: None
    """
    if not is_authenticated():
        st.session_state.auth_manager.attempt_reauthentication()
        if not is_authenticated():
            display_login_signup()
            return

    # Ensure input forms are enabled
    enable_form()

    # Display configuration panels
    display_configuration_panels()

    # Display state management
    display_state_management_management()

    # Load Configuration
    if st.button("Load Configuration", use_container_width=True):
        load_system_components(checkpointer)
        st.success("Configuration loaded successfully.")
        st.rerun()

    # Display current model configuration
    st.markdown("---")
    st.markdown("<h2>Current Model Configuration</h2>", unsafe_allow_html=True)
    display_model_configuration()
    st.markdown("---")

    # Display Chatbot Conversation Sandbox
    st.markdown(
        """<h2>Conversation Sandbox</h2>
        <p>Use the form below to ask a question to the chatbot using your 
        specified configuration settings.</p>""",
        unsafe_allow_html=True,
    )

    # Status container for feedback
    status_container = st.empty()

    # If the conversation_chain is loaded, show the query form and chat history
    if (
        "conversation_chain" in st.session_state
        and st.session_state["conversation_chain"]
    ):
        form = st.form(key="conversation_form", clear_on_submit=True)
        user_query = form.text_area(
            "Ask a question",
            key="user_query_input",
            disabled=st.session_state.is_processing_query,
        )
        if form.form_submit_button(
            label="Submit",
            disabled=st.session_state.is_processing_query
            or not st.session_state["conversation_chain"],
            on_click=disable_form,
            use_container_width=True,
        ):
            status_container.markdown(
                "**Status:** Processing...", unsafe_allow_html=True
            )
            process_query(
                st.session_state["conversation_chain"],
                user_query,
            )
            enable_form()
            st.rerun()

        display_state_management(form)
    else:
        st.warning("Please load the configuration before asking a question.")


def display_state_management_management():
    """
    Manages display and configuration for chat history management using Streamlit components.

    This function provides an interface for toggling and setting parameters related
    to memory limit type, memory management strategy, and editable configuration
    for the memory limit value. It uses Streamlit's expander layout to group settings
    pertinent to memory strategies and thresholds. It also addresses warning
    conditions for the absence of a conversation chain.

    :raises KeyError: Raised internally by Streamlit if incorrect keys are accessed
        or not initialized in ``st.session_state``.
    :raises ValueError: Raised by Streamlit during value range issues in inputs.
    """
    with st.expander("Chat History Management"):
        # Toggle between message_count vs token_length
        if "memory_limit_type" not in st.session_state:
            st.session_state["memory_limit_type"] = "message_count"
        st.session_state["memory_limit_type"] = st.selectbox(
            "Memory Limit Type",
            ["token_length", "message_count"],
            index=["token_length", "message_count"].index(
                st.session_state["memory_limit_type"]
            ),
            format_func=lambda x: (
                "Number of Messages" if x == "message_count" else "Total Tokens"
            ),
            help="Number of messages vs total tokens in conversation. Note: if you use"
            " Number of Messages, this will include all intermediate messages"
            " generated by the system (such as tool calls).",
        )

        # Memory Strategy
        if "memory_strategy" not in st.session_state:
            st.session_state["memory_strategy"] = "summarize"
        st.session_state["memory_strategy"] = st.selectbox(
            "Memory Strategy",
            ["summarize", "trim"],
            format_func=lambda x: "Summarize" if x == "summarize" else "Trim",
            index=["summarize", "trim"].index(st.session_state["memory_strategy"]),
            help="Summarize will take the removed messages and create a single summary"
            " message to retain the most important information about what was removed."
            " Trim will remove the messages and not retain any information about them.",
        )

        # The numeric input is always 'k', but we label it differently
        if st.session_state["memory_limit_type"] == "message_count":
            if st.session_state["memory_strategy"] == "trim":
                label = "Number of messages to keep before truncating (k)"
                trim_label = "Number of messages to trim to (trim_k)"
            else:
                label = "Number of messages to keep before summarizing (k)"
                trim_label = "Number of messages to summarize to (trim_k)"
        else:
            if st.session_state["memory_strategy"] == "trim":
                label = "Total token limit to keep before truncating (k)"
                trim_label = "Total token limit to keep after trimming (trim_k)"
            else:
                label = "Total token limit to keep before summarizing (k)"
                trim_label = "Total token limit to keep after summarizing (trim_k)"

        if "memory_limit_value" not in st.session_state:
            if st.session_state["memory_limit_type"] == "message_count":
                st.session_state["memory_limit_value"] = 15
            else:
                st.session_state["memory_limit_value"] = 10000
        st.session_state["memory_limit_value"] = st.number_input(
            label,
            min_value=4,
            max_value=2097152,
            value=st.session_state["memory_limit_value"],
            key="memory_limit_value_setting",
            help="If using Number of Messages to trim, this is how many messages to keep before trimming."
            " If using Total Tokens, this is how many tokens total are allowed before trimming.",
        )

        if "memory_limit_trim_value" not in st.session_state:
            if st.session_state["memory_limit_type"] == "message_count":
                st.session_state["memory_limit_trim_value"] = 15
            else:
                st.session_state["memory_limit_trim_value"] = 10000
        st.session_state["memory_limit_trim_value"] = st.number_input(
            trim_label,
            min_value=4,
            max_value=2097152,
            value=st.session_state["memory_limit_trim_value"],
            key="memory_limit_trim_value_setting",
            help="If using Number of Messages to trim, this is how many messages to trim to, after the"
            "trim limit is met. If using Total Tokens, this is how many tokens total are allowed "
            "to be retained after trimming.",
        )

        # If no conversation chain loaded, warn the user
        if "conversation_chain" not in st.session_state:
            st.warning("No conversation chain loaded.")
            return


def display_state_management(question_form):
    """
    Handles the visualization and management of the current state of an ongoing
    conversation. This feature provides functionalities such as viewing the
    conversation's overall state, including message history, conversation summaries,
    and tool call history. Additionally, it supports exporting the current state
    as a JSON file for persistence and allows users to import conversation states from
    JSON files, enabling effective testing and history replication.

    This function also provides functionality to inspect individual steps in the
    conversation process, such as displaying processing messages between interactions.

    :param question_form: A Streamlit form object used for rendering UI components
        related to displaying and managing the conversation's state.
    :return: None
    """
    st.markdown(
        f"""<h2>Conversation State Management</h2>
            <p>View the current state for the conversation, including conversation
            summary (if present), message history, and tool call history. You can also
            export the current state as JSON, and import state from a JSON file to
            overwrite the current state of the conversation. This allows you to effectively
            save and load chat histories between conversations, which is a useful tool during
            testing!</p>""",
        unsafe_allow_html=True,
    )

    if "conversation_chain" in st.session_state:
        config = get_state_config()

        # Retrieve the latest state
        latest_state = st.session_state["conversation_chain"].get_state(config)

        if latest_state is None:
            st.warning("No saved state to display.")
            return

        # Dynamically find the last human message and the last AI message.
        # These will be the most recent human message and the most recent AI message in the conversation.
        latest_state_messages = latest_state.values.get("messages", [])
        last_human_message = next(
            (
                msg
                for msg in reversed(latest_state_messages)
                if isinstance(msg, HumanMessage)
            ),
            None,
        )
        if last_human_message:
            question_form.markdown(format_message_with_citations(last_human_message))
            last_human_message_index = latest_state_messages.index(last_human_message)
        else:
            last_human_message_index = None

        last_ai_message = next(
            (
                msg
                for msg in reversed(latest_state_messages)
                if isinstance(msg, AIMessage)
            ),
            None,
        )
        if last_ai_message:
            last_ai_message_index = latest_state_messages.index(last_ai_message)
        else:
            last_ai_message_index = None

        processing_messages = []
        if last_human_message_index and last_ai_message_index:
            processing_messages = latest_state_messages[
                last_human_message_index + 1 : last_ai_message_index
            ]
        if len(processing_messages) > 0:
            with question_form.expander("Intermediate Steps"):
                for i, message in enumerate(processing_messages):
                    st.text_area(
                        f"Processing Message {i+1} (Type: {message.__class__.__name__})",
                        value=format_message_with_citations(message),
                        height=150,
                        disabled=True,
                        key=f"latest_processing_{i}",
                    )

        if last_ai_message:
            question_form.markdown(format_message_with_citations(last_ai_message))

        # Create three buttons. The first button will start a new conversation by
        # clearing the existing state. The second button will export the current state as JSON.
        # The third button will import a JSON file to overwrite the current state.

        # Clear button
        if st.button("Clear Conversation State", use_container_width=True):
            st.session_state["conversation_chain"].update_state(
                config, clear_state(latest_state)
            )
            st.session_state["state_cleared"] = True
            st.rerun()
        if st.session_state.get("state_cleared", False):
            st.success("Conversation state cleared!")
            st.session_state.update({"state_cleared": False})

        # Export button
        if st.button("Export Conversation State as JSON", use_container_width=True):
            st.download_button(
                label="Download Conversation State as JSON",
                data=JsonPlusSerializer().dumps(latest_state),
                file_name="conversation_state.json",
                mime="application/json",
                use_container_width=True,
            )

        # Import uploader
        uploaded_file = st.file_uploader(
            "Import Conversation State from JSON",
            type="json",
            on_change=lambda: st.session_state.update({"state_imported": False}),
        )
        if uploaded_file is not None:
            if not st.session_state.get("state_imported", False):
                imported_state = JsonPlusSerializer().loads(uploaded_file.read())
                # If imported_state is a list, take first value as state
                if isinstance(imported_state, list):
                    imported_state = imported_state[0]

                # Clear state first to remove existing messages and summary
                st.session_state["conversation_chain"].update_state(
                    config, clear_state(latest_state)
                )

                # Next, import the new state
                st.session_state["conversation_chain"].update_state(
                    config,
                    {
                        "messages": imported_state.get("messages", []),
                        "summary": imported_state.get("summary", ""),
                    },
                )
                st.session_state["state_imported"] = True
                st.rerun()
            elif "state_imported" in st.session_state:
                st.success("Configuration imported successfully!")

        container = st.expander("Current Conversation State", expanded=False)

        # Display the current state as JSON inside the container
        container.json(latest_state, expanded=True)


def display_model_configuration():
    """
    Displays detailed configuration information of the currently loaded language model,
    chat history, and tools in the session state. This function groups related settings
    under expandable sections for better organization. If the model configuration is not
    loaded, it displays a warning message.

    :param session_state: A built-in Streamlit session variable that maintains the state
        and stores model-related configurations, memory settings, and other tool
        information.

    :return: None
    """
    if "model_config" in st.session_state:
        with st.expander("LangChain/LangGraph Configuration Details"):
            st.text_area(
                "Model Configuration",
                value=str(st.session_state["model_config"]),
            )
            st.text_area(
                "Checkpointer Configuration",
                value=st.session_state["conversation_chain"].checkpointer,
            )

        with st.expander("Chat History Configuration"):
            if "memory_limit_type" in st.session_state:
                st.markdown(
                    f"**Memory Limit Type:** {st.session_state.get('memory_limit_type')}"
                )
            if "memory_strategy" in st.session_state:
                st.markdown(
                    f"**Memory Strategy Type:** {st.session_state.get('memory_strategy')}"
                )
            if "memory_limit_value" in st.session_state:
                st.markdown(
                    f"**Chat History 'k' Value:** {st.session_state.get('memory_limit_value')}"
                )
            if "memory_limit_trim_value" in st.session_state:
                st.markdown(
                    f"""**Chat History 'trim_k' Value:** {st.session_state.get('memory_limit_trim_value')}"""
                )

        if st.session_state.get("supports_tools", True):
            with st.expander("Tool Configuration"):
                for tool in st.session_state.get("selected_tools", []):
                    st.markdown(f"**{tool} Prompt:**")
                    st.text_area(
                        f"{tool} Prompt",
                        value=st.session_state.get(f"{tool}_prompt"),
                        height=100,
                        disabled=True,
                    )
    else:
        st.warning("Model configuration not loaded.")


def load_system_components(checkpointer):
    """
    Load and initialize all necessary components for a conversational system,
    including the model, tools, memory strategy, and conversation chain. The
    function also enables user input through a form interface. Various session
    state configurations are utilized to customize the behavior.

    :param checkpointer: Optional. A checkpoint-saving mechanism to save the
        state of the conversation chain. If not provided, defaults to the
        `MemorySaver` class.
    :type checkpointer: Optional[Any]

    :return: None
    """
    # Load the model that will be used for the conversation chain
    model_kwargs = st.session_state.get("model_kwargs", {})

    # Check if the llm config supports structured output
    if st.session_state.get("supports_structured_output", False):
        # If it does check if the mime type is json
        if st.session_state.get("response_mime_type") == "application/json":
            custom_schema = st.session_state.get("custom_response_schema")
            # If the custom schema is json and exists
            if custom_schema:
                try:
                    # Load the json schema
                    parsed_schema = json.loads(custom_schema)
                    model_kwargs["response_schema"] = parsed_schema
                except json.JSONDecodeError:
                    # Use default string schema if custom schema is invalid
                    model_kwargs["response_schema"] = {"type": "string"}
            else:
                # Default to string response
                model_kwargs["response_schema"] = {"type": "string"}

        # Handle MIME type
        response_mime_type = st.session_state.get("response_mime_type", "text/plain")
        if response_mime_type == "text/plain":
            model_kwargs["response_mime_type"] = "text/plain"
        elif response_mime_type == "application/json":
            model_kwargs["response_mime_type"] = "application/json"

    model = load_model(
        model_type=st.session_state["model_type"],
        model_name=st.session_state["model_id"],
        max_tokens=st.session_state.get("max_output_tokens", None),
        temperature=st.session_state.get("temperature", None),
        top_p=st.session_state.get("top_p", None),
        top_k=st.session_state.get("top_k", None),
        seed=st.session_state.get("seed_value", None),
        **model_kwargs,
    )
    st.session_state["model_config"] = model

    if st.session_state.get("supports_tools", True):
        # Load the active tools and their prompts
        active_tools = st.session_state.get("selected_tools", [])
        tool_prompts = {
            tool: st.session_state.get(f"{tool}_prompt") for tool in active_tools
        }
        tool_params = {}
        for tool in active_tools:
            if "params" in TOOLS[tool]:
                tool_params[tool] = {}
                for param_key in TOOLS[tool]["params"]:
                    tool_params[tool][param_key] = st.session_state.get(
                        f"{tool}_{param_key}",
                        TOOLS[tool]["params"][param_key]["default"],
                    )
            if "kwargs" in TOOLS[tool]:
                tool_kwargs = TOOLS[tool]["kwargs"]
                for key, value in tool_kwargs.items():
                    if key not in tool_params[tool]:
                        tool_params[tool][key] = value
    else:
        active_tools = None
        tool_prompts = None
        tool_params = None
    tools = initialize_tools(
        active_tools=active_tools,
        tool_prompts=tool_prompts,
        tool_params=tool_params,
    )

    # Grab the memory strategy toggles
    memory_strategy = st.session_state.get("memory_strategy")
    memory_limit_type = st.session_state.get("memory_limit_type")
    memory_limit_value = st.session_state.get("memory_limit_value")
    memory_limit_trim_value = st.session_state.get("memory_limit_trim_value")

    # If no checkpointer is provided, use MemorySaver for checkpoints
    if checkpointer is None:
        checkpointer = MemorySaver()

    # Load the conversation chain
    node_kwargs = {
        # Use the selected model for the agent
        "llm_model": model,
        # Use the currently selected system prompt as the persona for the agent
        "persona": st.session_state.get("persona"),
        # Use the currently selected tools for the agent
        "tools": tools,
        # Use the selected model for conversation summarization
        "summarize_llm_model": model,
        # Use the currently configured memory strategy for summarization within the agent
        "memory_strategy": memory_strategy,
        "memory_limit_type": memory_limit_type,
        "k": memory_limit_value,
        "trim_k": memory_limit_trim_value,
        "current_user": st.session_state.get("user_profile"),
    }

    # Add per-user state node to the graph nodes since UI is per-user
    graph_definition = DEFAULT_GRAPH_DEFINITION.copy()
    graph_definition.insert(1, "per_user_state")

    # Create the langgraph agent
    conversation_agent = build_agent_graph(
        # Use the checkpointer for state persistence to allow for conversation history
        # to be saved and restored
        checkpoint_saver=checkpointer,
        # Use the default graph definition
        graph_definition=graph_definition,
        # Pass in the node kwargs for the agent which are used to initialize
        # each node in the defined graph
        node_kwargs=node_kwargs,
        # Use the default State definition for the agent
        state=State,
    )

    # Store the conversation chain in session state
    st.session_state["conversation_chain"] = conversation_agent

    # Enable the form to allow user queries
    enable_form()
