# BiliCore Streamlit Implementation Documentation

## Overview

BiliCore provides a comprehensive Streamlit-based web interface for benchmarking and interacting with Large Language Models (LLMs). The Streamlit application serves as the primary user interface for the BiliCore framework, enabling researchers and developers to:

- Configure and compare different LLM providers (AWS Bedrock, Google Vertex AI, Azure OpenAI, OpenAI, local models)
- Customize chatbot behavior with configurable prompts and personas
- Select and configure external tools for enhanced conversational capabilities
- Manage conversation state and memory strategies
- Export and import configurations for reproducible testing

This document provides a detailed technical overview of the Streamlit implementation architecture.

---

## Table of Contents

1. [Application Architecture](#application-architecture)
2. [Entry Point and Initialization](#entry-point-and-initialization)
3. [UI Component Structure](#ui-component-structure)
4. [Session State Management](#session-state-management)
5. [Authentication System](#authentication-system)
6. [Configuration Panels](#configuration-panels)
7. [Chat Interface](#chat-interface)
8. [LangGraph Integration](#langgraph-integration)
9. [Query Processing](#query-processing)
10. [Checkpointing and State Persistence](#checkpointing-and-state-persistence)
11. [Utility Functions](#utility-functions)
12. [Configuration Options](#configuration-options)
13. [Component Diagrams](#component-diagrams)

---

## Application Architecture

### Directory Structure

```
bili/
├── streamlit_app.py              # Main entry point
├── streamlit_ui/
│   ├── __init__.py
│   ├── query/
│   │   ├── __init__.py
│   │   └── streamlit_query_handler.py   # Query processing logic
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── auth_ui.py                   # Authentication UI components
│   │   ├── chat_interface.py            # Main chat interface
│   │   ├── configuration_panels.py      # LLM/tool configuration panels
│   │   └── ui_auth_manager.py           # Streamlit-specific auth manager
│   └── utils/
│       ├── __init__.py
│       ├── state_management.py          # Session state utilities
│       └── streamlit_utils.py           # Caching decorators
├── config/
│   ├── llm_config.py                    # LLM model configurations
│   └── tool_config.py                   # Tool configurations
├── prompts/
│   └── default_prompts.json             # Default prompt templates
├── checkpointers/
│   └── checkpointer_functions.py        # Checkpointer initialization
└── loaders/
    └── langchain_loader.py              # LangGraph agent builder
```

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Application                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐   │
│  │   auth_ui    │───>│  chat_interface  │───>│ configuration_panels │   │
│  │              │    │                  │    │                     │   │
│  │ - Login      │    │ - Message Input  │    │ - LLM Selection     │   │
│  │ - Signup     │    │ - Chat Display   │    │ - Tool Config       │   │
│  │ - Sign Out   │    │ - State Mgmt     │    │ - Prompt Editor     │   │
│  └──────────────┘    └──────────────────┘    └─────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│                    ┌──────────────────┐                                 │
│                    │  Query Handler   │                                 │
│                    │                  │                                 │
│                    │ process_query()  │                                 │
│                    └────────┬─────────┘                                 │
│                             │                                           │
└─────────────────────────────┼───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        LangGraph Agent                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ START → persona_and_summary → inject_datetime → react_agent →   │   │
│  │         update_timestamp → trim_summarize → normalize_state → END│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐     │
│  │ LLM Model   │    │   Tools     │    │     Checkpointer        │     │
│  │ (Bedrock,   │    │ (OpenSearch,│    │ (PostgreSQL, MongoDB,   │     │
│  │  Vertex,    │    │  Weather,   │    │  Memory)                │     │
│  │  Azure...)  │    │  SERP...)   │    │                         │     │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Entry Point and Initialization

### Main Entry Point (`bili/streamlit_app.py`)

The application entry point performs the following initialization sequence:

```python
def main():
    # 1. Configure Streamlit page layout and branding
    configure_streamlit()

    # 2. Initialize authentication manager with providers
    st.session_state.auth_manager = initialize_auth_manager(
        auth_provider_name="sqlite",
        profile_provider_name="sqlite",
        role_provider_name="sqlite",
    )

    # 3. Verify user authentication
    check_auth()

    # 4. Get appropriate checkpointer (PostgreSQL, MongoDB, or Memory)
    checkpointer = get_checkpointer()

    # 5. Run main application page
    run_app_page(checkpointer)
```

### Page Configuration

The `configure_streamlit()` function sets up:

- **Page Title**: "Bili Core Sandbox Application"
- **Page Icon**: Custom logo from `bili/images/logo.png`
- **Layout**: Wide mode for better utilization of screen space
- **Welcome Content**: Introduction text and links to GitHub repository

### Running the Application

```bash
# From the bili-core directory
streamlit run bili/streamlit_app.py

# Or using the alias defined in the Docker container
streamlit
```

---

## UI Component Structure

### Component Hierarchy

```
streamlit_app.py (main)
│
├── configure_streamlit()
│   └── Page configuration, logo, welcome text
│
├── check_auth()
│   ├── is_authenticated() → Check session state
│   └── display_login_signup() → Login/Signup form
│
└── run_app_page(checkpointer)
    │
    ├── display_configuration_panels()
    │   ├── Import/Export Configuration (expander)
    │   ├── LLM Configuration (expander)
    │   │   ├── Model Type Selection
    │   │   ├── Model Name Selection
    │   │   ├── Temperature, Top-p, Top-k sliders
    │   │   ├── Seed value input
    │   │   ├── Max output tokens
    │   │   ├── Structured output (JSON schema)
    │   │   └── Thinking budget (Gemini 2.5)
    │   ├── Prompt Customization (expander)
    │   │   ├── Template selection
    │   │   └── System prefix editor
    │   └── Tool Selection (expander)
    │       ├── Enable/disable toggles
    │       ├── Tool prompts
    │       └── Tool parameters
    │
    ├── display_state_management_management()
    │   ├── Memory Limit Type (message_count/token_length)
    │   ├── Memory Strategy (summarize/trim)
    │   ├── Memory limit values (k and trim_k)
    │   └── Warning if no conversation chain loaded
    │
    ├── Load Configuration Button
    │   └── load_system_components(checkpointer)
    │
    ├── display_model_configuration()
    │   ├── LangChain/LangGraph Configuration (expander)
    │   ├── Chat History Configuration (expander)
    │   └── Tool Configuration (expander)
    │
    └── Conversation Sandbox
        ├── Query form (text area + submit button)
        ├── display_state_management(form)
        │   ├── Last human/AI message display
        │   ├── Intermediate steps expander
        │   ├── Clear Conversation State button
        │   ├── Export Conversation State button
        │   ├── Import Conversation State uploader
        │   └── Current Conversation State (JSON expander)
        └── Warning if configuration not loaded
```

---

## Session State Management

### Core Session State Variables

The Streamlit application uses `st.session_state` extensively to maintain state across reruns. Key session state variables include:

#### Authentication State
| Variable | Type | Description |
|----------|------|-------------|
| `auth_manager` | `UIAuthManager` | Authentication manager instance |
| `user_info` | `dict` | User account information (email, uid, etc.) |
| `user_profile` | `dict` | User profile data |
| `role` | `str` | User role ("researcher", "admin", "user") |
| `auth_warning` | `str` | Warning message for auth errors |
| `auth_success` | `str` | Success message for auth operations |
| `needs_profile_creation` | `bool` | Flag for incomplete profiles |

#### Model Configuration State
| Variable | Type | Description |
|----------|------|-------------|
| `model_type` | `str` | LLM provider key (e.g., "remote_aws_bedrock") |
| `model_name` | `str` | Display name of selected model |
| `model_id` | `str` | API identifier for the model |
| `model_config` | `object` | Loaded LLM model instance |
| `model_kwargs` | `dict` | Additional model parameters |
| `temperature` | `float` | LLM temperature setting |
| `top_p` | `float` | Nucleus sampling parameter |
| `top_k` | `int` | Top-k sampling parameter |
| `seed_value` | `int` | Random seed for reproducibility |
| `max_output_tokens` | `int` | Maximum tokens in response |
| `thinking_budget` | `int` | Extended thinking budget (Gemini 2.5) |

#### Memory Configuration State
| Variable | Type | Description |
|----------|------|-------------|
| `memory_limit_type` | `str` | "message_count" or "token_length" |
| `memory_strategy` | `str` | "summarize" or "trim" |
| `memory_limit_value` | `int` | Threshold before memory management (k) |
| `memory_limit_trim_value` | `int` | Target after memory management (trim_k) |

#### Conversation State
| Variable | Type | Description |
|----------|------|-------------|
| `conversation_chain` | `CompiledStateGraph` | LangGraph agent instance |
| `is_processing_query` | `bool` | Form disable flag during processing |
| `state_cleared` | `bool` | Flag for state clear confirmation |
| `state_imported` | `bool` | Flag for state import confirmation |

#### Tool Configuration State
| Variable | Type | Description |
|----------|------|-------------|
| `selected_tools` | `list[str]` | List of enabled tool names |
| `supports_tools` | `bool` | Whether selected model supports tools |
| `{tool}_enabled` | `bool` | Enable flag for each tool |
| `{tool}_prompt` | `str` | Custom prompt for each tool |
| `{tool}_{param}` | `varies` | Tool-specific parameters |

#### Prompt Configuration State
| Variable | Type | Description |
|----------|------|-------------|
| `selected_prompt_template` | `str` | Selected template name |
| `persona` | `str` | System prompt/persona text |
| `prompt_description` | `str` | Description of selected template |

### State Management Utilities (`bili/streamlit_ui/utils/state_management.py`)

```python
def disable_form():
    """Set processing flag to disable form during query processing."""
    st.session_state.is_processing_query = True

def enable_form():
    """Clear processing flag to re-enable form."""
    st.session_state.is_processing_query = False

def get_state_config():
    """Create configuration dict for LangGraph state management."""
    email = st.session_state.get("user_info", {}).get("email")
    return {
        "configurable": {
            "thread_id": f"{email}",
        },
    }
```

### State Persistence Pattern

The application follows a consistent pattern for state persistence:

```python
# 1. Initialize default value if not in session state
if "memory_limit_type" not in st.session_state:
    st.session_state["memory_limit_type"] = "message_count"

# 2. Create widget with session state key
st.session_state["memory_limit_type"] = st.selectbox(
    "Memory Limit Type",
    ["token_length", "message_count"],
    index=["token_length", "message_count"].index(
        st.session_state["memory_limit_type"]
    ),
)
```

---

## Authentication System

### Architecture

The authentication system uses a provider-based architecture with three types of providers:

1. **Auth Provider**: Handles user authentication (sign-in, sign-up, password reset)
2. **Profile Provider**: Manages user profile data
3. **Role Provider**: Handles user role assignment and verification

### UIAuthManager (`bili/streamlit_ui/ui/ui_auth_manager.py`)

The `UIAuthManager` extends the base `AuthManager` class with Streamlit-specific UI integration:

```python
class UIAuthManager(AuthManager):
    def sign_in(self, email, password, first_name=None, last_name=None):
        """Sign in user and update Streamlit session state."""

    def create_account(self, email, password, first_name, last_name, existing_user):
        """Create new account or complete profile for existing user."""

    def reset_password(self, email):
        """Send password reset email."""

    def sign_out(self):
        """Clear session state and sign out user."""

    def delete_account(self, password):
        """Delete user account after password verification."""

    def attempt_reauthentication(self):
        """Attempt to reauthenticate using stored auth_info."""
```

### Authentication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Authentication Flow                          │
└─────────────────────────────────────────────────────────────────┘

User visits app
       │
       ▼
┌──────────────┐     No      ┌──────────────────┐
│is_authenticated()├────────>│ display_login_   │
│              │             │ signup()         │
└──────┬───────┘             └────────┬─────────┘
       │ Yes                          │
       ▼                              ▼
┌──────────────┐              ┌───────────────┐
│ Show welcome │              │ Login/Signup  │
│ message +    │              │ Form          │
│ Sign Out btn │              └───────┬───────┘
└──────────────┘                      │
                                      ▼
                              ┌───────────────┐
                              │ sign_in() or  │
                              │ create_account│
                              └───────┬───────┘
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                    ┌──────▼──────┐      ┌──────▼──────┐
                    │ Success:    │      │ Failure:    │
                    │ Set user_   │      │ Set auth_   │
                    │ info, role  │      │ warning     │
                    │ st.rerun()  │      │ st.rerun()  │
                    └─────────────┘      └─────────────┘
```

### Role-Based Access

The application supports three user roles:

- **admin**: Full access to all features
- **researcher**: Full access to benchmarking features
- **user**: Pending approval, limited access

```python
def is_authenticated():
    if "user_info" in st.session_state and "role" in st.session_state:
        if st.session_state.role in ["researcher", "admin"]:
            return True
    return False
```

---

## Configuration Panels

### LLM Configuration Panel

The LLM configuration panel (`display_configuration_panels()` in `configuration_panels.py`) provides comprehensive model configuration:

#### Model Selection
- **LLM Type**: Dropdown for provider selection (AWS Bedrock, Google Vertex AI, Azure OpenAI, OpenAI, Local models)
- **LLM Model**: Dropdown for specific model selection within provider

#### Model Parameters (dynamically shown based on model capabilities)
| Parameter | Widget | Condition |
|-----------|--------|-----------|
| Temperature | Number input (0.0 - max) | `supports_temperature` |
| Top-p | Number input (0.0 - 1.0) | `supports_top_p` |
| Top-k | Number input (1 - max) | `supports_top_k` |
| Seed Value | Number input | `supports_seed` |
| Max Output Tokens | Number input | `supports_max_output_tokens` |
| Max Retries | Number input | `supports_max_retries` |
| Response MIME Type | Selectbox | `supports_structured_output` |
| JSON Schema | Text area | `supports_structured_output` + JSON MIME |
| Thinking Budget | Number input | `supports_thinking_budget` |

#### Supported LLM Providers

```
LLM_MODELS = {
    "remote_aws_bedrock": {
        "models": [
            "Amazon Nova Pro/Premier/Lite/Micro",
            "Amazon Titan Text G1",
            "AI21 Jamba 1.5",
            "Anthropic Claude 3/3.5/3.7/4 series",
            "Cohere Command R/R+",
            "DeepSeek-R1",
            "Meta Llama 3/3.1/3.2/3.3/4 series",
            "Mistral models",
            "TwelveLabs Pegasus"
        ]
    },
    "remote_google_vertex": {
        "models": [
            "Gemini 2.5 Pro/Flash/Flash Lite",
            "Gemini 2.0 Flash/Flash Lite",
            "Gemini 1.5 Pro/Flash",
            "Gemini 1.0 Pro"
        ]
    },
    "remote_azure_openai": {
        "models": [
            "GPT-4.1/4.1 mini/4.1 nano",
            "GPT-4o/4o mini",
            "GPT-4 Turbo",
            "o1/o1-mini/o3/o3-mini/o3-pro/o4-mini",
            "GPT-3.5 Turbo"
        ]
    },
    "remote_openai": {
        "models": [
            "GPT-4o/4o mini",
            "GPT-4 Turbo",
            "o1/o1-mini/o3-mini",
            "GPT-3.5 Turbo"
        ]
    },
    "local_llamacpp": { "models": ["LlamaCpp Local Model"] },
    "local_huggingface": { "models": ["HuggingFace Local Model"] }
}
```

### Prompt Customization Panel

- **Template Selection**: Dropdown to select from predefined prompt templates loaded from `default_prompts.json`
- **System Prefix Editor**: Text area for editing the persona/system prompt

### Tool Selection Panel

For each available tool:
- **Enable/Disable Toggle**: Checkbox to enable or disable the tool
- **Tool Prompt**: Text area for customizing the tool's description
- **Tool Parameters**: Dynamic inputs based on tool configuration

#### Available Tools

```python
TOOLS = {
    "local_faiss_retriever": {...},      # FAISS in-memory vector store
    "aws_opensearch_retriever": {...},   # Amazon OpenSearch
    "weather_api_tool": {...},           # OpenWeatherMap API
    "weather_gov_api_tool": {...},       # weather.gov API
    "free_weather_api_tool": {...},      # weatherapi.com
    "serp_api_tool": {...},              # Search engine results
    "mock_tool": {...},                  # Testing tool
}
```

### Import/Export Configuration

The configuration panel supports:
- **Import**: Upload a JSON file to restore all configuration settings
- **Export**: Download current configuration as JSON file

Exported configuration includes:
- Model type, name, ID, and all parameters
- Memory strategy settings
- Selected tools and their prompts
- System persona/prompt

---

## Chat Interface

### Main Chat Interface (`bili/streamlit_ui/ui/chat_interface.py`)

#### `run_app_page(checkpointer)`

Main function that orchestrates the entire chat interface:

```python
def run_app_page(checkpointer=None):
    # 1. Authentication check
    if not is_authenticated():
        st.session_state.auth_manager.attempt_reauthentication()
        if not is_authenticated():
            display_login_signup()
            return

    # 2. Enable input forms
    enable_form()

    # 3. Display configuration panels
    display_configuration_panels()

    # 4. Display memory management options
    display_state_management_management()

    # 5. Load Configuration button
    if st.button("Load Configuration"):
        load_system_components(checkpointer)
        st.success("Configuration loaded successfully.")
        st.rerun()

    # 6. Display current model configuration
    display_model_configuration()

    # 7. Conversation Sandbox
    if "conversation_chain" in st.session_state:
        # Show query form and chat history
        form = st.form(key="conversation_form", clear_on_submit=True)
        user_query = form.text_area("Ask a question", ...)
        if form.form_submit_button("Submit", ...):
            process_query(st.session_state["conversation_chain"], user_query)
            enable_form()
            st.rerun()

        display_state_management(form)
    else:
        st.warning("Please load the configuration before asking a question.")
```

#### State Management Display (`display_state_management`)

Provides comprehensive conversation state visualization:

1. **Message Display**: Shows the last human message and AI response
2. **Intermediate Steps**: Expander showing tool calls and processing messages between user input and final response
3. **State Operations**:
   - Clear Conversation State button
   - Export Conversation State as JSON
   - Import Conversation State from JSON
4. **State Inspector**: JSON expander showing current conversation state

#### Memory Management Configuration (`display_state_management_management`)

Configures how conversation history is managed:

- **Memory Limit Type**:
  - `message_count`: Limit by number of messages
  - `token_length`: Limit by total tokens

- **Memory Strategy**:
  - `summarize`: Compress old messages into a summary
  - `trim`: Simply remove old messages

- **Memory Limit Values**:
  - `k`: Threshold before triggering memory management
  - `trim_k`: Target size after memory management

---

## LangGraph Integration

### Agent Building (`bili/loaders/langchain_loader.py`)

The Streamlit UI builds a LangGraph agent using `build_agent_graph()`:

```python
def load_system_components(checkpointer):
    # 1. Load LLM model with configured parameters
    model = load_model(
        model_type=st.session_state["model_type"],
        model_name=st.session_state["model_id"],
        max_tokens=st.session_state.get("max_output_tokens"),
        temperature=st.session_state.get("temperature"),
        top_p=st.session_state.get("top_p"),
        top_k=st.session_state.get("top_k"),
        seed=st.session_state.get("seed_value"),
        **model_kwargs,
    )

    # 2. Initialize selected tools
    tools = initialize_tools(
        active_tools=active_tools,
        tool_prompts=tool_prompts,
        tool_params=tool_params,
    )

    # 3. Configure node kwargs
    node_kwargs = {
        "llm_model": model,
        "persona": st.session_state.get("persona"),
        "tools": tools,
        "summarize_llm_model": model,
        "memory_strategy": memory_strategy,
        "memory_limit_type": memory_limit_type,
        "k": memory_limit_value,
        "trim_k": memory_limit_trim_value,
        "current_user": st.session_state.get("user_profile"),
        "model_type": st.session_state.get("model_type"),
    }

    # 4. Build the agent graph
    conversation_agent = build_agent_graph(
        checkpoint_saver=checkpointer,
        graph_definition=graph_definition,
        node_kwargs=node_kwargs,
        state=State,
    )

    st.session_state["conversation_chain"] = conversation_agent
```

### Default Graph Definition

```
START
  │
  ▼
add_persona_and_summary ──► inject_current_datetime ──► react_agent
                                                            │
                                                            ▼
                                                    update_timestamp
                                                            │
                                                            ▼
                                                    trim_summarize
                                                            │
                                                            ▼
                                                    normalize_state
                                                            │
                                                            ▼
                                                           END
```

### State Schema

```python
class State(MessagesState):
    summary: str                    # Conversation summary
    owner: str                      # User identifier
    previous_message_time: datetime # Last message timestamp
    current_message_time: datetime  # Current message timestamp
    delta_time: float              # Time between messages
    disable_summarization: bool    # Summarization toggle
    template_dict: dict            # Prompt templates
    title: str                     # Conversation title
    tags: List[str]               # Conversation tags
    llm_config: dict              # Runtime LLM config
```

---

## Query Processing

### Query Handler (`bili/streamlit_ui/query/streamlit_query_handler.py`)

```python
def process_query(conversation_chain, user_query):
    # 1. Get state configuration (thread_id based on user email)
    config = get_state_config()

    # 2. Convert query to HumanMessage
    input_message = HumanMessage(content=user_query)

    # 3. Invoke the conversation chain
    result = conversation_chain.invoke(
        {"messages": [input_message], "verbose": False},
        config
    )

    # 4. Extract and return the final AI message
    if isinstance(result, dict) and "messages" in result:
        final_msg = result["messages"][-1]
        return final_msg.pretty_repr()

    return "No response or invalid format."
```

### Processing Flow

```
User Input
    │
    ▼
┌───────────────────┐
│ disable_form()    │  ← Prevent duplicate submissions
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ process_query()   │
│                   │
│ 1. Get config     │
│ 2. Create message │
│ 3. Invoke chain   │
│ 4. Extract result │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ enable_form()     │  ← Re-enable form
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ st.rerun()        │  ← Refresh UI with new state
└───────────────────┘
```

---

## Checkpointing and State Persistence

### Checkpointer Selection (`bili/checkpointers/checkpointer_functions.py`)

The application automatically selects the appropriate checkpointer based on environment variables:

```python
def get_checkpointer():
    # Priority 1: PostgreSQL
    if os.getenv("POSTGRES_CONNECTION_STRING"):
        return get_pg_checkpointer()

    # Priority 2: MongoDB
    if os.getenv("MONGO_CONNECTION_STRING"):
        return get_mongo_checkpointer()

    # Priority 3: In-memory (fallback)
    return QueryableMemorySaver()
```

### Supported Checkpointers

| Checkpointer | Use Case | Persistence |
|--------------|----------|-------------|
| PostgresSaver | Production, multi-user | Persistent |
| MongoDBSaver | Production, flexible schema | Persistent |
| QueryableMemorySaver | Development, testing | Session-only |

### State Configuration

Thread identification uses user email for consistent state retrieval:

```python
def get_state_config():
    email = st.session_state.get("user_info", {}).get("email")
    return {
        "configurable": {
            "thread_id": f"{email}",
        },
    }
```

---

## Utility Functions

### Conditional Caching (`bili/streamlit_ui/utils/streamlit_utils.py`)

Custom decorators for environment-aware caching:

```python
@conditional_cache_resource()
def expensive_resource_function():
    """Only cached when running in Streamlit environment."""
    pass

@conditional_cache_data()
def expensive_data_function():
    """Only cached when running in Streamlit environment."""
    pass
```

These decorators check for `STREAMLIT_SERVER_ADDRESS` environment variable to determine whether to apply Streamlit caching.

### Message Formatting (`bili/utils/langgraph_utils.py`)

```python
def format_message_with_citations(message):
    """Format AIMessage with citation metadata if present."""
    formatted = message.pretty_repr()

    citations = message.response_metadata.get("citation_metadata", {}).get("citations", [])
    if citations:
        formatted += "\n\n**Citations:**\n"
        for citation in citations:
            formatted += f"- [{citation['title']}]({citation['uri']})\n"

    return formatted

def clear_state(state):
    """Remove all messages from state and return cleared state."""
    messages = state.get("messages", state.values.get("messages", []))
    messages_to_remove = [RemoveMessage(id=msg.id) for msg in messages]
    return {"messages": messages_to_remove, "summary": ""}
```

---

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_CONNECTION_STRING` | PostgreSQL connection string | None |
| `MONGO_CONNECTION_STRING` | MongoDB connection string | None |
| `ENV` | Environment mode ("development" shows local models) | None |
| `DEFAULT_PROMPT_PATH` | Path to prompt templates JSON | `bili/prompts/default_prompts.json` |
| `TOKENIZERS_PARALLELISM` | Disable tokenizer parallelism | "false" |

### Configuration Files

#### `bili/prompts/default_prompts.json`
```json
{
  "templates": {
    "default": {
      "description": "...",
      "persona": "..."
    }
  }
}
```

#### `bili/config/llm_config.py`
Contains `LLM_MODELS` dictionary with all supported model configurations.

#### `bili/config/tool_config.py`
Contains `TOOLS` dictionary with all available tool configurations.

---

## Component Diagrams

### Session State Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         st.session_state                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │ Auth State  │    │Model Config │    │ Memory Cfg  │                 │
│  │             │    │             │    │             │                 │
│  │ user_info   │    │ model_type  │    │ memory_     │                 │
│  │ role        │    │ model_id    │    │ strategy    │                 │
│  │ auth_manager│    │ temperature │    │ memory_     │                 │
│  └──────┬──────┘    │ top_p/top_k │    │ limit_type  │                 │
│         │           │ ...         │    │ k, trim_k   │                 │
│         │           └──────┬──────┘    └──────┬──────┘                 │
│         │                  │                  │                         │
│         ▼                  ▼                  ▼                         │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │                    load_system_components()               │         │
│  └────────────────────────────┬─────────────────────────────┘         │
│                               │                                        │
│                               ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │              conversation_chain (CompiledStateGraph)         │      │
│  │                                                              │      │
│  │  ┌─────────┐  ┌───────┐  ┌────────────┐  ┌────────────┐    │      │
│  │  │ LLM     │  │ Tools │  │Checkpointer│  │   State    │    │      │
│  │  │ Model   │  │       │  │            │  │   Schema   │    │      │
│  │  └─────────┘  └───────┘  └────────────┘  └────────────┘    │      │
│  └─────────────────────────────────────────────────────────────┘      │
│                                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Request Processing Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Request Processing Lifecycle                       │
└─────────────────────────────────────────────────────────────────────────┘

1. User submits query via form
   │
   ▼
2. disable_form() - Prevent double submission
   │
   ▼
3. process_query(conversation_chain, user_query)
   │
   ├──► get_state_config() → {thread_id: user_email}
   │
   ├──► HumanMessage(content=user_query)
   │
   └──► conversation_chain.invoke({messages: [...]}, config)
        │
        │    ┌─────────────────────────────────────────────┐
        └───►│ LangGraph Execution Pipeline:               │
             │                                             │
             │ add_persona_and_summary                     │
             │    └── Inject persona, load summary         │
             │                                             │
             │ inject_current_datetime                     │
             │    └── Add timestamp to system message      │
             │                                             │
             │ react_agent                                 │
             │    └── LLM reasoning + tool calls           │
             │                                             │
             │ update_timestamp                            │
             │    └── Track message timing                 │
             │                                             │
             │ trim_summarize                              │
             │    └── Apply memory management strategy     │
             │                                             │
             │ normalize_state                             │
             │    └── Clean up tool calls                  │
             └─────────────────────────────────────────────┘
                                │
                                ▼
4. Result returned to Streamlit
   │
   ▼
5. enable_form() - Re-enable form
   │
   ▼
6. st.rerun() - Refresh UI with updated state
   │
   ▼
7. display_state_management() - Show conversation history
```

---

## Customization and Extension

### Adding New LLM Providers

1. Add provider configuration to `bili/config/llm_config.py`:
```python
LLM_MODELS["new_provider"] = {
    "name": "Provider Name",
    "description": "Description",
    "model_help": "https://docs.example.com",
    "models": [
        {
            "model_name": "Model Display Name",
            "model_id": "model-api-id",
            "supports_temperature": True,
            # ... other capabilities
        }
    ]
}
```

2. Add loader support in `bili/loaders/llm_loader.py`

### Adding New Tools

1. Add tool configuration to `bili/config/tool_config.py`:
```python
TOOLS["new_tool"] = {
    "description": "Tool description",
    "enabled": False,
    "default_prompt": "Tool prompt for LLM",
    "params": {
        "param1": {
            "description": "Parameter description",
            "default": "default_value",
            "type": "str"
        }
    }
}
```

2. Implement tool in `bili/tools/new_tool.py`
3. Register in `bili/loaders/tools_loader.py`

### Adding Custom Graph Nodes

1. Create node in `bili/nodes/custom_node.py`:
```python
from functools import partial
from bili.graph_builder.classes.node import Node

def custom_node_function(state, **kwargs):
    # Node logic
    return state

def custom_node(**node_kwargs):
    return Node(
        name="custom_node",
        function=partial(custom_node_function, **node_kwargs)
    )
```

2. Register in `bili/loaders/langchain_loader.py`:
```python
GRAPH_NODE_REGISTRY["custom_node"] = custom_node
```

3. Add to graph definition in `load_system_components()`:
```python
graph_definition.insert(index, custom_node(edges=["next_node"]))
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Form disabled after error | Manually call `enable_form()` or refresh page |
| Configuration not loading | Check console for errors, verify API credentials |
| Authentication loop | Clear browser cookies, check auth provider config |
| Memory issues with large conversations | Reduce `k` value or use `trim` strategy |
| Tools not appearing | Verify `supports_tools` for selected model |

### Debug Mode

Enable LangChain debug logging by setting log level to DEBUG:
```python
import logging
logging.getLogger("bili").setLevel(logging.DEBUG)
```

---

## References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [BiliCore GitHub Repository](https://github.com/msu-denver/bili-core)
