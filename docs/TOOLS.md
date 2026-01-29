# Tools Framework

This document describes the BiliCore tools system for extending agent capabilities with external services and data sources.

## Overview

BiliCore provides an extensible tools framework that allows agents to:

- Retrieve information from vector databases (FAISS, OpenSearch)
- Fetch real-time data from external APIs (weather, search)
- Perform custom operations via user-defined tools

Tools are dynamically loaded based on configuration and can be enhanced with middleware.

## Available Tools

| Tool | Description | Default Enabled |
|------|-------------|-----------------|
| `aws_opensearch_retriever` | Amazon OpenSearch similarity search | Yes |
| `free_weather_api_tool` | Weather data from weatherapi.com | Yes |
| `serp_api_tool` | Search engine results via SERP API | Yes |
| `local_faiss_retriever` | Local FAISS vector search | No |
| `weather_api_tool` | OpenWeatherMap API | No |
| `weather_gov_api_tool` | US weather from weather.gov | No |
| `mock_tool` | Testing/simulation tool | No |

## Tool Configuration

### Configuration File: `bili/config/tool_config.py`

Each tool is defined with:

```python
TOOLS = {
    "tool_name": {
        "description": "Human-readable description",
        "enabled": True,  # Default enabled state
        "default_prompt": "Instructions for the LLM on how to use this tool",
        "params": {
            "param_name": {
                "description": "Parameter description",
                "default": "default_value",
                "type": "str",  # str, int, float, etc.
                "choices": ["option1", "option2"],  # Optional
            }
        },
        "kwargs": {
            # Additional static configuration
        }
    }
}
```

### Example: OpenSearch Retriever Config

```python
"aws_opensearch_retriever": {
    "description": "Retrieves facts using Amazon OpenSearch for similarity searches.",
    "enabled": True,
    "default_prompt": "This tool utilizes Amazon OpenSearch to retrieve facts...",
    "params": {
        "index_name": {
            "description": "Name of the Amazon OpenSearch index to query",
            "default": "amazon_titan-embed-text-v2",
            "choices": [
                "amazon_titan-embed-text-v2",
                "azure_text-embedding-3-large",
                "vertex_text-embedding-005",
            ],
            "type": "str",
        },
        "k": {
            "description": "Number of similar facts to retrieve",
            "default": 10,
            "type": "int",
        },
        "score_threshold": {
            "description": "Minimum similarity score required",
            "default": 0.0,
            "type": "float",
        },
    },
    "kwargs": {
        "index_mapping": {
            "vertex_text-embedding-005": {
                "provider": "vertex",
                "model_name": "text-embedding-005",
            },
            # ... more mappings
        }
    },
}
```

## Tool Registry

### Location: `bili/loaders/tools_loader.py`

The `TOOL_REGISTRY` maps tool names to initialization functions:

```python
TOOL_REGISTRY = {
    "faiss_retriever": lambda name, prompt, params: create_retriever_tool(
        init_faiss(params.get("path", "data")),
        name,
        prompt,
        **params,
    ),
    "weather_api_tool": lambda name, prompt, params: init_weather_api_tool(
        name, prompt, **params
    ),
    "serp_api_tool": lambda name, prompt, params: init_serp_api_tool(
        name, prompt, **params
    ),
    "weather_gov_api_tool": lambda name, prompt, params: init_weather_gov_api_tool(
        name, prompt, **params
    ),
    "free_weather_api_tool": lambda name, prompt, params: init_weather_tool(
        name, prompt, **params
    ),
    "mock_tool": lambda name, prompt, params: init_mock_tool(
        name, prompt, **params
    ),
    "aws_opensearch_retriever": lambda name, prompt, params: init_amazon_opensearch(
        name, prompt,
        _embedding_function=load_embedding_function(...),
        **params,
    ),
}
```

## Tool Initialization

### Basic Usage

```python
from bili.loaders.tools_loader import initialize_tools

tools = initialize_tools(
    active_tools=["free_weather_api_tool", "serp_api_tool"],
    tool_prompts={
        "free_weather_api_tool": "Get weather for any city",
        "serp_api_tool": "Search the web for current information"
    },
    tool_params={
        "serp_api_tool": {"num_results": 5}
    }
)
```

### Using Default Prompts

If a tool has a `default_prompt` in config, you can omit it:

```python
tools = initialize_tools(
    active_tools=["free_weather_api_tool"],  # Uses default prompt
    tool_prompts={},
    tool_params={}
)
```

### With Middleware

Apply middleware to all tools or specific tools:

```python
from bili.loaders.middleware_loader import initialize_middleware

middleware = initialize_middleware(
    active_middleware=["model_call_limit"],
    middleware_params={"model_call_limit": {"run_limit": 5}}
)

# Apply same middleware to all tools
tools = initialize_tools(
    active_tools=["weather_api_tool", "serp_api_tool"],
    tool_prompts={...},
    tool_middleware=middleware  # List applies to all
)

# Or apply different middleware per tool
tools = initialize_tools(
    active_tools=["weather_api_tool", "serp_api_tool"],
    tool_prompts={...},
    tool_middleware={
        "weather_api_tool": middleware,
        "serp_api_tool": []  # No middleware
    }
)
```

## Tool Implementations

### FAISS Vector Search

**Location**: `bili/tools/faiss_memory_indexing.py`

Local vector similarity search using FAISS:

```python
from bili.tools.faiss_memory_indexing import init_faiss

# Initialize retriever from local documents
retriever = init_faiss(data_dir="data/my_documents")

# Use in tool initialization
tools = initialize_tools(
    active_tools=["faiss_retriever"],
    tool_prompts={"faiss_retriever": "Search local knowledge base"},
    tool_params={"faiss_retriever": {"path": "data/my_documents"}}
)
```

**Security**: Path validation ensures only allowed directories can be indexed:
```python
ALLOWED_PREFIXES = [
    os.path.join(PARENT_DIR, "data"),
    "/app/bili/data",
]
```

### Amazon OpenSearch

**Location**: `bili/tools/amazon_opensearch.py`

Cloud-based similarity search with multiple embedding providers:

```python
tools = initialize_tools(
    active_tools=["aws_opensearch_retriever"],
    tool_prompts={"aws_opensearch_retriever": "Search sustainability knowledge base"},
    tool_params={
        "aws_opensearch_retriever": {
            "index_name": "amazon_titan-embed-text-v2",
            "k": 10,
            "score_threshold": 0.5
        }
    }
)
```

Supported embedding providers:
- Amazon Titan (`amazon_titan-embed-text-v2`)
- Azure OpenAI (`azure_text-embedding-3-large`)
- Google Vertex AI (`vertex_text-embedding-005`)

### Weather APIs

**Free Weather API** (`bili/tools/api_free_weather_api.py`):
```python
tools = initialize_tools(
    active_tools=["free_weather_api_tool"],
    tool_prompts={
        "free_weather_api_tool": "Get current weather. Use city name only."
    }
)
```

**OpenWeatherMap** (`bili/tools/api_open_weather.py`):
```python
tools = initialize_tools(
    active_tools=["weather_api_tool"],
    tool_prompts={
        "weather_api_tool": "Get weather using 'City,State' or ZIP code format"
    },
    tool_params={
        "weather_api_tool": {"api_key": os.environ["OPENWEATHER_API_KEY"]}
    }
)
```

**Weather.gov** (`bili/tools/api_weather_gov.py`):
```python
tools = initialize_tools(
    active_tools=["weather_gov_api_tool"],
    tool_prompts={
        "weather_gov_api_tool": "Get US weather forecast using lat,lon format"
    }
)
```

### SERP API (Web Search)

**Location**: `bili/tools/api_serp.py`

```python
tools = initialize_tools(
    active_tools=["serp_api_tool"],
    tool_prompts={
        "serp_api_tool": "Search the web for current information"
    },
    tool_params={
        "serp_api_tool": {"api_key": os.environ["SERP_API_KEY"]}
    }
)
```

### Mock Tool (Testing)

**Location**: `bili/tools/mock_tool.py`

For testing tool interactions without external dependencies:

```python
tools = initialize_tools(
    active_tools=["mock_tool"],
    tool_prompts={"mock_tool": "Test tool for development"},
    tool_params={
        "mock_tool": {
            "mock_response": "This is a simulated response",
            "response_time": 0.5  # Simulate latency
        }
    }
)
```

## Creating Custom Tools

### Step 1: Create the Tool Module

```python
# bili/tools/my_custom_tool.py
from langchain_core.tools import Tool
from bili.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

def init_my_custom_tool(name: str, prompt: str, **params):
    """
    Initialize a custom tool.

    Args:
        name: Tool name for LangChain
        prompt: Tool description for the LLM
        **params: Additional parameters from tool_params
    """
    api_key = params.get("api_key")
    middleware = params.get("middleware", [])

    def _execute_tool(query: str) -> str:
        """The actual tool logic."""
        LOGGER.debug(f"Executing custom tool with query: {query}")

        # Your tool implementation here
        result = f"Processed: {query}"

        return result

    return Tool(
        name=name,
        description=prompt,
        func=_execute_tool
    )
```

### Step 2: Add to Tool Config

```python
# bili/config/tool_config.py
TOOLS = {
    # ... existing tools ...
    "my_custom_tool": {
        "description": "Description of what the tool does",
        "enabled": True,
        "default_prompt": "Instructions for the LLM on when/how to use this tool",
        "params": {
            "api_key": {
                "description": "API key for the service",
                "default": None,
                "type": "str",
            }
        }
    }
}
```

### Step 3: Register the Tool

```python
# bili/loaders/tools_loader.py
from bili.tools.my_custom_tool import init_my_custom_tool

TOOL_REGISTRY = {
    # ... existing tools ...
    "my_custom_tool": lambda name, prompt, params: init_my_custom_tool(
        name, prompt, **params
    ),
}
```

### Step 4: Use the Tool

```python
tools = initialize_tools(
    active_tools=["my_custom_tool"],
    tool_prompts={"my_custom_tool": "Custom instructions for the LLM"},
    tool_params={"my_custom_tool": {"api_key": "your-api-key"}}
)
```

## Integration with LangGraph

Tools are passed to the agent via `node_kwargs`:

```python
from bili.loaders.langchain_loader import build_agent_graph
from bili.loaders.tools_loader import initialize_tools

# Initialize tools
tools = initialize_tools(
    active_tools=["free_weather_api_tool", "serp_api_tool"],
    tool_prompts={...}
)

# Build agent with tools
agent = build_agent_graph(
    node_kwargs={
        "llm_model": my_llm,
        "tools": tools,  # Tools passed here
        "persona": "You are a helpful assistant with access to weather and search"
    }
)
```

The `react_agent_node` automatically creates a ReAct agent with the provided tools:

```python
# Inside react_agent_node.py
if tools is not None:
    agent = create_agent(
        model=llm_model,
        state_schema=state,
        tools=tools,
        middleware=middleware or (),
    )
```

## Error Handling

### Unknown Tool Warning

If an unrecognized tool is requested, a warning is logged:

```python
LOGGER.warning("Skipping unrecognized tool: %s", tool)
```

### Missing Prompt Error

If a tool has no default prompt and none is provided:

```python
raise ValueError(
    f"Tool '{tool}' does not have a default prompt and no prompt was provided."
)
```

### Path Validation (FAISS)

Invalid paths raise an error:

```python
raise ValueError(
    f"Invalid path: {data_dir}. Path must start with 'data/' "
    f"or one of the allowed prefixes {ALLOWED_PREFIXES}"
)
```

## Best Practices

1. **Always provide clear prompts**: Help the LLM understand when and how to use the tool
2. **Use default prompts**: Leverage tool_config.py defaults for consistency
3. **Validate inputs**: Especially for tools that access file systems or external services
4. **Log tool usage**: Use the logger for debugging and monitoring
5. **Handle errors gracefully**: Return informative error messages the LLM can understand
6. **Apply middleware judiciously**: Use rate limiting for expensive external APIs

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md) - Overall system architecture
- [LANGGRAPH.md](./LANGGRAPH.md) - LangGraph workflow documentation
- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/agents/tools/)
