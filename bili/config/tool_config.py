"""
Module: config

This module initializes and configures various tools for use within the
application. The tools are defined in the centralized configuration file and
include options for fact retrieval, foundational knowledge retrieval, weather
data fetching, and search engine results retrieval. Each tool is initialized
based on its specific parameters and prompts, which are provided as arguments
to the `initialize_tools` function.

Functions:
    - initialize_tools(active_tools, tool_prompts, tool_params):
      Initializes and caches tools based on the active tools and their specific
      prompts.

Dependencies:
    - langchain.agents.agent_toolkits: Imports create_retriever_tool for
      creating retriever tools.
    - bili.config.tool_config: Imports TOOLS for accessing the tool configurations.
    - bili.tools.api_open_weather: Imports init_weather_api_tool for
      initializing the weather API tool.
    - bili.tools.api_serp: Imports init_serp_api_tool for initializing the
      SERP API tool.
    - bili.tools.api_weather_gov: Imports init_weather_gov_api_tool for
      initializing the weather.gov API tool.
    - bili.tools.faiss_memory_indexing: Imports init_faiss for initializing
      the FAISS memory indexing tool.
    - bili.tools.mock_tool: Imports init_mock_tool for initializing the mock
      tool.

Usage:
    This module is intended to be used within the application to initialize and
    configure various tools based on the provided configuration. It supports
    tools for fact retrieval, weather data fetching, and search engine results
    retrieval.

Example:
    from bili.config.tool_config import initialize_tools

    # Initialize tools with active tools, prompts, and parameters
    tools = initialize_tools(active_tools, tool_prompts, tool_params)
"""

# Available Tools
TOOLS = {
    "local_faiss_retriever": {
        "description": "Retrieves facts using FAISS in-memory data store.",
        "enabled": False,
        "default_prompt": "This tool, powered by FAISS, retrieves precise, domain-related facts "
        "from local in-memory vector databases. It is optimized for answering "
        "fact-based queries using pre-indexed data.",
        "params": {
            "path": {
                "description": "Path to the directory containing source data for FAISS to "
                "index and retrieve facts from. Files in this directory will "
                "be parsed and vectorized into the FAISS index.",
                "default": "data/provider",
                "type": "str",
            }
        },
    },
    "aws_opensearch_retriever": {
        "description": "Retrieves facts using Amazon OpenSearch for similarity searches.",
        "enabled": True,
        "default_prompt": "This tool utilizes Amazon OpenSearch to retrieve facts based on "
        "similarity searches. It identifies and ranks facts most relevant "
        "to the query by contextual alignment.",
        "params": {
            "index_name": {
                "description": "Name of the Amazon OpenSearch index to query for facts.",
                "default": "amazon_titan-embed-text-v2",
                "choices": [
                    "amazon_titan-embed-text-v2",
                    "azure_text-embedding-3-large",
                    "vertex_text-embedding-005",
                ],
                "type": "str",
            },
            "k": {
                "description": "Number of similar facts to retrieve for each query.",
                "default": 10,
                "type": "int",
            },
            "score_threshold": {
                "description": "Minimum similarity score required for a fact "
                "to be considered relevant.",
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
                "amazon_titan-embed-text-v2": {
                    "provider": "amazon",
                    "model_name": "amazon.titan-embed-text-v2:0",
                },
                "azure_text-embedding-3-large": {
                    "provider": "azure",
                    "model_name": "text-embedding-3-large",
                },
            }
        },
    },
    "weather_api_tool": {
        "description": "Fetches weather data via OpenWeatherMap API.",
        "enabled": True,
        "default_prompt": "This tool retrieves weather data using the OpenWeatherMap API. "
        "The input query should provide the location in a comma-separated "
        "format (e.g., 'Denver,CO') to accurately convert to geographic "
        "coordinates for weather retrieval. Alternatively, you can input a US 5-digit ZIP code (e.g., '80202')",
    },
    "weather_gov_api_tool": {
        "description": "US weather data from weather.gov.",
        "enabled": False,
        "default_prompt": "This tool provides detailed weather forecasts for US locations via "
        "the weather.gov API. Queries must be formatted as "
        "'latitude,longitude' (e.g., '39.7392,-104.9903').",
    },
    "serp_api_tool": {
        "description": "Search engine results using SERP API.",
        "enabled": True,
        "default_prompt": "The Search Engine Results Page (SERP) API enables real-time retrieval "
        "of search engine results. This tool is useful for answering queries "
        "requiring up-to-date information from the internet.",
    },
    "mock_tool": {
        "description": "Simulates a tool by returning predefined mock responses. "
        "Use for testing purposes.",
        "enabled": False,
        "default_prompt": "This tool provides mock responses to simulate tool interaction. "
        "It is intended for testing system behaviors where tool invocation is "
        "needed, but the actual tool is not implemented. Accepts input queries "
        "and provides predefined responses to validate the system's handling of "
        "tool interactions.",
        "params": {
            "mock_response": {
                "description": "Mock response that the tool will return",
                "default": "This is a mock response from the Test Tool",
            },
            "response_time": {
                "description": "Simulated time delay in seconds before returning the response",
                "default": 0.0,
                "type": "float",
            },
        },
    },
}
