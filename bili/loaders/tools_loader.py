"""
Module: tools_loader

This module initializes and configures various tools for use within the
application. The tools are defined in a centralized configuration file and
include options for fact retrieval, foundational knowledge retrieval, weather
data fetching, and search engine results retrieval. Each tool is initialized
based on its specific parameters and prompts, which are provided as arguments
to the `initialize_tools` function.

Functions:
    - initialize_tools(active_tools, tool_prompts, tool_params):
      Initializes and caches tools based on the active tools and their specific
      prompts.

Dependencies:
    - langchain.agents.agent_toolkits: Imports `create_retriever_tool` for
      creating retriever tools.
    - bili.loaders.embeddings_loader: Imports `load_embedding_function` for
      loading embedding functions.
    - bili.tools.amazon_opensearch: Imports `init_amazon_opensearch` for
      initializing Amazon OpenSearch tools.
    - bili.tools.api_open_weather: Imports `init_weather_api_tool` for
      initializing weather API tools.
    - bili.tools.api_serp: Imports `init_serp_api_tool` for initializing SERP
      API tools.
    - bili.tools.api_weather_gov: Imports `init_weather_gov_api_tool` for
      initializing Weather.gov API tools.
    - bili.config.tool_config: Imports `TOOLS` for tool configurations.
    - bili.tools.faiss_memory_indexing: Imports `init_faiss` for initializing
      FAISS retrievers.
    - bili.tools.mock_tool: Imports `init_mock_tool` for initializing mock
      tools.
    - bili.utils.logging_utils: Imports `get_logger` for logging.

Usage:
    This module is intended to be used within the application to initialize and
    configure various tools based on the provided configuration. It provides a
    function to dynamically initialize tools with specific prompts and
    parameters.

    The `TOOL_REGISTRY` dictionary maps tool names to their corresponding
    initialization functions. This allows you to override the default tool
    initialization behavior or add new tools. An example of adding a new tool
    to the registry would be:
    `TOOL_REGISTRY["new_tool"] = lambda name, prompt, params: init_new_tool(
    name, prompt, **params)`

Example:
    from bili.loaders.tools_loader import initialize_tools

    # Initialize tools
    tools = initialize_tools(
        active_tools=["faiss_retriever", "weather_api_tool"],
        tool_prompts={"faiss_retriever_prompt": "Retrieve documents",
            "weather_api_tool_prompt": "Fetch weather data"},
        tool_params={"faiss_retriever": {"path": "data"},
            "weather_api_tool": {"api_key": "your_api_key"}}
    )
"""

from langchain_classic.agents.agent_toolkits import create_retriever_tool

from bili.config.tool_config import TOOLS
from bili.loaders.embeddings_loader import load_embedding_function
from bili.tools.amazon_opensearch import init_amazon_opensearch
from bili.tools.api_free_weather_api import init_weather_tool
from bili.tools.api_open_weather import init_weather_api_tool
from bili.tools.api_serp import init_serp_api_tool
from bili.tools.api_weather_gov import init_weather_gov_api_tool
from bili.tools.faiss_memory_indexing import init_faiss
from bili.tools.mock_tool import init_mock_tool
from bili.utils.logging_utils import get_logger

# Get the logger instance for the module
LOGGER = get_logger(__name__)

# Define a registry of tool initialization functions
# This allows for dynamic initialization of tools based on the provided configuration
# and for users to override the default tool initialization behavior or define new tools
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
    "mock_tool": lambda name, prompt, params: init_mock_tool(name, prompt, **params),
    "aws_opensearch_retriever": lambda name, prompt, params: init_amazon_opensearch(
        name,
        prompt,
        _embedding_function=load_embedding_function(
            params["index_mapping"][params["index_name"]]["provider"],
            params["index_mapping"][params["index_name"]]["model_name"],
        ),
        **params,
    ),
}


def initialize_tools(active_tools, tool_prompts, tool_params=None):
    """
    Initializes and configures a list of tools based on the provided parameters.

    This function takes a list of active tools, their corresponding prompts, and their
    parameters. It checks if each tool exists in the `TOOL_REGISTRY`. If it does, the tool
    is initialized using the associated prompt and parameters and added to the collection
    of tools. If a tool is not recognized, a warning is logged. The initialized tools are
    then returned as a list.

    :param active_tools: List of tool names to be activated and initialized.
    :type active_tools: list[str]
    :param tool_prompts: Dictionary mapping tool names to their respective prompt values.
    :type tool_prompts: dict[str, str]
    :param tool_params: Dictionary mapping tool names to their respective parameter
        configurations.
    :type tool_params: dict[str, dict]
    :return: A list of initialized tool objects.
    :rtype: list
    """
    if tool_params is None:
        tool_params = {}
    LOGGER.debug("Initializing tools: %s", active_tools)

    tools = []
    for tool in active_tools:
        if tool in TOOL_REGISTRY:
            # If there is a default prompt, retrieve it, otherwise set to None
            if tool in TOOLS and "default_prompt" in TOOLS[tool]:
                default_prompt = TOOLS[tool]["default_prompt"]
            else:
                default_prompt = None

            # Check if a custom prompt is set, otherwise use the default prompt if available
            # If there is no custom prompt and no default prompt, raise an error
            if f"{tool}_prompt" in tool_prompts:
                prompt = tool_prompts[f"{tool}_prompt"]
            elif tool in tool_prompts:
                prompt = tool_prompts[tool]
            elif default_prompt:
                prompt = default_prompt
            else:
                raise ValueError(
                    f"Tool '{tool}' does not have a default prompt and no prompt was provided."
                )

            tools.append(
                TOOL_REGISTRY[tool](
                    tool,
                    prompt,
                    tool_params.get(tool, {}),
                )
            )
        else:
            LOGGER.warning("Skipping unrecognized tool: %s", tool)

    return tools
