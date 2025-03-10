"""
flask_app.py
------------

This module sets up a Flask application with authentication and authorization mechanisms,
integrates a language model for conversational AI, and provides various routes for user
interaction and model invocation.

Functions:
----------
- login():
    Handles user login, sets authentication tokens as cookies, and returns user information.

- protected_route():
    A protected route that requires authentication and specific roles to access user information.

- nova_pro_route():
    A route that invokes the Nova Pro language model to process user prompts and return responses.

Dependencies:
-------------
- os: Provides a way to interact with the operating system.
- pathlib.Path: Offers a way to handle file system paths.
- flask: A micro web framework for Python.
- langchain_core.messages.HumanMessage: Represents a human message in the conversation.
- bili.checkpointers.checkpointer_functions.get_checkpointer: Retrieves the appropriate checkpointer.
- bili.flask_api.flask_utils: Contains utility functions and decorators for Flask.
- bili.loaders.langchain_loader.load_langgraph_agent: Loads the LangGraph agent.
- bili.loaders.llm_loader.load_model: Loads the language model.
- bili.streamlit_ui.ui.auth_ui.initialize_auth_manager: Initializes the authentication manager.
- bili.utils.file_utils.load_from_json: Loads data from a JSON file.

Usage:
------
To use this module, run the Flask application as shown in the example below:

Example:
--------
from flask import Flask
from bili.flask_app import app

if __name__ == '__main__':
    app.run()
"""

import os
from pathlib import Path

from flask import Flask, g, jsonify, make_response, request

from bili.checkpointers.checkpointer_functions import get_checkpointer
from bili.config.tool_config import TOOLS
from bili.flask_api.flask_utils import (
    add_unauthorized_handler,
    auth_required,
    handle_agent_prompt,
    set_token_cookies,
)
from bili.loaders.langchain_loader import load_langgraph_agent
from bili.loaders.llm_loader import load_model
from bili.streamlit_ui.ui.auth_ui import initialize_auth_manager
from bili.utils.file_utils import load_from_json

app = Flask(__name__)

# Initialize the appropriate AuthManager
AUTH_MANAGER = initialize_auth_manager(
    auth_provider_name="firebase",
    profile_provider_name="default",
    role_provider_name="default",
)

# Initialize JWT token refresh on 401 Unauthorized
add_unauthorized_handler(AUTH_MANAGER, app)


@app.route("/login", methods=["POST"])
def login():
    """
    Handles the login functionality for users by verifying the provided
    email and password. If the credentials are valid, it returns a response
    containing an ID token and user information, while also setting secure
    cookies for authentication in subsequent requests. In case of errors,
    an appropriate error response is returned.

    :raises Exception: If there are issues during the authentication process.
    :param data: The JSON payload with keys "email" and "password" provided
        by the client for authentication.
    :type data: dict
    :return: A JSON response containing the authentication token and user
        details if the login is successful, or an error message and status
        code otherwise.
    """
    data = request.get_json()
    if not data or "email" not in data or "password" not in data:
        return jsonify({"error": "Email and password are required"}), 400

    email = data["email"]
    password = data["password"]

    try:
        # Authenticate user
        auth_response = AUTH_MANAGER.auth_provider.sign_in(email, password)

        # Get user details
        user_info = AUTH_MANAGER.auth_provider.get_account_info(auth_response["uid"])

        # Extract tokens
        id_token = auth_response["token"]
        refresh_token = auth_response["refreshToken"]

        # Create response, include ID token and user info. Token will be used
        # by API clients to authenticate subsequent requests. For web requests,
        # the secure cookies created below will be used for authentication.
        resp = make_response(jsonify({"token": id_token, "user": user_info}))

        # Set cookies
        set_token_cookies(resp, id_token, refresh_token)

        return resp

    except Exception as e:
        return jsonify({"error": str(e)}), 401


@app.route("/me", methods=["GET"])
@auth_required(AUTH_MANAGER)
def me():
    """
    Handles the retrieval of the currently authenticated user's information. This route
    requires authentication and returns the user's data in JSON format.

    Raises an error if the user is not authenticated or an issue occurs during
    authorization.

    :raises Unauthorized: If the user is not authenticated.
    :raises Exception: For other generic issues that might occur.
    :return: JSON response containing the authenticated user's information.
    :rtype: flask.Response
    """
    return jsonify({"user": g.user})


# ----- LLM example routes -----

# Get the appropriate checkpointer (PostgresSaver or MemorySaver) for state persistence
checkpointer = get_checkpointer()

# Create the nova pro llm model instance using the remote AWS Bedrock provider
nova_pro_llm = load_model(
    model_type="remote_aws_bedrock",
    **{
        "model_name": "amazon.nova-pro-v1:0",
        "max_tokens": 5000,
        "temperature": 0.7,
        # "top_p": 0.5,
        # "top_k": 50,
        # "seed": 42,
    },
)

# Initialize tools list
# For this demo we will enable the weather and serp tools
active_tools = ["weather_api_tool", "serp_api_tool"]
tool_prompts = {
    "weather_api_tool": TOOLS["weather_api_tool"]["default_prompt"],
    "serp_api_tool": TOOLS["serp_api_tool"]["default_prompt"],
}

# Create langgraph agent using the created LLM model and active tools
# Load default prompts from JSON that will be used in the api
default_prompts = load_from_json(
    os.getenv(
        "DEFAULT_PROMPT_PATH",
        Path(__file__).parent.joinpath("prompts", "default_prompts.json").as_posix(),
    ),
    "templates",
)

# Create the langgraph agent
conversation_agent = load_langgraph_agent(
    # Use the nova pro llm model
    llm_model=nova_pro_llm,
    # Use the default prompts as defined in the default_prompts.json file
    langgraph_system_prefix=default_prompts["default"]["langgraph_system_prefix"],
    # Use the active tools and prompts
    active_tools=active_tools,
    tool_prompts=tool_prompts,
    # Use a summarization strategy with a memory limit of 15 messages
    memory_strategy="summarize",
    memory_limit_type="message_count",
    k=15,
    # Use the checkpointer for state persistence to allow for conversation history
    # to be saved and restored
    checkpoint_saver=checkpointer,
)


@app.route("/nova", methods=["GET"])
@auth_required(AUTH_MANAGER, required_roles=["admin", "researcher"])
def nova_pro_route():
    """
    Handles requests sent to the `/nova` endpoint. Authenticates user roles and processes
    a prompt provided in the request body. Generates a response using a conversational
    agent and returns the AI's reply to the user.

    :param prompt: The input text extracted from the client's JSON request body, provided
        under the key "prompt".

    :raises KeyError: If the provided JSON object does not include a "prompt" key.
    :raises TypeError: If the prompt is not a string.

    :return: A JSON response containing the AI-generated reply as a string under the
        "response" key. Returns a 500 status with a generic error message if the
        conversational agent execution fails or yields no output.
    :rtype: flask.Response
    """
    # Get the prompt from the request body JSON data via param "prompt"
    prompt = request.get_json().get("prompt", "")

    # Invoke agent with the provided prompt
    return handle_agent_prompt(g.user, conversation_agent, prompt)


if __name__ == "__main__":
    app.run()
