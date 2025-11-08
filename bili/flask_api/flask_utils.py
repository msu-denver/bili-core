"""
flask_utils.py
--------------

This module provides utility functions and decorators for integrating
authentication and authorization mechanisms into a Flask application.
It includes middleware for handling token refresh on 401 Unauthorized
responses, setting authentication tokens as cookies, and a decorator
for enforcing authentication and role-based access control on protected routes.

Functions:
----------
- add_unauthorized_handler(auth_manager, app):
    Adds middleware to intercept 401 Unauthorized responses and handle token refresh.

- set_token_cookies(resp, id_token, refresh_token=None):
    Sets provided tokens as cookies in the response object.

- auth_required(auth_manager, required_roles=None):
    A decorator function to enforce authentication and optionally
    validate role-based access control (RBAC) for the decorated Flask route function.

- per_user_agent(checkpoint_saver, graph_definition, node_kwargs, state=State, custom_node_registry=None):
    Builds a decorator to create a per-user agent graph for a Flask application.

- handle_agent_prompt(user, conversation_agent, prompt):
    Processes a user prompt using a conversation agent and retrieves the generated response.

Dependencies:
-------------
- functools.wraps: Used to preserve the original function's metadata when applying decorators.
- flask.request: Provides access to the request object for extracting headers and cookies.
- flask.make_response: Used to create response objects with updated cookies.
- flask.jsonify: Used to create JSON responses for error handling.
- flask.g: Used to store user information in Flask's global context.
- bili.utils.langgraph_utils.State: Represents the initial state of the graph.
- langchain_core.messages.HumanMessage: Represents a human message in the conversation.
- langgraph.checkpoint.base.BaseCheckpointSaver: Base class for checkpoint savers.
- bili.auth.auth_manager.AuthManager: Manages authentication and authorization.
- bili.loaders.langchain_loader.build_agent_graph: Builds and compiles a state graph for a LangGraph-based agent.
- bili.loaders.langchain_loader.GRAPH_NODE_REGISTRY: Default registry for graph nodes.

Usage:
------
To use the utilities provided in this module, import the necessary functions and apply them to your Flask
application as shown in the examples below:

Example:
--------
from flask import Flask
from bili.flask_api.flask_utils import add_unauthorized_handler, auth_required, per_user_agent

app = Flask(__name__)
auth_manager = get_auth_manager()

# Add middleware to handle token refresh on 401 responses
add_unauthorized_handler(auth_manager, app)

@app.route('/protected')
@auth_required(auth_manager, required_roles=['admin'])
def protected_route():
    return jsonify({"message": "This is a protected route"})

if __name__ == '__main__':
    app.run()
"""

from functools import wraps
from dataclasses import replace

from flask import g, jsonify, make_response, request
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from bili.auth.auth_manager import AuthManager
from bili.loaders.langchain_loader import GRAPH_NODE_REGISTRY, build_agent_graph
from bili.nodes.per_user_state import per_user_state_node
from bili.nodes.add_persona_and_summary import persona_and_summary_node
from bili.utils.langgraph_utils import State
from bili.utils.logging_utils import get_logger

# Get the logger for the module
LOGGER = get_logger(__name__)


def add_unauthorized_handler(auth_manager, app):
    """
    Adds middleware to intercept 401 Unauthorized responses and handle token refresh.
    This middleware checks for a refresh token in cookies when a 401 response occurs.
    If a refresh token is available, it attempts to refresh the token via the auth manager
    and retries the original request with the new token. If the retry succeeds, the
    appropriate token cookies are updated for subsequent requests.

    :param auth_manager: The authentication manager that provides the refresh token mechanism.
    :param app: The Flask application instance to which the middleware is added.

    :return: None
    """

    @app.after_request
    def refresh_token_on_401(response):
        """
        Middleware that intercepts 401 responses.
        If a 401 is detected and a refresh token is available, it attempts to refresh the token
        and retries the original request.
        """
        if (
            response.status_code == 401
        ):  # Only retry if a 401 Unauthorized response occurs
            refresh_token = request.cookies.get("refresh_token")
            if not refresh_token:
                return response  # No refresh token, return original response

            try:
                # Get the new ID token
                tokens = auth_manager.auth_provider.refresh_token(refresh_token)
                new_id_token = tokens["id_token"]
                new_refresh_token = tokens["refresh_token"]

                # Rebuild the original request with the new token
                new_headers = dict(request.headers)
                new_headers["Authorization"] = f"Bearer {new_id_token}"

                with app.test_request_context(
                    request.path,  # Original path
                    method=request.method,
                    headers=new_headers,
                    data=request.get_data(),
                    query_string=request.query_string,
                    content_type=request.content_type,
                ):
                    retry_response = app.full_dispatch_request()

                # If retry succeeds, update the cookie and return the new response
                if retry_response.status_code != 401:
                    resp = make_response(
                        retry_response.get_data(), retry_response.status_code
                    )
                    set_token_cookies(resp, new_id_token, new_refresh_token)
                    return resp

            except Exception as e:
                LOGGER.error(f"Token refresh failed: {str(e)}")  # Log failure

        return response  # If refresh fails, just return the original 401 response


def set_token_cookies(resp, id_token, refresh_token=None):
    """
    Sets provided tokens as cookies in the response object. This function adds
    security configurations, ensuring that the cookies are appropriately set for
    secure contexts. The `id_token` is accessible through client-side scripts,
    while the `refresh_token`, when provided, is set as HTTP-only to enhance
    security against client-side access.

    :param resp: Response object to which cookies will be added.
    :type resp: Response
    :param id_token: ID token that is used for authentication purposes.
    :type id_token: str
    :param refresh_token: Refresh token that is used for retrieving new ID tokens.
    This is optional.
    :type refresh_token: str, optional
    :return: The response object with the added cookies.
    :rtype: Response
    """
    resp.set_cookie(
        "id_token", id_token, httponly=False, secure=True, samesite="Strict"
    )
    if refresh_token:
        resp.set_cookie(
            "refresh_token",
            refresh_token,
            httponly=True,
            secure=True,
            samesite="Strict",
        )
    return resp


def auth_required(auth_manager: AuthManager, required_roles=None):
    """
    A decorator function to enforce authentication and optionally validate
    role-based access control (RBAC) for the decorated Flask route function.
    This function ensures that the incoming request contains a valid
    authorization token, verifies it using the provided `auth_manager`, and
    optionally checks whether the user has sufficient roles for the operation.

    Authentication tokens are expected in the "Authorization" header with an
    optional "Bearer " prefix. If the header is not present, the function
    attempts to retrieve an "id_token" from cookies. If a valid token is not
    provided or user lacks required roles, an appropriate JSON response with
    an HTTP error status code is returned. Validated user information is stored
    in Flask's global context (`g.user`) to be accessed within the view logic
    of the decorated route.

    :param auth_manager: An authentication provider responsible for verifying the
        JWT tokens and optionally checking role-based access control.
        Must expose `auth_provider.verify_jwt_token` for token validation
        and `role_provider.is_authorized` for role checks.
    :type auth_manager: object
    :param required_roles: A list of roles required for accessing the endpoint.
        If omitted or None, role-based authorization is skipped.
    :type required_roles: list or None
    :return: A decorator that wraps the target Flask route function, enforcing
        token-based authentication and role authorization.
    :rtype: callable
    """

    def decorator(func):
        """
        A decorator function to enforce authentication and authorization on Flask routes.

        This function wraps a Flask route to ensure that the incoming request contains a valid
        JWT token in the headers. It verifies the token using the provided `AuthManager` instance
        and checks if the user has the necessary permissions to access the route.

        The decorator extracts the user data from the token and stores it in Flask's
        global context (`g.user`). If the user is not authorized, it returns a 403 Forbidden
        response. If the token is invalid or missing, it returns a 401 Unauthorized response.

        Usage:
            @app.route('/protected')
            @auth_required(auth_manager, required_roles=['admin'])
            def protected_route():
                return jsonify({"message": "This is a protected route"})

        :param func: The Flask route function to be wrapped by the decorator.
        :type func: function
        :return: A wrapped function with authentication and authorization enforcement.
        :rtype: function
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Wrapper function to enforce authentication and authorization on Flask routes.

            This function is used within the `auth_required` decorator to ensure that the
            incoming request contains a valid JWT token in the headers. It verifies the token
            using the provided `AuthManager` instance and checks if the user has the necessary
            permissions to access the route.

            The wrapper extracts the user data from the token and stores it in Flask's global
            context (`g.user`). If the user is not authorized, it returns a 403 Forbidden response.
            If the token is invalid or missing, it returns a 401 Unauthorized response.

            :param args: Positional arguments passed to the wrapped route function.
            :param kwargs: Keyword arguments passed to the wrapped route function.
            :return: The response from the wrapped route function if authentication and authorization
                     are successful, otherwise a JSON response with an error message and appropriate
                     HTTP status code.
            :rtype: Response
            """
            token = request.headers.get("Authorization")

            # If Authorization header is missing, try getting ID token from cookies
            if not token:
                token = request.cookies.get("id_token")
            else:
                token = token.replace(
                    "Bearer ", ""
                )  # Remove "Bearer " prefix if present

            if not token:
                return jsonify({"error": "Missing authentication token"}), 401

            try:
                user_data = auth_manager.auth_provider.verify_jwt_token(token)
                g.user = user_data  # Store user info in Flask's global context

                # Role-based authorization check
                if required_roles and not auth_manager.role_provider.is_authorized(
                    user_data.get("uid"), token, required_roles
                ):
                    return (
                        jsonify({"error": "Forbidden: Insufficient permissions"}),
                        403,
                    )

                user_profile = auth_manager.profile_provider.get_user_profile(
                    g.user["uid"], token
                )
                g.user_profile = (
                    user_profile  # Store user profile in Flask's global context
                )

                return func(*args, **kwargs)  # Proceed with the original route function

            except ValueError as e:
                return jsonify({"error": str(e)}), 401

        return wrapper

    return decorator


def per_user_agent(
    checkpoint_saver: BaseCheckpointSaver,
    graph_definition: list,
    node_kwargs: dict,
    state: type = State,
    custom_node_registry: dict = None,
):
    """
    Builds a decorator to create a per-user agent graph for a Flask application. This
    decorator facilitates constructing a graph with nodes defined by a graph
    definition, initializing edges between them, and compiling it into a reusable
    graph structure. Additionally, it adds the current user from Flask's global
    context to the node_kwargs, enabling the graph to be tailored to a specific user.

    :param checkpoint_saver: An instance of BaseCheckpointSaver utilized to
        handle checkpoint saving mechanisms for the graph.
    :param graph_definition: A list of node identifiers as strings that define
        the structure and sequence of nodes in the user-agent graph.
    :param node_kwargs: A dictionary containing key-value pairs that provide
        configuration properties or arguments required by the nodes in the graph.
    :param state: A class or type representing the shared state passed through
        the graph. Defaults to `State` if not explicitly specified.
    :param custom_node_registry: An optional dictionary that maps custom node
        names to their implementations. If not provided, the default
        `GRAPH_NODE_REGISTRY` is used.
    :return: A decorator function that, when applied to another function, supplies
        it with a per-user agent graph tailored to the active user and defined by
        the provided parameters.
    """
    if custom_node_registry is None:
        custom_node_registry = GRAPH_NODE_REGISTRY

    def decorator(func):
        """
        Decorator function to build a per-user agent graph using the provided
        parameters. This function initializes the graph with nodes defined by the
        provided graph definition, sets up edges between them, and compiles it
        using the provided checkpoint saver. The annotation will add to the
        node_kwargs dictionary the current user object from Flask's global context.
        This will allow the graph to be built with the current user information if it
        supports it.

        :param func: The function to be decorated.
        :return: A wrapped function that builds a per-user agent graph.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Add current user to node_kwargs. Check for user_profile first, and
            # use user if it is not available
            if hasattr(g, "user_profile"):
                node_kwargs["current_user"] = g.user_profile
            else:
                # Use g.user if user_profile is not available
                node_kwargs["current_user"] = g.user

            # Check that the graph definition contains the per_user_state node
            # If not, insert it between persona_and_summary and inject_current_datetime
            if per_user_state_node not in graph_definition:

                # Create modified copies with updated edges
                persona_node_modified = replace(persona_and_summary_node, edges=["per_user_state"])
                per_user_state_modified = replace(per_user_state_node, edges=["inject_current_datetime"])

                # Rebuild graph definition: persona → per_user_state → rest
                graph_definition = [
                    persona_node_modified,
                    per_user_state_modified,
                ] + graph_definition[1:]  # Skip original persona_and_summary_node

            # Build the agent graph using the provided parameters
            agent_graph = build_agent_graph(
                checkpoint_saver=checkpoint_saver,
                custom_node_registry=custom_node_registry,
                graph_definition=graph_definition,
                node_kwargs=node_kwargs,
                state=state,
            )
            g.agent = agent_graph  # Store the agent graph in Flask's global context
            return func(*args, **kwargs)

        return wrapper

    return decorator


def handle_agent_prompt(user, conversation_agent, prompt):
    """
    Processes a user prompt using a conversation agent by associating the prompt
    with a specific thread ID based on the user's email. The function converts
    the prompt into a usable format for the conversation agent and retrieves
    the generated response, returning it in JSON format.

    :param user: The authenticated user details as a dictionary, must include
        the "email" key.
    :type user: dict
    :param conversation_agent: The conversation agent instance responsible for
        handling user messages.
    :type conversation_agent: Any
    :param prompt: The input prompt or message provided by the user.
    :type prompt: str
    :return: A JSON response containing the content generated by the conversation
        agent if successful or an error message otherwise.
    :rtype: flask.Response
    """
    # Associate the correct thread id based on the authenticated user
    email = user["email"]
    config = {
        "configurable": {
            # "thread_id": f"{email}_{thread_id}",
            "thread_id": f"{email}",
        },
    }

    # Convert prompt into a HumanMessage
    input_message = HumanMessage(content=prompt)

    # Process the user prompt using the conversation agent
    result = conversation_agent.invoke(
        {"messages": [input_message], "verbose": False}, config
    )

    # result is typically a dict containing "messages": [...], where the last is the AIMessage
    if isinstance(result, dict) and "messages" in result:
        final_msg = result["messages"][-1]
        return jsonify({"response": final_msg.content})
    else:
        return jsonify({"response": "No response"}), 500
