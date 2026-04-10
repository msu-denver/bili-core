"""Tests for Flask application routes.

Uses the Flask test client with a real SQLite auth provider
and temporary database to test all routes including login,
protected endpoints, and LLM-related routes.
"""

# pylint: disable=attribute-defined-outside-init

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask, g, jsonify, make_response, request
from langchain_core.messages import HumanMessage

from bili.auth.auth_manager import AuthManager
from bili.flask_api.flask_utils import (
    add_unauthorized_handler,
    auth_required,
    handle_agent_prompt,
    handle_agent_prompt_stream,
    set_token_cookies,
)

_TEST_EMAIL = "testuser@example.com"
_TEST_PASSWORD = "SecurePass123!"
_JWT_SECRET = "flask-app-test-secret-key"


def _setup_auth(tmp_path):
    """Create an AuthManager backed by a temp SQLite database."""
    db_path = str(tmp_path / "test.db")
    os.environ["JWT_SECRET_KEY"] = _JWT_SECRET
    os.environ["PROFILE_DB_PATH"] = db_path
    return AuthManager(
        auth_provider_name="sqlite",
        profile_provider_name="sqlite",
        role_provider_name="sqlite",
    )


def _register_test_user(auth_manager):
    """Create a test user with profile and return sign-in data."""
    auth_manager.auth_provider.create_user(_TEST_EMAIL, _TEST_PASSWORD)
    result = auth_manager.auth_provider.sign_in(_TEST_EMAIL, _TEST_PASSWORD)
    auth_manager.profile_provider.create_user_profile(
        uid=result["uid"],
        email=_TEST_EMAIL,
        first_name="Test",
        last_name="User",
        token=result["token"],
    )
    return result


def _build_test_app(auth_manager, mock_agent):
    """Build a minimal Flask app mirroring production routes."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    add_unauthorized_handler(auth_manager, app)
    roles = ["admin", "researcher"]

    @app.route("/login", methods=["POST"])
    def login():
        """Handle user login."""
        data = request.get_json()
        if not data or "email" not in data or "password" not in data:
            return (
                jsonify({"error": "Email and password are required"}),
                400,
            )
        try:
            auth_resp = auth_manager.auth_provider.sign_in(
                data["email"], data["password"]
            )
            user_info = auth_manager.auth_provider.get_account_info(auth_resp["uid"])
            resp = make_response(
                jsonify({"token": auth_resp["token"], "user": user_info})
            )
            set_token_cookies(resp, auth_resp["token"], auth_resp["refreshToken"])
            return resp
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 401

    @app.route("/me", methods=["GET"])
    @auth_required(auth_manager)
    def me_route():
        """Return current user info."""
        return jsonify({"user": g.user})

    @app.route("/nova_global", methods=["GET"])
    @auth_required(auth_manager, required_roles=roles)
    def nova_global():
        """Invoke the global conversation agent."""
        request_data = request.get_json()
        if not request_data:
            return (
                jsonify({"error": "Request body is required"}),
                400,
            )
        prompt = request_data.get("prompt", "")
        conv_id = request_data.get("conversation_id")
        return handle_agent_prompt(g.user, mock_agent, prompt, conv_id)

    @app.route("/nova_per_user", methods=["GET"])
    @auth_required(auth_manager, required_roles=roles)
    def nova_per_user():
        """Invoke the per-user conversation agent."""
        request_data = request.get_json()
        if not request_data:
            return (
                jsonify({"error": "Request body is required"}),
                400,
            )
        prompt = request_data.get("prompt", "")
        conv_id = request_data.get("conversation_id")
        return handle_agent_prompt(g.user, mock_agent, prompt, conv_id)

    @app.route("/nova_stream", methods=["POST"])
    @auth_required(auth_manager, required_roles=roles)
    def nova_stream():
        """Stream a response from the conversation agent."""
        request_data = request.get_json()
        if not request_data:
            return (
                jsonify({"error": "Request body is required"}),
                400,
            )
        prompt = request_data.get("prompt", "")
        conv_id = request_data.get("conversation_id")
        return handle_agent_prompt_stream(g.user, mock_agent, prompt, conv_id)

    return app


def _auth_header(token):
    """Return an Authorization header dict for the given token."""
    return {"Authorization": f"Bearer {token}"}


def _make_mock_agent():
    """Return a MagicMock that mimics a LangGraph agent."""
    agent = MagicMock()
    agent.invoke.return_value = {"messages": [MagicMock(content="Hello from the LLM")]}
    return agent


class TestLoginRoute:
    """Tests for POST /login."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Set up test app and client for each test."""
        self.auth_manager = _setup_auth(tmp_path)
        self.mock_agent = _make_mock_agent()
        _register_test_user(self.auth_manager)
        app = _build_test_app(self.auth_manager, self.mock_agent)
        self.client = app.test_client()

    def test_valid_credentials_returns_token_and_user(self):
        """Successful login returns token and user info."""
        resp = self.client.post(
            "/login",
            json={
                "email": _TEST_EMAIL,
                "password": _TEST_PASSWORD,
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "token" in data
        assert isinstance(data["token"], str)
        assert len(data["token"]) > 0
        assert "user" in data
        assert data["user"]["email"] == _TEST_EMAIL

    def test_valid_login_sets_cookies(self):
        """Successful login sets id_token and refresh_token cookies."""
        resp = self.client.post(
            "/login",
            json={
                "email": _TEST_EMAIL,
                "password": _TEST_PASSWORD,
            },
        )
        assert resp.status_code == 200
        set_cookie_headers = resp.headers.getlist("Set-Cookie")
        cookie_names = [h.split("=")[0] for h in set_cookie_headers]
        assert "id_token" in cookie_names
        assert "refresh_token" in cookie_names

    def test_invalid_password_returns_401(self):
        """Wrong password returns 401 with error message."""
        resp = self.client.post(
            "/login",
            json={
                "email": _TEST_EMAIL,
                "password": "wrong-password",
            },
        )
        assert resp.status_code == 401
        data = resp.get_json()
        assert "error" in data

    def test_nonexistent_user_returns_401(self):
        """Non-existent user returns 401."""
        resp = self.client.post(
            "/login",
            json={
                "email": "nobody@example.com",
                "password": "anything",
            },
        )
        assert resp.status_code == 401

    def test_missing_email_returns_400(self):
        """Request without email field returns 400."""
        resp = self.client.post(
            "/login",
            json={"password": _TEST_PASSWORD},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["error"] == "Email and password are required"

    def test_missing_password_returns_400(self):
        """Request without password field returns 400."""
        resp = self.client.post(
            "/login",
            json={"email": _TEST_EMAIL},
        )
        assert resp.status_code == 400

    def test_empty_body_returns_400(self):
        """Request with no JSON body returns 400."""
        resp = self.client.post(
            "/login",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_login_response_token_is_valid_jwt(self):
        """The token in the login response can be verified."""
        resp = self.client.post(
            "/login",
            json={
                "email": _TEST_EMAIL,
                "password": _TEST_PASSWORD,
            },
        )
        data = resp.get_json()
        decoded = self.auth_manager.auth_provider.verify_jwt_token(data["token"])
        assert decoded["email"] == _TEST_EMAIL


class TestMeRoute:
    """Tests for GET /me."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Set up test app with authenticated user."""
        auth_manager = _setup_auth(tmp_path)
        sign_in = _register_test_user(auth_manager)
        self.token = sign_in["token"]
        app = _build_test_app(auth_manager, _make_mock_agent())
        self.client = app.test_client()

    def test_with_valid_token_returns_user(self):
        """Authenticated request returns user info."""
        resp = self.client.get("/me", headers=_auth_header(self.token))
        assert resp.status_code == 200
        data = resp.get_json()
        assert "user" in data
        assert data["user"]["email"] == _TEST_EMAIL

    def test_without_token_returns_401(self):
        """Request without token returns 401."""
        resp = self.client.get("/me")
        assert resp.status_code == 401
        data = resp.get_json()
        assert data["error"] == "Missing authentication token"

    def test_with_invalid_token_returns_401(self):
        """Request with garbage token returns 401."""
        resp = self.client.get(
            "/me",
            headers=_auth_header("not.a.valid.token"),
        )
        assert resp.status_code == 401
        data = resp.get_json()
        assert "error" in data

    def test_bearer_prefix_is_stripped(self):
        """Token with Bearer prefix is handled correctly."""
        resp = self.client.get(
            "/me",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        assert resp.status_code == 200

    def test_token_without_bearer_prefix_works(self):
        """Raw token in Authorization header also works."""
        resp = self.client.get(
            "/me",
            headers={"Authorization": self.token},
        )
        assert resp.status_code == 200

    def test_token_from_cookie_works(self):
        """Token provided via id_token cookie is accepted."""
        self.client.set_cookie("id_token", self.token, domain="localhost")
        resp = self.client.get("/me")
        assert resp.status_code == 200


class TestNovaGlobalRoute:
    """Tests for GET /nova_global."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Set up test app with authenticated user and agent."""
        auth_manager = _setup_auth(tmp_path)
        sign_in = _register_test_user(auth_manager)
        self.token = sign_in["token"]
        self.mock_agent = _make_mock_agent()
        app = _build_test_app(auth_manager, self.mock_agent)
        self.client = app.test_client()

    def test_with_valid_token_returns_response(self):
        """Authenticated request with prompt gets LLM response."""
        resp = self.client.get(
            "/nova_global",
            headers=_auth_header(self.token),
            json={"prompt": "Hello"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["response"] == "Hello from the LLM"

    def test_without_token_returns_401(self):
        """Unauthenticated request returns 401."""
        resp = self.client.get("/nova_global", json={"prompt": "Hello"})
        assert resp.status_code == 401

    def test_missing_body_returns_400(self):
        """Request with no JSON body returns 400."""
        resp = self.client.get(
            "/nova_global",
            headers=_auth_header(self.token),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_with_conversation_id(self):
        """Request with conversation_id uses scoped thread."""
        resp = self.client.get(
            "/nova_global",
            headers=_auth_header(self.token),
            json={
                "prompt": "Hello",
                "conversation_id": "conv_42",
            },
        )
        assert resp.status_code == 200
        call_config = self.mock_agent.invoke.call_args[0][1]
        thread_id = call_config["configurable"]["thread_id"]
        assert thread_id == f"{_TEST_EMAIL}_conv_42"

    def test_without_conversation_id_uses_email(self):
        """Request without conversation_id defaults to email."""
        resp = self.client.get(
            "/nova_global",
            headers=_auth_header(self.token),
            json={"prompt": "Hello"},
        )
        assert resp.status_code == 200
        call_config = self.mock_agent.invoke.call_args[0][1]
        thread_id = call_config["configurable"]["thread_id"]
        assert thread_id == _TEST_EMAIL

    def test_agent_invoke_receives_human_message(self):
        """The agent receives a HumanMessage with the prompt."""
        self.client.get(
            "/nova_global",
            headers=_auth_header(self.token),
            json={"prompt": "What is AI?"},
        )
        call_input = self.mock_agent.invoke.call_args[0][0]
        messages = call_input["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "What is AI?"


class TestNovaPerUserRoute:
    """Tests for GET /nova_per_user."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Set up test app with authenticated user and agent."""
        auth_manager = _setup_auth(tmp_path)
        sign_in = _register_test_user(auth_manager)
        self.token = sign_in["token"]
        self.mock_agent = _make_mock_agent()
        app = _build_test_app(auth_manager, self.mock_agent)
        self.client = app.test_client()

    def test_with_valid_token_returns_response(self):
        """Authenticated request gets LLM response."""
        resp = self.client.get(
            "/nova_per_user",
            headers=_auth_header(self.token),
            json={"prompt": "Hello"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["response"] == "Hello from the LLM"

    def test_without_token_returns_401(self):
        """Unauthenticated request returns 401."""
        resp = self.client.get("/nova_per_user", json={"prompt": "Hello"})
        assert resp.status_code == 401

    def test_missing_body_returns_400(self):
        """Request with no JSON body returns 400."""
        resp = self.client.get(
            "/nova_per_user",
            headers=_auth_header(self.token),
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestNovaStreamRoute:
    """Tests for POST /nova_stream."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Set up test app with authenticated user and agent."""
        auth_manager = _setup_auth(tmp_path)
        sign_in = _register_test_user(auth_manager)
        self.token = sign_in["token"]
        self.mock_agent = _make_mock_agent()
        app = _build_test_app(auth_manager, self.mock_agent)
        self.client = app.test_client()

    def test_without_token_returns_401(self):
        """Unauthenticated request returns 401."""
        resp = self.client.post("/nova_stream", json={"prompt": "Hello"})
        assert resp.status_code == 401

    def test_missing_body_returns_400(self):
        """Request with no JSON body returns 400."""
        resp = self.client.post(
            "/nova_stream",
            headers=_auth_header(self.token),
            content_type="application/json",
        )
        assert resp.status_code == 400

    @patch("bili.flask_api.flask_utils.stream_agent_response")
    def test_with_prompt_returns_event_stream(self, mock_stream):
        """Valid request returns text/event-stream response."""
        mock_stream.return_value = iter(
            [
                'event: token\ndata: {"content": "Hi"}\n\n',
                "event: done\ndata: {}\n\n",
            ]
        )
        resp = self.client.post(
            "/nova_stream",
            headers=_auth_header(self.token),
            json={"prompt": "Hello"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.content_type

    @patch("bili.flask_api.flask_utils.stream_agent_response")
    def test_stream_body_contains_tokens(self, mock_stream):
        """Streamed body includes token and done events."""
        mock_stream.return_value = iter(
            [
                'event: token\ndata: {"content": "world"}\n\n',
                "event: done\ndata: {}\n\n",
            ]
        )
        resp = self.client.post(
            "/nova_stream",
            headers=_auth_header(self.token),
            json={"prompt": "Hello"},
        )
        body = resp.get_data(as_text=True)
        assert "event: token" in body
        assert "event: done" in body
        assert "world" in body


class TestActualFlaskAppModule:
    """Tests that import the real flask_app module and verify routes.

    These tests reload flask_app.py with all heavy dependencies
    mocked so that no real LLM, DB, or API key is needed.
    """

    def _reload_flask_app(self):
        """Reload flask_app with all external deps mocked."""
        patches = {
            "bili.iris.checkpointers.pg_checkpointer": MagicMock(),
            "bili.iris.loaders.llm_loader": MagicMock(),
            "bili.iris.loaders.tools_loader": MagicMock(),
            "bili.iris.loaders.langchain_loader": MagicMock(),
            "bili.iris.config.tool_config": MagicMock(),
            "bili.utils.file_utils": MagicMock(),
            "bili.utils.langgraph_utils": MagicMock(),
        }
        # Mock tool config TOOLS dict
        patches["bili.iris.config.tool_config"].TOOLS = {
            "weather_api_tool": {"default_prompt": "weather"},
            "serp_api_tool": {"default_prompt": "search"},
        }
        # Mock langchain_loader module attrs
        patches["bili.iris.loaders.langchain_loader"].DEFAULT_GRAPH_DEFINITION = []
        patches["bili.iris.loaders.langchain_loader"].build_agent_graph.return_value = (
            MagicMock()
        )
        # Mock load_from_json
        patches["bili.utils.file_utils"].load_from_json.return_value = {
            "default": {"persona": "You are helpful."}
        }
        # Mock State
        patches["bili.utils.langgraph_utils"].State = dict

        saved = {}
        for mod_name, mock_mod in patches.items():
            saved[mod_name] = sys.modules.get(mod_name)
            sys.modules[mod_name] = mock_mod

        # Also patch get_auth_manager and checkpointer
        with patch(
            "bili.auth.auth_manager.get_auth_manager",
            return_value=MagicMock(),
        ):
            if "bili.flask_app" in sys.modules:
                del sys.modules["bili.flask_app"]

            import bili.flask_app as flask_app_mod  # pylint: disable=import-outside-toplevel

        # Restore modules
        for mod_name, orig in saved.items():
            if orig is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = orig

        return flask_app_mod

    def test_app_has_all_routes(self):
        """The real flask_app.app registers all 5 expected routes."""
        flask_app_mod = self._reload_flask_app()
        rules = [r.rule for r in flask_app_mod.app.url_map.iter_rules()]
        assert "/login" in rules
        assert "/me" in rules
        assert "/nova_global" in rules
        assert "/nova_per_user" in rules
        assert "/nova_stream" in rules

    def test_app_is_flask_instance(self):
        """The app object is a Flask instance."""
        flask_app_mod = self._reload_flask_app()
        assert isinstance(flask_app_mod.app, Flask)

    def test_route_methods(self):
        """Routes accept the correct HTTP methods."""
        flask_app_mod = self._reload_flask_app()
        rules = {r.rule: r.methods for r in flask_app_mod.app.url_map.iter_rules()}
        assert "POST" in rules["/login"]
        assert "GET" in rules["/me"]
        assert "GET" in rules["/nova_global"]
        assert "GET" in rules["/nova_per_user"]
        assert "POST" in rules["/nova_stream"]


class TestRoleAuthorization:
    """Tests for role-based access control on protected routes."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Set up a restricted Flask app for role tests."""
        self.auth_manager = _setup_auth(tmp_path)
        app = Flask(__name__)
        app.config["TESTING"] = True
        mgr = self.auth_manager

        @app.route("/admin_only", methods=["GET"])
        @auth_required(mgr, required_roles=["admin"])
        def admin_only():
            """Admin-only route."""
            return jsonify({"ok": True})

        @app.route("/researcher_ok", methods=["GET"])
        @auth_required(mgr, required_roles=["admin", "researcher"])
        def researcher_ok():
            """Researcher-accessible route."""
            return jsonify({"ok": True})

        self.client = app.test_client()

    def test_user_without_required_role_gets_403(self):
        """User with researcher role gets 403 on admin-only route."""
        sign_in = _register_test_user(self.auth_manager)
        resp = self.client.get(
            "/admin_only",
            headers=_auth_header(sign_in["token"]),
        )
        assert resp.status_code == 403
        data = resp.get_json()
        assert "Forbidden" in data["error"]

    def test_user_with_matching_role_gets_200(self):
        """User with researcher role passes researcher check."""
        sign_in = _register_test_user(self.auth_manager)
        resp = self.client.get(
            "/researcher_ok",
            headers=_auth_header(sign_in["token"]),
        )
        assert resp.status_code == 200
