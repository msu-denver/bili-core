"""Tests for Flask utility functions.

Covers cookie helpers, the auth_required decorator, the SSE event
formatter, and the handle_agent_prompt function.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask, g, jsonify, make_response

from bili.auth.auth_manager import AuthManager
from bili.flask_api.flask_utils import (
    _sse_event,
    add_unauthorized_handler,
    auth_required,
    handle_agent_prompt,
    handle_agent_prompt_stream,
    set_token_cookies,
    stream_agent_response,
)

_JWT_SECRET = "utils-test-secret-key"
_TEST_EMAIL = "util_test@example.com"
_TEST_PASSWORD = "UtilPass123!"


def _setup_auth(tmp_path):
    """Create an AuthManager with a test user and profile."""
    db_path = str(tmp_path / "utils_test.db")
    os.environ["JWT_SECRET_KEY"] = _JWT_SECRET
    os.environ["PROFILE_DB_PATH"] = db_path

    auth_manager = AuthManager(
        auth_provider_name="sqlite",
        profile_provider_name="sqlite",
        role_provider_name="sqlite",
    )
    auth_manager.auth_provider.create_user(_TEST_EMAIL, _TEST_PASSWORD)
    sign_in = auth_manager.auth_provider.sign_in(_TEST_EMAIL, _TEST_PASSWORD)
    auth_manager.profile_provider.create_user_profile(
        uid=sign_in["uid"],
        email=_TEST_EMAIL,
        first_name="Util",
        last_name="Tester",
        token=sign_in["token"],
    )
    return auth_manager, sign_in


class TestSetTokenCookies:
    """Tests for set_token_cookies()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Create a Flask app for request-context tests."""
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True

    def _make_response_in_context(self):
        """Create a response object inside a request context."""
        return make_response("ok")

    def test_sets_id_token_cookie(self):
        """The id_token cookie is set on the response."""
        with self.app.test_request_context():
            resp = self._make_response_in_context()
            set_token_cookies(resp, "my-id-token")
            headers = resp.headers.getlist("Set-Cookie")
            id_cookies = [h for h in headers if "id_token=" in h]
            assert len(id_cookies) == 1
            assert "my-id-token" in id_cookies[0]

    def test_sets_refresh_token_cookie(self):
        """The refresh_token cookie is set when provided."""
        with self.app.test_request_context():
            resp = self._make_response_in_context()
            set_token_cookies(resp, "my-id-token", "my-refresh-token")
            headers = resp.headers.getlist("Set-Cookie")
            ref_cookies = [h for h in headers if "refresh_token=" in h]
            assert len(ref_cookies) == 1
            assert "my-refresh-token" in ref_cookies[0]

    def test_no_refresh_cookie_when_omitted(self):
        """No refresh_token cookie when refresh_token is None."""
        with self.app.test_request_context():
            resp = self._make_response_in_context()
            set_token_cookies(resp, "my-id-token")
            headers = resp.headers.getlist("Set-Cookie")
            ref_cookies = [h for h in headers if "refresh_token=" in h]
            assert len(ref_cookies) == 0

    def test_id_token_cookie_is_not_httponly(self):
        """The id_token cookie is accessible to JavaScript."""
        with self.app.test_request_context():
            resp = self._make_response_in_context()
            set_token_cookies(resp, "tok123")
            headers = resp.headers.getlist("Set-Cookie")
            id_cookie = [h for h in headers if "id_token=" in h][0]
            assert "HttpOnly" not in id_cookie

    def test_refresh_token_cookie_is_httponly(self):
        """The refresh_token cookie is HTTP-only for security."""
        with self.app.test_request_context():
            resp = self._make_response_in_context()
            set_token_cookies(resp, "tok", "ref")
            headers = resp.headers.getlist("Set-Cookie")
            ref_cookie = [h for h in headers if "refresh_token=" in h][0]
            assert "HttpOnly" in ref_cookie

    def test_cookies_are_samesite_strict(self):
        """Both cookies use SameSite=Strict."""
        with self.app.test_request_context():
            resp = self._make_response_in_context()
            set_token_cookies(resp, "tok", "ref")
            headers = resp.headers.getlist("Set-Cookie")
            for header in headers:
                assert "SameSite=Strict" in header

    def test_returns_response_object(self):
        """set_token_cookies returns the response for chaining."""
        with self.app.test_request_context():
            resp = self._make_response_in_context()
            returned = set_token_cookies(resp, "tok")
            assert returned is resp


class TestAuthRequiredDecorator:
    """Tests for the auth_required decorator."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Set up auth manager and Flask app for each test."""
        self.auth_manager, sign_in = _setup_auth(tmp_path)
        self.token = sign_in["token"]
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True

    def test_missing_token_returns_401(self):
        """Request without any token returns 401."""

        @self.app.route("/p")
        @auth_required(self.auth_manager)
        def protected():
            """Protected route."""
            return jsonify({"ok": True})

        client = self.app.test_client()
        resp = client.get("/p")
        assert resp.status_code == 401
        data = resp.get_json()
        assert data["error"] == "Missing authentication token"

    def test_invalid_token_returns_401(self):
        """Request with invalid token returns 401."""

        @self.app.route("/p2")
        @auth_required(self.auth_manager)
        def protected2():
            """Protected route."""
            return jsonify({"ok": True})

        client = self.app.test_client()
        resp = client.get(
            "/p2",
            headers={"Authorization": "Bearer garbage"},
        )
        assert resp.status_code == 401

    def test_valid_token_sets_g_user(self):
        """Valid token populates g.user with decoded data."""
        captured = {}

        @self.app.route("/check")
        @auth_required(self.auth_manager)
        def check_user():
            """Route that exposes g.user."""
            captured["user"] = dict(g.user)
            return jsonify({"email": g.user["email"]})

        client = self.app.test_client()
        resp = client.get(
            "/check",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        assert resp.status_code == 200
        assert captured["user"]["email"] == _TEST_EMAIL

    def test_role_check_forbids_unauthorized_role(self):
        """User without required role gets 403."""

        @self.app.route("/admin")
        @auth_required(self.auth_manager, required_roles=["admin"])
        def admin_only():
            """Admin-only route."""
            return jsonify({"ok": True})

        client = self.app.test_client()
        resp = client.get(
            "/admin",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        assert resp.status_code == 403
        data = resp.get_json()
        assert "Forbidden" in data["error"]

    def test_role_check_allows_matching_role(self):
        """User with matching role is allowed through."""

        @self.app.route("/researcher")
        @auth_required(self.auth_manager, required_roles=["researcher"])
        def researcher_ok():
            """Researcher route."""
            return jsonify({"ok": True})

        client = self.app.test_client()
        resp = client.get(
            "/researcher",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        assert resp.status_code == 200

    def test_no_role_requirement_skips_check(self):
        """When required_roles is None, role check is skipped."""

        @self.app.route("/open")
        @auth_required(self.auth_manager)
        def no_role():
            """Route with no role requirement."""
            return jsonify({"ok": True})

        client = self.app.test_client()
        resp = client.get(
            "/open",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        assert resp.status_code == 200

    def test_cookie_token_is_accepted(self):
        """Token from id_token cookie is accepted."""

        @self.app.route("/cookie")
        @auth_required(self.auth_manager)
        def cookie_auth():
            """Cookie-authenticated route."""
            return jsonify({"ok": True})

        client = self.app.test_client()
        client.set_cookie("id_token", self.token, domain="localhost")
        resp = client.get("/cookie")
        assert resp.status_code == 200


class TestSseEvent:
    """Tests for the _sse_event helper."""

    def test_formats_token_event(self):
        """Token event has correct SSE format."""
        result = _sse_event("token", {"content": "Hello"})
        assert result.startswith("event: token\n")
        assert "data: " in result
        assert result.endswith("\n\n")
        data_line = result.split("\n")[1]
        payload = json.loads(data_line.replace("data: ", ""))
        assert payload["content"] == "Hello"

    def test_formats_done_event(self):
        """Done event has correct SSE format with empty data."""
        result = _sse_event("done", {})
        assert "event: done\n" in result
        data_line = result.split("\n")[1]
        payload = json.loads(data_line.replace("data: ", ""))
        assert payload == {}

    def test_data_is_valid_json(self):
        """The data portion is valid JSON."""
        result = _sse_event("test", {"key": "value", "num": 42})
        data_line = result.split("\n")[1]
        raw_json = data_line.replace("data: ", "")
        parsed = json.loads(raw_json)
        assert parsed["key"] == "value"
        assert parsed["num"] == 42


class TestHandleAgentPrompt:
    """Tests for handle_agent_prompt()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up Flask app and mock agent for each test."""
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.mock_agent = MagicMock()
        self.mock_agent.invoke.return_value = {
            "messages": [MagicMock(content="Agent reply")]
        }

    def test_returns_agent_response(self):
        """Successful invocation returns the agent content."""
        user = {"email": "a@b.com", "uid": "u1"}
        with self.app.app_context():
            resp = handle_agent_prompt(user, self.mock_agent, "Hi")
        data = resp.get_json()
        assert data["response"] == "Agent reply"

    def test_user_without_identifier_returns_400(self):
        """User dict with no email or uid returns 400."""
        with self.app.app_context():
            result = handle_agent_prompt({}, self.mock_agent, "Hi")
        _, status = result
        assert status == 400

    def test_agent_empty_result_returns_500(self):
        """Agent returning non-dict falls back to 500."""
        self.mock_agent.invoke.return_value = "unexpected"
        user = {"email": "a@b.com"}
        with self.app.app_context():
            result = handle_agent_prompt(user, self.mock_agent, "Hi")
        _, status = result
        assert status == 500

    def test_uid_used_when_email_missing(self):
        """Falls back to uid when email is absent."""
        user = {"uid": "user-123"}
        with self.app.app_context():
            handle_agent_prompt(user, self.mock_agent, "Hi")
        config = self.mock_agent.invoke.call_args[0][1]
        assert config["configurable"]["thread_id"] == "user-123"


class TestStreamAgentResponse:
    """Tests for stream_agent_response()."""

    @patch("bili.iris.loaders.streaming_utils.stream_agent")
    def test_yields_token_and_done_events(self, mock_stream_fn):
        """Generator yields SSE token events then a done event."""
        mock_stream_fn.return_value = iter(["Hello", " World"])
        events = list(stream_agent_response(MagicMock(), "Hi", "thread-1"))
        token_events = [e for e in events if e.startswith("event: token")]
        done_events = [e for e in events if e.startswith("event: done")]
        assert len(token_events) == 2
        assert len(done_events) == 1

    @patch("bili.iris.loaders.streaming_utils.stream_agent")
    def test_token_event_contains_content(self, mock_stream_fn):
        """Each token event carries the streamed content."""
        mock_stream_fn.return_value = iter(["chunk"])
        events = list(stream_agent_response(MagicMock(), "Hi", "t1"))
        token_event = events[0]
        data_line = token_event.split("\n")[1]
        payload = json.loads(data_line.replace("data: ", ""))
        assert payload["content"] == "chunk"

    @patch("bili.iris.loaders.streaming_utils.stream_agent")
    def test_done_event_has_empty_payload(self, mock_stream_fn):
        """The final done event carries an empty dict payload."""
        mock_stream_fn.return_value = iter(["tok"])
        events = list(stream_agent_response(MagicMock(), "Hi", "t1"))
        done_event = events[-1]
        data_line = done_event.split("\n")[1]
        payload = json.loads(data_line.replace("data: ", ""))
        assert payload == {}


class TestHandleAgentPromptStream:
    """Tests for handle_agent_prompt_stream()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Create a Flask app for request-context tests."""
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True

    @patch("bili.flask_api.flask_utils.stream_agent_response")
    def test_returns_event_stream_content_type(self, mock_stream):
        """Response has text/event-stream content type."""
        mock_stream.return_value = iter(["event: done\ndata: {}\n\n"])
        user = {"email": "x@y.com"}
        with self.app.test_request_context():
            resp = handle_agent_prompt_stream(user, MagicMock(), "Hi")
        assert "text/event-stream" in resp.content_type

    @patch("bili.flask_api.flask_utils.stream_agent_response")
    def test_sets_no_cache_header(self, mock_stream):
        """Response includes Cache-Control: no-cache."""
        mock_stream.return_value = iter(["event: done\ndata: {}\n\n"])
        user = {"email": "x@y.com"}
        with self.app.test_request_context():
            resp = handle_agent_prompt_stream(user, MagicMock(), "Hi")
        cache_ctrl = resp.headers.get("Cache-Control")
        assert cache_ctrl == "no-cache"

    @patch("bili.flask_api.flask_utils.stream_agent_response")
    def test_sets_no_buffering_header(self, mock_stream):
        """Response includes X-Accel-Buffering: no."""
        mock_stream.return_value = iter(["event: done\ndata: {}\n\n"])
        user = {"email": "x@y.com"}
        with self.app.test_request_context():
            resp = handle_agent_prompt_stream(user, MagicMock(), "Hi")
        buffering = resp.headers.get("X-Accel-Buffering")
        assert buffering == "no"

    def test_missing_user_identifier_returns_400(self):
        """User without email or uid returns 400."""
        with self.app.app_context():
            result = handle_agent_prompt_stream({}, MagicMock(), "Hi")
        _, status = result
        assert status == 400


class TestAddUnauthorizedHandler:
    """Tests for add_unauthorized_handler middleware."""

    def test_non_401_responses_pass_through(self):
        """Responses that are not 401 are not intercepted."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        auth_manager = MagicMock()
        add_unauthorized_handler(auth_manager, app)

        @app.route("/ok")
        def ok_route():
            """Simple 200 route."""
            return jsonify({"status": "ok"})

        client = app.test_client()
        resp = client.get("/ok")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_401_without_refresh_cookie_passes_through(self):
        """A 401 with no refresh_token cookie is returned as-is."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        auth_manager = MagicMock()
        add_unauthorized_handler(auth_manager, app)

        @app.route("/fail")
        def fail_route():
            """Route that always returns 401."""
            return jsonify({"error": "unauthorized"}), 401

        client = app.test_client()
        resp = client.get("/fail")
        assert resp.status_code == 401
