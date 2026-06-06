"""Tests for bili.streamlit_ui.ui.auth_ui module.

Tests non-Streamlit logic (initialize_auth_manager and is_authenticated
session-state checking) plus the Streamlit rendering functions
display_login_signup and check_auth via AppTest.
"""

# pylint: disable=import-outside-toplevel

from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from bili.streamlit_ui.tests.conftest import FakeSessionState

# =========================================================================
# initialize_auth_manager
# =========================================================================


class TestInitializeAuthManager:
    """Tests for the initialize_auth_manager factory function."""

    @patch(
        "bili.streamlit_ui.ui.auth_ui.conditional_cache_resource",
    )
    def test_returns_ui_auth_manager_instance(self, mock_cache):
        """initialize_auth_manager returns a UIAuthManager."""
        mock_cache.return_value = lambda fn: fn

        from bili.streamlit_ui.ui.ui_auth_manager import UIAuthManager

        with patch(
            "bili.streamlit_ui.ui.auth_ui.conditional_cache_resource",
            return_value=lambda fn: fn,
        ):
            import importlib

            import bili.streamlit_ui.ui.auth_ui as auth_ui_mod

            importlib.reload(auth_ui_mod)

            with patch.object(
                UIAuthManager,
                "__init__",
                return_value=None,
            ):
                result = auth_ui_mod.initialize_auth_manager(
                    auth_provider_name="sqlite",
                    profile_provider_name="sqlite",
                    role_provider_name="sqlite",
                )
                assert isinstance(result, UIAuthManager)

    @patch(
        "bili.streamlit_ui.ui.auth_ui.conditional_cache_resource",
    )
    def test_passes_provider_names_through(self, mock_cache):
        """Provider names are forwarded to UIAuthManager."""
        mock_cache.return_value = lambda fn: fn

        with patch(
            "bili.streamlit_ui.ui.auth_ui.conditional_cache_resource",
            return_value=lambda fn: fn,
        ):
            import importlib

            import bili.streamlit_ui.ui.auth_ui as auth_ui_mod

            importlib.reload(auth_ui_mod)

            with patch(
                "bili.streamlit_ui.ui.ui_auth_manager.UIAuthManager.__init__",
                return_value=None,
            ) as mock_init:
                auth_ui_mod.initialize_auth_manager(
                    auth_provider_name="firebase",
                    profile_provider_name="firebase",
                    role_provider_name="firebase",
                )
                mock_init.assert_called_once_with(
                    auth_provider_name="firebase",
                    profile_provider_name="firebase",
                    role_provider_name="firebase",
                )


# =========================================================================
# is_authenticated
# =========================================================================


class TestIsAuthenticated:
    """Tests for the is_authenticated function."""

    @patch("bili.streamlit_ui.ui.auth_ui.st")
    def test_returns_true_for_researcher(self, mock_st):
        """Returns True when role is 'researcher'."""
        mock_st.session_state = FakeSessionState(
            user_info={"email": "u@test.com"},
            role="researcher",
        )
        from bili.streamlit_ui.ui.auth_ui import is_authenticated

        assert is_authenticated() is True

    @patch("bili.streamlit_ui.ui.auth_ui.st")
    def test_returns_true_for_admin(self, mock_st):
        """Returns True when role is 'admin'."""
        mock_st.session_state = FakeSessionState(
            user_info={"email": "u@test.com"},
            role="admin",
        )
        from bili.streamlit_ui.ui.auth_ui import is_authenticated

        assert is_authenticated() is True

    @patch("bili.streamlit_ui.ui.auth_ui.st")
    def test_returns_false_for_user_role(self, mock_st):
        """Returns False when role is 'user' (not approved)."""
        mock_st.session_state = FakeSessionState(
            user_info={"email": "u@test.com"},
            role="user",
        )
        from bili.streamlit_ui.ui.auth_ui import is_authenticated

        assert is_authenticated() is False

    @patch("bili.streamlit_ui.ui.auth_ui.st")
    def test_returns_false_when_no_user_info(self, mock_st):
        """Returns False when user_info is absent."""
        mock_st.session_state = FakeSessionState()
        from bili.streamlit_ui.ui.auth_ui import is_authenticated

        assert is_authenticated() is False

    @patch("bili.streamlit_ui.ui.auth_ui.st")
    def test_returns_false_when_no_role(self, mock_st):
        """Returns False when role key is absent."""
        mock_st.session_state = FakeSessionState(
            user_info={"email": "u@test.com"},
        )
        from bili.streamlit_ui.ui.auth_ui import is_authenticated

        assert is_authenticated() is False


# =========================================================================
# display_login_signup -- rendering and button flows
# =========================================================================


class TestDisplayLoginSignup:
    """Tests for the display_login_signup rendering function."""

    def test_renders_welcome_and_widgets(self):
        """The login/signup view renders the welcome heading and inputs."""
        at = AppTest.from_string(
            """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui.auth_ui import display_login_signup
st.session_state.auth_manager = MagicMock()
display_login_signup()
""",
            default_timeout=15,
        )
        at.run()
        assert not at.exception
        all_md = " ".join(m.value for m in at.markdown)
        assert "Welcome to BiliCore" in all_md
        assert len(at.selectbox) >= 1
        assert len(at.text_input) >= 2

    def test_shows_auth_warning_and_clears_it(self):
        """A pending auth_warning is displayed then cleared from state."""
        at = AppTest.from_string(
            """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui.auth_ui import display_login_signup
st.session_state.auth_manager = MagicMock()
st.session_state.auth_warning = "Bad credentials"
display_login_signup()
st.markdown(f"warn_cleared:{st.session_state.auth_warning == ''}")
""",
            default_timeout=15,
        )
        at.run()
        assert not at.exception
        assert any("Bad credentials" in w.value for w in at.warning)
        assert "warn_cleared:True" in " ".join(m.value for m in at.markdown)

    def test_shows_auth_success_and_clears_it(self):
        """A pending auth_success is displayed then cleared from state."""
        at = AppTest.from_string(
            """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui.auth_ui import display_login_signup
st.session_state.auth_manager = MagicMock()
st.session_state.auth_success = "Signed in"
display_login_signup()
st.markdown(f"ok_cleared:{st.session_state.auth_success == ''}")
""",
            default_timeout=15,
        )
        at.run()
        assert not at.exception
        assert any("Signed in" in s.value for s in at.success)
        assert "ok_cleared:True" in " ".join(m.value for m in at.markdown)

    def test_needs_profile_creation_defaults_to_signup(self):
        """When needs_profile_creation is set the Signup tab is preselected."""
        at = AppTest.from_string(
            """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui.auth_ui import display_login_signup
st.session_state.auth_manager = MagicMock()
st.session_state.needs_profile_creation = True
display_login_signup()
st.markdown(f"choice:{at_choice if False else st.session_state.get('_x','')}")
""",
            default_timeout=15,
        )
        at.run()
        assert not at.exception
        # index=1 means the Signup option is selected by default
        assert at.selectbox[0].value == "Signup"

    def test_login_button_calls_sign_in(self):
        """Clicking Log In calls auth_manager.sign_in with email and password."""
        at = AppTest.from_string(
            """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui.auth_ui import display_login_signup
mgr = MagicMock()
mgr.sign_in.return_value = {"uid": "u1"}
st.session_state.auth_manager = mgr
display_login_signup()
""",
            default_timeout=15,
        )
        at.run()
        assert not at.exception
        at.text_input[0].set_value("user@example.com")
        at.text_input[1].set_value("secret")
        login_buttons = [b for b in at.button if b.label == "Log In"]
        assert login_buttons
        login_buttons[0].click()
        at.run()
        assert not at.exception
        mgr = at.session_state["auth_manager"]
        mgr.sign_in.assert_called_once_with("user@example.com", "secret")
        assert at.session_state["password"] == "secret"
        assert at.session_state["auth_info"] == {"uid": "u1"}

    def test_forgot_password_button_calls_reset(self):
        """Clicking Forgot Password calls auth_manager.reset_password."""
        at = AppTest.from_string(
            """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui.auth_ui import display_login_signup
st.session_state.auth_manager = MagicMock()
display_login_signup()
""",
            default_timeout=15,
        )
        at.run()
        assert not at.exception
        at.text_input[0].set_value("forgot@example.com")
        forgot_buttons = [b for b in at.button if b.label == "Forgot Password"]
        assert forgot_buttons
        forgot_buttons[0].click()
        at.run()
        assert not at.exception
        mgr = at.session_state["auth_manager"]
        mgr.reset_password.assert_called_once_with("forgot@example.com")

    def test_signup_create_account_button_calls_create(self):
        """Selecting Signup and clicking Create Account calls create_account."""
        at = AppTest.from_string(
            """
from unittest.mock import MagicMock
import streamlit as st
from bili.streamlit_ui.ui.auth_ui import display_login_signup
st.session_state.auth_manager = MagicMock()
display_login_signup()
""",
            default_timeout=15,
        )
        at.run()
        assert not at.exception
        at.selectbox[0].set_value("Signup")
        at.run()
        assert not at.exception
        at.text_input[0].set_value("new@example.com")
        at.text_input[1].set_value("pw123")
        # first_name / last_name inputs appear after the password input
        at.text_input[2].set_value("Jane")
        at.text_input[3].set_value("Roe")
        create_buttons = [b for b in at.button if b.label == "Create Account"]
        assert create_buttons
        create_buttons[0].click()
        at.run()
        assert not at.exception
        mgr = at.session_state["auth_manager"]
        mgr.create_account.assert_called_once_with(
            "new@example.com", "pw123", "Jane", "Roe", False
        )


# =========================================================================
# check_auth
# =========================================================================


class TestCheckAuth:
    """Tests for the check_auth function."""

    def test_unauthenticated_renders_login_and_stops(self):
        """When not authenticated check_auth renders login and stops the script."""
        at = AppTest.from_string(
            """
from unittest.mock import MagicMock, patch
import streamlit as st
from bili.streamlit_ui.ui import auth_ui
st.session_state.auth_manager = MagicMock()
with patch.object(auth_ui, "is_authenticated", return_value=False):
    auth_ui.check_auth()
st.markdown("after_stop")
""",
            default_timeout=15,
        )
        at.run()
        assert not at.exception
        # check_auth renders the login/signup view (welcome heading present).
        all_md = " ".join(m.value for m in at.markdown)
        assert "Welcome to BiliCore" in all_md
        # st.stop() halts execution, so the trailing markdown never renders.
        assert "after_stop" not in all_md

    def test_authenticated_shows_welcome_and_signout_button(self):
        """When authenticated check_auth greets the user and shows Sign Out."""
        at = AppTest.from_string(
            """
from unittest.mock import MagicMock, patch
import streamlit as st
from bili.streamlit_ui.ui import auth_ui
st.session_state.auth_manager = MagicMock()
st.session_state.user_info = {"email": "u@test.com"}
with patch.object(auth_ui, "is_authenticated", return_value=True):
    auth_ui.check_auth()
""",
            default_timeout=15,
        )
        at.run()
        assert not at.exception
        assert any("Welcome u@test.com" in s.value for s in at.success)
        assert any(b.label == "Sign Out" for b in at.button)

    def test_sign_out_button_calls_sign_out(self):
        """Clicking Sign Out calls auth_manager.sign_out and shows confirmation."""
        at = AppTest.from_string(
            """
from unittest.mock import MagicMock, patch
import streamlit as st
from bili.streamlit_ui.ui import auth_ui
# setdefault keeps the SAME mock object across reruns so the assertion
# observes the call made during the button-click rerun.
st.session_state.setdefault("auth_manager", MagicMock())
st.session_state.user_info = {"email": "u@test.com"}
with patch.object(auth_ui, "is_authenticated", return_value=True):
    auth_ui.check_auth()
""",
            default_timeout=15,
        )
        at.run()
        assert not at.exception
        signout = [b for b in at.button if b.label == "Sign Out"]
        assert signout
        signout[0].click()
        at.run()
        assert not at.exception
        mgr = at.session_state["auth_manager"]
        mgr.sign_out.assert_called_once()
