"""Tests for bili.streamlit_ui.ui.auth_ui module.

Tests non-Streamlit logic: initialize_auth_manager and
is_authenticated session-state checking.
"""

# pylint: disable=import-outside-toplevel

from unittest.mock import patch

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
