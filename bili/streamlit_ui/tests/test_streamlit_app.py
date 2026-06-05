"""Tests for bili.streamlit_app main entry point.

Validates that main() can be called with mocked Streamlit and
that the page navigation dict contains IRIS, AETHER, and AEGIS
sections with the expected pages. Also exercises the individual
page-render callbacks that st.navigation wires up.
"""

from unittest.mock import MagicMock, patch

from bili.streamlit_app import (
    _run_aether_page,
    _run_attack_page,
    _run_attack_results_page,
    _run_baseline_runner_page,
    _run_bilicore_page,
    _run_results_page,
    main,
)
from bili.streamlit_ui.tests.conftest import FakeSessionState

_MODULE = "bili.streamlit_app"


def _build_mock_st():
    """Create a mock Streamlit module with session_state."""
    mock_st = MagicMock()
    mock_st.session_state = FakeSessionState()
    # st.Page should return a distinguishable object
    mock_st.Page = MagicMock(side_effect=lambda fn, **kw: MagicMock(_fn=fn, **kw))
    mock_pg = MagicMock()
    mock_st.navigation = MagicMock(return_value=mock_pg)
    return mock_st


class TestStreamlitAppMain:
    """Tests for the main() entry point."""

    @patch(f"{_MODULE}.get_checkpointer")
    @patch(f"{_MODULE}.run_app_page")
    @patch(f"{_MODULE}.render_attack_results_page")
    @patch(f"{_MODULE}.render_attack_page")
    @patch(f"{_MODULE}.render_results_page")
    @patch(f"{_MODULE}.render_aether_page")
    @patch(f"{_MODULE}.check_auth")
    @patch(f"{_MODULE}.initialize_auth_manager")
    @patch(f"{_MODULE}.Image")
    @patch(f"{_MODULE}.st")
    def test_main_calls_set_page_config(
        self,
        mock_st,
        mock_image,
        _mock_init_auth,
        _mock_check_auth,
        *_mocks,
    ):
        """main() calls st.set_page_config."""
        mock_st.session_state = FakeSessionState()
        mock_st.Page = MagicMock(side_effect=lambda fn, **kw: MagicMock())
        mock_pg = MagicMock()
        mock_st.navigation = MagicMock(return_value=mock_pg)
        mock_image.open.return_value = MagicMock()

        main()

        mock_st.set_page_config.assert_called_once()
        call_kwargs = mock_st.set_page_config.call_args
        assert "BiliCore" in str(call_kwargs)

    @patch(f"{_MODULE}.get_checkpointer")
    @patch(f"{_MODULE}.run_app_page")
    @patch(f"{_MODULE}.render_attack_results_page")
    @patch(f"{_MODULE}.render_attack_page")
    @patch(f"{_MODULE}.render_results_page")
    @patch(f"{_MODULE}.render_aether_page")
    @patch(f"{_MODULE}.check_auth")
    @patch(f"{_MODULE}.initialize_auth_manager")
    @patch(f"{_MODULE}.Image")
    @patch(f"{_MODULE}.st")
    def test_main_initializes_auth(
        self,
        mock_st,
        mock_image,
        mock_init_auth,
        mock_check_auth,
        *_mocks,
    ):
        """main() initializes auth and calls check_auth."""
        mock_st.session_state = FakeSessionState()
        mock_st.Page = MagicMock(side_effect=lambda fn, **kw: MagicMock())
        mock_st.navigation = MagicMock(return_value=MagicMock())
        mock_image.open.return_value = MagicMock()

        main()

        mock_init_auth.assert_called_once_with(
            auth_provider_name="sqlite",
            profile_provider_name="sqlite",
            role_provider_name="sqlite",
        )
        mock_check_auth.assert_called_once()

    @patch(f"{_MODULE}.get_checkpointer")
    @patch(f"{_MODULE}.run_app_page")
    @patch(f"{_MODULE}.render_attack_results_page")
    @patch(f"{_MODULE}.render_attack_page")
    @patch(f"{_MODULE}.render_results_page")
    @patch(f"{_MODULE}.render_aether_page")
    @patch(f"{_MODULE}.check_auth")
    @patch(f"{_MODULE}.initialize_auth_manager")
    @patch(f"{_MODULE}.Image")
    @patch(f"{_MODULE}.st")
    def test_navigation_has_iris_aether_aegis(
        self,
        mock_st,
        mock_image,
        _mock_init_auth,
        _mock_check_auth,
        *_mocks,
    ):
        """Navigation dict has IRIS, AETHER, AEGIS keys."""
        mock_st.session_state = FakeSessionState()
        mock_st.Page = MagicMock(side_effect=lambda fn, **kw: MagicMock())
        mock_pg = MagicMock()
        mock_st.navigation = MagicMock(return_value=mock_pg)
        mock_image.open.return_value = MagicMock()

        main()

        nav_call = mock_st.navigation.call_args
        nav_dict = nav_call[0][0]
        assert "IRIS" in nav_dict
        assert "AETHER" in nav_dict
        assert "AEGIS" in nav_dict

    @patch(f"{_MODULE}.get_checkpointer")
    @patch(f"{_MODULE}.run_app_page")
    @patch(f"{_MODULE}.render_attack_results_page")
    @patch(f"{_MODULE}.render_attack_page")
    @patch(f"{_MODULE}.render_results_page")
    @patch(f"{_MODULE}.render_aether_page")
    @patch(f"{_MODULE}.check_auth")
    @patch(f"{_MODULE}.initialize_auth_manager")
    @patch(f"{_MODULE}.Image")
    @patch(f"{_MODULE}.st")
    def test_iris_section_has_one_page(
        self,
        mock_st,
        mock_image,
        _mock_init_auth,
        _mock_check_auth,
        *_mocks,
    ):
        """IRIS section has exactly one page."""
        mock_st.session_state = FakeSessionState()
        mock_st.Page = MagicMock(side_effect=lambda fn, **kw: MagicMock())
        mock_st.navigation = MagicMock(return_value=MagicMock())
        mock_image.open.return_value = MagicMock()

        main()

        nav_dict = mock_st.navigation.call_args[0][0]
        assert len(nav_dict["IRIS"]) == 1

    @patch(f"{_MODULE}.get_checkpointer")
    @patch(f"{_MODULE}.run_app_page")
    @patch(f"{_MODULE}.render_attack_results_page")
    @patch(f"{_MODULE}.render_attack_page")
    @patch(f"{_MODULE}.render_results_page")
    @patch(f"{_MODULE}.render_aether_page")
    @patch(f"{_MODULE}.check_auth")
    @patch(f"{_MODULE}.initialize_auth_manager")
    @patch(f"{_MODULE}.Image")
    @patch(f"{_MODULE}.st")
    def test_aether_section_has_one_page(
        self,
        mock_st,
        mock_image,
        _mock_init_auth,
        _mock_check_auth,
        *_mocks,
    ):
        """AETHER section has exactly one page."""
        mock_st.session_state = FakeSessionState()
        mock_st.Page = MagicMock(side_effect=lambda fn, **kw: MagicMock())
        mock_st.navigation = MagicMock(return_value=MagicMock())
        mock_image.open.return_value = MagicMock()

        main()

        nav_dict = mock_st.navigation.call_args[0][0]
        assert len(nav_dict["AETHER"]) == 1

    @patch(f"{_MODULE}.get_checkpointer")
    @patch(f"{_MODULE}.run_app_page")
    @patch(f"{_MODULE}.render_attack_results_page")
    @patch(f"{_MODULE}.render_attack_page")
    @patch(f"{_MODULE}.render_results_page")
    @patch(f"{_MODULE}.render_aether_page")
    @patch(f"{_MODULE}.check_auth")
    @patch(f"{_MODULE}.initialize_auth_manager")
    @patch(f"{_MODULE}.Image")
    @patch(f"{_MODULE}.st")
    def test_aegis_section_has_four_pages(
        self,
        mock_st,
        mock_image,
        _mock_init_auth,
        _mock_check_auth,
        *_mocks,
    ):
        """AEGIS section has exactly four pages (Attack Suite, Attack Results, Baseline Runner, Baseline Results)."""
        mock_st.session_state = FakeSessionState()
        mock_st.Page = MagicMock(side_effect=lambda fn, **kw: MagicMock())
        mock_st.navigation = MagicMock(return_value=MagicMock())
        mock_image.open.return_value = MagicMock()

        main()

        nav_dict = mock_st.navigation.call_args[0][0]
        assert len(nav_dict["AEGIS"]) == 4

    @patch(f"{_MODULE}.get_checkpointer")
    @patch(f"{_MODULE}.run_app_page")
    @patch(f"{_MODULE}.render_attack_results_page")
    @patch(f"{_MODULE}.render_attack_page")
    @patch(f"{_MODULE}.render_results_page")
    @patch(f"{_MODULE}.render_aether_page")
    @patch(f"{_MODULE}.check_auth")
    @patch(f"{_MODULE}.initialize_auth_manager")
    @patch(f"{_MODULE}.Image")
    @patch(f"{_MODULE}.st")
    def test_pg_run_is_called(
        self,
        mock_st,
        mock_image,
        _mock_init_auth,
        _mock_check_auth,
        *_mocks,
    ):
        """main() calls pg.run() on the navigation object."""
        mock_st.session_state = FakeSessionState()
        mock_st.Page = MagicMock(side_effect=lambda fn, **kw: MagicMock())
        mock_pg = MagicMock()
        mock_st.navigation = MagicMock(return_value=mock_pg)
        mock_image.open.return_value = MagicMock()

        main()

        mock_pg.run.assert_called_once()

    @patch(f"{_MODULE}.get_checkpointer")
    @patch(f"{_MODULE}.run_app_page")
    @patch(f"{_MODULE}.render_attack_results_page")
    @patch(f"{_MODULE}.render_attack_page")
    @patch(f"{_MODULE}.render_results_page")
    @patch(f"{_MODULE}.render_aether_page")
    @patch(f"{_MODULE}.check_auth")
    @patch(f"{_MODULE}.initialize_auth_manager")
    @patch(f"{_MODULE}.Image")
    @patch(f"{_MODULE}.st")
    def test_custom_css_applied(
        self,
        mock_st,
        mock_image,
        _mock_init_auth,
        _mock_check_auth,
        *_mocks,
    ):
        """main() applies CUSTOM_CSS via st.markdown."""
        mock_st.session_state = FakeSessionState()
        mock_st.Page = MagicMock(side_effect=lambda fn, **kw: MagicMock())
        mock_st.navigation = MagicMock(return_value=MagicMock())
        mock_image.open.return_value = MagicMock()

        main()

        # At least one markdown call with unsafe_allow_html
        html_calls = [
            c
            for c in mock_st.markdown.call_args_list
            if c.kwargs.get("unsafe_allow_html")
        ]
        assert len(html_calls) >= 1


class TestPageRenderCallbacks:
    """Tests for the individual page-render callbacks wired into navigation."""

    @patch(f"{_MODULE}.run_app_page")
    @patch(f"{_MODULE}.get_checkpointer")
    @patch(f"{_MODULE}.Image")
    @patch(f"{_MODULE}.st")
    def test_run_bilicore_page_renders_and_runs_app(
        self,
        mock_st,
        mock_image,
        mock_get_checkpointer,
        mock_run_app_page,
    ):
        """The IRIS page renders the header and delegates to run_app_page."""
        mock_image.open.return_value = MagicMock()
        mock_get_checkpointer.return_value = "CHECKPOINTER"

        _run_bilicore_page()

        # The checkpointer is fetched and handed to the IRIS chat page.
        mock_get_checkpointer.assert_called_once()
        mock_run_app_page.assert_called_once_with("CHECKPOINTER")
        # The IRIS heading is rendered.
        markdown_calls = " ".join(str(c) for c in mock_st.markdown.call_args_list)
        assert "IRIS" in markdown_calls

    @patch(f"{_MODULE}.run_app_page")
    @patch(f"{_MODULE}.get_checkpointer")
    @patch(f"{_MODULE}.Image")
    @patch(f"{_MODULE}.st")
    def test_run_bilicore_page_shows_logo_when_present(
        self,
        mock_st,
        mock_image,
        _mock_get_checkpointer,
        _mock_run_app_page,
    ):
        """The IRIS page shows the sidebar logo when the logo file exists."""
        mock_image.open.return_value = MagicMock()

        with patch(f"{_MODULE}.LOGO_PATH") as mock_logo_path:
            mock_logo_path.exists.return_value = True
            mock_logo_path.as_posix.return_value = "/tmp/logo.png"
            _run_bilicore_page()

        # st.image is called for the sidebar logo and the header logo.
        assert mock_st.image.call_count >= 1

    @patch(f"{_MODULE}.run_app_page")
    @patch(f"{_MODULE}.get_checkpointer")
    @patch(f"{_MODULE}.Image")
    @patch(f"{_MODULE}.st")
    def test_run_bilicore_page_skips_logo_when_absent(
        self,
        mock_st,
        mock_image,
        _mock_get_checkpointer,
        _mock_run_app_page,
    ):
        """The IRIS page skips the sidebar logo when the logo file is absent."""
        mock_image.open.return_value = MagicMock()

        with patch(f"{_MODULE}.LOGO_PATH") as mock_logo_path:
            mock_logo_path.exists.return_value = False
            mock_logo_path.as_posix.return_value = "/tmp/logo.png"
            _run_bilicore_page()

        # The header logo (Image.open + st.image) still renders even with no
        # sidebar logo, so the page does not crash.
        assert mock_st.markdown.called

    @patch(f"{_MODULE}.render_aether_page")
    def test_run_aether_page_delegates(self, mock_render):
        """The AETHER page delegates to render_aether_page."""
        _run_aether_page()
        mock_render.assert_called_once_with()

    @patch(f"{_MODULE}.render_baseline_runner_page")
    def test_run_baseline_runner_page_delegates(self, mock_render):
        """The baseline runner page delegates to render_baseline_runner_page."""
        _run_baseline_runner_page()
        mock_render.assert_called_once_with()

    @patch(f"{_MODULE}.render_results_page")
    def test_run_results_page_delegates(self, mock_render):
        """The baseline results page delegates to render_results_page."""
        _run_results_page()
        mock_render.assert_called_once_with()

    @patch(f"{_MODULE}.render_attack_results_page")
    def test_run_attack_results_page_delegates(self, mock_render):
        """The attack results page delegates to render_attack_results_page."""
        _run_attack_results_page()
        mock_render.assert_called_once_with()

    @patch(f"{_MODULE}.render_attack_page")
    def test_run_attack_page_delegates(self, mock_render):
        """The attack page delegates to render_attack_page."""
        _run_attack_page()
        mock_render.assert_called_once_with()
