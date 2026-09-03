"""Tests for the command-line entry point.

The exit code is the whole contract for a scheduled job, and its one hard
requirement is that a fetch failure can never produce the code that means "no
divergence". That is what distinguishes a broken job from a passing one.
"""

import io
import json

from bili.iris.config.catalog_divergence import cli
from bili.iris.config.catalog_divergence.datasets import MODELS_DEV_URL
from bili.iris.config.catalog_divergence.report import STICKY_MARKER


def run(argv):
    """Run the CLI and capture its text output.

    :param argv: The argument vector.
    :returns: A ``(exit_code, text)`` pair.
    :rtype: tuple
    """
    buffer = io.StringIO()
    code = cli.main(argv, stdout=buffer)
    return code, buffer.getvalue()


class TestParser:
    """The flags the scheduled job and a maintainer both use."""

    def test_it_defaults_to_the_network(self):
        """With no file flags the check reads the live upstreams."""
        args = cli.build_parser().parse_args([])
        assert args.models_dev_file is None
        assert args.litellm_file is None

    def test_the_timeout_is_configurable(self):
        """A slow upstream must not need a code change to survive."""
        assert cli.build_parser().parse_args(["--timeout", "5"]).timeout == 5


class TestExitCodes:
    """Three codes, because three things can happen."""

    def test_a_clean_comparison_exits_zero(self, models_dev_path, litellm_path):
        """Everything read, real coverage, and nothing over-claimed.

        This runs against the recorded slices rather than a hand-made pair,
        because the CLI always compares the SHIPPED catalog and a toy dataset
        resolves almost none of it, which is now a reported failure in its own
        right. A clean exit has to be earned against data that actually
        covers the catalog.
        """
        code, text = run(
            [
                "--models-dev-file",
                str(models_dev_path),
                "--litellm-file",
                str(litellm_path),
            ]
        )
        assert code == cli.EXIT_OK
        assert "INCOMPLETE" not in text
        assert "BELOW THE RECORDED FLOORS" not in text

    def test_datasets_that_resolve_nothing_do_not_exit_zero(self, tmp_path):
        """A well-formed document that covers nothing is a broken check.

        An upstream renaming a provider key still serves valid JSON, so
        nothing fails to parse and every lookup simply stops hitting. Without
        this the run would report no divergence from a check that had
        silently stopped working, which is the same silence a failed fetch
        would produce.
        """
        md = tmp_path / "md.json"
        md.write_text(
            json.dumps({"a-renamed-provider": {"models": {"m": {}}}}),
            encoding="utf-8",
        )
        ll = tmp_path / "ll.json"
        ll.write_text(
            json.dumps({"m": {"litellm_provider": "a-renamed-provider"}}),
            encoding="utf-8",
        )
        code, text = run(["--models-dev-file", str(md), "--litellm-file", str(ll)])
        assert code == cli.EXIT_UNAVAILABLE
        assert code != cli.EXIT_OK
        assert "BELOW THE RECORDED FLOORS" in text

    def test_an_error_finding_exits_one(self, tmp_path):
        """An over-claimed capability is what non-zero is reserved for.

        The shipped catalog declares image input for this model, so an
        upstream that states a text-only array is a real over-claim reaching
        the real comparison, not a report injected into the CLI.
        """
        md = tmp_path / "md.json"
        md.write_text(
            json.dumps(
                {"openai": {"models": {"gpt-4o": {"modalities": {"input": ["text"]}}}}}
            ),
            encoding="utf-8",
        )
        ll = tmp_path / "ll.json"
        ll.write_text(
            json.dumps({"gpt-4o": {"litellm_provider": "openai"}}), encoding="utf-8"
        )
        code, text = run(["--models-dev-file", str(md), "--litellm-file", str(ll)])
        assert code == cli.EXIT_ERRORS
        assert "INCOMPLETE" not in text

    def test_an_unreadable_dataset_exits_two_and_never_zero(self, tmp_path):
        """A failed fetch must never be reported as "no divergence".

        This is the case the three-code split exists for: with nothing to
        compare against there are no findings, and a two-code scheme would
        return the same value it returns for a clean catalog.
        """
        code, text = run(
            [
                "--models-dev-file",
                str(tmp_path / "missing.json"),
                "--litellm-file",
                str(tmp_path / "missing.json"),
            ]
        )
        assert code == cli.EXIT_UNAVAILABLE
        assert code != cli.EXIT_OK
        assert "INCOMPLETE" in text

    def test_an_error_outranks_an_unreadable_dataset(self, tmp_path):
        """A real error must not be masked by a partial run.

        The report still states that the run was partial; the exit code
        carries the more actionable of the two facts.
        """
        md = tmp_path / "md.json"
        md.write_text(
            json.dumps(
                {"openai": {"models": {"gpt-4o": {"modalities": {"input": ["text"]}}}}}
            ),
            encoding="utf-8",
        )
        code, text = run(
            ["--models-dev-file", str(md), "--litellm-file", str(tmp_path / "no.json")]
        )
        assert code == cli.EXIT_ERRORS
        assert "INCOMPLETE" in text


class TestOutputs:
    """The two forms the CLI can emit."""

    def test_the_text_report_goes_to_the_given_stream(self, tmp_path):
        """The report is written where the caller asked."""
        _, text = run(
            [
                "--models-dev-file",
                str(tmp_path / "no.json"),
                "--litellm-file",
                str(tmp_path / "no.json"),
            ]
        )
        assert "Model catalog divergence report" in text

    def test_quiet_suppresses_the_text_but_not_the_exit_code(self, tmp_path):
        """A caller that only wants the verdict can have just the verdict."""
        code, text = run(
            [
                "--quiet",
                "--models-dev-file",
                str(tmp_path / "no.json"),
                "--litellm-file",
                str(tmp_path / "no.json"),
            ]
        )
        assert text == ""
        assert code == cli.EXIT_UNAVAILABLE

    def test_the_issue_body_is_written_to_the_given_path(self, tmp_path):
        """The scheduled job posts this file; it must not render it itself.

        Rebuilding the report inside the job's shell would be untested code
        in a place nothing exercises, so the rendering stays here.
        """
        out = tmp_path / "body.md"
        run(
            [
                "--quiet",
                "--issue-body",
                str(out),
                "--run-url",
                "https://ci.example/run/1",
                "--models-dev-file",
                str(tmp_path / "no.json"),
                "--litellm-file",
                str(tmp_path / "no.json"),
            ]
        )
        body = out.read_text(encoding="utf-8")
        assert "INCOMPLETE" in body
        assert "https://ci.example/run/1" in body

    def test_the_issue_body_carries_the_sticky_marker(self, tmp_path):
        """The job finds its own issue by this marker, so it has to be there.

        Emitting it here rather than appending it in the job is what keeps
        the marker the job looks for and the marker the body carries from
        drifting apart.
        """
        out = tmp_path / "body.md"
        run(
            [
                "--quiet",
                "--issue-body",
                str(out),
                "--models-dev-file",
                str(tmp_path / "no.json"),
                "--litellm-file",
                str(tmp_path / "no.json"),
            ]
        )
        assert STICKY_MARKER in out.read_text(encoding="utf-8")

    def test_no_issue_body_is_written_unless_asked(self, tmp_path):
        """The default run writes nothing but its report."""
        run(
            [
                "--quiet",
                "--models-dev-file",
                str(tmp_path / "no.json"),
                "--litellm-file",
                str(tmp_path / "no.json"),
            ]
        )
        assert not list(tmp_path.glob("*.md"))

    def test_json_is_written_to_the_given_path(self, tmp_path):
        """The machine-readable form is what the scheduled job consumes."""
        out = tmp_path / "report.json"
        run(
            [
                "--quiet",
                "--json",
                str(out),
                "--models-dev-file",
                str(tmp_path / "no.json"),
                "--litellm-file",
                str(tmp_path / "no.json"),
            ]
        )
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["complete"] is False
        assert len(data["unavailable"]) == 2


class TestNetworkDefault:  # pylint: disable=too-few-public-methods
    """With no file flags the check reaches for the live upstreams."""

    def test_it_fetches_and_degrades_rather_than_raising(self, monkeypatch):
        """An offline runner produces a reported failure, not a traceback."""
        seen = []

        def _raise(request, **_kwargs):
            seen.append(getattr(request, "full_url", request))
            raise OSError("offline")

        monkeypatch.setattr("urllib.request.urlopen", _raise)
        code, text = run([])
        assert code == cli.EXIT_UNAVAILABLE
        assert "INCOMPLETE" in text
        assert MODELS_DEV_URL in seen


def test_the_three_exit_codes_are_distinct():
    """Collapsing any two would lose a distinction a scheduled job branches on.

    In particular the success code has to be the only zero, or a partial run
    and a clean one become indistinguishable to every caller.
    """
    codes = [cli.EXIT_OK, cli.EXIT_ERRORS, cli.EXIT_UNAVAILABLE]
    assert len(set(codes)) == 3
    assert cli.EXIT_OK == 0
    assert 0 not in (cli.EXIT_ERRORS, cli.EXIT_UNAVAILABLE)
