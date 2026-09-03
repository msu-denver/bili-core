"""Tests for provider prompt caching on the native react-agent path.

``build_react_agent_node`` enables prompt caching by default on the native
(``create_agent``) path for the providers that need explicit wiring: for an
Anthropic model it appends ``AnthropicPromptCachingMiddleware``, and for a
Claude/Nova model on AWS Bedrock it appends the Bedrock cache-point middleware,
innermost, so a multi-call agent run re-reads its stable prefix from cache.
Every other model is left untouched, byte-for-byte.

These are kept in a dedicated module (rather than the main react-agent test
file) so the top-level provider imports stay isolated and the main file stays
small.
"""

from unittest.mock import MagicMock, patch

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_aws import ChatBedrockConverse

from bili.iris.nodes.react_agent_node import build_react_agent_node
from bili.iris.providers.bedrock_cache import BedrockSystemCachePointMiddleware


def _chat():
    """Return an offline ``ChatAnthropic`` (placeholder key, no network)."""
    return ChatAnthropic(model="claude-opus-4-8", api_key="sk-ant-test", max_tokens=16)


def _bedrock(model_id="us.anthropic.claude-sonnet-4-6"):
    """Return an offline ``ChatBedrockConverse`` (region set, no AWS call)."""
    return ChatBedrockConverse(model_id=model_id, region_name="us-east-1")


class TestNativePathPromptCaching:
    """The native path enables Anthropic caching by default and only for
    Anthropic models."""

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_injects_caching_middleware_for_anthropic(self, mock_create_agent):
        """An Anthropic model gets a caching middleware appended (innermost)."""
        mock_create_agent.return_value = MagicMock()
        caller_mw = [MagicMock()]

        build_react_agent_node(
            tools=[MagicMock()],
            llm_model=_chat(),
            middleware=caller_mw,
        )

        passed = mock_create_agent.call_args.kwargs["middleware"]
        # The caller's middleware is preserved and the caching middleware is
        # appended last so it applies to the final request.
        assert passed[0] is caller_mw[0]
        assert isinstance(passed[-1], AnthropicPromptCachingMiddleware)
        assert len(passed) == 2

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_no_caching_middleware_for_non_anthropic(self, mock_create_agent):
        """A non-Anthropic model's middleware list is passed through unchanged."""
        mock_create_agent.return_value = MagicMock()
        caller_mw = [MagicMock()]

        build_react_agent_node(
            tools=[MagicMock()],
            llm_model=MagicMock(),
            middleware=caller_mw,
        )

        passed = mock_create_agent.call_args.kwargs["middleware"]
        # Identity preserved: no wrapping, no appending, for non-Anthropic models.
        assert passed is caller_mw
        assert not any(isinstance(m, AnthropicPromptCachingMiddleware) for m in passed)

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_no_double_add_when_already_present(self, mock_create_agent):
        """A caller-supplied caching middleware is not duplicated."""
        mock_create_agent.return_value = MagicMock()
        existing = AnthropicPromptCachingMiddleware()

        build_react_agent_node(
            tools=[MagicMock()],
            llm_model=_chat(),
            middleware=[existing],
        )

        passed = mock_create_agent.call_args.kwargs["middleware"]
        caching = [m for m in passed if isinstance(m, AnthropicPromptCachingMiddleware)]
        assert caching == [existing]

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_caching_with_no_caller_middleware(self, mock_create_agent):
        """With no caller middleware, only the caching middleware is passed."""
        mock_create_agent.return_value = MagicMock()

        build_react_agent_node(
            tools=[MagicMock()],
            llm_model=_chat(),
            middleware=None,
        )

        passed = mock_create_agent.call_args.kwargs["middleware"]
        assert len(passed) == 1
        assert isinstance(passed[0], AnthropicPromptCachingMiddleware)


class TestNativePathBedrockCaching:
    """The native path enables Bedrock caching for Claude/Nova models and leaves
    non-caching Bedrock families untouched."""

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_injects_caching_middleware_for_claude_bedrock(self, mock_create_agent):
        """A Claude Bedrock model gets the Bedrock cache middleware appended."""
        mock_create_agent.return_value = MagicMock()
        caller_mw = [MagicMock()]

        build_react_agent_node(
            tools=[MagicMock()],
            llm_model=_bedrock(),
            middleware=caller_mw,
        )

        passed = mock_create_agent.call_args.kwargs["middleware"]
        assert passed[0] is caller_mw[0]
        assert isinstance(passed[-1], BedrockSystemCachePointMiddleware)
        assert len(passed) == 2

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_no_caching_for_non_caching_bedrock_family(self, mock_create_agent):
        """A Bedrock family without caching support is passed through unchanged."""
        mock_create_agent.return_value = MagicMock()
        caller_mw = [MagicMock()]

        build_react_agent_node(
            tools=[MagicMock()],
            llm_model=_bedrock("us.meta.llama3-3-70b-instruct-v1:0"),
            middleware=caller_mw,
        )

        passed = mock_create_agent.call_args.kwargs["middleware"]
        assert passed is caller_mw
        assert not any(isinstance(m, BedrockSystemCachePointMiddleware) for m in passed)

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_no_double_add_when_already_present(self, mock_create_agent):
        """A caller-supplied Bedrock cache middleware is not duplicated."""
        mock_create_agent.return_value = MagicMock()
        existing = BedrockSystemCachePointMiddleware()

        build_react_agent_node(
            tools=[MagicMock()],
            llm_model=_bedrock(),
            middleware=[existing],
        )

        passed = mock_create_agent.call_args.kwargs["middleware"]
        caching = [
            m for m in passed if isinstance(m, BedrockSystemCachePointMiddleware)
        ]
        assert caching == [existing]

    @patch("bili.iris.nodes.react_agent_node.create_agent")
    def test_caching_with_no_caller_middleware(self, mock_create_agent):
        """With no caller middleware, only the Bedrock cache middleware is passed."""
        mock_create_agent.return_value = MagicMock()

        build_react_agent_node(
            tools=[MagicMock()],
            llm_model=_bedrock(),
            middleware=None,
        )

        passed = mock_create_agent.call_args.kwargs["middleware"]
        assert len(passed) == 1
        assert isinstance(passed[0], BedrockSystemCachePointMiddleware)
