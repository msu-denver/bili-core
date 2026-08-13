"""Tests for operator steering — user-initiated mid-run redirect.

Steering lets a human supervising a long-running run inject a directive that
the next agent observes at the next superstep boundary, without killing the
run and starting over. It is the opposite direction from HITL / ``ask_user``
(where an agent pauses to ask and a human answers).

These tests EXECUTE the behavior against a real compiled graph and are able to
FAIL:

- The observation tests back each agent with a fake echo model whose output
  literally contains every message it saw, so "the next node observed the
  injected directive" is a concrete assertion on that node's output rather
  than an inference from reading the code.
- The additivity tests assert the load-bearing safety property: an unused
  steer channel changes nothing. Steering disabled compiles with no interrupt
  points (byte-for-byte the non-steering path), and an enabled-but-empty
  steerable run produces the same agent nodes, in the same order, seeing the
  same messages, as a plain streamed run.

All tests run without any LLM API calls: the echo tests patch model creation
and the additivity/error tests use stub agents.
"""

# pylint: disable=missing-function-docstring,protected-access
# pylint: disable=unused-argument,too-few-public-methods
#   unused-argument: the ``echo_models`` fixture is requested for its
#     monkeypatch side effect, not referenced in the test body.
#   too-few-public-methods: ``_EchoModel`` is a one-method test double.

import pytest
from langchain_core.messages import (  # pylint: disable=import-error
    AIMessage,
    HumanMessage,
)

from bili.aether.runtime.executor import MASExecutor
from bili.aether.schema import AgentSpec, MASConfig, WorkflowType

# Real agent node names for a sequential config built by _echo_config /
# _stub_config below; used to filter out the ``__interrupt__`` bookkeeping
# sentinels the graph emits at each pause.
_AGENT_IDS = {f"agent_{i}" for i in range(8)}


# =========================================================================
# Helpers
# =========================================================================


class _EchoModel:
    """A minimal fake chat model that echoes the content it was handed.

    The direct-LLM agent node calls ``llm.invoke(messages)`` and reads
    ``response.content``, so an agent backed by this model produces an output
    that literally contains every message it observed — which is what lets a
    test assert that an injected directive was seen by the next node.
    """

    def invoke(self, messages):
        seen = " || ".join(str(getattr(m, "content", "")) for m in messages)
        return AIMessage(content=f"ECHO[{seen}]")


@pytest.fixture(name="echo_models")
def _echo_models(monkeypatch):
    """Back every model-bearing agent with the fake echo model.

    ``create_llm`` is imported inside the agent-node builder at compile time,
    so patching the module attribute is picked up by ``initialize()``.
    """
    monkeypatch.setattr(
        "bili.aether.compiler.llm_resolver.create_llm",
        lambda agent: _EchoModel(),
    )


def _agent(agent_id: str, **kwargs) -> AgentSpec:
    defaults = {"role": "reviewer", "objective": f"Objective for {agent_id}"}
    defaults.update(kwargs)
    return AgentSpec(agent_id=agent_id, **defaults)


def _echo_config(n_agents: int = 2, **kwargs) -> MASConfig:
    """Sequential config whose agents run on the fake echo model."""
    agents = [_agent(f"agent_{i}", model_name="echo-model") for i in range(n_agents)]
    defaults = {
        "mas_id": "steer_echo",
        "name": "Steering Echo MAS",
        "workflow_type": WorkflowType.SEQUENTIAL,
        "agents": agents,
        # checkpoint_enabled=False proves steering attaches its own saver.
        "checkpoint_enabled": False,
    }
    defaults.update(kwargs)
    return MASConfig(**defaults)


def _stub_config(n_agents: int = 2, **kwargs) -> MASConfig:
    """Sequential config with stub agents (no model)."""
    agents = [_agent(f"agent_{i}") for i in range(n_agents)]
    defaults = {
        "mas_id": "steer_stub",
        "name": "Steering Stub MAS",
        "workflow_type": WorkflowType.SEQUENTIAL,
        "agents": agents,
        "checkpoint_enabled": False,
    }
    defaults.update(kwargs)
    return MASConfig(**defaults)


def _agent_updates(updates):
    """Reduce a stream of ``(node, state_update)`` to agent nodes + contents.

    Filters out the ``__interrupt__`` sentinels emitted at each pause so two
    runs can be compared on their agent-observable behavior.
    """
    return [
        (node, [str(m.content) for m in upd.get("messages", [])])
        for (node, upd) in updates
        if node in _AGENT_IDS
    ]


# =========================================================================
# Observation: an injected directive is seen by the next node
# =========================================================================


class TestDirectiveObservation:
    """A directive injected mid-run is observed by the next node."""

    def test_queued_directive_is_observed_by_next_node(self, echo_models):
        config = _echo_config(n_agents=2)
        executor = MASExecutor(config, enable_steering=True)
        executor.initialize()

        gen = executor.run_streaming_steerable(
            {"messages": [HumanMessage(content="start")]}, thread_id="t1"
        )

        # Drive the run to the first boundary (after agent_0 completes), then
        # inject a directive from "outside" before agent_1 runs.
        updates = []
        for node, upd in gen:
            updates.append((node, upd))
            if node == "agent_0":
                executor.submit_steer("EMPHASIZE_XYZ")
                break
        updates.extend(gen)

        by_node = {n: u for (n, u) in updates if n in _AGENT_IDS}
        assert "agent_1" in by_node
        agent1_output = " ".join(str(m.content) for m in by_node["agent_1"]["messages"])
        # The next agent's echoed output contains the injected directive,
        # proving it read the directive from state at the start of its step.
        assert "EMPHASIZE_XYZ" in agent1_output

    def test_directive_absent_from_a_prior_node(self, echo_models):
        # A directive injected after agent_0 must NOT appear in agent_0's own
        # output — it is observed by the NEXT node, not retroactively.
        config = _echo_config(n_agents=2)
        executor = MASExecutor(config, enable_steering=True)
        executor.initialize()

        gen = executor.run_streaming_steerable(
            {"messages": [HumanMessage(content="start")]}, thread_id="t1b"
        )
        updates = []
        for node, upd in gen:
            updates.append((node, upd))
            if node == "agent_0":
                executor.submit_steer("LATE_DIRECTIVE")
                break
        updates.extend(gen)

        by_node = {n: u for (n, u) in updates if n in _AGENT_IDS}
        agent0_output = " ".join(str(m.content) for m in by_node["agent_0"]["messages"])
        assert "LATE_DIRECTIVE" not in agent0_output

    def test_steer_method_injects_and_is_observed(self, echo_models):
        # The explicit steer(message, thread_id) path: drive a base streamed
        # run to its pause, then inject one directive and resume.
        config = _echo_config(n_agents=2)
        executor = MASExecutor(config, enable_steering=True)
        executor.initialize()

        first = list(
            executor.run_streaming(
                {"messages": [HumanMessage(content="start")]}, thread_id="t2"
            )
        )
        nodes = [n for (n, _) in first]
        # interrupt_after pauses the run after agent_0; agent_1 has not run.
        assert "agent_0" in nodes
        assert "agent_1" not in nodes

        resumed = list(executor.steer("REDIRECT_ABC", thread_id="t2"))
        by_node = {n: u for (n, u) in resumed if n in _AGENT_IDS}
        assert "agent_1" in by_node
        agent1_output = " ".join(str(m.content) for m in by_node["agent_1"]["messages"])
        assert "REDIRECT_ABC" in agent1_output


# =========================================================================
# Additivity: an unused steer channel changes nothing
# =========================================================================


class TestSteeringIsAdditive:
    """Steering, unused, leaves existing behavior unchanged."""

    def test_steering_disabled_adds_no_interrupt_points(self):
        config = _stub_config(n_agents=2)
        executor = MASExecutor(config)  # enable_steering defaults to False
        executor.initialize()

        # Compilation is byte-for-byte the non-steering path: no interrupt
        # points, and no directive queue was allocated.
        assert not (
            getattr(executor._compiled_graph, "interrupt_after_nodes", []) or []
        )
        assert executor._steer_queue is None

    def test_steering_disabled_run_is_unchanged(self):
        # A regression pin on the existing sequential-stub behavior.
        config = _stub_config(n_agents=2)
        executor = MASExecutor(config)
        executor.initialize()
        result = executor.run(save_results=False)

        assert result.success
        assert len(result.agent_results) == 2
        for agent_result in result.agent_results:
            assert agent_result.output.get("status") == "stub"

    def test_enabled_but_empty_queue_matches_plain_run(self, echo_models):
        # An enabled steerable run with nothing queued yields the same agent
        # nodes, in the same order, seeing the same messages, as a plain run.
        config = _echo_config(n_agents=3)

        plain_exec = MASExecutor(config)
        plain_exec.initialize()
        plain = list(
            plain_exec.run_streaming(
                {"messages": [HumanMessage(content="start")]}, thread_id="plain"
            )
        )

        steer_exec = MASExecutor(config, enable_steering=True)
        steer_exec.initialize()
        steerable = list(
            steer_exec.run_streaming_steerable(
                {"messages": [HumanMessage(content="start")]}, thread_id="steer"
            )
        )

        assert _agent_updates(plain) == _agent_updates(steerable)

    def test_steering_enables_interrupt_after_and_forces_checkpointer(self):
        config = _stub_config(n_agents=2)
        executor = MASExecutor(config, enable_steering=True)
        executor.initialize()

        after = set(
            getattr(executor._compiled_graph, "interrupt_after_nodes", []) or []
        )
        assert {"agent_0", "agent_1"}.issubset(after)
        # Steering requires the update_state + resume seam, so a checkpointer
        # is attached even though checkpoint_enabled=False.
        assert executor._checkpointer is not None


# =========================================================================
# Guards
# =========================================================================


class TestSteeringGuards:
    """Clear errors when steering is used incorrectly."""

    def test_submit_steer_without_steering_raises(self):
        executor = MASExecutor(_stub_config(n_agents=1))
        with pytest.raises(RuntimeError, match="Steering is not enabled"):
            executor.submit_steer("directive")

    def test_run_streaming_steerable_without_steering_raises(self):
        executor = MASExecutor(_stub_config(n_agents=1))
        executor.initialize()
        with pytest.raises(RuntimeError, match="Steering is not enabled"):
            list(executor.run_streaming_steerable())

    def test_steer_before_initialize_raises(self):
        executor = MASExecutor(_stub_config(n_agents=1), enable_steering=True)
        with pytest.raises(RuntimeError, match="not initialized"):
            list(executor.steer("directive", thread_id="t"))

    def test_run_streaming_steerable_before_initialize_raises(self):
        executor = MASExecutor(_stub_config(n_agents=1), enable_steering=True)
        with pytest.raises(RuntimeError, match="not initialized"):
            list(executor.run_streaming_steerable())
