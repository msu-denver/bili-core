"""Tests for always-on AETHER checkpointing (PR #300).

Covers the behavioral matrix added when MASExecutor.initialize() was changed
to attach a checkpointer regardless of whether a user_id is present:

    checkpoint_enabled=True  + user_id → full checkpointer (existing behaviour)
    checkpoint_enabled=True  + no user_id → JSONL or memory fallback (NEW)
    checkpoint_enabled=False + user_id → no checkpointer (existing behaviour)
    checkpoint_enabled=False + no user_id → no checkpointer (existing behaviour)
    human_in_loop + checkpoint_enabled=False → MemorySaver override (existing)

Also tests audit_view() over the MASExecutor + JSONLCheckpointSaver stack.
"""

# pylint: disable=missing-function-docstring, protected-access

from unittest.mock import MagicMock

import pytest

from bili.aether.runtime.audit import audit_view
from bili.aether.runtime.executor import MASExecutor
from bili.aether.schema import AgentSpec, MASConfig, WorkflowType
from bili.iris.checkpointers.jsonl_checkpointer import JSONLCheckpointSaver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(agent_id: str, **kwargs) -> AgentSpec:
    defaults = {"role": "test_role", "objective": f"Objective for {agent_id}"}
    defaults.update(kwargs)
    return AgentSpec(agent_id=agent_id, **defaults)


def _seq_config(
    mas_id: str = "test_ck",
    n_agents: int = 2,
    checkpoint_enabled: bool = True,
    checkpoint_config: dict | None = None,
    **kwargs,
) -> MASConfig:
    agents = [_agent(f"agent_{i}") for i in range(n_agents)]
    cfg = {
        "mas_id": mas_id,
        "name": "Test Checkpointing MAS",
        "workflow_type": WorkflowType.SEQUENTIAL,
        "agents": agents,
        "checkpoint_enabled": checkpoint_enabled,
        "checkpoint_config": checkpoint_config or {"type": "memory"},
    }
    cfg.update(kwargs)
    return MASConfig(**cfg)


# ---------------------------------------------------------------------------
# initialize() behavioural matrix
# ---------------------------------------------------------------------------


class TestInitializeBehavioralMatrix:
    """Always-on checkpointing: checkpointer is attached when enabled."""

    def test_checkpoint_enabled_with_user_id_attaches_checkpointer(self, tmp_path):
        """checkpoint_enabled=True + user_id → _checkpointer is not None."""
        config = _seq_config(checkpoint_enabled=True)
        executor = MASExecutor(config, user_id="alice@test.com")
        executor.initialize()
        assert executor._checkpointer is not None

    def test_checkpoint_enabled_without_user_id_attaches_checkpointer(
        self, tmp_path, monkeypatch
    ):
        """checkpoint_enabled=True + no user_id → _checkpointer is not None (NEW)."""
        config = _seq_config(
            checkpoint_enabled=True,
            checkpoint_config={"type": "memory"},
        )
        executor = MASExecutor(config, user_id=None)
        executor.initialize()
        assert executor._checkpointer is not None

    def test_checkpoint_disabled_with_user_id_no_checkpointer(self):
        """checkpoint_enabled=False + user_id → _checkpointer remains None."""
        config = _seq_config(checkpoint_enabled=False)
        executor = MASExecutor(config, user_id="alice@test.com")
        executor.initialize()
        assert executor._checkpointer is None

    def test_checkpoint_disabled_without_user_id_no_checkpointer(self):
        """checkpoint_enabled=False + no user_id → _checkpointer remains None."""
        config = _seq_config(checkpoint_enabled=False)
        executor = MASExecutor(config, user_id=None)
        executor.initialize()
        assert executor._checkpointer is None

    def test_human_in_loop_always_gets_checkpointer(self):
        """human_in_loop=True with is_human agents attaches MemorySaver even when disabled."""
        # HITL override requires at least one agent with is_human=True so that
        # human_nodes is non-empty inside initialize().
        human_agent = AgentSpec(
            agent_id="human_reviewer",
            role="human",
            objective="Review agent output before proceeding",
            is_human=True,
        )
        ai_agent = AgentSpec(
            agent_id="agent_0", role="test_role", objective="Produce output for review"
        )
        config = MASConfig(
            mas_id="hitl_test",
            name="HITL Test",
            workflow_type=WorkflowType.SEQUENTIAL,
            agents=[ai_agent, human_agent],
            checkpoint_enabled=False,
            checkpoint_config={"type": "memory"},
            human_in_loop=True,
            human_escalation_condition="False",
        )
        executor = MASExecutor(config, user_id=None)
        executor.initialize()
        assert executor._checkpointer is not None

    def test_jsonl_backend_attached_via_config(self, tmp_path):
        """checkpoint_config type='jsonl' → JSONLCheckpointSaver is attached."""
        path = str(tmp_path / "test.jsonl")
        config = _seq_config(
            checkpoint_enabled=True,
            checkpoint_config={"type": "jsonl", "path": path},
        )
        executor = MASExecutor(config, user_id=None)
        executor.initialize()
        assert isinstance(executor._checkpointer, JSONLCheckpointSaver)
        assert executor._checkpointer.path == path

    def test_file_alias_backend_attached_via_config(self, tmp_path):
        """checkpoint_config type='file' → JSONLCheckpointSaver is attached."""
        path = str(tmp_path / "test.jsonl")
        config = _seq_config(
            checkpoint_enabled=True,
            checkpoint_config={"type": "file", "path": path},
        )
        executor = MASExecutor(config, user_id=None)
        executor.initialize()
        assert isinstance(executor._checkpointer, JSONLCheckpointSaver)

    def test_initialize_is_idempotent(self):
        """Calling initialize() twice does not raise."""
        config = _seq_config(checkpoint_enabled=True)
        executor = MASExecutor(config)
        executor.initialize()
        executor.initialize()
        assert executor._checkpointer is not None


# ---------------------------------------------------------------------------
# _create_checkpointer_local helper
# ---------------------------------------------------------------------------


class TestCreateCheckpointerLocal:
    """_create_checkpointer_local() returns a usable checkpointer for no-user-id runs."""

    def test_returns_checkpointer(self, tmp_path):
        config = _seq_config(
            checkpoint_enabled=True,
            checkpoint_config={"type": "memory"},
        )
        executor = MASExecutor(config, user_id=None)
        checkpointer = executor._create_checkpointer_local()
        assert checkpointer is not None

    def test_jsonl_path_forwarded(self, tmp_path):
        path = str(tmp_path / "local.jsonl")
        config = _seq_config(
            checkpoint_enabled=True,
            checkpoint_config={"type": "jsonl", "path": path},
        )
        executor = MASExecutor(config, user_id=None)
        checkpointer = executor._create_checkpointer_local()
        assert isinstance(checkpointer, JSONLCheckpointSaver)
        assert checkpointer.path == path


# ---------------------------------------------------------------------------
# JSONL env-var tier in get_checkpointer()
# ---------------------------------------------------------------------------


class TestCheckpointerFunctionsJSONLTier:
    """get_checkpointer() and get_async_checkpointer() honour JSONL_CHECKPOINT_PATH."""

    def test_jsonl_tier_returns_jsonl_saver(self, tmp_path, monkeypatch):
        """With JSONL_CHECKPOINT_PATH set and no PG/Mongo, get_checkpointer() returns JSONL."""
        from bili.iris.checkpointers.checkpointer_functions import get_checkpointer

        path = str(tmp_path / "env.jsonl")
        monkeypatch.setenv("JSONL_CHECKPOINT_PATH", path)
        monkeypatch.delenv("POSTGRES_CONNECTION_STRING", raising=False)
        monkeypatch.delenv("MONGODB_URI", raising=False)

        saver = get_checkpointer()
        assert isinstance(saver, JSONLCheckpointSaver)
        assert saver.path == path

    def test_jsonl_tier_async(self, tmp_path, monkeypatch):
        """get_async_checkpointer() also returns JSONLCheckpointSaver when env var set."""
        import asyncio

        from bili.iris.checkpointers.checkpointer_functions import (
            get_async_checkpointer,
        )

        path = str(tmp_path / "env_async.jsonl")
        monkeypatch.setenv("JSONL_CHECKPOINT_PATH", path)
        monkeypatch.delenv("POSTGRES_CONNECTION_STRING", raising=False)
        monkeypatch.delenv("MONGODB_URI", raising=False)

        saver = asyncio.run(get_async_checkpointer())
        assert isinstance(saver, JSONLCheckpointSaver)


# ---------------------------------------------------------------------------
# AETHER checkpointer_factory jsonl / file aliases
# ---------------------------------------------------------------------------


class TestCheckpointerFactoryAliases:
    """create_checkpointer_from_config handles 'jsonl' and 'file' aliases."""

    def test_jsonl_alias(self, tmp_path):
        from bili.aether.integration.checkpointer_factory import (
            create_checkpointer_from_config,
        )

        path = str(tmp_path / "alias.jsonl")
        saver = create_checkpointer_from_config({"type": "jsonl", "path": path})
        assert isinstance(saver, JSONLCheckpointSaver)

    def test_file_alias(self, tmp_path):
        from bili.aether.integration.checkpointer_factory import (
            create_checkpointer_from_config,
        )

        path = str(tmp_path / "alias_file.jsonl")
        saver = create_checkpointer_from_config({"type": "file", "path": path})
        assert isinstance(saver, JSONLCheckpointSaver)

    def test_keep_last_n_forwarded(self, tmp_path):
        from bili.aether.integration.checkpointer_factory import (
            create_checkpointer_from_config,
        )

        path = str(tmp_path / "prune.jsonl")
        saver = create_checkpointer_from_config(
            {"type": "jsonl", "path": path, "keep_last_n": 5}
        )
        assert saver.keep_last_n == 5

    def test_user_id_forwarded(self, tmp_path):
        from bili.aether.integration.checkpointer_factory import (
            create_checkpointer_from_config,
        )

        path = str(tmp_path / "uid.jsonl")
        saver = create_checkpointer_from_config(
            {"type": "jsonl", "path": path}, user_id="alice@test.com"
        )
        assert saver.user_id == "alice@test.com"


# ---------------------------------------------------------------------------
# audit_view() tests
# ---------------------------------------------------------------------------


def _mk_checkpoint(cp_id, agent_id=None, msg=None, ts=None):
    """Build a minimal checkpoint dict for audit_view tests."""
    agent_outputs = (
        {agent_id: {"message": msg or f"output from {agent_id}"}} if agent_id else {}
    )
    return {
        "id": cp_id,
        "ts": ts or "2024-01-01T00:00:00+00:00",
        "v": 1,
        "channel_values": {
            "messages": [],
            "agent_outputs": agent_outputs,
            "communication_log": [],
        },
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }


def _mk_config(thread_id, cp_id=None, ns=""):
    cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns}}
    if cp_id:
        cfg["configurable"]["checkpoint_id"] = cp_id
    return cfg


def _mk_metadata(ts=None):
    return {"source": "test", "step": 0, "writes": {}, "ts": ts}


class TestAuditView:
    """audit_view() builds a readable timeline from checkpoint history."""

    def test_empty_thread_returns_empty_list(self, tmp_path):
        """audit_view on a thread with no checkpoints returns []."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "audit.jsonl"))
        result = audit_view(saver, thread_id="nonexistent")
        assert result == []

    def test_single_superstep_timeline(self, tmp_path):
        """A single checkpoint produces a one-entry timeline."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "audit.jsonl"))
        saver.put(
            _mk_config("run-1"),
            _mk_checkpoint("cp-1", agent_id="agent_0"),
            _mk_metadata("2024-01-01T00:00:00+00:00"),
            {},
        )
        timeline = audit_view(saver, thread_id="run-1")
        assert len(timeline) == 1
        assert timeline[0]["step"] == 1
        assert timeline[0]["agent_id"] == "agent_0"

    def test_output_summary_truncated_to_200_chars(self, tmp_path):
        """output_summary is at most 200 characters."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "audit.jsonl"))
        long_output = "x" * 500
        ck = _mk_checkpoint("cp-1")
        ck["channel_values"]["agent_outputs"] = {"agent_0": {"message": long_output}}
        saver.put(_mk_config("run-1"), ck, _mk_metadata(), {})
        timeline = audit_view(saver, thread_id="run-1")
        assert timeline[0]["output_summary"] is not None
        assert len(timeline[0]["output_summary"]) <= 200

    def test_two_supersteps_delta(self, tmp_path):
        """Second step shows only the newly changed agent_outputs delta."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "audit.jsonl"))
        # Step 1: agent_0 acts
        ck1 = _mk_checkpoint("cp-1", agent_id="agent_0", msg="first")
        saver.put(_mk_config("run-1"), ck1, _mk_metadata(), {})

        # Step 2: agent_1 acts (agent_0 output unchanged)
        ck2 = _mk_checkpoint("cp-1")
        ck2["id"] = "cp-2"
        ck2["channel_values"]["agent_outputs"] = {
            "agent_0": {"message": "first"},  # unchanged
            "agent_1": {"message": "second"},  # new
        }
        saver.put(_mk_config("run-1", "cp-1"), ck2, _mk_metadata(), {})

        timeline = audit_view(saver, thread_id="run-1")
        assert len(timeline) == 2
        # Step 2 only shows agent_1 in raw_agent_outputs
        step2 = timeline[1]
        assert "agent_1" in step2["raw_agent_outputs"]
        assert "agent_0" not in step2["raw_agent_outputs"]

    def test_communication_log_diff(self, tmp_path):
        """messages_sent shows only newly appended communication_log entries."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "audit.jsonl"))
        ck1 = _mk_checkpoint("cp-1")
        ck1["channel_values"]["communication_log"] = [{"from": "agent_0", "msg": "A"}]
        saver.put(_mk_config("run-1"), ck1, _mk_metadata(), {})

        ck2 = _mk_checkpoint("cp-1")
        ck2["id"] = "cp-2"
        ck2["channel_values"]["communication_log"] = [
            {"from": "agent_0", "msg": "A"},
            {"from": "agent_1", "msg": "B"},
        ]
        saver.put(_mk_config("run-1", "cp-1"), ck2, _mk_metadata(), {})

        timeline = audit_view(saver, thread_id="run-1")
        step1 = timeline[0]
        step2 = timeline[1]
        assert len(step1["messages_sent"]) == 1
        assert step1["messages_sent"][0]["msg"] == "A"
        assert len(step2["messages_sent"]) == 1
        assert step2["messages_sent"][0]["msg"] == "B"

    def test_chronological_order(self, tmp_path):
        """Timeline is chronological (oldest first)."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "audit.jsonl"))
        for i in range(4):
            # Use distinct agent IDs so each checkpoint has new agent activity
            # and is not filtered by the no-activity guard in audit_view.
            ck = _mk_checkpoint(f"cp-{i}", agent_id=f"agent_{i}")
            saver.put(_mk_config("run-1"), ck, _mk_metadata(), {})
        timeline = audit_view(saver, thread_id="run-1")
        steps = [e["step"] for e in timeline]
        assert steps == sorted(steps)
        assert steps[0] == 1

    def test_checkpoint_id_in_timeline_entry(self, tmp_path):
        """Each timeline entry includes checkpoint_id."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "audit.jsonl"))
        # Include agent activity so the entry is not filtered by the
        # no-activity guard in audit_view.
        saver.put(
            _mk_config("run-1"),
            _mk_checkpoint("cp-sentinel", agent_id="agent_sentinel"),
            _mk_metadata(),
            {},
        )
        timeline = audit_view(saver, thread_id="run-1")
        # checkpoint_id comes from config configurable, not checkpoint.id
        assert timeline[0]["checkpoint_id"] is not None

    def test_output_summary_uses_content_key(self, tmp_path):
        """output_summary falls back to 'content' key when 'message' is absent."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "audit.jsonl"))
        ck = _mk_checkpoint("cp-1")
        ck["channel_values"]["agent_outputs"] = {"agent_0": {"content": "response"}}
        saver.put(_mk_config("run-1"), ck, _mk_metadata(), {})
        timeline = audit_view(saver, thread_id="run-1")
        assert timeline[0]["output_summary"] == "response"

    def test_output_summary_non_dict_agent_output(self, tmp_path):
        """output_summary handles a raw string agent output."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "audit.jsonl"))
        ck = _mk_checkpoint("cp-1")
        ck["channel_values"]["agent_outputs"] = {"agent_0": "plain string output"}
        saver.put(_mk_config("run-1"), ck, _mk_metadata(), {})
        timeline = audit_view(saver, thread_id="run-1")
        assert timeline[0]["output_summary"] == "plain string output"

    def test_ts_extracted_from_checkpoint(self, tmp_path):
        """ts is extracted from the checkpoint dict when not in metadata."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "audit.jsonl"))
        # Include agent activity so the entry is not filtered by the
        # no-activity guard in audit_view.
        ck = _mk_checkpoint("cp-1", agent_id="agent_ts")
        ck["ts"] = "2024-06-01T12:00:00+00:00"
        saver.put(
            _mk_config("run-1"), ck, {"source": "test", "step": 0, "writes": {}}, {}
        )
        timeline = audit_view(saver, thread_id="run-1")
        assert timeline[0]["ts"] == "2024-06-01T12:00:00+00:00"

    def test_message_serialization_to_log_dict(self):
        """messages_sent: items with to_log_dict() are serialized via that method.

        Uses a mock saver to bypass JSONL encoding; the to_log_dict path is
        exercised in the audit_view serialization loop.
        """
        from langgraph.checkpoint.base import CheckpointTuple

        msg = MagicMock()
        msg.to_log_dict.return_value = {"from": "agent_0", "msg": "via_to_log_dict"}

        ck = _mk_checkpoint("cp-1")
        ck["channel_values"]["communication_log"] = [msg]

        tup = CheckpointTuple(
            config={"configurable": {"thread_id": "run-1", "checkpoint_id": "cp-1"}},
            checkpoint=ck,
            metadata=_mk_metadata(),
            parent_config=None,
            pending_writes=[],
        )
        mock_saver = MagicMock()
        mock_saver.list.return_value = [tup]

        timeline = audit_view(mock_saver, thread_id="run-1")
        assert len(timeline) == 1
        assert timeline[0]["messages_sent"][0]["msg"] == "via_to_log_dict"

    def test_audit_view_raises_on_checkpointer_error(self, tmp_path):
        """audit_view propagates errors from list()."""
        saver = MagicMock()
        saver.list.side_effect = RuntimeError("list failed")
        with pytest.raises(RuntimeError, match="list failed"):
            audit_view(saver, thread_id="run-1")

    def test_messages_sent_dict_passthrough(self, tmp_path):
        """communication_log entries that are already dicts are passed through unchanged."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "audit.jsonl"))
        ck = _mk_checkpoint("cp-1")
        ck["channel_values"]["communication_log"] = [
            {"from": "a", "to": "b", "text": "hi"}
        ]
        saver.put(_mk_config("run-1"), ck, _mk_metadata(), {})
        timeline = audit_view(saver, thread_id="run-1")
        assert timeline[0]["messages_sent"][0]["text"] == "hi"

    def test_messages_sent_raw_fallback(self, tmp_path):
        """communication_log entries with no to_log_dict / __dict__ are str-wrapped."""
        saver = JSONLCheckpointSaver(path=str(tmp_path / "audit.jsonl"))
        ck = _mk_checkpoint("cp-1")
        # A simple int has neither to_log_dict nor a useful __dict__
        ck["channel_values"]["communication_log"] = [42]
        saver.put(_mk_config("run-1"), ck, _mk_metadata(), {})
        timeline = audit_view(saver, thread_id="run-1")
        # Serialization to JSONL converts int to encoded blob; timeline still has entry
        assert isinstance(timeline[0]["messages_sent"], list)


# ---------------------------------------------------------------------------
# audit_view with mock saver (tests that bypass JSONL encode/decode path)
# ---------------------------------------------------------------------------


class TestAuditViewMockSaver:
    """audit_view() called directly on a mock saver that yields CheckpointTuples."""

    def _make_tuple(self, cp_id, thread_id, agent_id=None, comm_entries=None):
        """Build a minimal CheckpointTuple-compatible object."""
        from langgraph.checkpoint.base import CheckpointTuple  # type: ignore

        agent_outputs = {agent_id: {"message": f"out-{agent_id}"}} if agent_id else {}
        checkpoint = {
            "id": cp_id,
            "ts": "2024-01-01T00:00:00+00:00",
            "v": 1,
            "channel_values": {
                "messages": [],
                "agent_outputs": agent_outputs,
                "communication_log": comm_entries or [],
            },
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        config = {"configurable": {"thread_id": thread_id, "checkpoint_id": cp_id}}
        return CheckpointTuple(
            config=config,
            checkpoint=checkpoint,
            metadata={"source": "test", "step": 0, "writes": {}},
            parent_config=None,
            pending_writes=[],
        )

    def test_current_agent_from_channel_values(self):
        """acting_agent is read from channel_values.current_agent when present."""
        mock_saver = MagicMock()
        tup = self._make_tuple("cp-1", "t1", agent_id="agent_0")
        # Inject current_agent explicitly
        tup.checkpoint["channel_values"]["current_agent"] = "agent_explicit"
        mock_saver.list.return_value = [tup]

        timeline = audit_view(mock_saver, thread_id="t1")
        assert timeline[0]["agent_id"] == "agent_explicit"

    def test_two_supersteps_chronological(self):
        """list() output (most-recent-first) is reversed to chronological in timeline."""
        mock_saver = MagicMock()
        tup1 = self._make_tuple("cp-1", "t1", agent_id="agent_0")
        tup2 = self._make_tuple("cp-2", "t1", agent_id="agent_1")
        # list() returns most-recent first (cp-2, cp-1)
        mock_saver.list.return_value = [tup2, tup1]

        timeline = audit_view(mock_saver, thread_id="t1")
        assert len(timeline) == 2
        assert timeline[0]["step"] == 1
        assert timeline[1]["step"] == 2

    def test_communication_log_hasattr_dict_fallback(self):
        """comm_log entries with __dict__ are serialized via vars()."""
        mock_saver = MagicMock()

        class _Msg:
            def __init__(self):
                self.from_agent = "a"
                self.content = "hi"

        tup = self._make_tuple("cp-1", "t1", comm_entries=[_Msg()])
        mock_saver.list.return_value = [tup]

        timeline = audit_view(mock_saver, thread_id="t1")
        assert timeline[0]["messages_sent"][0]["from_agent"] == "a"
