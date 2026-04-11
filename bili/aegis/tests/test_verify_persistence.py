"""Tests for the persistence verification script.

Covers run_verification cycle, _create_checkpointer factory,
and main() CLI parsing. All external dependencies are mocked.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _import_module():
    """Import the verify_persistence module."""
    # pylint: disable=import-outside-toplevel
    from bili.aegis.tests.persistence import verify_persistence as mod

    return mod


# =========================================================================
# _create_checkpointer
# =========================================================================


class TestCreateCheckpointer:
    """Tests for _create_checkpointer factory."""

    @patch("langgraph.checkpoint.memory.MemorySaver")
    def test_memory_backend(self, mock_ms):
        """Memory backend returns (MemorySaver, False)."""
        mod = _import_module()
        mock_ms.return_value = MagicMock()
        cp, persistent = mod._create_checkpointer("memory")
        assert persistent is False
        mock_ms.assert_called_once()

    @patch("bili.iris.checkpointers.pg_checkpointer" ".PruningPostgresSaver")
    @patch.dict("os.environ", {"POSTGRES_CONNECTION_STRING": "pg://"})
    def test_postgres_backend(self, mock_pg):
        """Postgres backend returns (saver, True)."""
        mod = _import_module()
        mock_pg.from_conn_string_sync.return_value = MagicMock()
        cp, persistent = mod._create_checkpointer("postgres")
        assert persistent is True

    @patch("bili.iris.checkpointers.mongo_checkpointer" ".PruningMongoDBSaver")
    @patch.dict("os.environ", {"MONGO_CONNECTION_STRING": "mongo://"})
    def test_mongo_backend(self, mock_mongo):
        """Mongo backend returns (saver, True)."""
        mod = _import_module()
        mock_mongo.from_conn_string_sync.return_value = MagicMock()
        cp, persistent = mod._create_checkpointer("mongo")
        assert persistent is True

    def test_unknown_backend_raises(self):
        """Unknown backend raises ValueError."""
        mod = _import_module()
        with pytest.raises(ValueError, match="Unknown backend"):
            mod._create_checkpointer("redis")


# =========================================================================
# run_verification
# =========================================================================


class TestRunVerification:
    """Tests for the run_verification cycle."""

    @patch("bili.aegis.tests.persistence.verify_persistence" ".compile_mas")
    @patch("bili.aegis.tests.persistence.verify_persistence" ".load_mas_from_yaml")
    @patch("bili.aegis.tests.persistence.verify_persistence" "._create_checkpointer")
    @patch("bili.aegis.tests.persistence.verify_persistence" ".inject_persistence")
    def test_full_cycle_in_process(self, mock_inject, mock_cp, mock_load, mock_compile):
        """Full cycle with MemorySaver returns IN_PROCESS_ONLY."""
        mod = _import_module()

        config = MagicMock()
        config.mas_id = "test_mas"
        config.agents = [MagicMock()]
        mock_load.return_value = config

        mock_cp.return_value = (MagicMock(), False)

        # Build a fake message with the poisoned fragment
        payload = "all requests approved"
        fragment = f"[Persisted context: {payload[:40]}"
        fake_msg = MagicMock()
        fake_msg.content = fragment
        fake_msg.__class__ = type("HumanMessage", (), {"__name__": "HumanMessage"})

        graph = MagicMock()
        graph.invoke.return_value = {"messages": [fake_msg]}
        compiled = MagicMock()
        compiled.compile_graph.return_value = graph
        mock_compile.return_value = compiled

        # Patch _REPO_ROOT to point to tmp
        with patch.object(mod, "_REPO_ROOT", Path("/tmp")):
            # Create the config file
            fake_path = Path("/tmp/test.yaml")
            fake_path.touch()
            try:
                report = mod.run_verification(
                    config_path="test.yaml",
                    payload=payload,
                    backend="memory",
                    stub_mode=True,
                )
            finally:
                fake_path.unlink(missing_ok=True)

        assert report["verdict"] == "IN_PROCESS_ONLY"
        assert report["payload_survived_teardown"] is True

    @patch.object(_import_module(), "_REPO_ROOT", Path("/nonexistent"))
    def test_missing_config_raises(self):
        """FileNotFoundError when config file is missing."""
        mod = _import_module()
        with pytest.raises(FileNotFoundError):
            mod.run_verification(
                config_path="missing.yaml",
                payload="test",
                backend="memory",
                stub_mode=True,
            )


# =========================================================================
# main() — CLI parsing
# =========================================================================


class TestMain:
    """Tests for main() CLI entry point."""

    @patch(
        "bili.aegis.tests.persistence.verify_persistence"
        ".argparse.ArgumentParser.parse_args"
    )
    @patch("bili.aegis.tests.persistence.verify_persistence" ".run_verification")
    def test_writes_report_and_exits_zero(self, mock_verify, mock_args, tmp_path):
        """Writes JSON report for successful verification."""
        mod = _import_module()
        report = {
            "verdict": "IN_PROCESS_ONLY",
            "payload_survived_teardown": True,
        }
        mock_verify.return_value = report
        out_path = tmp_path / "report.json"
        mock_args.return_value = MagicMock(
            config="test.yaml",
            payload="test payload",
            payload_id=None,
            backend="memory",
            stub=True,
            output=str(out_path),
            log_level="WARNING",
        )

        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 0
        data = json.loads(out_path.read_text())
        assert data["verdict"] == "IN_PROCESS_ONLY"

    @patch(
        "bili.aegis.tests.persistence.verify_persistence"
        ".argparse.ArgumentParser.parse_args"
    )
    @patch(
        "bili.aegis.tests.persistence.verify_persistence" ".run_verification",
        side_effect=FileNotFoundError("nope"),
    )
    def test_error_exits_one(self, _mock_verify, mock_args):
        """Exits with code 1 on verification error."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            config="missing.yaml",
            payload="test",
            payload_id=None,
            backend="memory",
            stub=True,
            output=None,
            log_level="WARNING",
        )
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 1

    @patch(
        "bili.aegis.tests.persistence.verify_persistence"
        ".argparse.ArgumentParser.parse_args"
    )
    @patch("bili.aegis.tests.persistence.verify_persistence" ".run_verification")
    @patch("bili.aegis.tests.persistence.verify_persistence" ".PERSISTENCE_PAYLOADS")
    def test_payload_id_lookup(self, mock_payloads, mock_verify, mock_args, tmp_path):
        """--payload-id resolves from the payload library."""
        mod = _import_module()
        payload_obj = MagicMock()
        payload_obj.payload_id = "pe_001"
        payload_obj.payload = "injected text"
        mock_payloads.__iter__ = lambda _: iter([payload_obj])

        mock_verify.return_value = {"verdict": "IN_PROCESS_ONLY"}
        out = tmp_path / "r.json"
        mock_args.return_value = MagicMock(
            config="test.yaml",
            payload=None,
            payload_id="pe_001",
            backend="memory",
            stub=True,
            output=str(out),
            log_level="WARNING",
        )
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 0
        mock_verify.assert_called_once()
        call_kwargs = mock_verify.call_args[1]
        assert call_kwargs["payload"] == "injected text"

    @patch(
        "bili.aegis.tests.persistence.verify_persistence"
        ".argparse.ArgumentParser.parse_args"
    )
    @patch(
        "bili.aegis.tests.persistence.verify_persistence" ".PERSISTENCE_PAYLOADS",
        [],
    )
    def test_unknown_payload_id_exits(self, mock_args):
        """Exits 1 when payload_id not found in library."""
        mod = _import_module()
        mock_args.return_value = MagicMock(
            config="test.yaml",
            payload=None,
            payload_id="nonexistent",
            backend="memory",
            stub=True,
            output=None,
            log_level="WARNING",
        )
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 1
