"""Tests for bili.aegis.tests._helpers shared utilities."""

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

from bili.aegis.tests._helpers import (
    CONFIG_PATHS,
    config_fingerprint,
    model_id_safe,
    yaml_hash,
)


class TestConfigPaths:
    """Tests for the CONFIG_PATHS constant."""

    def test_is_non_empty_list(self):
        """CONFIG_PATHS is a non-empty list."""
        assert isinstance(CONFIG_PATHS, list)
        assert len(CONFIG_PATHS) > 0

    def test_all_entries_are_strings(self):
        """Every entry in CONFIG_PATHS is a string."""
        for path in CONFIG_PATHS:
            assert isinstance(path, str)

    def test_all_entries_end_with_yaml(self):
        """Every entry in CONFIG_PATHS ends with .yaml."""
        for path in CONFIG_PATHS:
            assert path.endswith(".yaml")


class TestYamlHash:
    """Tests for yaml_hash()."""

    def test_returns_hex_string(self):
        """yaml_hash returns a 12-character hex string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_file = root / "test.yaml"
            yaml_file.write_text("key: value\n")

            result = yaml_hash(root, "test.yaml")

            assert isinstance(result, str)
            assert len(result) == 12
            # Verify it is valid hex
            int(result, 16)

    def test_matches_manual_sha256(self):
        """yaml_hash matches a manually computed SHA-256 prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            yaml_file = root / "cfg.yaml"
            content = b"agents:\n  - id: a\n"
            yaml_file.write_bytes(content)

            expected = hashlib.sha256(content).hexdigest()[:12]
            assert yaml_hash(root, "cfg.yaml") == expected

    def test_different_content_different_hash(self):
        """Different file contents produce different hashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.yaml").write_text("content_a")
            (root / "b.yaml").write_text("content_b")

            assert yaml_hash(root, "a.yaml") != yaml_hash(root, "b.yaml")


class TestConfigFingerprint:
    """Tests for config_fingerprint()."""

    def test_returns_expected_keys(self):
        """config_fingerprint returns a dict with the required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "c.yaml").write_text("x: 1")

            config = _fake_config(
                mas_id="test-mas",
                agents=[
                    _fake_agent("a1", "gpt-4o", 0.7),
                    _fake_agent("a2", None, None),
                ],
            )

            result = config_fingerprint(config, "c.yaml", root)

            assert set(result.keys()) == {
                "yaml_hash",
                "config_path",
                "config_name",
                "model_name",
                "temperature",
            }

    def test_config_path_and_name(self):
        """config_path and config_name are set correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "f.yaml").write_text("data")

            config = _fake_config(
                mas_id="my-mas",
                agents=[_fake_agent("a", "m1", 0.5)],
            )
            result = config_fingerprint(config, "f.yaml", root)

            assert result["config_path"] == "f.yaml"
            assert result["config_name"] == "my-mas"

    def test_stub_model_name_when_none(self):
        """Agents with model_name=None produce 'stub' in the model list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "s.yaml").write_text("x")

            config = _fake_config(
                mas_id="s",
                agents=[_fake_agent("a", None, None)],
            )
            result = config_fingerprint(config, "s.yaml", root)
            assert result["model_name"] == "stub"

    def test_temperature_per_agent(self):
        """Temperature dict has one entry per agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "t.yaml").write_text("x")

            config = _fake_config(
                mas_id="t",
                agents=[
                    _fake_agent("a1", "m", 0.2),
                    _fake_agent("a2", "m", None),
                ],
            )
            result = config_fingerprint(config, "t.yaml", root)
            assert result["temperature"] == {"a1": 0.2, "a2": 0.0}


class TestModelIdSafe:
    """Tests for model_id_safe()."""

    def test_none_returns_stub(self):
        """None model_id returns 'stub'."""
        assert model_id_safe(None) == "stub"

    def test_sanitizes_special_chars(self):
        """Colons, dots, slashes, hyphens become underscores."""
        result = model_id_safe("us.anthropic.claude-3-5-haiku-20241022-v1:0")
        assert ":" not in result
        assert "." not in result
        assert "/" not in result
        assert "-" not in result

    def test_result_is_lowercase(self):
        """Output is always lowercase."""
        assert model_id_safe("GPT-4o") == "gpt_4o"

    def test_preserves_alphanumeric(self):
        """Alphanumeric characters are preserved (lowered)."""
        assert model_id_safe("abc123") == "abc123"


# =========================================================================
# Test helpers
# =========================================================================


def _fake_agent(agent_id, model_name, temperature):
    """Build a minimal stand-in for an AgentSpec."""
    return SimpleNamespace(
        agent_id=agent_id, model_name=model_name, temperature=temperature
    )


def _fake_config(mas_id, agents):
    """Build a minimal stand-in for a MASConfig."""
    return SimpleNamespace(mas_id=mas_id, agents=agents)
