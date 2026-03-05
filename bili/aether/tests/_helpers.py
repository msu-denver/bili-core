"""Shared utilities and constants for AETHER test runners.

Centralises the three helper functions that were previously copy-pasted
across ``run_baseline.py`` and ``run_injection_suite.py``:

- :func:`find_repo_root` — locates the repository root by walking up until
  a ``.git`` directory is found.
- :func:`yaml_hash` — produces a short SHA-256 fingerprint of a YAML config
  file for reproducibility anchoring.
- :func:`config_fingerprint` — builds the full reproducibility dict embedded
  in every result JSON.

All functions take explicit parameters (no hidden module-level state) so
they can be called from any runner regardless of working directory.
"""

import hashlib
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: MAS config YAML paths used by both the injection and jailbreak suites.
#: Paths are relative to the repo root.
CONFIG_PATHS: list[str] = [
    "bili/aether/config/examples/simple_chain.yaml",
    "bili/aether/config/examples/hierarchical_voting.yaml",
    "bili/aether/config/examples/supervisor_moderation.yaml",
    "bili/aether/config/examples/consensus_network.yaml",
    "bili/aether/config/examples/custom_escalation.yaml",
]


def find_repo_root() -> Path:
    """Walk up from this file until a ``.git`` directory is found.

    Returns:
        Absolute path to the repository root.

    Raises:
        RuntimeError: If no ``.git`` directory is found before the filesystem
            root (e.g. the project is not inside a git repository).
    """
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / ".git").is_dir():
            return p
        p = p.parent
    raise RuntimeError(
        "Could not locate repo root: no .git directory found above "
        f"{Path(__file__).resolve()}"
    )


def yaml_hash(repo_root: Path, yaml_path: str) -> str:
    """Return a 12-char SHA-256 hex digest of the YAML file content.

    Args:
        repo_root: Absolute path to the repository root.
        yaml_path: Path to the YAML file, relative to *repo_root*.

    Returns:
        First 12 hex characters of the file's SHA-256 digest.
    """
    return hashlib.sha256((repo_root / yaml_path).read_bytes()).hexdigest()[:12]


def config_fingerprint(config, yaml_path: str, repo_root: Path) -> dict:
    """Build the reproducibility anchor embedded in every result file.

    Includes ``config_path`` and ``config_name`` for MAS visualiser
    cross-reference (Tasks 15.7 / 15.8).

    ``model_name`` is a sorted, comma-joined string of unique model names
    across all agents (``"stub"`` when ``model_name`` is ``None``).

    ``temperature`` is a per-agent dict because agents routinely differ
    (e.g. judge at 0.0, others at 0.2).

    Args:
        config:     A loaded ``MASConfig`` instance.
        yaml_path:  Path to the YAML file, relative to *repo_root*.
        repo_root:  Absolute path to the repository root.

    Returns:
        A dict with keys: ``yaml_hash``, ``config_path``, ``config_name``,
        ``model_name``, ``temperature``.
    """
    model_names = sorted(
        {a.model_name if a.model_name else "stub" for a in config.agents}
    )
    temps = {
        a.agent_id: a.temperature if a.temperature is not None else 0.0
        for a in config.agents
    }
    return {
        "yaml_hash": yaml_hash(repo_root, yaml_path),
        "config_path": yaml_path,
        "config_name": config.mas_id,
        "model_name": ",".join(model_names),
        "temperature": temps,
    }
