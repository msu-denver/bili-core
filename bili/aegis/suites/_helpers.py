"""Shared utilities and constants for AETHER test runners.

Centralises the helper functions that were previously copy-pasted
across ``run_baseline.py`` and ``run_injection_suite.py``:

- :func:`find_repo_root` — locates the repository root by walking up until
  a ``.git`` directory is found.
- :func:`yaml_hash` — produces a short SHA-256 fingerprint of a YAML config
  file for reproducibility anchoring.
- :func:`config_fingerprint` — builds the full reproducibility dict embedded
  in every result JSON.
- :func:`model_id_safe` — converts a model_id string to a filesystem-safe
  directory name (used by the cross-model suite and its pytest fixtures).

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

#: Default path to the baseline results directory, relative to the repo root.
#: Used by all attack suite runners as the default value for ``--baseline-results``
#: so that Tier 3 semantic evaluation runs automatically in real mode without
#: requiring the flag to be passed explicitly.
DEFAULT_BASELINE_RESULTS_DIR: str = "bili/aegis/suites/baseline/results"


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


def model_id_safe(model_id: str | None) -> str:
    """Convert a model_id to a filesystem-safe directory name.

    Replaces characters that are illegal or awkward in directory names
    (``:`` ``.`` ``/`` ``-``) with underscores.  Returns ``"stub"`` when
    *model_id* is ``None`` (stub-mode runs).

    Args:
        model_id: Provider model identifier (e.g.
            ``"us.anthropic.claude-3-5-haiku-20241022-v1:0"``), or ``None``
            for stub runs.

    Returns:
        A lowercase alphanumeric-and-underscore string safe for use as a
        filesystem directory component.
    """
    if model_id is None:
        return "stub"
    return (
        model_id.replace(":", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace("-", "_")
        .lower()
    )


def next_run_dir(base_dir: Path) -> Path:
    """Return the next run directory path (``run_001``, ``run_002``, …) under *base_dir*.

    Creates *base_dir* if it does not exist.  Does **not** create the run
    directory itself — the caller is responsible for creating it when the
    first result is written.

    Args:
        base_dir: The ``mas_id``-level results directory.

    Returns:
        Path to the next run directory, e.g. ``base_dir / "run_003"``.

    Note:
        This uses a non-atomic read-then-create pattern. Safe for
        single-user local development but would need a file lock or
        atomic mkdir for concurrent multi-user deployments.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        d
        for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_") and d.name[4:].isdigit()
    )
    run_dir = base_dir / f"run_{len(existing) + 1:03d}"
    # Create immediately to reduce race window
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def latest_run_dir(base_dir: Path) -> "Path | None":
    """Return the most recent run directory under *base_dir*, or ``None``.

    Returns ``None`` when *base_dir* does not exist or contains no
    ``run_NNN`` subdirectories (i.e. the flat legacy structure is in use).

    Args:
        base_dir: The ``mas_id``-level results directory.

    Returns:
        Path to the most recent ``run_NNN`` directory, or ``None``.
    """
    if not base_dir.exists():
        return None
    existing = sorted(
        d
        for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_") and d.name[4:].isdigit()
    )
    return existing[-1] if existing else None
