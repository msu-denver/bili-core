"""
YAML and dict loaders for MAS configurations.

Provides functions to load a complete MASConfig from a YAML file or a
Python dictionary.  Pydantic handles all validation and enum coercion
(enums use the ``(str, Enum)`` pattern, so string values like
``"content_reviewer"`` are accepted directly).

Usage::

    from bili.aether.config import load_mas_from_yaml, load_mas_from_dict

    # From YAML
    config = load_mas_from_yaml("path/to/config.yaml")

    # From dict
    config = load_mas_from_dict({
        "mas_id": "my_mas",
        "name": "My MAS",
        "workflow_type": "sequential",
        "agents": [...]
    })
"""

import os
from pathlib import Path
from typing import Union

import yaml

from bili.aether.schema import MASConfig


def load_mas_from_yaml(path: Union[str, Path]) -> MASConfig:
    """Load a MASConfig from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated ``MASConfig`` instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If the file contains invalid YAML syntax.
        pydantic.ValidationError: If the parsed data fails schema validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MAS config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping at top level in {path}, "
            f"got {type(data).__name__}"
        )

    return load_mas_from_dict(data)


def load_mas_from_dict(data: dict) -> MASConfig:
    """Load a MASConfig from a Python dictionary.

    The dictionary structure must match the ``MASConfig`` schema.
    Enum fields accept lowercase string values (e.g. ``"sequential"``,
    ``"content_reviewer"``).  Nested ``agents``, ``channels``, and
    ``workflow_edges`` lists are parsed into their respective Pydantic models
    automatically.

    Args:
        data: Dictionary matching the MASConfig schema.

    Returns:
        A validated ``MASConfig`` instance.

    Raises:
        pydantic.ValidationError: If the data fails schema validation.
    """
    return MASConfig(**data)


# =============================================================================
# CLI smoke-test: load all example YAML configs
# =============================================================================

if __name__ == "__main__":
    _examples_dir = os.path.join(os.path.dirname(__file__), "examples")

    _example_files = [
        "simple_chain.yaml",
        "hierarchical_voting.yaml",
        "supervisor_moderation.yaml",
        "consensus_network.yaml",
        "custom_escalation.yaml",
    ]

    for fname in _example_files:
        fpath = os.path.join(_examples_dir, fname)
        if not os.path.exists(fpath):
            print(f"SKIP  {fname}  (file not found)")
            continue
        cfg = load_mas_from_yaml(fpath)
        print(f"OK    {fname}  →  {cfg}")
