"""Shared fixtures for the catalog-divergence tests.

The recorded dataset slices are parsed once per session: they are the whole
upstream record set for every mapped provider, so re-parsing them per test is
several seconds of nothing.
"""

from pathlib import Path

import pytest

from bili.iris.config.catalog_divergence.datasets import load_litellm, load_models_dev

FIXTURE_DIR = Path(__file__).parent / "fixtures"
MODELS_DEV_FIXTURE = FIXTURE_DIR / "models_dev_2026-09-03.json"
LITELLM_FIXTURE = FIXTURE_DIR / "litellm_2026-09-03.json"


@pytest.fixture(scope="session")
def models_dev_path() -> Path:
    """Path to the recorded models.dev slice.

    :returns: The fixture path.
    :rtype: Path
    """
    return MODELS_DEV_FIXTURE


@pytest.fixture(scope="session")
def litellm_path() -> Path:
    """Path to the recorded LiteLLM slice.

    :returns: The fixture path.
    :rtype: Path
    """
    return LITELLM_FIXTURE


@pytest.fixture(scope="session")
def models_dev_dataset():
    """The parsed models.dev slice.

    :returns: A parsed dataset.
    """
    return load_models_dev(MODELS_DEV_FIXTURE)


@pytest.fixture(scope="session")
def litellm_dataset():
    """The parsed LiteLLM slice.

    :returns: A parsed dataset.
    """
    return load_litellm(LITELLM_FIXTURE)
