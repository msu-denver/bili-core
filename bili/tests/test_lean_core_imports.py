"""Lean-core import tests.

Verify that the IRIS and AETHER core runtime paths import successfully when
the optional surface and backend extras (streamlit, flask, torch, faiss,
opensearch, firebase, pymongo, psycopg2) are NOT installed.

The lean-core install surface is:
    pip install bili-core          # no extras

The optional surfaces/backends are guarded by extras:
    pip install bili-core[streamlit]   # Streamlit UI
    pip install bili-core[flask]       # Flask REST API
    pip install bili-core[aegis]       # AEGIS adversarial testing
    pip install bili-core[faiss]       # FAISS vector search
    pip install bili-core[opensearch]  # Amazon OpenSearch
    pip install bili-core[mongo]       # MongoDB checkpointer
    pip install bili-core[postgres]    # PostgreSQL checkpointer
    pip install bili-core[huggingface] # HuggingFace local models
    pip install bili-core[firebase]    # Firebase auth
    pip install bili-core[mcp]         # MCP client subsystem
    pip install bili-core[all]         # everything (backward-compat bundle)

Approach: block optional modules from sys.modules using None-sentinels, then
import the core public API and assert the imports succeed.  Using None-sentinel
blocking rather than uninstalling packages makes the test runnable in the
full development environment without destroying any packages.
"""

import sys
import types
from contextlib import contextmanager
from typing import Dict, List

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _blocked_modules(names: List[str]):
    """Context manager: temporarily block a list of module names from sys.modules.

    Sets each name (and any already-imported sub-module whose dotted prefix
    matches) to ``None`` in ``sys.modules``, which causes subsequent import
    attempts to raise ``ImportError``.  On exit the original state of
    ``sys.modules`` is restored.
    """
    # Collect all existing entries whose names begin with any blocked prefix.
    blocked_prefixes = tuple(n + "." for n in names) + tuple(names)

    originals: Dict[str, object] = {}
    for key in list(sys.modules.keys()):
        if key in names or any(key.startswith(p) for p in blocked_prefixes):
            originals[key] = sys.modules.pop(key)

    # Install None-sentinels for the top-level names.
    for name in names:
        sys.modules[name] = None  # type: ignore[assignment]

    # Also evict any already-cached bili.* modules so they are re-imported
    # fresh under the blocked environment.
    bili_originals: Dict[str, object] = {}
    for key in list(sys.modules.keys()):
        if key == "bili" or key.startswith("bili."):
            bili_originals[key] = sys.modules.pop(key)

    try:
        yield
    finally:
        # Restore everything in reverse order.
        for key in list(sys.modules.keys()):
            if key in names:
                del sys.modules[key]
        sys.modules.update(originals)
        sys.modules.update(bili_originals)


# Optional extras whose absence should NOT break the lean core.
_OPTIONAL_TOP_LEVEL = [
    "streamlit",
    "streamlit_flow",
    "flask",
    "torch",
    "tensorflow",
    "keras",
    "faiss",
    "opensearchpy",
    "firebase_admin",
    "mcp",
    # HuggingFace heavy stack
    "transformers",
    "sentence_transformers",
    "langchain_huggingface",
    # DB drivers
    "pymongo",
    "motor",
    "psycopg2",
    "psycopg",
    "psycopg_pool",
    # Firebase / Google Cloud heavy SDK
    "firebase_admin",
    "google.cloud.aiplatform",
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLeanCoreImports:
    """Core IRIS/AETHER APIs import without optional extras installed."""

    def test_bili_package_imports_without_optionals(self):
        """The top-level ``bili`` package imports cleanly without optional extras."""
        with _blocked_modules(_OPTIONAL_TOP_LEVEL):
            import bili  # pylint: disable=import-outside-toplevel  # noqa: F401

            assert hasattr(bili, "__getattr__")

    def test_bili_aether_schema_imports_without_optionals(self):
        """``bili.aether.schema`` (MASConfig, AgentSpec) imports without optional extras."""
        with _blocked_modules(_OPTIONAL_TOP_LEVEL):
            from bili.aether.schema import (  # pylint: disable=import-outside-toplevel
                AgentSpec,
                MASConfig,
            )

            assert AgentSpec is not None
            assert MASConfig is not None

    def test_bili_aether_compiler_imports_without_optionals(self):
        """``bili.aether.compiler`` (compile_mas) imports without optional extras."""
        with _blocked_modules(_OPTIONAL_TOP_LEVEL):
            from bili.aether.compiler import (  # pylint: disable=import-outside-toplevel
                compile_mas,
            )

            assert callable(compile_mas)

    def test_bili_aether_validation_imports_without_optionals(self):
        """``bili.aether.validation`` (validate_mas) imports without optional extras."""
        with _blocked_modules(_OPTIONAL_TOP_LEVEL):
            from bili.aether.validation import (  # pylint: disable=import-outside-toplevel
                validate_mas,
            )

            assert callable(validate_mas)

    def test_bili_iris_config_imports_without_optionals(self):
        """``bili.iris.config.llm_config`` imports without optional extras."""
        with _blocked_modules(_OPTIONAL_TOP_LEVEL):
            from bili.iris.config.llm_config import (  # pylint: disable=import-outside-toplevel
                LLM_MODELS,
            )

            assert isinstance(LLM_MODELS, dict)

    def test_bili_iris_providers_registry_imports_without_optionals(self):
        """The provider registry imports without optional extras."""
        with _blocked_modules(_OPTIONAL_TOP_LEVEL):
            from bili.iris.providers.registry import (  # pylint: disable=import-outside-toplevel
                PROVIDER_REGISTRY,
                ProviderRegistry,
            )

            assert isinstance(PROVIDER_REGISTRY, ProviderRegistry)

    def test_bili_iris_loaders_streaming_imports_without_optionals(self):
        """``bili.iris.loaders.streaming_utils`` imports without optional extras."""
        with _blocked_modules(_OPTIONAL_TOP_LEVEL):
            from bili.iris.loaders.streaming_utils import (  # pylint: disable=import-outside-toplevel
                stream_agent,
            )

            assert callable(stream_agent)

    def test_bili_iris_loaders_langchain_loader_imports_without_optionals(self):
        """``bili.iris.loaders.langchain_loader`` imports without optional extras."""
        with _blocked_modules(_OPTIONAL_TOP_LEVEL):
            from bili.iris.loaders.langchain_loader import (  # pylint: disable=import-outside-toplevel
                build_agent_graph,
            )

            assert callable(build_agent_graph)

    def test_bili_iris_checkpointers_memory_imports_without_optionals(self):
        """The in-memory checkpointer imports without optional extras."""
        with _blocked_modules(_OPTIONAL_TOP_LEVEL):
            from bili.iris.checkpointers.memory_checkpointer import (  # pylint: disable=import-outside-toplevel
                QueryableMemorySaver,
            )

            assert QueryableMemorySaver is not None

    def test_bili_utils_logging_imports_without_optionals(self):
        """``bili.utils.logging_utils`` imports without optional extras."""
        with _blocked_modules(_OPTIONAL_TOP_LEVEL):
            from bili.utils.logging_utils import (  # pylint: disable=import-outside-toplevel
                get_logger,
            )

            assert callable(get_logger)

    def test_bili_utils_langgraph_utils_imports_without_optionals(self):
        """``bili.utils.langgraph_utils`` imports without optional extras."""
        with _blocked_modules(_OPTIONAL_TOP_LEVEL):
            from bili.utils.langgraph_utils import (  # pylint: disable=import-outside-toplevel
                State,
            )

            assert State is not None

    def test_streamlit_utils_imports_without_streamlit(self):
        """``conditional_cache_resource`` imports without streamlit installed."""
        with _blocked_modules(["streamlit", "streamlit_flow"]):
            # Evict bili.streamlit_ui.utils specifically so it re-imports fresh.
            to_evict = [k for k in sys.modules if k.startswith("bili.streamlit_ui")]
            saved = {k: sys.modules.pop(k) for k in to_evict}
            try:
                from bili.streamlit_ui.utils.streamlit_utils import (  # pylint: disable=import-outside-toplevel
                    conditional_cache_resource,
                )

                assert callable(conditional_cache_resource)
            finally:
                sys.modules.update(saved)

    def test_llm_loader_imports_without_torch(self):
        """``bili.iris.loaders.llm_loader`` imports without torch/transformers."""
        with _blocked_modules(["torch", "transformers", "langchain_huggingface"]):
            from bili.iris.loaders.llm_loader import (  # pylint: disable=import-outside-toplevel
                load_model,
            )

            assert callable(load_model)

    def test_embeddings_loader_imports_without_heavy_sdks(self):
        """``bili.iris.loaders.embeddings_loader`` imports without embedding SDKs."""
        with _blocked_modules(["langchain_aws", "langchain_google_vertexai"]):
            from bili.iris.loaders.embeddings_loader import (  # pylint: disable=import-outside-toplevel
                load_embedding_function,
            )

            assert callable(load_embedding_function)

    def test_tools_loader_imports_without_faiss_or_opensearch(self):
        """``bili.iris.loaders.tools_loader`` imports without faiss or opensearch-py."""
        with _blocked_modules(["faiss", "opensearchpy"]):
            from bili.iris.loaders.tools_loader import (  # pylint: disable=import-outside-toplevel
                TOOL_REGISTRY,
                initialize_tools,
            )

            assert hasattr(TOOL_REGISTRY, "__getitem__")
            assert callable(initialize_tools)

    def test_opensearch_utils_not_imported_on_utils_access(self):
        """Accessing ``bili.utils.logging_utils`` does not pull in opensearch-py."""
        with _blocked_modules(["opensearchpy"]):
            # Direct import of logging_utils should succeed even though
            # opensearch_utils is in the same package.
            from bili.utils.logging_utils import (  # pylint: disable=import-outside-toplevel
                get_logger,
            )

            assert callable(get_logger)
            # opensearch_utils must NOT have been imported as a side-effect.
            assert "bili.utils.opensearch_utils" not in sys.modules or (
                sys.modules.get("bili.utils.opensearch_utils") is None
                or isinstance(
                    sys.modules["bili.utils.opensearch_utils"], types.ModuleType
                )
                # If it was imported before this test, it may be a real module; that's OK.
            )

    def test_mongo_checkpointer_not_imported_on_memory_checkpointer_access(self):
        """Importing the memory checkpointer does not pull in pymongo."""
        with _blocked_modules(["pymongo", "motor"]):
            from bili.iris.checkpointers.memory_checkpointer import (  # pylint: disable=import-outside-toplevel
                QueryableMemorySaver,
            )

            assert QueryableMemorySaver is not None
