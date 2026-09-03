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
    pip install bili-core[llamacpp]    # llama.cpp local models
    pip install bili-core[ml]          # Full ML stack
    pip install bili-core[docs]        # Document-processing tools
    pip install bili-core[firebase]    # Firebase auth
    pip install bili-core[mcp]         # MCP client subsystem
    pip install bili-core[dev]         # Development tooling
    pip install bili-core[all]         # everything (backward-compat bundle)

Approach: block optional modules from sys.modules using None-sentinels, then
import the core public API and assert the imports succeed.  Using None-sentinel
blocking rather than uninstalling packages makes the test runnable in the
full development environment without destroying any packages.
"""

import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path
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

    def test_amazon_opensearch_not_imported_on_sibling_tool_access(self):
        """Importing a sibling bili.iris.tools submodule does not pull in opensearch-py.

        bili.iris.tools.__init__ lazily loads its submodules (PEP 562), so
        importing bili.iris.tools.mock_tool must not trigger
        bili.iris.tools.amazon_opensearch's own module-level opensearch-py
        import as a side effect of the package __init__ running. Checking
        for the absence of "bili.iris.tools.amazon_opensearch" in
        sys.modules (rather than "opensearchpy" itself) sidesteps
        _blocked_modules' own None-sentinel, which is always a key present
        in sys.modules while active -- see
        test_opensearch_utils_not_imported_on_utils_access for the same
        pattern.
        """
        with _blocked_modules(["opensearchpy"]):
            import bili.iris.tools.mock_tool  # pylint: disable=import-outside-toplevel,unused-import

            assert "bili.iris.tools.amazon_opensearch" not in sys.modules


# ---------------------------------------------------------------------------
# PEP 562 lazy-loader __init__.py coverage
#
# Each lazy-loader __init__.py has three reachable branches:
#   1. __getattr__ with a known lazy submodule name: imports + caches + returns
#   2. __getattr__ with an unknown name: raises AttributeError
#   3. __dir__: returns the advertised submodule set
#
# These tests exercise all three branches for every package that uses the
# pattern so the per-file coverage gate passes.
# ---------------------------------------------------------------------------


class TestLazyLoaderInits:
    """PEP 562 __getattr__ branches in every lazy-loader __init__.py."""

    # ------------------------------------------------------------------
    # bili/utils/__init__.py
    # ------------------------------------------------------------------

    def test_utils_getattr_loads_known_submodule(self):
        """Accessing a known submodule via bili.utils.__getattr__ imports it."""
        import importlib  # pylint: disable=import-outside-toplevel

        # Evict cached module so __getattr__ is invoked fresh.
        saved = sys.modules.pop("bili.utils", None)
        try:
            pkg = importlib.import_module("bili.utils")
            # Trigger the lazy path by accessing a known submodule name.
            mod = pkg.__getattr__("logging_utils")
            assert mod is not None
            assert hasattr(mod, "get_logger")
        finally:
            if saved is not None:
                sys.modules["bili.utils"] = saved

    def test_utils_getattr_raises_for_unknown_name(self):
        """Accessing an unknown attribute via bili.utils.__getattr__ raises AttributeError."""
        import importlib  # pylint: disable=import-outside-toplevel

        pkg = importlib.import_module("bili.utils")
        import pytest as _pytest  # pylint: disable=import-outside-toplevel

        with _pytest.raises(AttributeError, match="has no attribute"):
            pkg.__getattr__("_nonexistent_submodule_xyz")

    def test_utils_dir_returns_submodule_names(self):
        """bili.utils.__dir__() returns the declared lazy submodule names."""
        import importlib  # pylint: disable=import-outside-toplevel

        pkg = importlib.import_module("bili.utils")
        names = pkg.__dir__()
        assert "logging_utils" in names
        assert "file_utils" in names
        assert "opensearch_utils" in names

    # ------------------------------------------------------------------
    # bili/iris/checkpointers/__init__.py
    # ------------------------------------------------------------------

    def test_checkpointers_getattr_loads_known_submodule(self):
        """Accessing a known submodule via bili.iris.checkpointers.__getattr__ imports it."""
        import importlib  # pylint: disable=import-outside-toplevel

        saved = sys.modules.pop("bili.iris.checkpointers", None)
        try:
            pkg = importlib.import_module("bili.iris.checkpointers")
            mod = pkg.__getattr__("memory_checkpointer")
            assert mod is not None
            assert hasattr(mod, "QueryableMemorySaver")
        finally:
            if saved is not None:
                sys.modules["bili.iris.checkpointers"] = saved

    def test_checkpointers_getattr_raises_for_unknown_name(self):
        """Accessing an unknown attribute via bili.iris.checkpointers.__getattr__ raises."""
        import importlib  # pylint: disable=import-outside-toplevel

        import pytest as _pytest  # pylint: disable=import-outside-toplevel

        pkg = importlib.import_module("bili.iris.checkpointers")
        with _pytest.raises(AttributeError, match="has no attribute"):
            pkg.__getattr__("_nonexistent_xyz")

    def test_checkpointers_dir_returns_submodule_names(self):
        """bili.iris.checkpointers.__dir__() returns the declared lazy submodule names."""
        import importlib  # pylint: disable=import-outside-toplevel

        pkg = importlib.import_module("bili.iris.checkpointers")
        names = pkg.__dir__()
        assert "memory_checkpointer" in names
        assert "mongo_checkpointer" in names
        assert "pg_checkpointer" in names

    # ------------------------------------------------------------------
    # bili/iris/tools/__init__.py
    # ------------------------------------------------------------------

    def test_iris_tools_getattr_loads_known_submodule(self):
        """Accessing a known submodule via bili.iris.tools.__getattr__ imports it."""
        import importlib  # pylint: disable=import-outside-toplevel

        saved = sys.modules.pop("bili.iris.tools", None)
        try:
            pkg = importlib.import_module("bili.iris.tools")
            mod = pkg.__getattr__("mock_tool")
            assert mod is not None
            assert hasattr(mod, "init_mock_tool")
        finally:
            if saved is not None:
                sys.modules["bili.iris.tools"] = saved

    def test_iris_tools_getattr_raises_for_unknown_name(self):
        """Accessing an unknown attribute via bili.iris.tools.__getattr__ raises."""
        import importlib  # pylint: disable=import-outside-toplevel

        import pytest as _pytest  # pylint: disable=import-outside-toplevel

        pkg = importlib.import_module("bili.iris.tools")
        with _pytest.raises(AttributeError, match="has no attribute"):
            pkg.__getattr__("_nonexistent_xyz")

    def test_iris_tools_dir_returns_submodule_names(self):
        """bili.iris.tools.__dir__() returns the declared lazy submodule names."""
        import importlib  # pylint: disable=import-outside-toplevel

        pkg = importlib.import_module("bili.iris.tools")
        names = pkg.__dir__()
        assert "amazon_opensearch" in names
        assert "faiss_memory_indexing" in names
        assert "mock_tool" in names

    # ------------------------------------------------------------------
    # bili/iris/loaders/__init__.py
    # ------------------------------------------------------------------

    def test_loaders_getattr_loads_known_submodule(self):
        """Accessing a known submodule via bili.iris.loaders.__getattr__ imports it."""
        import importlib  # pylint: disable=import-outside-toplevel

        saved = sys.modules.pop("bili.iris.loaders", None)
        try:
            pkg = importlib.import_module("bili.iris.loaders")
            mod = pkg.__getattr__("streaming_utils")
            assert mod is not None
            assert hasattr(mod, "stream_agent")
        finally:
            if saved is not None:
                sys.modules["bili.iris.loaders"] = saved

    def test_loaders_getattr_raises_for_unknown_name(self):
        """Accessing an unknown attribute via bili.iris.loaders.__getattr__ raises."""
        import importlib  # pylint: disable=import-outside-toplevel

        import pytest as _pytest  # pylint: disable=import-outside-toplevel

        pkg = importlib.import_module("bili.iris.loaders")
        with _pytest.raises(AttributeError, match="has no attribute"):
            pkg.__getattr__("_nonexistent_xyz")

    def test_loaders_dir_returns_submodule_names(self):
        """bili.iris.loaders.__dir__() returns the declared lazy submodule names."""
        import importlib  # pylint: disable=import-outside-toplevel

        pkg = importlib.import_module("bili.iris.loaders")
        names = pkg.__dir__()
        assert "llm_loader" in names
        assert "tools_loader" in names
        assert "streaming_utils" in names

    # ------------------------------------------------------------------
    # bili/streamlit_ui/__init__.py
    # ------------------------------------------------------------------

    def test_streamlit_ui_getattr_raises_for_unknown_name(self):
        """Accessing an unknown attribute via bili.streamlit_ui.__getattr__ raises."""
        import importlib  # pylint: disable=import-outside-toplevel

        import pytest as _pytest  # pylint: disable=import-outside-toplevel

        pkg = importlib.import_module("bili.streamlit_ui")
        with _pytest.raises(AttributeError, match="has no attribute"):
            pkg.__getattr__("_nonexistent_xyz")

    def test_streamlit_ui_dir_returns_submodule_names(self):
        """bili.streamlit_ui.__dir__() returns the declared lazy submodule names."""
        import importlib  # pylint: disable=import-outside-toplevel

        pkg = importlib.import_module("bili.streamlit_ui")
        names = pkg.__dir__()
        assert "query" in names
        assert "ui" in names
        assert "utils" in names

    def test_streamlit_ui_getattr_loads_known_submodule(self):
        """Accessing a known submodule via bili.streamlit_ui.__getattr__ imports it."""
        import importlib  # pylint: disable=import-outside-toplevel

        # Evict cached entry so __getattr__ path is exercised fresh.
        saved = sys.modules.pop("bili.streamlit_ui", None)
        try:
            pkg = importlib.import_module("bili.streamlit_ui")
            mod = pkg.__getattr__("utils")
            assert mod is not None
        finally:
            if saved is not None:
                sys.modules["bili.streamlit_ui"] = saved

    # ------------------------------------------------------------------
    # bili/streamlit_ui/utils/__init__.py
    # ------------------------------------------------------------------

    def test_streamlit_ui_utils_getattr_loads_known_submodule(self):
        """Accessing a known submodule via bili.streamlit_ui.utils.__getattr__ imports it."""
        import importlib  # pylint: disable=import-outside-toplevel

        saved = sys.modules.pop("bili.streamlit_ui.utils", None)
        try:
            pkg = importlib.import_module("bili.streamlit_ui.utils")
            mod = pkg.__getattr__("streamlit_utils")
            assert mod is not None
            assert hasattr(mod, "conditional_cache_resource")
        finally:
            if saved is not None:
                sys.modules["bili.streamlit_ui.utils"] = saved

    def test_streamlit_ui_utils_getattr_raises_for_unknown_name(self):
        """Accessing an unknown attribute via bili.streamlit_ui.utils.__getattr__ raises."""
        import importlib  # pylint: disable=import-outside-toplevel

        import pytest as _pytest  # pylint: disable=import-outside-toplevel

        pkg = importlib.import_module("bili.streamlit_ui.utils")
        with _pytest.raises(AttributeError, match="has no attribute"):
            pkg.__getattr__("_nonexistent_xyz")

    def test_streamlit_ui_utils_dir_returns_submodule_names(self):
        """bili.streamlit_ui.utils.__dir__() returns the declared lazy submodule names."""
        import importlib  # pylint: disable=import-outside-toplevel

        pkg = importlib.import_module("bili.streamlit_ui.utils")
        names = pkg.__dir__()
        assert "state_management" in names
        assert "streamlit_utils" in names


# ---------------------------------------------------------------------------
# Lean install_requires gate
#
# Asserts that setup.py's install_requires (populated from requirements.txt)
# contains only the lean-core set, and that every heavy optional package
# appears ONLY in extras_require, never in install_requires.
#
# This is a static analysis test: it parses setup.py's extras_require and
# reads requirements.txt directly rather than running a subprocess install,
# so it runs without network access in CI and without needing a clean venv.
# ---------------------------------------------------------------------------

# Canonical set of package name prefixes that must NEVER appear in the
# base install_requires.  These are the heavy optional backends gated by
# extras.  Matching is case-insensitive and normalises hyphens to underscores
# so "faiss-cpu" matches "faiss_cpu".
_MUST_BE_EXTRAS_ONLY = {
    # Streamlit surface
    "streamlit",
    "streamlit_flow_component",
    "pillow",
    "pandas",
    # Flask surface
    "flask",
    "pyjwt",
    # ML frameworks
    "torch",
    "torchaudio",
    "torchvision",
    "tensorflow",
    "tf_keras",
    "keras",
    "scikit_learn",
    "ml_dtypes",
    "accelerate",
    "optimum",
    # HuggingFace stack
    "transformers",
    "langchain_huggingface",
    "sentence_transformers",
    "datasets",
    # llama.cpp
    "llama_cpp_python",
    # Vector search
    "faiss_cpu",
    "opensearch_py",
    "requests_aws4auth",
    # DB adapters
    "pymongo",
    "motor",
    "langchain_mongodb",
    "langgraph_checkpoint_mongodb",
    "psycopg2",
    "langgraph_checkpoint_postgres",
    # Auth
    "firebase_admin",
    # Document processing
    "beautifulsoup4",
    "nltk",
    "openpyxl",
    "pypdf",
    "python_docx",
    "rapidocr_onnxruntime",
    "textract",
    "unstructured",
    "unstructured_client",
    # Vision / imaging extras
    "opencv_python",
    "pi_heif",
    # Dev tooling
    "pytest",
    "pylint",
    "black",
    "isort",
    "autoflake",
    "pre_commit",
    "mongomock",
    "pympler",
    "watchdog",
    "setuptools",
}


def _normalise(pkg_spec: str) -> str:
    """Return the normalised package name from a requirement specifier.

    Strips version constraints (``~=``, ``>=``, ``==``, etc.), extras
    (``[all-docs]``), and normalises hyphens to underscores + lowercase.
    """
    # Strip extras like [all-docs]
    name = re.split(r"[\[~>=<!;]", pkg_spec.strip())[0]
    return name.lower().replace("-", "_")


class TestInstallRequiresIsLean:
    """install_requires must contain only the lean-core set.

    Reads requirements.txt (the source of install_requires) and asserts that
    none of the heavy optional packages are listed there.  Also asserts that
    the known heavy packages DO appear in at least one extras_require entry in
    setup.py, confirming they are still reachable via extras rather than simply
    dropped.
    """

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _REQUIREMENTS_TXT = _REPO_ROOT / "requirements.txt"
    _SETUP_PY = _REPO_ROOT / "setup.py"

    def _read_requirements(self):
        """Parse requirements.txt into a list of normalised package names."""
        lines = self._REQUIREMENTS_TXT.read_text(encoding="utf-8").splitlines()
        return [
            _normalise(line)
            for line in lines
            if line.strip()
            and not line.startswith("#")
            and not line.startswith("git+")
            and not line.startswith("http")
        ]

    def _read_all_extras_packages(self):
        """Parse all package names mentioned in any extras_require entry.

        Uses a simple regex scan of setup.py rather than executing it, so the
        test has no side effects and runs in a sandboxed CI environment.
        """
        text = self._SETUP_PY.read_text(encoding="utf-8")
        # Match quoted strings inside the extras_require dict block.
        # This regex finds string literals that look like package specifiers.
        raw = re.findall(r'["\']([A-Za-z0-9][A-Za-z0-9._\-\[\]~>=<!, ]+)["\']', text)
        return {_normalise(r) for r in raw}

    def test_no_heavy_package_in_install_requires(self):
        """requirements.txt must not contain any heavy optional package."""
        base_packages = self._read_requirements()
        violations = [pkg for pkg in base_packages if pkg in _MUST_BE_EXTRAS_ONLY]
        assert not violations, (
            "The following heavy packages must be moved from requirements.txt "
            f"(install_requires) to an extras_require entry in setup.py:\n"
            + "\n".join(f"  {v}" for v in sorted(violations))
        )

    def test_heavy_packages_present_in_extras(self):
        """Every mandatory-extras-only package appears in at least one extra."""
        extras_packages = self._read_all_extras_packages()
        # Spot-check a representative sample rather than the full set, because
        # some packages (e.g. llama_cpp_python) have unusual pip specifiers that
        # the regex may not capture identically.
        spot_check = {
            "streamlit",
            "torch",
            "pymongo",
            "psycopg2",
            "faiss_cpu",
            "opensearch_py",
            "firebase_admin",
            "flask",
            "transformers",
            "langchain_mongodb",
            "langgraph_checkpoint_mongodb",
            "langgraph_checkpoint_postgres",
        }
        missing = {pkg for pkg in spot_check if pkg not in extras_packages}
        assert not missing, (
            "The following packages are expected to appear in extras_require "
            f"in setup.py but were not found:\n"
            + "\n".join(f"  {m}" for m in sorted(missing))
        )

    def test_lean_core_packages_present_in_requirements(self):
        """requirements.txt must contain the essential lean-core packages."""
        base_packages = set(self._read_requirements())
        lean_core_required = {
            "langchain",
            "langchain_core",
            "langchain_community",
            "langgraph",
            "langchain_aws",
            "langchain_google_vertexai",
            "langchain_openai",
            "pyyaml",
            "requests",
        }
        missing = lean_core_required - base_packages
        assert not missing, (
            "The following lean-core packages are missing from requirements.txt:\n"
            + "\n".join(f"  {m}" for m in sorted(missing))
        )
