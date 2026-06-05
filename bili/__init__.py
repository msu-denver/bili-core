"""BiliCore - Framework for benchmarking and building dynamic RAG implementations.

Subpackages are loaded lazily (PEP 562) to avoid importing heavy dependencies
(langgraph, torch, cloud SDKs, etc.) when only lightweight modules are needed.
"""

# The lazily loadable top-level subpackages. These must be real importable
# packages under bili/. The v5.0.0 refactor moved the former top-level
# checkpointers/config/graph_builder/loaders/nodes/tools packages under
# bili/iris/, so the lazy list is the three components (iris, aether, aegis)
# plus the shared subpackages.
_LAZY_SUBMODULES = {
    "aegis",
    "aether",
    "auth",
    "flask_api",
    "iris",
    "streamlit_ui",
    "utils",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_SUBMODULES)


__all__ = [
    "aether",
    "auth",
    "checkpointers",
    "config",
    "flask_api",
    "graph_builder",
    "loaders",
    "nodes",
    "tools",
    "utils",
]
