"""BiliCore - Framework for benchmarking and building dynamic RAG implementations.

Subpackages are loaded lazily (PEP 562) to avoid importing heavy dependencies
(langgraph, torch, cloud SDKs, etc.) when only lightweight modules are needed.
"""

_LAZY_SUBMODULES = {
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
