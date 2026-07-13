"""bili.iris.tools: built-in LangChain tool implementations.

Submodules are loaded lazily so that importing bili.iris.tools (or any one
sibling submodule) does not eagerly pull in heavy optional backends:
- ``amazon_opensearch`` requires ``opensearch-py`` / ``requests-aws4auth``
  (OpenSearch extra)
- ``faiss_memory_indexing`` requires ``faiss-cpu`` (FAISS extra)

Only the modules the caller actually accesses are imported.
"""

_LAZY_SUBMODULES = {
    "amazon_opensearch",
    "api_open_weather",
    "api_serp",
    "api_weather_gov",
    "ask_user",
    "faiss_memory_indexing",
    "hitl",
    "mock_tool",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib  # pylint: disable=import-outside-toplevel

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(_LAZY_SUBMODULES)


__all__ = sorted(_LAZY_SUBMODULES)
