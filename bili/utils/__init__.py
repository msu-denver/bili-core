"""bili.utils — shared utility helpers.

Submodules are loaded lazily so that importing bili.utils does not eagerly
pull in heavy optional backends (opensearch-py, boto3, requests-aws4auth).
``opensearch_utils`` is only needed when the ``[opensearch]`` extra is
installed and an OpenSearch-backed tool is actually used; eager import would
break lean-core installs.
"""

# Lazy attribute mapping: module name -> load the submodule itself
_LAZY_SUBMODULES = {
    "file_utils",
    "langgraph_utils",
    "logging_utils",
    "opensearch_utils",
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
