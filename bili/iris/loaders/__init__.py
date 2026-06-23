"""bili.iris.loaders — workflow, LLM, tool, and embedding loader utilities.

Submodules are loaded lazily so that importing bili.iris.loaders does not
eagerly pull in heavy optional backends:
- ``tokenizer_loader`` requires ``transformers`` (HuggingFace extra)
- ``llm_loader`` requires ``torch`` and ``transformers`` (HuggingFace extra)
- ``embeddings_loader`` requires provider SDKs (langchain-aws, etc.)
- ``tools_loader`` requires FAISS / OpenSearch tool backends

Only the modules the caller actually accesses are imported.
"""

_LAZY_SUBMODULES = {
    "embeddings_loader",
    "langchain_loader",
    "llm_loader",
    "middleware_loader",
    "streaming_utils",
    "tokenizer_loader",
    "tools_loader",
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
