"""bili.streamlit_ui — Streamlit web-UI layer.

Subpackages are loaded lazily so that importing bili.streamlit_ui (which
happens as a side-effect of the conditional_cache_resource import chain in
core runtime modules) does not eagerly pull in ``streamlit`` itself.
``query`` → ``streamlit_query_handler`` → ``state_management`` → ``import
streamlit as st``.  Making this package lazy breaks that chain.
"""

_LAZY_SUBMODULES = {"query", "ui", "utils"}


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
