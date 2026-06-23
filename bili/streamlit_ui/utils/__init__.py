"""bili.streamlit_ui.utils — Streamlit utility helpers.

Submodules are loaded lazily so that importing bili.streamlit_ui.utils (which
happens as a side-effect of importing ``streamlit_utils`` in core runtime
modules) does not eagerly pull in ``state_management``.  ``state_management``
imports ``streamlit`` at module scope and would break lean-core installs that
do not have the ``[streamlit]`` extra installed.
"""

_LAZY_SUBMODULES = {"state_management", "streamlit_utils"}


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
