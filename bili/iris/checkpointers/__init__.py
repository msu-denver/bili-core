"""bili.iris.checkpointers — state persistence backends.

Submodules are loaded lazily so that importing bili.iris.checkpointers does
not eagerly pull in ``pymongo``/``motor`` (MongoDB extra) or
``psycopg2``/``psycopg_pool`` (PostgreSQL extra).  Only
``memory_checkpointer`` and ``versioning`` are pure-Python; the database
backends are optional and guarded by the ``[mongo]`` and ``[postgres]``
extras respectively.
"""

_LAZY_SUBMODULES = {
    "base_checkpointer",
    "checkpointer_functions",
    "memory_checkpointer",
    "mongo_checkpointer",
    "pg_checkpointer",
    "versioning",
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
