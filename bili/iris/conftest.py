"""Shared pytest setup for the IRIS test suites.

Importing ``bili.iris.tools`` before ``bili.iris.loaders`` triggers a circular
import: the tools package ``__init__`` imports ``faiss_memory_indexing``, which
imports ``bili.iris.loaders.embeddings_loader``; loading the loaders package in
turn runs ``tools_loader``, which imports ``init_faiss`` from the
still-initializing tools package. pytest imports a test module's parent
packages before the module itself, so collecting any module under
``bili/iris/tools/tests/`` in isolation would import ``bili.iris.tools`` first
and hit the cycle.

pytest loads conftest files from the rootdir down toward each test file's
directory and imports them before collecting test modules in or under that
directory. Importing the loaders package here, at the ``bili/iris`` level,
establishes the correct order so collection succeeds whether the tools suite
runs alone or alongside the rest of the suite.
"""

import bili.iris.loaders  # noqa: F401  pylint: disable=unused-import
