"""Tests for the bili package lazy-submodule loader (PEP 562)."""

import importlib
import types

import pytest

import bili


class TestLazySubmoduleLoader:
    """bili.__getattr__ lazily imports declared subpackages."""

    def test_getattr_imports_known_submodule(self):
        """Accessing a declared submodule returns the imported module."""
        module = bili.__getattr__("utils")
        assert isinstance(module, types.ModuleType)
        assert module.__name__ == "bili.utils"

    def test_getattr_caches_module_in_globals(self):
        """A loaded submodule is cached so later access is a direct attribute."""
        bili.__getattr__("utils")
        assert isinstance(vars(bili).get("utils"), types.ModuleType)

    def test_getattr_unknown_name_raises_attribute_error(self):
        """An undeclared attribute name raises AttributeError."""
        with pytest.raises(AttributeError):
            bili.__getattr__("definitely_not_a_submodule")

    def test_dir_lists_exactly_the_lazy_submodules(self):
        """dir(bili) advertises exactly the lazily loadable submodules."""
        assert set(bili.__dir__()) == set(bili._LAZY_SUBMODULES)

    def test_every_declared_submodule_actually_imports(self):
        """Every name in _LAZY_SUBMODULES must resolve to a real package.

        Regression guard for stale entries: the v5.0.0 refactor moved several
        former top-level packages (config, checkpointers, loaders, nodes,
        tools, graph_builder) under bili/iris/, leaving the lazy list pointing
        at modules that no longer exist. This asserts each declared submodule
        imports cleanly, so a stale entry fails loudly here.
        """
        for name in bili._LAZY_SUBMODULES:
            module = importlib.import_module(f"bili.{name}")
            assert isinstance(module, types.ModuleType)

    def test_all_matches_lazy_submodules(self):
        """__all__ must equal _LAZY_SUBMODULES so `from bili import *` works.

        A name in __all__ that is not in _LAZY_SUBMODULES would raise
        AttributeError from __getattr__ during a star import.
        """
        assert set(bili.__all__) == set(bili._LAZY_SUBMODULES)

    def test_star_import_resolves_every_name(self):
        """Every name in __all__ resolves via __getattr__ (star-import safe)."""
        for name in bili.__all__:
            assert isinstance(getattr(bili, name), types.ModuleType)
