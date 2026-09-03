"""The check must run on an interpreter with nothing installed.

The scheduled job that runs this against the live upstreams should not need a
dependency install. That is not a convenience: an install step is a second
thing that can fail, and a job whose install failed reports a red run that
says nothing about the catalog, which is the same confusion between "could not
run" and "found something" that the exit codes exist to prevent.

Measured: with no packages installed at all, the check parses both recorded
slices and produces its report. This pins that property by reading what the
package imports, because the property is invisible on a development machine
where every dependency happens to be present.
"""

import ast
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parent.parent

#: The only non-stdlib import the check is allowed: the catalog it reads.
ALLOWED_EXTERNAL = {"bili.iris.config.llm_config"}

RUNTIME_MODULES = sorted(p for p in PACKAGE_DIR.glob("*.py") if p.name != "__init__.py")


def _imported_roots(path: Path):
    """Return the top-level module name of every import in *path*.

    A relative import inside the package resolves to nothing external and is
    skipped.

    :param path: The module to read.
    :returns: The set of imported names, absolute ones only.
    :rtype: set
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import, inside this package
                continue
            if node.module:
                names.add(node.module)
    return names


class TestNoThirdPartyImports:
    """Nothing here may reach for a package a bare interpreter lacks."""

    def test_the_package_has_runtime_modules_to_check(self):
        """A glob that matched nothing would pass every assertion below."""
        assert len(RUNTIME_MODULES) >= 5

    @pytest.mark.parametrize("path", RUNTIME_MODULES, ids=lambda p: p.name)
    def test_every_import_is_stdlib_or_the_catalog(self, path):
        """An import outside the standard library breaks the no-install run."""
        for name in sorted(_imported_roots(path)):
            if name in ALLOWED_EXTERNAL:
                continue
            root = name.split(".")[0]
            assert root in sys.stdlib_module_names, (
                f"{path.name} imports {name!r}, which is neither stdlib nor "
                f"the catalog: the scheduled job would need an install step"
            )

    def test_the_catalog_itself_imports_nothing(self):
        """The one allowed import must not drag a dependency in behind it.

        The catalog is pure data today. If it grew an import, this check
        would inherit it and the no-install property would break somewhere
        that has nothing to do with this package.
        """
        catalog = PACKAGE_DIR.parent / "llm_config.py"
        assert _imported_roots(catalog) == set()
