"""The repo-root bootstrap must work in a git worktree, in every copy of it.

Suite runners and their conftests locate the repository root by walking up
until they find ``.git``, then put that directory on ``sys.path``.  This runs
*before* any ``bili.*`` import, because making ``bili`` importable is what the
bootstrap is for, so these copies cannot import a shared helper from inside
the package: the import they would use is the one that does not work yet.  The
duplication is therefore forced rather than careless, and what follows makes
it non-drifting instead of trying to remove it.

The defect this pins: a git worktree's ``.git`` is a **file** holding a
``gitdir:`` pointer, not a directory.  ``Path.is_dir()`` is false there, the
walk runs off the top of the filesystem, and every one of these modules raises
at import.  Measured before the fix, collecting from a worktree: 9 collection
errors, which is the whole of AEGIS's suite-backed tests.

Three properties are asserted, and the last is the one that would have caught
it:

* the private ``_find_repo_root`` copies are structurally identical to each
  other, so one cannot drift.  The public ``_helpers.find_repo_root`` is
  deliberately out of that comparison: it serves callers that already have
  ``bili`` importable, so it carries a Returns/Raises contract and a fuller
  message, and forcing it to match a bootstrap would make one of the two
  wrong for its own audience;
* no copy anywhere, that one included, tests ``.git`` for being a directory;
  and
* the predicate actually resolves a root through a worktree-shaped ``.git``
  **file**, executed against a real directory layout rather than inspected.

The first two are source checks and would both pass with every copy
identically wrong, which is precisely the state this file was written to end.
The third is what makes them mean something.
"""

# pylint: disable=missing-function-docstring

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from bili.aegis.suites._helpers import find_repo_root

#: Root of the checkout under test, found the same way the code under test
#: finds it, so this module inherits the fix rather than re-deriving it.
_REPO_ROOT = find_repo_root()

#: The bootstrap's name in the standalone runners; the canonical helper
#: exports it without the underscore.
_BOOTSTRAP_NAMES = {"_find_repo_root", "find_repo_root"}


def _bootstrap_sources() -> "dict[str, str]":
    """Every definition of the repo-root walk in the package, by location.

    Derived from the tree rather than listed, so a copy added later is
    covered without anyone remembering to extend a list.
    """
    found = {}
    for path in sorted((_REPO_ROOT / "bili").rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:  # pragma: no cover — unreadable source
            continue
        if ".git" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:  # pragma: no cover — not importable anyway
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in _BOOTSTRAP_NAMES:
                key = f"{path.relative_to(_REPO_ROOT)}::{node.name}"
                found[key] = ast.dump(
                    ast.parse(ast.unparse(node)).body[0],  # type: ignore[arg-type]
                )
    return found


class TestEveryCopyOfTheBootstrapAgrees:
    """The forced duplication must not be allowed to drift."""

    def test_the_bootstrap_exists_in_more_than_one_place(self):
        """Guard the guard: an empty discovery would satisfy the next test."""
        sources = _bootstrap_sources()
        private = {k: v for k, v in sources.items() if k.endswith("::_find_repo_root")}
        assert len(private) > 1, (
            "found no duplicated bootstrap to compare; the discovery is broken "
            f"or the shape changed. found={list(sources)}"
        )

    def test_every_private_copy_is_structurally_identical(self):
        """The ``_find_repo_root`` copies must all be the same function.

        Scoped to the private bootstrap deliberately.  The public
        ``_helpers.find_repo_root`` answers the same question for callers that
        already have ``bili`` importable, so it carries a documented
        Returns/Raises contract and a fuller error message; requiring it to be
        byte-identical to a bootstrap would force one of the two to be wrong
        for its own audience.  What both must share is the predicate, which
        the next test asserts across every copy including that one.
        """
        sources = {
            k: v
            for k, v in _bootstrap_sources().items()
            if k.endswith("::_find_repo_root")
        }
        if len(set(sources.values())) > 1:
            groups: "dict[str, list[str]]" = {}
            for loc, body in sources.items():
                groups.setdefault(body, []).append(loc)
            summary = "\n".join(
                f"  variant {i}: {locs}" for i, locs in enumerate(groups.values(), 1)
            )
            pytest.fail(
                "the repo-root bootstrap has drifted between copies; they must "
                f"stay identical because none can import the others:\n{summary}"
            )

    def test_no_copy_tests_dot_git_for_being_a_directory(self):
        """The regression guard, spanning every copy including the public one.

        This is the specific mistake that shipped eighteen times, and it is
        the one a future edit is most likely to reintroduce, because
        ``is_dir()`` reads as the more precise choice until you know a
        worktree exists.
        """
        offenders = []
        for path in sorted((_REPO_ROOT / "bili").rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):  # pragma: no cover — unreadable source
                continue
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "is_dir"
                ):
                    continue
                # Matched on the parsed tree rather than on the source text:
                # a textual scan for `".git"` and `.is_dir()` on one line
                # matches this very check, which is the shape of detector
                # that reports itself and hides a real offender behind its
                # own noise.
                names = {
                    sub.value
                    for sub in ast.walk(node.func.value)
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
                }
                if ".git" in names:
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
        assert not offenders, (
            "a .git directory test is back; a git worktree's .git is a FILE, "
            f"so this fails there: {offenders}"
        )


class TestTheBootstrapResolvesThroughAWorktree:
    """The behavioural half: run the predicate against a real layout."""

    @staticmethod
    def _run_bootstrap_from(module_path: Path) -> "subprocess.CompletedProcess[str]":
        """Execute the canonical bootstrap with ``__file__`` at *module_path*."""
        canonical = (
            _REPO_ROOT / "bili" / "aegis" / "suites" / "_helpers.py"
        ).read_text(encoding="utf-8")
        source = ast.unparse(
            next(
                node
                for node in ast.walk(ast.parse(canonical))
                if isinstance(node, ast.FunctionDef) and node.name == "find_repo_root"
            )
        )
        program = (
            "from pathlib import Path\n"
            f"__file__ = {str(module_path)!r}\n"
            f"{source}\n"
            "print(find_repo_root())\n"
        )
        return subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_a_worktree_dot_git_file_resolves(self, tmp_path):
        """``.git`` as a FILE must resolve, which is what a worktree has.

        Fails on ``is_dir()``; this is the case the whole change exists for.
        """
        root = tmp_path / "checkout"
        nested = root / "bili" / "aegis" / "suites"
        nested.mkdir(parents=True)
        # Exactly what `git worktree add` writes.
        (root / ".git").write_text(
            "gitdir: /somewhere/.git/worktrees/wt\n", encoding="utf-8"
        )
        result = self._run_bootstrap_from(nested / "_helpers.py")
        assert result.returncode == 0, (
            "the bootstrap failed against a worktree-shaped .git file: "
            f"{result.stderr[:400]}"
        )
        assert Path(result.stdout.strip()) == root

    def test_a_clone_dot_git_directory_still_resolves(self, tmp_path):
        """The ordinary case must keep working; the fix widens, not replaces."""
        root = tmp_path / "checkout"
        nested = root / "bili" / "aegis" / "suites"
        nested.mkdir(parents=True)
        (root / ".git").mkdir()
        result = self._run_bootstrap_from(nested / "_helpers.py")
        assert result.returncode == 0, result.stderr[:400]
        assert Path(result.stdout.strip()) == root

    def test_no_git_entry_still_raises(self, tmp_path):
        """Absent git metadata must stay a loud failure, not a wrong root.

        ``.exists()`` is broader than ``.is_dir()`` in a second way: it also
        matches a stray ``.git`` file that is no worktree pointer.  That trade
        is deliberate (a false root surfaces immediately as a bad ``sys.path``;
        an unrunnable suite does not), but the no-metadata case must still
        raise rather than silently selecting the filesystem root.
        """
        nested = tmp_path / "no_git" / "bili" / "aegis" / "suites"
        nested.mkdir(parents=True)
        result = self._run_bootstrap_from(nested / "_helpers.py")
        assert result.returncode != 0
        assert "Could not locate repo root" in result.stderr
