import os
import subprocess

from setuptools import find_packages, setup
from setuptools.command.install import install


class PostInstallCommand(install):
    """
    Post-installation command to handle additional setup steps after the standard
    installation process.

    This class extends the standard `install` command to include extra functionality,
    such as setting up pre-commit hooks and installing HTTP/Git-based dependencies.
    """

    def run(self):
        # Run the standard install process
        install.run(self)

        # Check if we're in build mode by looking for bdist_wheel in the command
        import sys

        is_wheel_build = "bdist_wheel" in sys.argv

        # Install git-based dependencies FIRST
        # Note: This may fail in isolated build environments (PEP 517)
        http_git_deps = read_http_git_requirements()
        if http_git_deps:
            print("Installing HTTP/Git-based dependencies separately...")
            try:
                subprocess.check_call(["pip", "install"] + http_git_deps)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to install Git/HTTP dependencies: {e}")
                print(
                    "Please install them manually with: pip install "
                    + " ".join(http_git_deps)
                )

        # THEN install pre-commit hooks
        if not is_wheel_build and os.path.isdir(".git"):
            print("Installing pre-commit hooks...")
            try:
                subprocess.check_call(["pre-commit", "install"])
            except:
                print("Warning: pre-commit install failed. Skipping hook setup.")
        elif not is_wheel_build:
            print("Skipping pre-commit hook installation (not a Git repository).")


def read_requirements():
    """
    Reads and processes the requirements file, returning a list of standard dependencies.

    Excludes:
    - Lines that start with `#` (comments)
    - Git-based (`git+`) and HTTP-based (`http`) dependencies (installed separately)
    """
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(req_file, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip()
            and not line.startswith("#")
            and not line.startswith("git+")
            and not line.startswith("http")
        ]


def read_http_git_requirements():
    """
    Reads `requirements.txt` and extracts only Git-based (`git+`) and HTTP-based (`http`) dependencies.

    These are installed separately after `setup.py` is executed.
    """
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(req_file, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and (line.startswith("git+") or line.startswith("http"))
        ]


setup(
    name="bili-core",
    version="5.3.2",
    # Detect runtime packages while excluding every test subpackage. Without
    # the exclude, find_packages() bundles 200+ .py test modules (under
    # bili/<component>/tests/ and bili/<component>/<subcomponent>/tests/)
    # into the wheel and sdist as a side effect of having __init__.py in
    # each tests/ dir. The exclude patterns match full dotted package paths:
    # "*.tests" catches packages like bili.aegis.tests and bili.aether.tests,
    # and "*.tests.*" catches their subpackages (bili.aegis.tests.injection,
    # bili.aegis.tests.injection.payloads, and so on). No runtime code
    # imports from any bili.*.tests module (verified by grep), so dropping
    # them from the distribution is safe; pytest still discovers them at
    # the source-tree level when developers run the suite locally or in CI.
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests", "tests.*"]),
    # find_packages() only collects .py modules, so non-Python runtime data
    # (prompts, images, aether example configs) must be declared explicitly or
    # the wheel ships without them and import-time file reads crash. These globs
    # are relative to the bili/ package directory and intentionally targeted so
    # the distribution does not also bundle the aegis test fixtures or docs.
    # Keep in sync with MANIFEST.in (which controls the sdist).
    package_data={
        "bili": [
            "prompts/*.json",
            "images/*",
            "aether/config/examples/*.yaml",
        ],
    },
    install_requires=read_requirements(),  # Load only standard dependencies
    cmdclass={
        "install": PostInstallCommand,
    },
)
