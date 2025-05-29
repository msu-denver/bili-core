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

        # Only install pre-commit hooks if this is a Git repository
        if os.path.isdir(".git"):
            print("Installing pre-commit hooks...")
            try:
                subprocess.check_call(["pre-commit", "install"])
            except:
                print("Warning: pre-commit install failed. Skipping hook setup.")
        else:
            print("Skipping pre-commit hook installation (not a Git repository).")

        # Install excluded HTTP/Git-based dependencies separately
        http_git_deps = read_http_git_requirements()
        if http_git_deps:
            print("Installing HTTP/Git-based dependencies separately...")
            subprocess.check_call(["pip", "install"] + http_git_deps)


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
    version="2.6.1",
    packages=find_packages(),  # Automatically detect all packages
    install_requires=read_requirements(),  # Load only standard dependencies
    cmdclass={
        "install": PostInstallCommand,
    },
)
