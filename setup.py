import os
import subprocess

from setuptools import find_packages, setup
from setuptools.command.install import install


class PostInstallCommand(install):
    """
    Post-installation command to handle additional setup steps after the standard
    installation process.

    This class extends the standard `install` command to include extra functionality,
    such as setting up pre-commit hooks. It can be useful for projects that require
    additional configuration or third-party tool initialization after the installation
    process is completed.
    """

    def run(self):
        install.run(self)  # Run the standard install process
        print("Installing pre-commit hooks...")
        subprocess.check_call(["pre-commit", "install"])


def read_requirements():
    """
    Reads and processes the requirements file, returning a list of dependencies.

    The function locates the `requirements.txt` file, reads its contents, and processes
    each line to return a list of all valid dependencies. It excludes lines that
    are empty or start with a comment (`#`).

    :return: A list of processed dependency strings from the requirements file.
    :rtype: list
    """
    req_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(req_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="bili-core",
    version="1.0.0",
    packages=find_packages(),  # Automatically detect all packages
    install_requires=read_requirements(),  # Load dependencies from requirements.txt
    cmdclass={
        "install": PostInstallCommand,
    },
)
