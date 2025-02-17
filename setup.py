import subprocess

from setuptools import setup
from setuptools.command.install import install


class PostInstallCommand(install):
    """
    Custom implementation of the install command to handle additional steps
    during the installation process.

    This class extends the standard installation procedure to include a step
    for installing pre-commit hooks automatically. It ensures that the
    pre-commit configurations are properly set up during the installation
    phase.
    """

    def run(self):
        # Install pre-commit hooks for all projects
        print("Installing pre-commit hooks...")
        # Install pre-commit hooks
        subprocess.check_call(["pre-commit", "install"])
        install.run(self)


setup(
    name="bili-core",
    version="1.0.0",
    packages=["bili"],  # Include all your project directories
    install_requires=[
        "pre-commit",  # Ensure pre-commit is installed
        "pylint",  # Include pylint as well
    ],
    cmdclass={
        "install": PostInstallCommand,
    },
)
