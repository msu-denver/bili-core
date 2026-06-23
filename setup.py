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
    extras_require={
        # ------------------------------------------------------------------ #
        # Provider extras — install the optional SDK for each new provider.  #
        # Usage: pip install bili-core[anthropic,mistral]                     #
        # ------------------------------------------------------------------ #
        "anthropic": ["langchain-anthropic>=0.3.0"],
        "mistral": ["langchain-mistralai>=0.2.0"],
        "cohere": ["langchain-cohere>=0.3.0"],
        "google-genai": ["langchain-google-genai>=2.0.0"],
        "deepseek": ["langchain-deepseek>=0.1.0"],
        "xai": ["langchain-xai>=0.2.0"],
        "groq": ["langchain-groq>=0.2.0"],
        # Convenience bundle for all new API providers
        "all-providers": [
            "langchain-anthropic>=0.3.0",
            "langchain-mistralai>=0.2.0",
            "langchain-cohere>=0.3.0",
            "langchain-google-genai>=2.0.0",
            "langchain-deepseek>=0.1.0",
            "langchain-xai>=0.2.0",
            "langchain-groq>=0.2.0",
        ],
        # ------------------------------------------------------------------ #
        # Surface extras — optional UI and API layers.                        #
        # ------------------------------------------------------------------ #
        # Streamlit web UI (bili/streamlit_app.py + bili/streamlit_ui/).
        # Usage: pip install bili-core[streamlit]
        "streamlit": [
            "streamlit~=1.51.0",
            "streamlit-flow-component>=1.3.0",
            "pillow~=10.4.0",
            "pandas==2.2.3",
        ],
        # Flask REST API (bili/flask_app.py + bili/flask_api/).
        # Usage: pip install bili-core[flask]
        "flask": [
            "flask~=3.1.2",
            "pyjwt==2.10.1",
        ],
        # ------------------------------------------------------------------ #
        # Security / adversarial testing (AEGIS).                             #
        # Usage: pip install bili-core[aegis]                                 #
        # ------------------------------------------------------------------ #
        "aegis": [
            "langchain-anthropic>=0.3.0",
        ],
        # ------------------------------------------------------------------ #
        # Tool-backend extras — heavy retrieval / search SDKs.               #
        # ------------------------------------------------------------------ #
        # FAISS in-memory vector search.
        # Usage: pip install bili-core[faiss]
        "faiss": [
            "faiss-cpu~=1.12.0",
            "sentence-transformers~=5.1.2",
        ],
        # Amazon OpenSearch vector search.
        # Usage: pip install bili-core[opensearch]
        "opensearch": [
            "opensearch-py~=3.0.0",
            "requests-aws4auth==1.3.1",
            "boto3~=1.40.19",
        ],
        # ------------------------------------------------------------------ #
        # Checkpointer-backend extras — database state persistence.          #
        # ------------------------------------------------------------------ #
        # MongoDB checkpointer (bili/iris/checkpointers/mongo_checkpointer.py).
        # Usage: pip install bili-core[mongo]
        "mongo": [
            "pymongo~=4.15.3",
            "motor~=3.7.0",
            "langgraph-checkpoint-mongodb~=0.3.0",
        ],
        # PostgreSQL checkpointer (bili/iris/checkpointers/pg_checkpointer.py).
        # Usage: pip install bili-core[postgres]
        "postgres": [
            "psycopg2~=2.9.11",
            "langgraph-checkpoint-postgres~=3.0.0",
        ],
        # ------------------------------------------------------------------ #
        # Local-model extras — heavy ML frameworks.                           #
        # ------------------------------------------------------------------ #
        # HuggingFace local inference (llm_loader + tokenizer_loader +
        # embeddings_loader sentence-transformer path).
        # Usage: pip install bili-core[huggingface]
        "huggingface": [
            "torch==2.6.0",
            "transformers~=4.57.1",
            "langchain-huggingface~=1.0.0",
            "sentence-transformers~=5.1.2",
            "accelerate~=1.11.0",
            "datasets~=4.3.0",
        ],
        # ------------------------------------------------------------------ #
        # Auth extras                                                          #
        # ------------------------------------------------------------------ #
        # Firebase auth provider.
        # Usage: pip install bili-core[firebase]
        "firebase": [
            "firebase-admin~=6.6.0",
        ],
        # ------------------------------------------------------------------ #
        # MCP client subsystem — lets bili-core agents consume tools from     #
        # MCP servers (stdio subprocess or HTTP/SSE transport).               #
        # Usage: pip install bili-core[mcp]                                   #
        # ------------------------------------------------------------------ #
        "mcp": ["mcp>=1.0"],
        # ------------------------------------------------------------------ #
        # Convenience bundle — installs all optional extras so that existing  #
        # consumers migrate from the old monolithic install with a single     #
        # word change:  pip install bili-core  →  pip install bili-core[all]  #
        # ------------------------------------------------------------------ #
        "all": [
            # Surfaces
            "streamlit~=1.51.0",
            "streamlit-flow-component>=1.3.0",
            "pillow~=10.4.0",
            "pandas==2.2.3",
            "flask~=3.1.2",
            "pyjwt==2.10.1",
            # AEGIS / security
            "langchain-anthropic>=0.3.0",
            # Provider extras
            "langchain-mistralai>=0.2.0",
            "langchain-cohere>=0.3.0",
            "langchain-google-genai>=2.0.0",
            "langchain-deepseek>=0.1.0",
            "langchain-xai>=0.2.0",
            "langchain-groq>=0.2.0",
            # Tools
            "faiss-cpu~=1.12.0",
            "sentence-transformers~=5.1.2",
            "opensearch-py~=3.0.0",
            "requests-aws4auth==1.3.1",
            "boto3~=1.40.19",
            # Checkpointers
            "pymongo~=4.15.3",
            "motor~=3.7.0",
            "langgraph-checkpoint-mongodb~=0.3.0",
            "psycopg2~=2.9.11",
            "langgraph-checkpoint-postgres~=3.0.0",
            # Local models
            "torch==2.6.0",
            "transformers~=4.57.1",
            "langchain-huggingface~=1.0.0",
            "accelerate~=1.11.0",
            "datasets~=4.3.0",
            # Auth
            "firebase-admin~=6.6.0",
            # MCP
            "mcp>=1.0",
        ],
    },
    cmdclass={
        "install": PostInstallCommand,
    },
)
