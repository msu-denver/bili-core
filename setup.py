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


# ---------------------------------------------------------------------------
# Optional extras
#
# Each entry is the single source of truth for its package list.  The [all]
# convenience bundle is composed from the individual extras below rather than
# re-listing version strings, so bumping a pin in one place keeps everything
# consistent.
#
# Install a specific extra:   pip install bili-core[streamlit]
# Install everything:         pip install bili-core[all]
# ---------------------------------------------------------------------------

_EXTRAS = {
    # ------------------------------------------------------------------ #
    # Provider extras — install the optional SDK for each new provider.  #
    # Usage: pip install bili-core[anthropic,mistral]                     #
    # ------------------------------------------------------------------ #
    "anthropic": ["langchain-anthropic~=1.0.0"],
    "mistral": ["langchain-mistralai>=0.2.0"],
    "cohere": ["langchain-cohere>=0.3.0"],
    "google-genai": ["langchain-google-genai>=2.0.0"],
    "deepseek": ["langchain-deepseek>=0.1.0"],
    "xai": ["langchain-xai>=0.2.0"],
    "groq": ["langchain-groq>=0.2.0"],
    # Convenience bundle for all API providers
    "all-providers": [
        "langchain-anthropic~=1.0.0",
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
    # AEGIS itself is lean-core-importable; this extra provides the       #
    # cross-model Anthropic support used in some attack strategies.       #
    # ------------------------------------------------------------------ #
    "aegis": [
        "langchain-anthropic~=1.0.0",
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
        "botocore~=1.40.19",
    ],
    # ------------------------------------------------------------------ #
    # Checkpointer-backend extras — database state persistence.          #
    # ------------------------------------------------------------------ #
    # MongoDB checkpointer (bili/iris/checkpointers/mongo_checkpointer.py).
    # Usage: pip install bili-core[mongo]
    "mongo": [
        "pymongo~=4.15.3",
        "motor~=3.7.0",
        "langchain-mongodb==0.8.0",
        "langgraph-checkpoint-mongodb~=0.3.0",
    ],
    # PostgreSQL checkpointer (bili/iris/checkpointers/pg_checkpointer.py).
    # Usage: pip install bili-core[postgres]
    "postgres": [
        "psycopg2~=2.9.11",
        "psycopg[binary]>=3.2",
        "psycopg-pool>=3.2",
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
        "torchaudio==2.6.0",
        "torchvision==0.21.0",
        "transformers~=4.57.1",
        "langchain-huggingface~=1.0.0",
        "sentence-transformers~=5.1.2",
        "accelerate~=1.11.0",
        "datasets~=4.3.0",
        "huggingface_hub~=0.34.0",
        "optimum~=2.0.0",
    ],
    # llama.cpp local inference (bili/iris/loaders/llm_loader.py LlamaCpp path).
    # Usage: pip install bili-core[llamacpp]
    "llamacpp": [
        "llama-cpp-python==0.3.7",
    ],
    # Ollama local server inference (bili/iris/providers/ollama_provider.py).
    # Talks to a running Ollama daemon over HTTP; native tool calling for
    # tool-capable models. Usage: pip install bili-core[ollama]
    "ollama": [
        "langchain-ollama>=0.2.0",
    ],
    # Full ML stack: Keras, TensorFlow, scikit-learn (AEGIS attack strategies,
    # baseline runners, and advanced embedding backends).
    # Usage: pip install bili-core[ml]
    "ml": [
        "keras~=3.12.0",
        "tensorflow~=2.18.0",
        "tf-keras~=2.18.0",
        "scikit-learn~=1.7.2",
        "ml-dtypes~=0.5.1",
        "numpy==1.26.4",
        "opencv-python==4.10.0.84",
        "pi-heif==0.21.0",
    ],
    # Document-processing tools (PDFs, DOCX, OCR, HTML extraction).
    # Used by AEGIS attack strategies and bili-core tool integrations.
    # Usage: pip install bili-core[docs]
    "docs": [
        "beautifulsoup4==4.12.3",
        "nltk==3.9.1",
        "openpyxl==3.1.5",
        "pypdf~=6.1.3",
        "python-docx~=1.2.0",
        "rapidocr-onnxruntime==1.3.24",
        "textract==1.5.0",
        "unstructured[all-docs]~=0.18.15",
        "unstructured-client==0.42.3",
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
    # MCP subsystem — two directions:                                     #
    # Client: consume tools from external MCP servers (#205).            #
    # Server: expose an agent's tools as an ephemeral MCP server for     #
    #   MCP-capable CLI models (#311).                                   #
    # Both require the mcp SDK; the server side additionally needs       #
    # uvicorn to run the ephemeral SSE server.                           #
    # Usage: pip install bili-core[mcp]                                  #
    # ------------------------------------------------------------------ #
    "mcp": ["mcp>=1.0", "uvicorn>=0.30"],
    # ------------------------------------------------------------------ #
    # Development tooling.                                                 #
    # Usage: pip install bili-core[dev]                                   #
    # ------------------------------------------------------------------ #
    "dev": [
        "autoflake==2.3.1",
        "black==24.10.0",
        "isort==5.13.2",
        "mongomock==4.3.0",
        "pre-commit==4.0.1",
        "pylint==3.3.1",
        "pylint-pydantic==0.3.2",
        "pympler==1.1",
        "pytest~=8.0.0",
        "pytest-cov~=7.0.0",
        "setuptools~=65.5.0",
        "watchdog==5.0.3",
    ],
}

# [all] is composed from every RUNTIME extra above.  This single source of
# truth means a pin bump in one extra propagates automatically; no separate
# copy to forget.  Two extras are excluded:
#   [all-providers] -- omitted to avoid duplicating individual provider entries.
#   [dev]           -- excluded because [all] is a runtime bundle; development
#                      tooling (pylint, black, pytest, …) should not be pulled
#                      in as transitive dependencies by consumers that install
#                      bili-core[all].  Install [dev] separately when needed.
_EXTRAS["all"] = sorted(
    {
        pkg
        for key, pkgs in _EXTRAS.items()
        if key not in ("all-providers", "dev")
        for pkg in pkgs
    }
)


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
    install_requires=read_requirements(),  # Load only lean-core dependencies
    extras_require=_EXTRAS,
    cmdclass={
        "install": PostInstallCommand,
    },
)
