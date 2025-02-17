<p align="center">
  <img src="bili/images/logo.png" alt="BiliCore Logo" width="150"/>
</p>

# BiliCore: A Framework for Benchmarking and Building Dynamic RAG Implementations

## Overview

**BiliCore** is an open-source, reusable framework designed to benchmark Large Language Models (LLMs) across cloud providers and local environments. It enables rapid, reproducible testing without requiring researchers to run models locally. Developed as part of the [**Colorado Sustainability Hub**](https://sustainabilityhub.co/). initiative, BiliCore contributes to the AI/ML community by providing a **flexible and extensible core library** built with **LangChain** and **LangGraph**. The framework emphasizes modular authentication, customizable tool selection, support for multiple LLMs across different providers, and a robust web interface in Streamlit that enables on-the-fly RAG customization during a user's session. This allows users to start a conversation with one LLM and switch to another mid-session without losing conversation history. Users can also test different personas using unique prompts, configure tools, adjust LLM parameters such as top-p, top-k, and seed values, and manage state with complete flexibility, including using truncation or summarization to preserve context when a customizable context window limit is reached.

This project is funded by the **[National Science Foundation (NSF)](https://www.nsf.gov/)** and the **[NAIRR Pilot](https://nairrpilot.org/)**, aligning with their mission to advance AI accessibility and sustainability.

---

## Key Features

### 1. **Benchmarking and RAG Configuration**
BiliCore empowers users to benchmark and customize RAG implementations through an intuitive interface:
- Select and test various **Large Language Models (LLMs)**
- Define **custom prompts** for conversational agents
- Adjust RAG parameters, such as model size and context window
- **Integrate external tools** for enhanced conversational capabilities
- **Manage chat history** and state persistence with MongoDB or PostgreSQL
- Optimize memory management strategies for efficient token usage

### 2. **Modular Authentication System**
BiliCore provides a flexible authentication system with **prebuilt authentication options**, including:
- **Firebase Authentication**: Utilize Firebase for user management and authentication.
- **Memory-Based Authentication**: A simple in-memory authentication suitable for testing and development.

To configure authentication:
1. In `bili/streamlit_app.py`, modify the `initialize_auth_manager` call to specify the desired authentication method.
2. If using Firebase, for instance, you can change `auth_provider_name` from `default`, which is the in-memory authentication provider, to `firebase`.

### 3. **Extensible Tools Framework**
BiliCore allows **customizable tool selection**, including:
- **[FAISS](https://github.com/facebookresearch/faiss)** for vectorized similarity search
- **[OpenSearch](https://opensearch.org/)** for vectorized search in AWS-hosted OpenSearch
- **[OpenWeather API](https://openweathermap.org/)** for weather-related queries
- **[Weather.gov API](https://www.weather.gov/documentation/services-web-api)** for real-time weather data
- **[SerpAPI](https://serpapi.com/)** for web search and scraping
- **MockTool** for testing new tool APIs before deployment

To use these tools:
1. Copy `scripts/development/secrets.template` to `scripts/development/secrets`.
2. Fill in values for tools, cloud providers, and LLMs in the `secrets` file.
3. For AWS services, store credentials in `env/bili_root/.aws/`.
4. For Google Cloud services, store credentials in `env/bili_root/.google/`.

### 4. **LangGraph-Powered Conversational AI**
BiliCore integrates **[LangGraph](https://www.langchain.com/langgraph)** to enable advanced conversational AI workflows:
- **Prebuilt multi-LLM capable RAG graph** for modularity and extensibility
- **Custom nodes** to extend graph functionality with personalized features
- **Flexible conversation flow control**, allowing structured interactions between LLMs

Users can extend the prebuilt LangGraph workflows by adding **custom nodes** in the `bili` module by modifying the call to `load_langgraph_agent` in `bili/loaders/langchain_loader.py`.

### 5. **Chat History and Memory Management**
Efficiently manages conversation states in LangGraph by persisting chat history with:
- **[MongoDB Checkpointer](https://langchain-ai.github.io/langgraph/how-tos/persistence_mongodb/)**: Supports flexible document-based state storage
- **[PostgreSQL Checkpointer](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/)**: Provides relational database storage for robust checkpointing
- **Trim Strategy**: Retains only the last `k` messages for simple memory management
- **Summarization Strategy**: Aggregates older messages into a coherent summary to optimize token usage

---

## Installation

### Prerequisites

1. **Docker**: Ensure Docker is installed and running. [Get Docker](https://docs.docker.com/get-docker/)
2. **Python**: Requires Python 3.11+. Python is provided as part of the Docker Compose setup in the `bili-core` container.
3. **MongoDB/PostgreSQL**: A MongoDB or PostgreSQL instance is required for state persistence. This is provided as part of the Docker Compose setup.
4. **Cloud Credentials**:
   - AWS keys must be placed in `env/bili_root/.aws/`
   - Google Cloud credentials must be placed in `env/bili_root/.google/`

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/msu-denver/bili-core.git
   cd bili-core
   ```
2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    pip install .
    ```
3. **Set Up Environment Variables**
    ```bash
    cp scripts/development/secrets.template scripts/development/secrets
    ```
   Edit `scripts/development/secrets` to include API keys for tools and LLMs that you plan to use.
4. **Start Docker Containers**
    ```bash
    cd scripts/development
    ./start-container[.bat]
    ```
5. **Attach to Development Container**
    ```bash
    cd scripts/development
    ./attach-container[.bat]
    ```
6. **Run BiliCore**
    ```bash
    streamlit
    ```
## Acknowledgments
This research is supported by the [National Science Foundation (NSF) (Grant No. 2318730)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2318730) and the [National Artificial Intelligence Research Resource (NAIRR) Pilot](https://nairrpilot.org/projects/awarded?_requestNumber=NAIRR240197). Their support has been instrumental in advancing AI accessibility and fostering innovation in sustainability-focused applications.

For more information, please visit the [Sustainability Hub Website](https://sustainabilityhub.co/).

