# BiliCore: A Framework for Building Dynamic RAG Implementations

## Overview

**BiliCore** is an open-source, reusable framework for rapidly building, testing, and deploying Retrieval-Augmented Generation (RAG) implementations. Developed as part of the Colorado Sustainability Hub initiative, this project aims to contribute to the AI/ML community by providing a flexible and extensible core library built with LangChain and LangGraph. The framework emphasizes dynamic chat history management, context window optimization, and ease of customization for various RAG use cases.

This project is funded by the [National Science Foundation](https://www.nsf.gov/) and [NAIRR Pilot](https://nairrpilot.org/), aligning with their mission to advance AI accessibility and sustainability.

---

## Key Features

### 1. **Dynamic RAG Configuration**
BiliCore empowers users to customize and test RAG configurations on the fly. Through an intuitive Streamlit interface, users can:
- Select and test various Large Language Models (LLMs)
- Define custom prompts for conversational agents
- Adjust RAG parameters, such as model size and context window
- Integrate external tools for enhanced conversational capabilities
- Manage chat history and state persistence with MongoDB or PostgreSQL
- Optimize memory management strategies for efficient token usage

### 2. **LangChain and LangGraph Integration**
BiliCore combines the power of [LangChain](https://www.langchain.com/) and [LangGraph](https://www.langchain.com/langgraph) to enable conversational AI with:
- Stateful conversation management using [MongoDB](https://www.mongodb.com/) or [PostgreSQL](https://www.postgresql.org/)
- Chat history summarization for optimal context window usage
- Customizable workflows for advanced conversational agents

### 3. **Chat History Persistence**
Efficiently manages conversation states by persisting chat history:
- **[MongoDBSaver](https://langchain-ai.github.io/langgraph/how-tos/persistence_mongodb/)**: Supports flexible document-based state storage
- **[PostgresSaver](https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres/)**: Provides relational database storage for robust checkpointing

### 4. **Extensible Tools Framework**
The system includes sample integrations for:
- [FAISS](https://github.com/facebookresearch/faiss) for vectorized similarity search
- [OpenSearch](https://opensearch.org/) for vectorized similarity search in AWS hosted OpenSearch
- [OpenWeather API](https://openweathermap.org/) for weather-related queries
- [Weather.gov](https://www.weather.gov/documentation/services-web-api) API for real-time weather data
- [SerpAPI](https://serpapi.com/) for web scraping and search
- MockTool for testing the integration of new tool APIs with BiliCore while still in development

### 5. **Summarization and Memory Management**
BiliCore supports dynamic memory management with:
- Trimming strategies to maintain manageable context windows
- Summarization to provide coherent conversation context while optimizing token usage

---

## Technical Details

### Deployment

#### Prerequisites

1. **Docker**: Ensure Docker is installed and running. For installation instructions, visit the [Docker documentation](https://docs.docker.com/get-docker/).
2. **Python**: Python 3.11+ is required.
3. **MongoDB/PostgreSQL**: A MongoDB or PostgreSQL instance is required for state persistence.
4. **Environment Variables**: Create a `secrets` file using the provided `scripts/development/secrets.template` for required environment variables.

#### Steps

1. **Start Docker Containers**
   ```bash
   cd scripts/development
   ./start-container
   ```

2. **Attach to Development Container**
   ```bash
   ./attach-container
   ```

3. **Install Dependencies**
   ```bash
   deps
   ```

4. **Run Streamlit Server**
   ```bash
   streamlit
   ```

---

## Features in Depth

### Memory Management
- **Trim Strategy**: Retain only the last `k` messages for simple memory management.
- **Summarization**: Aggregate older messages into a coherent summary to optimize token usage while maintaining context.

### Database Integration
- **MongoDBSaver**: Provides flexible NoSQL storage for chat histories.
- **PostgresSaver**: Supports relational storage with robust checkpointing.

### Tools Integration
- **FAISS**: Vectorized similarity search for document retrieval.
- **Weather APIs**: Real-time weather data from OpenWeather and Weather.gov.
- **SerpAPI**: Dynamic web search and scraping.
- **MockTool**: Test new tool APIs with BiliCore while still in development.
---

## Acknowledgments

This project is made possible through funding by the **[National Science Foundation (NSF)](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2318730)** and the **[NAIRR Pilot](https://nairrpilot.org/projects/awarded?_requestNumber=NAIRR240197)**. Their support has been instrumental in advancing AI accessibility and fostering innovation in sustainability-focused applications.

For more information, please visit the [Sustainability Hub Website](https://sustainabilityhub.co/).
