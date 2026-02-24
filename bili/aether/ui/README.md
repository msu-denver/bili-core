# AETHER MAS Visualizer

A read-only Streamlit application that visualizes AETHER multi-agent system YAML configurations as interactive node graphs.

## Installation

```bash
# From the bili-core root directory
pip install --upgrade --force-reinstall "langgraph~=1.0.2"
pip install streamlit-flow-component --ignore-installed blinker
```

## Usage

```bash
/app/bili-core/venv/bin/streamlit run bili/aether/ui/app.py
```

Note: the command `streamlit` will not work due to an alias that starts BiliCore. This is not important to change because AETHER will become a part of BiliCore, but for dev purposes this is worth noting for now.

The app will be available at [http://localhost:8502](http://localhost:8502).

## Features

- **YAML Selection**: Dropdown to browse all example MAS configurations
- **Interactive Graph**: Pan, zoom, and click nodes to inspect agent properties
- **Auto-Layout**: Layout algorithm adapts to workflow type:
  - Sequential: horizontal chain
  - Hierarchical: multi-tier vertical layout
  - Supervisor: hub-and-spoke pattern
  - Consensus: circular arrangement
  - Parallel: three-row layout (coordinator, workers, aggregator)
  - Custom/Deliberative: edge-based layered layout
- **Properties Panel**: Click any node to view agent details (role, objective, model, capabilities, tools, etc.)
- **Metadata Bar**: Summary metrics for the selected MAS configuration
- **Color-Coded Nodes**: Nodes are colored by agent role for quick visual identification
- **Edge Styling**: Solid lines for channels, dashed for workflow edges, animated for conditional edges

## Architecture

```
bili/aether/ui/
├── app.py                          # Main Streamlit entry point
├── .streamlit/config.toml          # Theme configuration
├── styles/
│   ├── bili_core_theme.py          # Theme constants and CSS
│   └── node_styles.py             # Role-to-style mapping
├── converters/
│   └── yaml_to_graph.py           # YAML -> streamlit-flow converter
└── components/
    └── graph_viewer.py            # Graph rendering + properties panel
```

## Troubleshooting

- **streamlit-flow-component not found**: Run `pip install streamlit-flow-component`
- **No YAML files shown**: Ensure example configs exist in `bili/aether/config/examples/`
- **YAML fails to load**: Check that the YAML is valid against the MASConfig schema

## Known Issues

- **Browser console "Invalid color" warnings** (`widgetBackgroundColor`, `widgetBorderColor`, `skeletonBackgroundColor`): Harmless. A known Streamlit 1.51.x platform bug where internal theme proto fields default to empty strings; `streamlit-flow-component` exposes this when it propagates the parent theme to its React Flow iframe. No fix available without upgrading Streamlit.
- **`bootstrap.min.css.map` terminal warning**: Suppressed via logging configuration in `app.py`. Root cause is a source map file absent from the `streamlit-flow-component` package distribution. Source maps are optional browser developer tools with no functional impact.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PRESENTATION LAYER                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Vue 3 Frontend (Port 5173)                                   │  │
│  │  • Chat Interface                                             │  │
│  │  • File Upload                                                │  │
│  │  • WebSocket Client (VueUse)                                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
           │                                    ▲
           │ ① POST /tokens/text               │ ⑧ WebSocket
           │    or /tokens/document             │    Response Stream
           │    (REST)                          │
           ▼                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER (FastAPI)                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Backend (Port 8000)                                  │  │
│  │  ┌─────────────────┐        ┌──────────────────────────────┐ │  │
│  │  │  REST Endpoints │        │  WebSocket Handler           │ │  │
│  │  │  • /tokens/text │        │  • /ws/{sessionId}          │ │  │
│  │  │  • /tokens/doc  │◄──────►│  • Session management       │ │  │
│  │  │  • /health      │        │  • Stream results to client │ │  │
│  │  └─────────────────┘        └──────────────────────────────┘ │  │
│  │                                                                │  │
│  │  ┌──────────────────────────────────────────────────────────┐ │  │
│  │  │  Background Task Orchestrator                            │ │  │
│  │  │  • Generate sessionId                                    │ │  │
│  │  │  • Spawn async task                                      │ │  │
│  │  │  • Call RAG Layer ④                                      │ │  │
│  │  │  • Push results to WebSocket ⑧                           │ │  │
│  │  └──────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                    ② Returns { sessionId }                          │
│                    ③ Client connects WebSocket                      │
└─────────────────────────────────────────────────────────────────────┘
           │                                    ▲
           │ ④ POST /query                     │ ⑦ Returns
           │    { prompt, document_id }         │    { result, sources }
           ▼                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG ORCHESTRATION LAYER                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  LlamaIndex RAG Service                                       │  │
│  │  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐  │  │
│  │  │  Query Engine   │  │  Retriever   │  │  Prompt Builder │  │  │
│  │  │  • Route query  │  │  • Embed     │  │  • Augment      │  │  │
│  │  │  • Orchestrate  │  │  • Search    │  │  • Format       │  │  │
│  │  └─────────────────┘  └──────────────┘  └─────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
           │                 ▲                         │
           │                 │                         │
           │ ⑤ Similarity    │                         │ ⑥ POST /generate
           │    Search       │ Retrieved               │    { prompt + context }
           ▼                 │ Chunks                  ▼
┌──────────────────────┐     │              ┌──────────────────────────┐
│   VECTOR STORAGE     │     │              │    LLM RUNTIME           │
│  ┌────────────────┐  │     │              │  ┌────────────────────┐  │
│  │ ChromaDB       │  │─────┘              │  │ Ollama             │  │
│  │ (Port 8001)    │  │                    │  │ (Port 11434)       │  │
│  │                │  │                    │  │                    │  │
│  │ • Embeddings   │  │                    │  │ • Llama 3.1 8B    │  │
│  │ • Historical   │  │                    │  │ • Model runtime   │  │
│  │   RFI chunks   │  │                    │  │ • Streaming resp  │  │
│  └────────────────┘  │                    │  └────────────────────┘  │
│                      │                    │                          │
│  Persistent Volume   │                    │  Persistent Volume       │
│  chroma-data:/chroma │                    │  ollama-data:/.ollama    │
└──────────────────────┘                    └──────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE LAYER                          │
│  Docker Compose Network: rfi-network                                │
│  • frontend → backend (HTTP/WS)                                     │
│  • backend → chromadb (HTTP)                                        │
│  • backend → ollama (HTTP)                                          │
└─────────────────────────────────────────────────────────────────────┘
```
