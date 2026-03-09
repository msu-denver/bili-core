# AETHER UI — Visualizer & Chat

Streamlit application for inspecting and interacting with AETHER multi-agent system configurations. Provides two pages in a single app: a read-only graph **Visualizer** and a multi-turn **Chat** interface.

## Installation

```bash
# From the bili-core root directory
pip install --upgrade --force-reinstall "langgraph~=1.0.2"
pip install streamlit-flow-component --ignore-installed blinker
```

## Entry Points

| Command | URL | Description |
|---------|-----|-------------|
| `/app/bili-core/venv/bin/streamlit run bili/aether/ui/app.py` | [http://localhost:8502](http://localhost:8502) | Combined app — Visualizer **and** Chat. **Recommended.** |
| `/app/bili-core/venv/bin/streamlit run bili/aether/ui/chat_app.py` | [http://localhost:8501](http://localhost:8501) | Chat interface only — standalone entry point |

Note: the `streamlit` alias starts BiliCore rather than invoking Streamlit directly. Use the venv path above.

## Navigation

When running `app.py`, the sidebar shows a horizontal **Visualizer | Chat** radio at the top. Selecting a page switches the entire main area and the page-specific sidebar controls beneath the radio. The two pages maintain independent session state — switching from Visualizer to Chat and back does not lose your selected graph or conversation history.

---

## Visualizer Page

A read-only graph viewer for exploring AETHER YAML configurations as interactive node diagrams.

### Features

- **YAML Selection**: Dropdown to browse all example MAS configurations
- **Interactive Graph**: Pan, zoom, and click nodes to inspect agent properties
- **Auto-Layout**: Layout algorithm adapts to workflow type:
  - Sequential: horizontal chain
  - Hierarchical: multi-tier vertical layout
  - Supervisor: hub-and-spoke pattern
  - Consensus: circular arrangement
  - Parallel: three-row layout (coordinator, workers, aggregator)
  - Custom/Deliberative: edge-based layered layout
- **Properties Panel**: Click any node to view agent details (role, objective, model, capabilities, tools, etc.). Includes a per-agent **model override selector** — choose any model from the configured LLM registry to override the agent's `model_name` for the current session without editing the YAML
- **Metadata Bar**: Summary metrics for the selected MAS configuration
- **Color-Coded Nodes**: Nodes are colored by agent role for quick visual identification
- **Edge Styling**: Solid lines for channels, dashed for workflow edges, animated for conditional edges
- **Send to Chat →** button: loads the current config into the Chat page. If model overrides have been applied in the Properties Panel, those overrides are patched onto the config before it is sent — the Chat page receives the modified config, not the original YAML config

---

## Chat Page

A multi-turn conversation interface that executes a compiled MAS against user messages in real time.

### Features

- **Multi-turn conversation** with any compiled MAS configuration
- **Live agent output panels**: expandable sections rendered as each agent node completes, collapsed automatically in conversation history
- **Stub mode**: configs without `model_name` execute without LLM calls; a banner in the sidebar indicates stub mode
- **Loading state**: "Initializing executor..." spinner during graph compilation on config selection
- **Config validation**: structural validation runs automatically on every config load and on every model patch. Errors block execution; warnings surface as non-blocking notices. Results are shown inline in the sidebar
- **Model picker**: switch any loaded config between stub mode and a real LLM without editing YAML (see [Model Settings](#model-settings))
- **Error isolation**: individual agent render failures surface as inline errors without aborting the turn
- **Config load errors**: surfaced in the main area so the sidebar remains clean

### Selecting a Config

**Example configs** — the selectbox lists all `.yaml` files from `bili/aether/config/examples/`, sorted alphabetically.

**Upload** — the file uploader accepts `.yaml` / `.yml` files directly in the browser:
- Validated at upload time: YAML syntax via `yaml.safe_load`, schema via Pydantic
- Successfully uploaded configs appear in the selectbox as `[Uploaded] filename.yaml`
- Uploads are session-only (never written to disk) and survive page switches and "New Conversation" resets

### Running a Conversation

1. Select a configuration from the sidebar dropdown
2. Wait for the "Initializing executor..." spinner to complete
3. Type a message in the chat input and press Enter
4. Agent output panels appear live as each node in the graph completes
5. The status bar updates to "Complete" or "Execution failed" at the end of each turn
6. Previous turns are preserved in the conversation history above the input

### Model Settings

After a config is loaded, a **Model Settings** section appears in the sidebar below the config metadata. It lets you switch between stub mode and a real LLM without editing or reloading the YAML.

| Control | Behaviour |
|---------|-----------|
| **Model selectbox** | Lists all LLMs from `bili/config/llm_config.py` (74 models), grouped by provider |
| **Apply to all** | Patches every agent in the loaded config with the selected model and reinitialises the executor. Conversation history is cleared |
| **Stub mode** | Clears `model_name` from all agents and reinitialises; stub mode indicator reappears |

**Important notes:**

- Patches are **in-memory only** — the original YAML on disk is never modified. Selecting a different config from the dropdown resets to the clean YAML
- The original loaded config is preserved as a base internally, so clicking **Apply to all** and then **Stub mode** (or vice versa) always patches from the unmodified YAML, not from a previously patched state
- Applying the same model twice does not trigger a redundant reinit (cache hit)
- If the config contains **pipeline agents**, a warning is shown because pipeline agents use their own internal node models and the top-level override may have no effect on them

### Conversation Controls

**New Conversation** — clears the conversation history and assigns a new `thread_id`. Any checkpointed state from the previous conversation is no longer referenced; the next turn starts fresh.

**Export Conversation** — available once at least one turn exists. Downloads the full conversation as a JSON file named `aether_chat_{mas_id}_{thread_id[:8]}.json`.

Export schema:

```json
{
  "mas_id": "simple_chain",
  "thread_id": "a1b2c3d4-...",
  "exported_at": "2026-03-08T12:00:00+00:00",
  "turns": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2026-03-08T12:00:01+00:00",
      "turn_index": 0,
      "agent_outputs": [
        {
          "agent_id": "agent_a",
          "output": { "messages": [{ "type": "AIMessage", "content": "..." }] }
        }
      ]
    }
  ]
}
```

Failed turns include an additional `"error"` field with the exception message; their `agent_outputs` contain any nodes that completed before the failure.

### Session State Keys (Chat)

| Key | Type | Description |
|-----|------|-------------|
| `chat_config` | `MASConfig` | Active configuration (may be model-patched) |
| `chat_config_base` | `MASConfig` | Original YAML-loaded config, preserved across model picker operations. Always the unpatched base |
| `chat_executor` | `MASExecutor` | Compiled executor (set after initialization) |
| `chat_yaml_path` | `str` | Cache key: bare file path, `"uploaded:{name}"`, or suffixed with `":model={model_id}"` / `":stub"` after model picker use |
| `chat_history` | `List[Dict]` | All completed turns |
| `chat_thread_id` | `str` | UUID for checkpoint continuity; reset on "New Conversation" |
| `chat_uploaded_configs` | `Dict[str, MASConfig]` | Session-only uploaded configs |
| `chat_load_error` | `str` | Error message from last failed config load |

---

## Architecture

```
bili/aether/ui/
├── app.py                  # Combined entry point (Visualizer + Chat navigation)
├── chat_app.py             # Chat interface (standalone or embedded by app.py)
├── .streamlit/config.toml  # Theme configuration
├── styles/
│   ├── bili_core_theme.py  # Theme constants and CSS
│   └── node_styles.py      # Role-to-style mapping
├── converters/
│   └── yaml_to_graph.py    # MASConfig → streamlit-flow nodes/edges
└── components/
    └── graph_viewer.py     # Graph rendering + properties panel
```

`chat_app.py` exposes two public functions used by `app.py` when the Chat page is active:

- `render_sidebar_content()` — chat sidebar controls (config selector, uploader, conversation buttons)
- `render_main()` — chat main area (history display + chat input)

`chat_app.render_page()` remains the standalone entry point and handles its own page configuration.

---

## Troubleshooting

- **streamlit-flow-component not found**: Run `pip install streamlit-flow-component`
- **No YAML files shown**: Ensure example configs exist in `bili/aether/config/examples/`
- **YAML fails to load**: Check that the YAML is valid against the MASConfig schema
- **Chat spinner hangs**: Executor initialization compiles the LangGraph; for real-LLM configs this requires valid provider credentials. Check the terminal for authentication errors.
- **"Execution failed" in chat**: A runtime error occurred during graph execution. The full traceback is logged to the terminal. Select a stub config (any config without `model_name`) to verify the UI works without credentials.
- **`ImportError: cannot import name 'MODEL_KEEP_SENTINEL'`** (or similar after updating): The venv has a stale non-editable install of bili-core shadowing the local source. Fix with:
  ```bash
  pip uninstall bili-core -y && pip install -e /app/bili-core --ignore-installed blinker
  ```
- **Chat stays in stub mode after applying a model**: Confirm the terminal shows "Creating LLM for agent..." log lines. If the executor initialises twice (two "MASExecutor initialized" lines with the second showing no LLM creation), the site-packages install is stale — run the reinstall command above.
- **Model picker shows no effect on pipeline agents**: Pipeline agents (e.g. `researcher` in `pipeline_agents.yaml`) use internal node models; the top-level `model_name` field has no effect. The UI shows a warning when this case is detected.

## Known Issues

- **Browser console "Invalid color" warnings** (`widgetBackgroundColor`, `widgetBorderColor`, `skeletonBackgroundColor`): Harmless. A known Streamlit 1.51.x platform bug where internal theme proto fields default to empty strings; `streamlit-flow-component` exposes this when it propagates the parent theme to its React Flow iframe. No fix available without upgrading Streamlit.
- **`bootstrap.min.css.map` terminal warning**: Suppressed via logging configuration in `app.py` and `chat_app.py`. Root cause is a source map file absent from the `streamlit-flow-component` package distribution. Source maps are optional browser developer tools with no functional impact.
