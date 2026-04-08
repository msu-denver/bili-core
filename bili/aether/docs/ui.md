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
| `streamlit` (container alias) | [http://localhost:8501](http://localhost:8501) | Unified app — AETHER Multi-Agent + Single-Agent RAG. **Recommended.** |
| `/app/bili-core/venv/bin/streamlit run bili/streamlit_app.py` | [http://localhost:8501](http://localhost:8501) | Same unified app (explicit path) |

## Navigation

The unified app uses `st.navigation()` with two pages: **AETHER Multi-Agent** and **Single-Agent RAG**. The sidebar shows page-specific controls (config selection, model overrides, etc.). The two pages maintain independent session state — switching between them does not lose your selected graph or conversation history.

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
- **In-session thread management**: maintain multiple conversations per session; rename, delete, filter, and switch threads freely; threads persist across config switches
- **Live agent output panels**: expandable sections rendered as each agent node completes, collapsed automatically in conversation history
- **Execution timeline**: live horizontal chip row showing agent progress during streaming; click completed chips to expand that agent's output panel in history
- **Dual export**: **Export JSON** (machine-readable) and **Export Markdown** (human-readable) available side-by-side once the active thread has at least one turn
- **Stub mode**: configs without `model_name` execute without LLM calls; a banner in the sidebar indicates stub mode
- **Loading state**: "Initializing executor..." spinner during graph compilation on config selection
- **Config validation**: structural validation runs automatically on every config load and on every model patch. Errors block execution; warnings surface as non-blocking notices. Results are shown inline in the sidebar
- **Model picker**: switch any loaded config between stub mode and a real LLM without editing YAML (see [Model Settings](#model-settings))
- **Error isolation**: individual agent render failures surface as inline errors without aborting the turn
- **Config load errors**: surfaced in the main area so the sidebar remains clean
- **Emoji avatars**: 👤 for user messages, 🤖 for MAS responses

### Selecting a Config

**Example configs** — the selectbox lists all `.yaml` files from `bili/aether/config/examples/`, sorted alphabetically.

**Upload** — the file uploader accepts `.yaml` / `.yml` files directly in the browser:
- Validated at upload time: YAML syntax via `yaml.safe_load`, schema via Pydantic
- Successfully uploaded configs appear in the selectbox as `[Uploaded] filename.yaml`
- Uploads are session-only (never written to disk) and survive page switches and "New Conversation" resets

### Thread Management

The Chat sidebar maintains a **Conversations** panel below the export buttons. Each conversation is stored as an in-session thread with its own message history, config reference, and creation timestamp.

**Creating threads**

- **New Conversation** button creates a new thread and makes it the active one. The previous thread is preserved in the Conversations panel and can be re-activated at any time by clicking it. Checkpointed LangGraph state from the previous thread is no longer referenced by new turns.
- The first message sent in a session auto-creates a thread if none exists.
- Threads are named automatically: `"{mas_id} – {HH:MM:SS}"` (local time).

**Switching threads**

Click any thread in the Conversations panel to make it active. The active thread is highlighted in the primary colour. Its message history is immediately restored in the main chat area.

**Filtering**

A search box at the top of the Conversations panel narrows the list by case-insensitive name match as you type.

**Renaming**

Click the ✏️ icon next to a thread to enter rename mode. An inline text input appears pre-filled with the current name. Click ✓ to save or ✕ to cancel without changes. Clicking a different thread while in rename mode also dismisses the rename input.

**Deleting**

Click the 🗑️ icon to delete a thread. If the deleted thread was the active one, the next most recently created thread is activated automatically. If no threads remain, the active state is cleared and the Conversations panel is hidden.

**Persistence**

Threads persist across config switches within a session — switching from one YAML config to another does not clear the thread list. Threads are **lost on page reload** (in-session only). Cross-session persistence backed by a database is a planned future feature.

### Running a Conversation

1. Select a configuration from the sidebar dropdown
2. Wait for the "Initializing executor..." spinner to complete
3. Type a message in the chat input and press Enter
4. A chip timeline appears above the "Running MAS…" status widget (see [Execution Timeline](#execution-timeline))
5. Agent output panels appear live as each node in the graph completes
6. The status bar updates to "Complete" or "Execution failed" at the end of each turn
7. Previous turns are preserved in the conversation history above the input

### Execution Timeline

During each streaming turn, a horizontal row of chip buttons appears above the "Running MAS…" status widget — one chip per agent defined in the config:

| Chip | State | Behaviour |
|------|-------|-----------|
| ○ `agent_id` | Pending — not yet started | Disabled |
| ⟳ `agent_id` | Active — currently executing | Disabled |
| ✓ `agent_id` | Completed | Clickable |

**Live updates** — the chip row updates in-place (no full page re-render) as each agent node yields a result.

**Stored turns** — once a turn is complete, the chip row persists in conversation history with all executed agents shown as ✓. Agents defined in the config that did not run in that turn are not shown in stored-turn timelines.

**Click-to-expand** — clicking a ✓ chip in a stored turn's chip row expands that agent's output panel inside the "Agent trace" expander for one render cycle. To expand a different panel, click its chip. This is a single-cycle action: the selection is consumed on render and does not persist to other turns.

**Supervisor workflows** — if the same agent executes multiple times within one turn (common in supervisor graphs that loop back to a worker), duplicate IDs are deduplicated in the stored-turn timeline. The agent appears as a single ✓ chip.

### Model Settings

After a config is loaded, a **Model Settings** section appears in the sidebar below the config metadata. It lets you switch between stub mode and a real LLM without editing or reloading the YAML.

| Control | Behaviour |
|---------|-----------|
| **Model selectbox** | Lists all LLMs from `bili/iris/config/llm_config.py` (74 models), grouped by provider |
| **Apply to all** | Patches every agent in the loaded config with the selected model and reinitialises the executor. Conversation history is cleared |
| **Stub mode** | Clears `model_name` from all agents and reinitialises; stub mode indicator reappears |

**Important notes:**

- Patches are **in-memory only** — the original YAML on disk is never modified. Selecting a different config from the dropdown resets to the clean YAML
- The original loaded config is preserved as a base internally, so clicking **Apply to all** and then **Stub mode** (or vice versa) always patches from the unmodified YAML, not from a previously patched state
- Applying the same model twice does not trigger a redundant reinit (cache hit)
- If the config contains **pipeline agents**, a warning is shown because pipeline agents use their own internal node models and the top-level override may have no effect on them

### Conversation Controls

**New Conversation** — creates a new thread and makes it active. The previous thread is preserved in the Conversations panel and can be re-activated by clicking it. Checkpointed LangGraph state from the previous thread is no longer referenced by new turns; the new thread starts fresh.

**Export JSON / Export Markdown** — appear side-by-side once the active thread has at least one turn. See [Exporting a Conversation](#exporting-a-conversation) for details.

### Exporting a Conversation

Both export buttons appear below the "New Conversation" button once the active thread has at least one turn. They are not visible on an empty thread.

**Export JSON** downloads `aether_chat_{mas_id}_{thread_id[:8]}.json`:

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
      "agent_trace": [
        {
          "agent_id": "agent_a",
          "output": { "messages": [{ "type": "AIMessage", "content": "..." }] }
        }
      ]
    }
  ]
}
```

Failed turns include an additional `"error"` field with the exception message; their `agent_trace` contains any nodes that completed before the failure. The `agent_trace` key is always present (defaulting to `[]`) even for turns created before the field was introduced.

**Export Markdown** downloads `aether_chat_{mas_id}_{thread_id[:8]}.md`:

```markdown
# Conversation — simple_chain
Thread: a1b2c3d4-...
Exported: 2026-03-08T12:00:00+00:00

## Turn 1

**User:** Hello

**MAS Response:**

> **agent_a:** The agent's response text here
```

Each agent output is rendered as a blockquote. Multi-line agent outputs have every line prefixed with `>` so the full response is blockquoted. User input is inserted verbatim — any markdown syntax typed by the user (headings, links, etc.) will render as markdown in the exported file.

### Session State Keys (Chat)

| Key | Type | Description |
|-----|------|-------------|
| `chat_config` | `MASConfig` | Active configuration (may be model-patched) |
| `chat_config_base` | `MASConfig` | Original YAML-loaded config, preserved across model picker operations. Always the unpatched base |
| `chat_executor` | `MASExecutor` | Compiled executor (set after initialization) |
| `chat_yaml_path` | `str` | Cache key: bare file path, `"uploaded:{name}"`, or suffixed with `":model={model_id}"` / `":stub"` after model picker use |
| `chat_threads` | `Dict[str, Dict]` | All in-session threads keyed by UUID. Each value: `{name, messages, mas_id, created_at}`. Persists across config switches; lost on page reload |
| `chat_thread_id` | `str` | UUID of the active thread; cleared on config load, re-set when user clicks a thread or sends a message |
| `chat_editing_thread` | `Optional[str]` | Thread ID currently in rename mode; `None` when idle |
| `chat_uploaded_configs` | `Dict[str, MASConfig]` | Session-only uploaded configs |
| `chat_load_error` | `str` | Error message from last failed config load |
| `aether_executing_node` | `Optional[str]` | Agent ID currently executing during a live streaming turn; absent between turns |
| `aether_execution_trace` | `List[str]` | Ordered list of completed agent IDs for the current live turn; reset at turn start |
| `aether_selected_trace_node` | `Optional[Tuple[int, str]]` | `(turn_index, agent_id)` tuple set by a timeline chip click; the stored turn matching `turn_index` auto-expands that agent's panel on the next render cycle, then pops the value so it does not persist to other turns |

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

- `render_sidebar_content()` — chat sidebar controls (config selector, uploader, conversation buttons, thread list)
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
- **Timeline chips not appearing**: The chip row requires at least one agent in `config.agents`. If the config loaded correctly but no chips appear, check that the YAML defines agents and that the executor initialized successfully.
- **Thread list shows threads from a previous config**: This is expected behaviour — threads persist across config switches so you can re-activate prior conversations. To start completely fresh, delete all threads manually or reload the page.

## Known Issues & Limitations

**Platform bugs (Streamlit)**
- **Browser console "Invalid color" warnings** (`widgetBackgroundColor`, `widgetBorderColor`, `skeletonBackgroundColor`): Harmless. A known Streamlit 1.51.x platform bug where internal theme proto fields default to empty strings; `streamlit-flow-component` exposes this when it propagates the parent theme to its React Flow iframe. No fix available without upgrading Streamlit.
- **`bootstrap.min.css.map` terminal warning**: Suppressed via logging configuration in `app.py` and `chat_app.py`. Root cause is a source map file absent from the `streamlit-flow-component` package distribution. Source maps are optional browser developer tools with no functional impact.

**Thread management**
- **Threads are in-session only**: all thread history is lost on page reload. Cross-session persistence backed by a database is a planned future feature.

**Execution timeline**
- **Click-to-expand is a single-cycle action**: clicking a chip auto-expands the matching agent panel for one render cycle only. The selection is immediately cleared after rendering so it does not leak into other turns that share the same agent ID. Re-click the chip to expand again.
- **Supervisor loop deduplication**: when the same agent executes multiple times in one turn (e.g. a supervisor that routes back to a worker repeatedly), stored-turn timelines show the agent as a single ✓ chip. The full execution order is preserved in the `agent_trace` field of the JSON export.

**Exports**
- **Markdown export inserts user input verbatim**: any markdown syntax typed by the user (e.g. `# heading`, `[link](url)`) will render as formatted markdown in the exported `.md` file. This is intentional — users are exporting their own conversations — but worth noting if exports are shared.
- **Agent output serialization**: `BaseMessage` objects are serialized to `{"type": ..., "content": ...}` for export. Complex message metadata (tool call arguments, structured outputs, etc.) beyond the `.content` string is not preserved in the export.

**Model settings**
- **Pipeline agent override limitation**: pipeline agents use internal node models; applying a top-level model override via the Model Settings panel may have no effect on them. The UI displays a warning when pipeline agents are detected in the loaded config.
