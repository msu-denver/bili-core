"""
Theme constants matching bili-core's .streamlit/config.toml dark theme.

Extracted from the project root .streamlit/config.toml and extended
with graph-specific derived colors for the AETHER visualization.
"""

# =============================================================================
# CORE THEME (from .streamlit/config.toml)
# =============================================================================

PRIMARY = "#55bfef"
BACKGROUND = "#2a2a2a"
SECONDARY_BG = "#3b3b3b"
TEXT_COLOR = "#FFFFFF"
FONT = "sans serif"

# =============================================================================
# GRAPH-SPECIFIC COLORS
# =============================================================================

EDGE_CHANNEL_COLOR = "#55bfef"
EDGE_WORKFLOW_COLOR = "#aaaaaa"
EDGE_CONDITIONAL_COLOR = "#f39c12"
NODE_BORDER_COLOR = "#55bfef"
GRAPH_BACKGROUND = "#2a2a2a"

# Protocol-specific edge colors
PROTOCOL_COLORS = {
    "direct": "#55bfef",
    "broadcast": "#f39c12",
    "request_response": "#2ecc71",
    "pubsub": "#9b59b6",
    "competitive": "#e74c3c",
    "consensus": "#1abc9c",
}

# =============================================================================
# CUSTOM CSS
# =============================================================================

CUSTOM_CSS = """
<style>
    .properties-header {
        color: #55bfef;
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .prop-label {
        color: #aaaaaa;
        font-size: 0.85em;
        margin-bottom: 2px;
    }
    .prop-value {
        color: #ffffff;
        font-weight: 500;
    }
    .capability-badge {
        display: inline-block;
        background-color: #3b3b3b;
        border: 1px solid #55bfef;
        border-radius: 12px;
        padding: 2px 8px;
        margin: 2px;
        font-size: 0.8em;
        color: #55bfef;
    }
    .inheritance-badge {
        display: inline-block;
        background-color: rgba(85, 191, 239, 0.15);
        border: 1px solid #55bfef;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8em;
        color: #55bfef;
    }
    .supervisor-badge {
        display: inline-block;
        background-color: rgba(230, 126, 34, 0.15);
        border: 1px solid #e67e22;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8em;
        color: #e67e22;
    }
</style>
"""
