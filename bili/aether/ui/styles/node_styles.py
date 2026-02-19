"""
Agent role -> node style mapping.

Since AETHER roles are free-form strings (not enums), this module uses
keyword-based matching to assign colors and icons to nodes. The first
matching rule wins, so more specific keywords should appear before
more general ones.
"""

from typing import Dict, List, Tuple

# Priority-ordered list: (keyword_in_role, hex_color, emoji_icon)
# First match wins, so order matters.
ROLE_STYLE_RULES: List[Tuple[str, str, str]] = [
    # Orchestration / supervision
    ("supervisor", "#e67e22", "\u2699\ufe0f"),  # gear
    ("coordinator", "#e67e22", "\u2699\ufe0f"),
    ("lead", "#e67e22", "\u2699\ufe0f"),
    # Decision-making
    ("judge", "#e74c3c", "\u2696\ufe0f"),  # scales
    ("vote", "#e74c3c", "\u2696\ufe0f"),
    ("arbiter", "#e74c3c", "\u2696\ufe0f"),
    # Adversarial / debate (check before generic advocate)
    ("block", "#c0392b", "\u274c"),  # red X
    ("allow", "#27ae60", "\u2714\ufe0f"),  # checkmark
    ("advocate", "#f39c12", "\U0001f4ac"),  # speech bubble
    ("debater", "#f39c12", "\U0001f4ac"),
    # Review
    ("reviewer", "#3498db", "\U0001f50d"),  # magnifying glass
    # Policy / compliance
    ("policy", "#9b59b6", "\U0001f4dc"),  # scroll
    ("compliance", "#9b59b6", "\U0001f4dc"),
    # Analysis / research
    ("analyst", "#1abc9c", "\U0001f4ca"),  # bar chart
    ("researcher", "#1abc9c", "\U0001f50e"),  # magnifying glass tilted
    ("fact_checker", "#1abc9c", "\u2714\ufe0f"),
    # Synthesis / aggregation
    ("synthesizer", "#2ecc71", "\U0001f4dd"),  # memo
    ("aggregator", "#2ecc71", "\U0001f4dd"),
    ("writer", "#2ecc71", "\U0001f4dd"),
    # Escalation
    ("appeals", "#e91e63", "\u26a0\ufe0f"),  # warning
    ("escalation", "#e91e63", "\u26a0\ufe0f"),
    # Human-in-loop
    ("human", "#ff5722", "\U0001f464"),  # bust
    ("expert", "#ff5722", "\U0001f464"),
    # Technical
    ("engineer", "#20c997", "\U0001f527"),  # wrench
    ("security", "#17a2b8", "\U0001f6e1\ufe0f"),  # shield
    ("code", "#20c997", "\U0001f4bb"),  # laptop
    # Support / community
    ("support", "#6f42c1", "\U0001f4de"),  # telephone
    ("community", "#fd7e14", "\U0001f465"),  # busts
    # Classification
    ("classifier", "#17a2b8", "\U0001f3f7\ufe0f"),  # label
    # Processing
    ("processor", "#3498db", "\u2699\ufe0f"),
]

DEFAULT_COLOR = "#55bfef"
DEFAULT_ICON = "\U0001f916"  # robot face


def get_node_style(role: str) -> Dict[str, str]:
    """Return style info for an agent role.

    Uses keyword matching against the role string (case-insensitive).

    Returns:
        Dict with keys: color, icon, font_color.
    """
    role_lower = role.lower()
    for keyword, color, icon in ROLE_STYLE_RULES:
        if keyword in role_lower:
            return {"color": color, "icon": icon, "font_color": "#ffffff"}
    return {"color": DEFAULT_COLOR, "icon": DEFAULT_ICON, "font_color": "#ffffff"}


def build_node_css(role: str) -> Dict[str, str]:
    """Build a CSS-style dict for StreamlitFlowNode.style.

    Args:
        role: Agent role string.

    Returns:
        Dict compatible with streamlit-flow's node style parameter.
    """
    style_info = get_node_style(role)
    return {
        "background": style_info["color"],
        "color": style_info["font_color"],
        "border": f"2px solid {style_info['color']}",
        "borderRadius": "8px",
        "padding": "10px",
        "fontSize": "12px",
        "width": "160px",
        "textAlign": "center",
        "boxShadow": f"0 2px 8px rgba(0,0,0,0.3)",
    }
