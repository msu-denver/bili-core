"""
Agent role -> node style mapping.

Since AETHER roles are free-form strings (not enums), this module uses
keyword-based matching to assign colors to nodes. The first matching
rule wins, so more specific keywords should appear before more general ones.
"""

from typing import Dict, List, Tuple

# Priority-ordered list: (keyword_in_role, hex_color)
# First match wins, so order matters.
ROLE_STYLE_RULES: List[Tuple[str, str]] = [
    # Orchestration / supervision
    ("supervisor", "#e67e22"),
    ("coordinator", "#e67e22"),
    ("lead", "#e67e22"),
    # Decision-making
    ("judge", "#e74c3c"),
    ("vote", "#e74c3c"),
    ("arbiter", "#e74c3c"),
    # Adversarial / debate (check before generic advocate)
    ("block", "#c0392b"),
    ("allow", "#27ae60"),
    ("advocate", "#f39c12"),
    ("debater", "#f39c12"),
    # Review
    ("reviewer", "#3498db"),
    # Policy / compliance
    ("policy", "#9b59b6"),
    ("compliance", "#9b59b6"),
    # Analysis / research
    ("analyst", "#1abc9c"),
    ("researcher", "#1abc9c"),
    ("fact_checker", "#1abc9c"),
    # Synthesis / aggregation
    ("synthesizer", "#2ecc71"),
    ("aggregator", "#2ecc71"),
    ("writer", "#2ecc71"),
    # Escalation
    ("appeals", "#e91e63"),
    ("escalation", "#e91e63"),
    # Human-in-loop
    ("human", "#ff5722"),
    ("expert", "#ff5722"),
    # Technical
    ("engineer", "#20c997"),
    ("security", "#17a2b8"),
    ("code", "#20c997"),
    # Support / community
    ("support", "#6f42c1"),
    ("community", "#fd7e14"),
    # Classification
    ("classifier", "#17a2b8"),
    # Processing
    ("processor", "#3498db"),
]

DEFAULT_COLOR = "#55bfef"


def get_node_style(role: str) -> Dict[str, str]:
    """Return style info for an agent role.

    Uses keyword matching against the role string (case-insensitive).

    Returns:
        Dict with keys: color, font_color.
    """
    role_lower = role.lower()
    for keyword, color in ROLE_STYLE_RULES:
        if keyword in role_lower:
            return {"color": color, "font_color": "#ffffff"}
    return {"color": DEFAULT_COLOR, "font_color": "#ffffff"}


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
