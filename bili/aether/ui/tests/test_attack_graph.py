"""Tests for bili.aether.ui.components.attack_graph.

The attack-graph component wraps ``convert_mas_to_graph`` and the third-party
``streamlit_flow`` component. The pure helpers (``build_node_states`` and
``_apply_style_overrides``) are tested directly. ``render_attack_graph`` is
driven through ``AppTest.from_string`` so it executes inside a real Streamlit
runtime with the flow component mocked.

Streamlit UI modules cannot be imported at module level because doing so
triggers ``st.set_page_config()`` and other runtime side-effects.
"""

# pylint: disable=import-outside-toplevel, protected-access, reimported
# pylint: disable=duplicate-code

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

_FM = {
    "streamlit_flow": MagicMock(),
    "streamlit_flow.elements": MagicMock(),
    "streamlit_flow.state": MagicMock(),
}


# ---------------------------------------------------------------------------
# build_node_states
# ---------------------------------------------------------------------------


def _make_obs(agent_id, influenced=False, resisted=False, received_payload=False):
    """Build an AgentObservation-like object exposing attribute access."""
    return SimpleNamespace(
        agent_id=agent_id,
        influenced=influenced,
        resisted=resisted,
        received_payload=received_payload,
    )


def test_build_node_states_influenced_takes_priority():
    """influenced wins over resisted and received in the precedence chain."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui.components.attack_graph import build_node_states

        obs = _make_obs("a0", influenced=True, resisted=True, received_payload=True)
        assert build_node_states([obs]) == {"a0": "influenced"}


def test_build_node_states_resisted_over_received():
    """resisted wins over received when not influenced."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui.components.attack_graph import build_node_states

        obs = _make_obs("a0", resisted=True, received_payload=True)
        assert build_node_states([obs]) == {"a0": "resisted"}


def test_build_node_states_received_only():
    """received maps to the received state."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui.components.attack_graph import build_node_states

        obs = _make_obs("a0", received_payload=True)
        assert build_node_states([obs]) == {"a0": "received"}


def test_build_node_states_clean_default():
    """An observation with no flags set maps to clean."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui.components.attack_graph import build_node_states

        obs = _make_obs("a0")
        assert build_node_states([obs]) == {"a0": "clean"}


def test_build_node_states_accepts_dicts():
    """build_node_states supports dict-shaped observations."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui.components.attack_graph import build_node_states

        observations = [
            {
                "agent_id": "a0",
                "influenced": False,
                "resisted": True,
                "received_payload": True,
            },
            {
                "agent_id": "a1",
                "influenced": True,
                "resisted": False,
                "received_payload": True,
            },
        ]
        assert build_node_states(observations) == {
            "a0": "resisted",
            "a1": "influenced",
        }


# ---------------------------------------------------------------------------
# _apply_style_overrides
# ---------------------------------------------------------------------------


def _fake_node(node_id):
    """Build a minimal node object with an id and a style dict."""
    return SimpleNamespace(id=node_id, style={"background": "#fff"})


def test_apply_style_overrides_target_border():
    """The target node receives the red target border."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui.components import attack_graph as ag

        nodes = [_fake_node("a0"), _fake_node("a1")]
        result = ag._apply_style_overrides(nodes, "a0", None)
        assert result[0].style["border"] == ag._TARGET_BORDER
        assert result[0].style["borderRadius"] == ag._BORDER_RADIUS
        # Untargeted node keeps original style, no border key.
        assert "border" not in result[1].style
        # Original node objects are not mutated.
        assert "border" not in nodes[0].style


def test_apply_style_overrides_state_beats_target():
    """A post-run state override takes priority over target selection."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui.components import attack_graph as ag

        nodes = [_fake_node("a0")]
        result = ag._apply_style_overrides(nodes, "a0", {"a0": "resisted"})
        assert result[0].style["border"] == ag._RESISTED_BORDER


def test_apply_style_overrides_clean_state_no_border():
    """A clean state leaves the role-based default style unchanged."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui.components import attack_graph as ag

        nodes = [_fake_node("a0")]
        result = ag._apply_style_overrides(nodes, None, {"a0": "clean"})
        assert "border" not in result[0].style


def test_apply_style_overrides_influenced_and_received():
    """influenced and received map to their respective borders."""
    with patch.dict("sys.modules", _FM):
        from bili.aether.ui.components import attack_graph as ag

        nodes = [_fake_node("a0"), _fake_node("a1")]
        result = ag._apply_style_overrides(
            nodes, None, {"a0": "influenced", "a1": "received"}
        )
        assert result[0].style["border"] == ag._INFLUENCED_BORDER
        assert result[1].style["border"] == ag._RECEIVED_BORDER


# ---------------------------------------------------------------------------
# render_attack_graph
# ---------------------------------------------------------------------------


def _render_script(target="agent_0", node_states="None", click="agent_0"):
    """Build an AppTest script that drives render_attack_graph once."""
    return f"""
from unittest.mock import MagicMock as _Mock
from unittest.mock import patch as _patch
import streamlit as st
from bili.aether.ui.tests.conftest import make_test_config as mk
fm = {{
    "streamlit_flow": _Mock(),
    "streamlit_flow.elements": _Mock(),
    "streamlit_flow.state": _Mock(),
}}
with _patch.dict("sys.modules", fm):
    from bili.aether.ui.components import attack_graph as ag

    returned_state = _Mock()
    returned_state.selected_id = {click!r}
    mock_flow = _Mock(return_value=returned_state)
    mock_state_cls = _Mock(side_effect=lambda n, e: _Mock(selected_id=None))
    ag.streamlit_flow = mock_flow
    ag.StreamlitFlowState = mock_state_cls

    cfg = mk(mas_id="atk")
    clicked = ag.render_attack_graph(cfg, {target!r}, {node_states})
    st.markdown(f"clicked:{{clicked}}")
"""


def test_render_attack_graph_returns_clicked_agent():
    """Clicking an agent node returns that agent's id."""
    at = AppTest.from_string(_render_script(click="agent_0"))
    at.run()
    assert not at.exception
    assert "clicked:agent_0" in " ".join(m.value for m in at.markdown)


def test_render_attack_graph_ignores_non_agent_click():
    """A selected id that is not an agent returns None."""
    at = AppTest.from_string(_render_script(click="edge_x"))
    at.run()
    assert not at.exception
    assert "clicked:None" in " ".join(m.value for m in at.markdown)


def test_render_attack_graph_shows_legend_after_run():
    """When node_states is provided the propagation legend caption renders."""
    at = AppTest.from_string(
        _render_script(node_states="{'agent_0': 'influenced'}", click="None")
    )
    at.run()
    assert not at.exception
    assert any("Influenced" in c.value for c in at.caption)


def test_render_attack_graph_no_legend_before_run():
    """With node_states None no legend caption is shown."""
    at = AppTest.from_string(_render_script(node_states="None", click="None"))
    at.run()
    assert not at.exception
    assert not any("Influenced" in c.value for c in at.caption)
