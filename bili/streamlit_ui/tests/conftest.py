"""Shared test fixtures and helpers for streamlit_ui tests."""


class FakeSessionState(dict):
    """Dict subclass supporting attribute access like Streamlit session_state."""

    def __getattr__(self, name):
        """Get attribute as dict key."""
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        """Set attribute as dict key."""
        self[name] = value

    def __delattr__(self, name):
        """Delete attribute as dict key."""
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
