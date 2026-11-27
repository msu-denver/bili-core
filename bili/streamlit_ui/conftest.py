"""Collection guard for every test under bili/streamlit_ui/."""

import pytest

# The Streamlit UI is the optional [streamlit] extra, which does not install
# on every supported Python. Without it the test modules here, and the
# bili.streamlit_ui.query package their collection imports, fail at import, so
# skip the whole tree instead of erroring at collection. The guard sits at this
# level because a conftest inside query/tests/ is itself imported through the
# bili.streamlit_ui.query package and would fail before it could skip.
pytest.importorskip("streamlit", reason="requires the [streamlit] extra")
