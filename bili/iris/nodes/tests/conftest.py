"""Root conftest: break circular import between nodes and loaders.

bili.iris.nodes.__init__ -> prepare_llm_config_node -> llm_loader
-> bili.iris.loaders.__init__ -> langchain_loader -> nodes (cycle).

We pre-seed sys.modules with a lightweight stub for the loaders
package so the eager __init__ import does not trigger the cycle.
"""

import sys
import types

_LOADERS = "bili.iris.loaders"
_LANGCHAIN = "bili.iris.loaders.langchain_loader"

if _LOADERS not in sys.modules:
    _pkg = types.ModuleType(_LOADERS)
    _pkg.__path__ = []
    _pkg.__package__ = _LOADERS
    sys.modules[_LOADERS] = _pkg

if _LANGCHAIN not in sys.modules:
    sys.modules[_LANGCHAIN] = types.ModuleType(_LANGCHAIN)

# Also stub llm_loader so prepare_llm_config_node can import
# prepare_runtime_config without pulling in the full loader chain.
_LLM_LOADER = "bili.iris.loaders.llm_loader"
if _LLM_LOADER not in sys.modules:
    _llm_mod = types.ModuleType(_LLM_LOADER)
    _llm_mod.prepare_runtime_config = lambda **kw: {}  # type: ignore
    sys.modules[_LLM_LOADER] = _llm_mod
