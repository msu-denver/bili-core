"""Which upstream records are authoritative for a catalog provider type.

A community capability dataset is keyed by *its* notion of a provider, and
several of those notions are resale gateways and routers that re-list another
vendor's model under the same id.  A flat ``model_id -> record`` index over
such a dataset therefore resolves a model to whichever provider happens to
come first in iteration order, and that record can disagree with the vendor's
own on the facts that matter.  Measured against the upstream data, ``o3-mini``
is present under seven providers: the vendor's record says it is a reasoning
model that does not take ``temperature``, and a gateway's record says the
opposite and adds an image input modality the model does not accept.

So lookup here is **provider-scoped and never global**.  Each catalog provider
type maps to an explicit allowlist of upstream provider ids, and a model found
only outside that allowlist is treated as *not found* rather than as a weaker
match.  The allowlist is curated framework data, not something discovered:
deciding which upstream is authoritative for a provider is a trust judgement,
and a gateway that re-lists a model is not an authority on it.

Advisory-only families
----------------------
:data:`ADVISORY_ONLY_PROVIDER_TYPES` names the provider types whose catalog
``model_id`` is not a vendor model id at all.  For a deployment-based provider
the id is a name the operator chose for a deployment, so an upstream record
that happens to carry the same string describes a different thing, and a
disagreement with it cannot establish that the catalog is wrong.  Findings for
those types are capped below ``ERROR`` (see :mod:`.compare`); they are still
reported, because they are worth a maintainer's eye, but they never fail the
run.

Id normalisation
----------------
The same model is spelled differently by the catalog and by an upstream.  A
managed-inference id carries a region prefix, a vendor prefix, and a version
suffix that the upstream key may omit; a local runtime id carries a ``:tag``;
a generative-API id may carry a ``models/`` path prefix.  :func:`id_candidates`
returns the spellings to try, most specific first, so the first hit is the
closest match rather than an arbitrary one.  Normalisation only ever *widens*
the search; it never rewrites the catalog.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

#: Catalog provider type -> authoritative models.dev provider ids.
MODELS_DEV_PROVIDERS: Dict[str, Sequence[str]] = {
    "remote_openai": ("openai",),
    "remote_anthropic": ("anthropic",),
    "remote_azure_openai": ("azure", "azure-cognitive-services"),
    "remote_google_genai": ("google",),
    "remote_google_vertex": ("google-vertex", "google-vertex-anthropic"),
    "remote_aws_bedrock": ("amazon-bedrock",),
    "remote_mistral": ("mistral",),
    "remote_cohere": ("cohere",),
    "remote_deepseek": ("deepseek",),
    "remote_xai": ("xai",),
    "remote_groq": ("groq",),
    "local_ollama": ("ollama-cloud",),
}

#: Catalog provider type -> authoritative LiteLLM ``litellm_provider`` values.
LITELLM_PROVIDERS: Dict[str, Sequence[str]] = {
    "remote_openai": ("openai",),
    "remote_anthropic": ("anthropic",),
    "remote_azure_openai": ("azure", "azure_ai"),
    "remote_google_genai": ("gemini",),
    "remote_google_vertex": (
        "vertex_ai-language-models",
        "vertex_ai-anthropic_models",
        "vertex_ai-mistral_models",
        "vertex_ai-ai21_models",
        "vertex_ai-llama_models",
        "vertex_ai-chat-models",
    ),
    "remote_aws_bedrock": ("bedrock", "bedrock_converse"),
    "remote_mistral": ("mistral",),
    "remote_cohere": ("cohere", "cohere_chat"),
    "remote_deepseek": ("deepseek",),
    "remote_xai": ("xai",),
    "remote_groq": ("groq",),
    "local_ollama": ("ollama", "ollama_chat"),
}

#: Provider types whose catalog ``model_id`` is an operator-chosen deployment
#: name rather than a vendor model id.  A same-string upstream record is not
#: about the same object, so findings here never reach ``ERROR``.
ADVISORY_ONLY_PROVIDER_TYPES = frozenset({"remote_azure_openai"})

#: Provider types with no upstream listing by construction: subprocess tools
#: with no API, and local weights addressed by a path.  These are expected to
#: match nothing, and reporting them as gaps would be noise.
UNLISTED_PROVIDER_TYPES = frozenset(
    {
        "cli",
        "cli_claude_code",
        "cli_codex",
        "cli_gemini_cli",
        "local_llamacpp",
        "local_huggingface",
    }
)

#: Entries per family resolved to at least one authoritative record, measured
#: against the 2026-09-03 recorded slices.
#:
#: This is framework data rather than a test constant because two different
#: readers need it. A test reads it as a regression net: a change here that
#: stops resolving model ids drops a family below its floor. The LIVE check
#: reads it to notice that the upstreams themselves have moved out from under
#: the mapping -- a provider key renamed, an id scheme changed -- which parses
#: cleanly, resolves nothing, and would otherwise report "no divergence" from a
#: check that had silently stopped working. One table, so the two readings
#: cannot drift.
#:
#: Re-measure and update these deliberately when the fixtures are re-captured.
RECORDED_MATCH_FLOORS: Dict[str, int] = {
    "remote_anthropic": 7,
    "remote_aws_bedrock": 48,
    "remote_azure_openai": 11,
    "remote_cohere": 3,
    "remote_deepseek": 2,
    "remote_google_genai": 7,
    "remote_google_vertex": 5,
    "remote_groq": 2,
    "remote_mistral": 3,
    "remote_openai": 9,
    "remote_xai": 1,
    "local_ollama": 0,
}

_REGION_PREFIX = re.compile(r"^(?:us|eu|apac|us-gov|global)\.")
_VERSION_SUFFIX = re.compile(r"-v\d+(?::\d+)?$")


def id_candidates(provider_type: str, model_id: str) -> List[str]:
    """Return the spellings of *model_id* to look up, most specific first.

    The literal id is always first, so an exact upstream key always wins over
    a normalised one.  Each subsequent candidate drops one piece of provider
    decoration that an upstream key may not carry.

    :param provider_type: The catalog provider type key.
    :param model_id: The id the catalog declares.
    :returns: Candidate ids, de-duplicated, in decreasing specificity.
    :rtype: List[str]
    """
    candidates = [model_id]

    if provider_type == "remote_aws_bedrock":
        stripped = _REGION_PREFIX.sub("", model_id)
        if stripped != model_id:
            candidates.append(stripped)
        # Drop the vendor prefix ("anthropic.", "amazon.", ...) from whatever
        # the region strip produced.
        latest = candidates[-1]
        if "." in latest:
            candidates.append(latest.split(".", 1)[1])
        for candidate in list(candidates):
            without_version = _VERSION_SUFFIX.sub("", candidate)
            if without_version != candidate:
                candidates.append(without_version)

    if provider_type == "local_ollama" and ":" in model_id:
        candidates.append(model_id.split(":", 1)[0])

    if provider_type in ("remote_google_genai", "remote_google_vertex"):
        if model_id.startswith("models/"):
            candidates.append(model_id.split("/", 1)[1])

    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered
