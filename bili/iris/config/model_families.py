"""Which provider owns a bare model name that several providers catalog.

The problem
-----------
:data:`bili.iris.config.llm_config.LLM_MODELS` is keyed by provider type, and
the same ``model_id`` legitimately appears under more than one of them: a
first-party API and a cloud re-host list the identical id.  ``gpt-4o`` is
cataloged by both ``remote_openai`` and ``remote_azure_openai``;
``gemini-2.5-flash`` by both ``remote_google_genai`` and
``remote_google_vertex``.

A caller that names a model by id alone therefore names two entries.  Before
this module the tie was broken by *dict insertion order* -- the provider that
happens to appear first in the catalog literal won.  That is not a decision
anybody made and nothing documents it, so the answer changed shape whenever
the catalog was reordered, and it silently contradicted the resolver's own
heuristic layer, which already states a preference for the same families
(``bili.aether.compiler.llm_resolver._HEURISTIC_RULES`` routes a bare
``gemini-`` id to the Developer API, while the catalog lookup that runs
*before* it routed the cataloged ones to Vertex).

Name-only selection is not an edge case.  ``AgentSpec`` carries
``model_name`` and ``fallback_models`` and has no provider field at all, so
for a declarative multi-agent run the model name is the *only* channel
through which a provider can be chosen.

The rule
--------
:data:`MODEL_FAMILY_OWNERS` states, once, which provider owns each model
family whose names collide.  The preference is the *publisher* of the family
rather than a re-host, because a bare canonical id is the publisher's own
spelling: a caller who types ``gpt-4o`` has typed OpenAI's id for it, and a
caller on a re-host reaches it by qualifying the name or by passing
``provider_type`` to :func:`bili.iris.loaders.llm_loader.load_model`
directly.

Matching is **longest prefix wins**, so the rules are order-independent and a
narrower rule cannot be shadowed by a broader one that happens to be written
first.  One family is owned by the re-host and says so: ``gpt-35-turbo`` is an
Azure *deployment* spelling (deployment names historically could not contain a
dot), and the direct API serves that model as ``gpt-3.5-turbo``, so the entry
that works for a bare ``gpt-35-turbo`` is the Azure one.

What this module deliberately does not do
-----------------------------------------
It resolves a **collision** and nothing else.  A model id that only one
provider catalogs already has one answer, and a preference must never move it
to a provider whose catalog does not list it: routing a bare
``gemini-2.5-pro`` to the Developer API because Google publishes Gemini would
name a model that provider has no entry for.  :func:`disambiguate` therefore
only ever returns a provider drawn from the candidates it was given, and
falls back to the caller's existing behaviour when the owner is not among
them.

An uncovered collision is not a hard failure either.  It resolves as it did
before (the first candidate) and is reported as ``ambiguous`` so the caller
can log it, because refusing at run time would break a working deployment
over a catalog edit somebody else made.  The build is where that gap is
loud: the tests derive every collision from the catalog and fail when one has
no rule.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

from bili.iris.config.llm_config import LLM_MODELS

__all__ = [
    "FamilyRule",
    "MODEL_FAMILY_OWNERS",
    "catalog_providers_for",
    "colliding_model_ids",
    "disambiguate",
    "owning_provider",
    "verify_family_owner_table",
]


class FamilyRule(NamedTuple):
    """One row of :data:`MODEL_FAMILY_OWNERS`.

    :param prefix: The lower-cased model-id prefix the rule claims.
    :param provider_type: The ``LLM_MODELS`` key that owns the family.
    :param reason: Why that provider owns it, phrased for a log line a user
        reads when a bare name was disambiguated for them.
    """

    prefix: str
    provider_type: str
    reason: str


#: The ONE table.  Every model family whose bare ids are cataloged by more
#: than one provider names its owner here, with the reason.  Longest matching
#: prefix wins, so the order below is presentation only.
#:
#: A rule is consulted ONLY when a name collides; a prefix that also spans
#: names cataloged by a single provider is harmless, because those names
#: never reach :func:`disambiguate`.
MODEL_FAMILY_OWNERS: Tuple[FamilyRule, ...] = (
    FamilyRule(
        "gpt-35-turbo",
        "remote_azure_openai",
        "this spelling is an Azure deployment name; the direct API serves "
        "the same model as 'gpt-3.5-turbo'",
    ),
    FamilyRule(
        "gpt-",
        "remote_openai",
        "OpenAI publishes the GPT family and a bare id is its own spelling; "
        "Azure OpenAI re-hosts it",
    ),
    FamilyRule(
        "o1",
        "remote_openai",
        "OpenAI publishes the o-series and a bare id is its own spelling; "
        "Azure OpenAI re-hosts it",
    ),
    FamilyRule(
        "o3",
        "remote_openai",
        "OpenAI publishes the o-series and a bare id is its own spelling; "
        "Azure OpenAI re-hosts it",
    ),
    FamilyRule(
        "gemini-",
        "remote_google_genai",
        "the Google AI Developer API is the direct route for a bare Gemini "
        "id; Vertex AI is reached by an explicit provider or a qualified name",
    ),
)


def verify_family_owner_table() -> None:
    """Refuse a table that cannot do its job, at import.

    The table is shipped data rather than user input, so the only way to trip
    this is an edit to this repository, and the failure lands on whoever made
    it.  Three faults are checked, each of which would otherwise be silent:

    - an empty prefix, which would claim every model,
    - a duplicate prefix, which would make one of the two rows unreachable,
    - a ``provider_type`` that is not a key of ``LLM_MODELS``, which would
      make the rule match and then never apply (:func:`disambiguate` only
      returns a provider that catalogs the model), so the collision would
      quietly keep resolving by catalog order.

    A row whose owner is real but which covers no live collision is NOT a
    fault here: the catalog can legitimately lose a collision, and a stale
    row does not mis-route anything.  The tests fail the build on one.

    :raises ValueError: If any rule is malformed.
    """
    seen: Dict[str, FamilyRule] = {}
    for rule in MODEL_FAMILY_OWNERS:
        if not rule.prefix:
            raise ValueError(
                "MODEL_FAMILY_OWNERS: a rule declares an empty prefix, which "
                "would claim every model name."
            )
        if rule.prefix != rule.prefix.lower():
            raise ValueError(
                f"MODEL_FAMILY_OWNERS: prefix {rule.prefix!r} is not lower-cased; "
                f"matching is case-insensitive and compares against a lower-cased id."
            )
        if rule.prefix in seen:
            raise ValueError(
                f"MODEL_FAMILY_OWNERS: prefix {rule.prefix!r} is declared twice "
                f"(owners {seen[rule.prefix].provider_type!r} and "
                f"{rule.provider_type!r}); one row would be unreachable."
            )
        if rule.provider_type not in LLM_MODELS:
            raise ValueError(
                f"MODEL_FAMILY_OWNERS: prefix {rule.prefix!r} names provider "
                f"{rule.provider_type!r}, which is not a key of LLM_MODELS; the "
                f"rule would match and never apply."
            )
        if not rule.reason:
            raise ValueError(
                f"MODEL_FAMILY_OWNERS: prefix {rule.prefix!r} declares no reason; "
                f"the reason is what a caller logs when a name is disambiguated."
            )
        seen[rule.prefix] = rule


def owning_provider(model_id: str) -> Optional[FamilyRule]:
    """Return the rule claiming *model_id*, or ``None``.

    Longest matching prefix wins, so a narrower family rule is never shadowed
    by a broader one regardless of the order they are written in.

    :param model_id: A bare model id (matched case-insensitively).
    :returns: The winning :class:`FamilyRule`, or ``None`` when no rule
        claims the id.
    """
    lower = model_id.lower()
    best: Optional[FamilyRule] = None
    for rule in MODEL_FAMILY_OWNERS:
        if lower.startswith(rule.prefix) and (
            best is None or len(rule.prefix) > len(best.prefix)
        ):
            best = rule
    return best


def catalog_providers_for(name: str) -> Tuple[str, ...]:
    """Return every provider type whose catalog carries *name*, in catalog order.

    A match is on the entry's ``model_id`` or its display ``model_name``,
    which is the same pair the resolver looks a name up by.  Membership is
    reported once per provider even when a provider lists the name twice.

    :param name: A model id or display name.
    :returns: The provider types cataloging it, in ``LLM_MODELS`` order.
    """
    providers: List[str] = []
    for provider_type, provider_info in LLM_MODELS.items():
        for entry in provider_info.get("models", []):
            if name in (entry.get("model_id", ""), entry.get("model_name", "")):
                providers.append(provider_type)
                break
    return tuple(providers)


def colliding_model_ids() -> Dict[str, Tuple[str, ...]]:
    """Return every model id cataloged by more than one provider.

    Derived from the catalog rather than listed, so a collision introduced by
    a catalog edit is visible to the tests that read this.

    :returns: A mapping of model id to the provider types cataloging it, in
        ``LLM_MODELS`` order.
    """
    seen: Dict[str, List[str]] = {}
    for provider_type, provider_info in LLM_MODELS.items():
        for entry in provider_info.get("models", []):
            model_id = entry.get("model_id", "")
            if not model_id:
                continue
            providers = seen.setdefault(model_id, [])
            if provider_type not in providers:
                providers.append(provider_type)
    return {
        model_id: tuple(providers)
        for model_id, providers in seen.items()
        if len(providers) > 1
    }


def disambiguate(
    model_id: str, candidates: Tuple[str, ...]
) -> Optional[Tuple[str, str]]:
    """Pick the owning provider from *candidates*, or ``None``.

    The returned provider is always drawn from *candidates*: a preference may
    reorder the providers that catalog a model and may never introduce one
    that does not.  ``None`` means the caller keeps whatever it would have
    done without a preference.

    :param model_id: The bare model id the candidates all catalog.
    :param candidates: The provider types cataloging *model_id*.
    :returns: ``(provider_type, reason)`` when a rule claims the id and its
        owner is among *candidates*; ``None`` otherwise.
    """
    rule = owning_provider(model_id)
    if rule is None or rule.provider_type not in candidates:
        return None
    return rule.provider_type, rule.reason


verify_family_owner_table()
