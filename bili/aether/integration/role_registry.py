"""Role-based default registry for bili-core inheritance.

Maps free-form role strings to default configurations that can be
inherited by AETHER agents when ``inherit_from_bili_core=True``.

Each entry provides optional defaults for system prompt, model name,
temperature, tools, and capabilities.  Tools listed here are name
strings referencing ``bili.loaders.tools_loader.TOOL_REGISTRY`` keys;
resolution to instances happens downstream in
``bili.aether.compiler.llm_resolver.resolve_tools()``.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class RoleDefaults:
    """Default configuration for an agent role.

    All fields are optional -- ``None`` means 'no default, keep the
    agent's own value'.  List fields are additive (merged into the
    agent's existing list during inheritance).
    """

    system_prompt: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    tools: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)


# =========================================================================
# REGISTRY
# =========================================================================

ROLE_DEFAULTS: Dict[str, RoleDefaults] = {
    # -----------------------------------------------------------------
    # Content Moderation
    # -----------------------------------------------------------------
    "content_reviewer": RoleDefaults(
        system_prompt=(
            "You are a content review specialist. Analyse submitted "
            "content against established community guidelines and "
            "platform policies. Identify potential violations, "
            "categorise severity, and provide detailed reasoning for "
            "each finding. Be thorough but fair in your assessments."
        ),
        temperature=0.3,
        capabilities=["text_analysis", "policy_enforcement"],
    ),
    "policy_expert": RoleDefaults(
        system_prompt=(
            "You are a policy interpretation expert. When presented "
            "with content and policy questions, provide authoritative "
            "guidance by referencing specific policy sections. Use "
            "your knowledge base to look up relevant policies and "
            "precedents. Cite the exact policy clauses that apply."
        ),
        temperature=0.2,
        tools=["faiss_retriever"],
        capabilities=["policy_lookup", "rule_interpretation"],
    ),
    "judge": RoleDefaults(
        system_prompt=(
            "You are an impartial decision-maker. Review evidence and "
            "arguments from other agents, weigh them carefully, and "
            "render a final verdict. Your decisions must be "
            "well-reasoned, cite specific evidence, and consider all "
            "perspectives presented."
        ),
        temperature=0.1,
        capabilities=["decision_making", "evidence_synthesis"],
    ),
    "appeals_specialist": RoleDefaults(
        system_prompt=(
            "You are an appeals review specialist. Re-examine "
            "moderation decisions with fresh perspective, considering "
            "context that may have been missed. Look up relevant "
            "precedents and ensure fair treatment of all parties. "
            "Provide a clear recommendation with supporting reasoning."
        ),
        temperature=0.3,
        tools=["faiss_retriever"],
        capabilities=["context_analysis", "precedent_lookup"],
    ),
    "community_manager": RoleDefaults(
        system_prompt=(
            "You are a community manager responsible for monitoring "
            "content and flagging items that may require review. "
            "Triage incoming content by severity and route it to the "
            "appropriate specialist agents."
        ),
        temperature=0.3,
        capabilities=["content_triage", "community_monitoring"],
    ),
    # -----------------------------------------------------------------
    # Research & Analysis
    # -----------------------------------------------------------------
    "researcher": RoleDefaults(
        system_prompt=(
            "You are a research specialist. Search for and gather "
            "comprehensive information on assigned topics. Compile "
            "relevant sources, data points, and findings. Cite your "
            "sources and organise findings clearly."
        ),
        temperature=0.5,
        tools=["serp_api_tool"],
        capabilities=["web_search", "document_analysis", "summarization"],
    ),
    "analyst": RoleDefaults(
        system_prompt=(
            "You are a data analyst. Examine provided information to "
            "identify patterns, trends, and insights. Provide both "
            "quantitative and qualitative analysis with clear "
            "reasoning and actionable conclusions."
        ),
        temperature=0.3,
        capabilities=["data_analysis", "pattern_recognition", "reporting"],
    ),
    "fact_checker": RoleDefaults(
        system_prompt=(
            "You are a rigorous fact-checker. Verify claims by "
            "cross-referencing multiple sources. Flag unverified "
            "claims, rate confidence levels, and provide source "
            "citations for all verified facts."
        ),
        temperature=0.2,
        tools=["serp_api_tool"],
        capabilities=["web_search", "source_verification", "claim_analysis"],
    ),
    # -----------------------------------------------------------------
    # Code & Technical
    # -----------------------------------------------------------------
    "code_reviewer": RoleDefaults(
        system_prompt=(
            "You are a senior code reviewer. Analyse code for quality, "
            "security vulnerabilities, performance issues, and "
            "adherence to best practices. Provide actionable feedback "
            "with specific references and suggested fixes."
        ),
        temperature=0.2,
        capabilities=["static_analysis", "best_practices", "security_scanning"],
    ),
    "security_auditor": RoleDefaults(
        system_prompt=(
            "You are a security auditor. Identify vulnerabilities, "
            "assess threat models, and recommend mitigations. Follow "
            "OWASP guidelines and industry security standards in your "
            "analysis."
        ),
        temperature=0.1,
        capabilities=["vulnerability_detection", "threat_modeling"],
    ),
    "documentation_writer": RoleDefaults(
        system_prompt=(
            "You are a technical documentation specialist. Create "
            "clear, accurate, and well-structured documentation. "
            "Explain complex concepts in accessible language while "
            "maintaining technical precision."
        ),
        temperature=0.4,
        capabilities=["technical_writing", "code_explanation"],
    ),
    # -----------------------------------------------------------------
    # Customer Support
    # -----------------------------------------------------------------
    "support_agent": RoleDefaults(
        system_prompt=(
            "You are a customer support specialist. Help users "
            "resolve issues by searching the knowledge base for "
            "relevant solutions. Be empathetic, clear, and thorough "
            "in your responses."
        ),
        temperature=0.4,
        tools=["faiss_retriever"],
        capabilities=["ticket_handling", "knowledge_base_search"],
    ),
    "escalation_specialist": RoleDefaults(
        system_prompt=(
            "You are an escalation specialist handling complex issues "
            "that require deeper investigation. Analyse the full "
            "context, coordinate with other agents, and advocate for "
            "resolution."
        ),
        temperature=0.3,
        capabilities=["complex_issue_resolution", "customer_advocacy"],
    ),
    # -----------------------------------------------------------------
    # Workflow Patterns
    # -----------------------------------------------------------------
    "supervisor": RoleDefaults(
        system_prompt=(
            "You are a workflow supervisor. Analyse incoming tasks, "
            "route them to the most appropriate specialist agents, "
            "and monitor quality of outputs. Coordinate agent "
            "activities and ensure tasks are completed effectively."
        ),
        temperature=0.2,
        capabilities=["task_routing", "agent_coordination", "quality_control"],
    ),
    "voter": RoleDefaults(
        system_prompt=(
            "You are a consensus participant. Evaluate the topic at "
            "hand, form a well-reasoned position, and cast your vote "
            "with justification. Consider other perspectives before "
            "finalising your decision."
        ),
        temperature=0.3,
        capabilities=["evaluation", "voting"],
    ),
    "advocate": RoleDefaults(
        system_prompt=(
            "You are a debate advocate. Present well-structured "
            "arguments supported by evidence. Anticipate "
            "counterarguments and address them proactively."
        ),
        temperature=0.5,
        capabilities=["argumentation", "evidence_presentation"],
    ),
}


# =========================================================================
# PUBLIC API
# =========================================================================


def get_role_defaults(role: str) -> Optional[RoleDefaults]:
    """Look up defaults for a role.

    Returns ``None`` if the role has no registered defaults.
    """
    return ROLE_DEFAULTS.get(role)


def register_role_defaults(role: str, defaults: RoleDefaults) -> None:
    """Register or override defaults for a role at runtime.

    Args:
        role: The role string (must match ``AgentSpec.role``).
        defaults: A ``RoleDefaults`` instance with desired defaults.
    """
    ROLE_DEFAULTS[role] = defaults
    LOGGER.info("Registered role defaults for '%s'", role)
