"""
Structure Analyzer.

Measures the distribution of context types and detects imbalance.
Uses simple threshold-based rules — no complex entropy for V0.1.

Detects:
  - Retrieval dominance (RAG flooding)
  - System prompt bloat
  - Memory explosion
  - Tool output sprawl
  - Missing context types
"""

from __future__ import annotations

from contextops.core.config import ContextOpsConfig
from contextops.core.models import (
    ContextBundle,
    ContextType,
    FindingSeverity,
    StructureFinding,
)

# ── Imbalance Issues ───────────────────────────────────────────────────────
ISSUES: dict[ContextType, dict[str, str | FindingSeverity]] = {
    ContextType.RETRIEVAL: {
        "issue": "Retrieval dominance",
        "detail": "RAG chunks consume {pct}% of context — likely noisy retrieval",
        "severity": FindingSeverity.HIGH,
    },
    ContextType.SYSTEM: {
        "issue": "System prompt bloat",
        "detail": "System prompt uses {pct}% of context — consider trimming instructions",
        "severity": FindingSeverity.MEDIUM,
    },
    ContextType.MEMORY: {
        "issue": "Memory explosion",
        "detail": "Memory entries consume {pct}% of context — prune old memories",
        "severity": FindingSeverity.HIGH,
    },
    ContextType.TOOL: {
        "issue": "Tool output sprawl",
        "detail": "Tool outputs use {pct}% of context — summarize tool responses",
        "severity": FindingSeverity.MEDIUM,
    },
}

# Minimum expected types for a "healthy" context
RECOMMENDED_MIN_TYPES: int = 2


def analyze_structure(bundle: ContextBundle, config: ContextOpsConfig | None = None) -> list[StructureFinding]:
    """
    Analyze the structural distribution of context types.

    Checks:
    1. Per-type ratio against config thresholds
    2. Whether context lacks diversity (too few types)

    Returns a list of StructureFinding, sorted by severity.
    """
    config = config or ContextOpsConfig.default()
    findings: list[StructureFinding] = []
    total_tokens = bundle.total_tokens

    if total_tokens == 0 or bundle.item_count <= 1:
        return findings

    # Calculate ratios per type
    type_tokens: dict[ContextType, int] = {}
    for item in bundle.items:
        type_tokens[item.type] = type_tokens.get(item.type, 0) + item.token_count

    # Configurable thresholds
    thresholds_map = {
        ContextType.RETRIEVAL: config.retrieval_max_ratio,
        ContextType.SYSTEM: config.system_max_ratio,
        ContextType.MEMORY: config.memory_max_ratio,
        ContextType.TOOL: config.tool_max_ratio,
    }

    # Check each type against thresholds
    for ctx_type, issue_info in ISSUES.items():
        tokens = type_tokens.get(ctx_type, 0)
        if tokens == 0:
            continue

        ratio = tokens / total_tokens
        max_ratio = thresholds_map[ctx_type]

        if ratio > max_ratio:
            findings.append(StructureFinding(
                issue=str(issue_info["issue"]),
                context_type=ctx_type,
                actual_ratio=ratio,
                threshold=max_ratio,
                severity=issue_info["severity"],  # type: ignore[arg-type]
            ))

    # Check for low type diversity
    unique_types = len(type_tokens)
    if unique_types < RECOMMENDED_MIN_TYPES and bundle.item_count > 1:
        # Find the dominant type
        dominant = max(type_tokens, key=lambda t: type_tokens[t])
        findings.append(StructureFinding(
            issue="Low context diversity",
            context_type=dominant,
            actual_ratio=type_tokens[dominant] / total_tokens,
            threshold=0.0,  # not a ratio threshold
            severity=FindingSeverity.LOW,
        ))

    # Sort by severity (critical first)
    severity_order = {
        FindingSeverity.CRITICAL: 0,
        FindingSeverity.HIGH: 1,
        FindingSeverity.MEDIUM: 2,
        FindingSeverity.LOW: 3,
    }
    findings.sort(key=lambda f: severity_order.get(f.severity, 99))

    return findings
