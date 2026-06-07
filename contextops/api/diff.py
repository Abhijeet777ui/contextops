"""
ContextOps Diff API.

Computes a deterministic delta between two context payloads.
Provides the data model and logic for `contextops diff`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from contextops.api.inspect import inspect_context
from contextops.core.models import AnalysisResult, Recommendation


@dataclass
class ContextDiffResult:
    """The computed difference between two context analysis results."""
    # Source results
    result_a: AnalysisResult
    result_b: AnalysisResult

    # Numeric Deltas (B - A)
    score_delta: int
    token_delta: int
    waste_delta: int
    cost_delta: float

    # Structure Deltas (B - A)
    structure_delta: dict[str, float]

    # Recommendation Lifecycle
    resolved_recommendations: list[Recommendation] = field(default_factory=list)
    new_recommendations: list[Recommendation] = field(default_factory=list)
    persisting_recommendations: list[Recommendation] = field(default_factory=list)

    @property
    def net_impact(self) -> str:
        """Categorical summary of the overall change."""
        if self.score_delta > 0:
            return "IMPROVEMENT"
        elif self.score_delta < 0:
            return "DEGRADATION"
        else:
            return "NEUTRAL"


def get_recommendation_id(rec: Recommendation) -> str:
    """
    Generate a deterministic, stable ID for a recommendation.
    
    This is critical for the set-based diff logic. We hash the normalized issue string.
    Do NOT use fuzzy matching or ML embeddings here.
    """
    # Normalize: lowercase and collapse all whitespace to single spaces
    normalized = re.sub(r'\s+', ' ', rec.issue.lower().strip())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]


def diff_contexts(
    raw_input_a: str | list[dict[str, Any]] | dict[str, Any],
    raw_input_b: str | list[dict[str, Any]] | dict[str, Any],
) -> ContextDiffResult:
    """
    Compare two context payloads and return a deterministic diff result.
    """
    result_a = inspect_context(raw_input_a)
    result_b = inspect_context(raw_input_b)

    return diff_analysis_results(result_a, result_b)


def diff_analysis_results(result_a: AnalysisResult, result_b: AnalysisResult) -> ContextDiffResult:
    """Compare two pre-computed AnalysisResult objects."""
    
    # 1. Numeric Deltas
    score_delta = result_b.score - result_a.score
    token_delta = result_b.token_breakdown.total_tokens - result_a.token_breakdown.total_tokens
    waste_delta = result_b.token_breakdown.wasted_tokens - result_a.token_breakdown.wasted_tokens
    cost_delta = result_b.token_breakdown.estimated_cost_usd - result_a.token_breakdown.estimated_cost_usd

    # 2. Structure Deltas
    structure_delta = {
        "redundancy": result_b.score_breakdown.redundancy_penalty - result_a.score_breakdown.redundancy_penalty,
        "density": result_b.score_breakdown.density_penalty - result_a.score_breakdown.density_penalty,
        "structure_imbalance": result_b.score_breakdown.structure_penalty - result_a.score_breakdown.structure_penalty,
        "concentration": result_b.score_breakdown.concentration_penalty - result_a.score_breakdown.concentration_penalty,
    }

    # 3. Recommendation Lifecycle
    dict_a = {get_recommendation_id(r): r for r in result_a.recommendations}
    dict_b = {get_recommendation_id(r): r for r in result_b.recommendations}

    ids_a = set(dict_a.keys())
    ids_b = set(dict_b.keys())

    resolved_ids = ids_a - ids_b
    new_ids = ids_b - ids_a
    persisting_ids = ids_a & ids_b

    resolved = [dict_a[rid] for rid in resolved_ids]
    new = [dict_b[nid] for nid in new_ids]
    persisting = [dict_b[pid] for pid in persisting_ids]  # Use B's updated version

    # Sort deterministically by severity/impact
    resolved.sort(key=lambda r: (-r.impact_score, -r.token_savings, r.issue))
    new.sort(key=lambda r: (-r.impact_score, -r.token_savings, r.issue))
    persisting.sort(key=lambda r: (-r.impact_score, -r.token_savings, r.issue))

    return ContextDiffResult(
        result_a=result_a,
        result_b=result_b,
        score_delta=score_delta,
        token_delta=token_delta,
        waste_delta=waste_delta,
        cost_delta=cost_delta,
        structure_delta=structure_delta,
        resolved_recommendations=resolved,
        new_recommendations=new,
        persisting_recommendations=persisting,
    )
