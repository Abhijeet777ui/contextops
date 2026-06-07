"""
ContextOps Core Engine.

The single orchestrator that:
  1. Runs all analyzers (tokens, redundancy, structure)
  2. Computes the 4-axis penalty score (100 - total penalty)
  3. Generates actionable recommendations (Next Best Action)
  4. Returns the final AnalysisResult (JSON-primary API contract)

This is intentionally ONE module, not a framework. V0.1 should feel
like one coherent system. We can modularize later.
"""

from __future__ import annotations

import math
from collections import defaultdict

from contextops.analyzers.density import compute_density_signal
from contextops.analyzers.redundancy import analyze_redundancy
from contextops.analyzers.structure import analyze_structure
from contextops.analyzers.tokens import analyze_tokens
from contextops.core.config import ContextOpsConfig
from contextops.core.models import (
    AnalysisResult,
    ContextBundle,
    ContextType,
    DensitySignal,
    FindingSeverity,
    Recommendation,
    RedundancyClassification,
    RedundancyFinding,
    ScoreBreakdown,
    StructureFinding,
    TokenBreakdown,
)


def analyze(
    bundle: ContextBundle,
    model: str = "gpt-4o",
    cost_per_1k: float = 0.005,
    config: ContextOpsConfig | None = None,
) -> AnalysisResult:
    """
    Run the full ContextOps analysis pipeline.

    Args:
        bundle: Normalized context bundle (from normalizer).
        model: Model name for tiktoken encoding.
        cost_per_1k: Cost per 1K input tokens in USD.
        config: Custom thresholds and mode configuration.

    Returns:
        Complete AnalysisResult ready for JSON serialization or CLI rendering.
    """
    config = config or ContextOpsConfig.default()

    # ── Step 1: Token counting ──────────────────────────────────────
    token_breakdown = analyze_tokens(bundle, model=model, cost_per_1k=cost_per_1k)

    # ── Step 2: Redundancy detection ────────────────────────────────
    redundancy_findings, final_wasted_tokens = analyze_redundancy(bundle)
    token_breakdown.wasted_tokens = final_wasted_tokens

    # ── Step 3: Structure analysis ──────────────────────────────────
    structure_findings = analyze_structure(bundle, config=config)

    # ── Step 3.5: Shadow Density analysis ───────────────────────────
    density_signal = compute_density_signal(bundle)

    # ── Step 4: Compute score ───────────────────────────────────────────
    score_breakdown = _compute_score(
        bundle, token_breakdown, redundancy_findings, structure_findings, density_signal
    )

    # Update wasted tokens in token breakdown (already set from analyze_redundancy)
    
    # ── Step 5: Generate recommendations ────────────────────────────
    recommendations = _generate_recommendations(
        bundle, redundancy_findings, structure_findings, score_breakdown
    )

    # ── Step 6: Assemble result ─────────────────────────────────────
    return AnalysisResult(
        score=score_breakdown.score,
        mode=config.mode,
        config_version=config.version,
        density_signal=density_signal,
        score_breakdown=score_breakdown,
        token_breakdown=token_breakdown,
        redundancy_findings=redundancy_findings,
        structure_findings=structure_findings,
        recommendations=recommendations,
        metadata={
            "item_count": bundle.item_count,
            "model": model,
            "version": "0.1.0",
        },
    )

# ── Scoring ─────────────────────────────────────────────────────────────


def _compute_score(
    bundle: ContextBundle,
    token_breakdown: TokenBreakdown,
    redundancy_findings: list[RedundancyFinding],
    structure_findings: list[StructureFinding],
    density_signal: DensitySignal,
) -> ScoreBreakdown:
    """
    Compute the 4-axis penalty score.

    Score = 100 - (redundancy + density + structure + concentration)
    Each penalty has a maximum cap to prevent any single axis from dominating.

    Signal contract: each axis reads only from its designated input.
    No cross-axis reading is permitted.
    """
    redundancy = _calc_redundancy_penalty(bundle, redundancy_findings, token_breakdown)
    density = _calc_density_penalty(density_signal)   # reads DensitySignal only
    structure = _calc_structure_penalty(structure_findings)
    concentration = _calc_concentration_penalty(bundle)

    return ScoreBreakdown(
        redundancy_penalty=redundancy,
        density_penalty=density,
        structure_penalty=structure,
        concentration_penalty=concentration,
    )


def _calc_redundancy_penalty(
    bundle: ContextBundle,
    findings: list[RedundancyFinding],
    token_breakdown: TokenBreakdown,
) -> float:
    """
    Redundancy penalty (0–30 pts).

    Formula:
        (waste_penalty_ratio × 0.6 + similarity_cluster_score × 0.4) × 30

    - waste_penalty_ratio: exponentially mapped from final_wasted_tokens
    - similarity_cluster_score: proportion of items involved in redundancy

    Signal contract: reads only redundancy analyzer outputs (wasted_tokens, findings).
    Must NOT read density_signal or any structural analyzer output.
    """
    wasted = token_breakdown.wasted_tokens
    if wasted == 0:
        return 0.0

    waste_penalty_ratio = 1 - math.exp(-0.001 * wasted)

    # Cluster score: what fraction of items are involved in redundancy?
    involved_ids: set[str] = set()
    for f in findings:
        if f.classification != RedundancyClassification.EXPECTED_OVERLAP:
            involved_ids.add(f.item_a_id)
            involved_ids.add(f.item_b_id)

    cluster_score = len(involved_ids) / max(1, bundle.item_count)

    penalty = (waste_penalty_ratio * 0.6 + cluster_score * 0.4) * 30.0
    return min(30.0, round(penalty, 2))


def _calc_density_penalty(density_signal: DensitySignal) -> float:
    """
    Structural density penalty (0–30 pts).

    Derived exclusively from DensitySignal — the structural analysis of raw context text.
    DensitySignal measures: format overhead (FO), whitespace waste (WL), entropy compression (EC).

    Formula: penalty = total_density_signal × 30
    where total_density_signal ∈ [0.0, 1.0] is the weighted combination:
        total = 0.4 * FO + 0.2 * WL + 0.4 * EC

    Signal contract: reads ONLY DensitySignal.
    Must NOT read wasted_tokens or any redundancy analyzer output.
    """
    penalty = density_signal.total_density_signal * 30.0
    return min(30.0, round(penalty, 2))


def _calc_structure_penalty(findings: list[StructureFinding]) -> float:
    """
    Structure imbalance penalty (0–20 pts).

    Based on how many imbalance findings exist and their severity.
    Each finding contributes points based on how far the ratio exceeds threshold.
    """
    if not findings:
        return 0.0

    _SEVERITY_MULTIPLIER = {
        FindingSeverity.LOW: 0.5,
        FindingSeverity.MEDIUM: 1.0,
        FindingSeverity.HIGH: 1.5,
        FindingSeverity.CRITICAL: 2.0,
    }

    total = 0.0
    for f in findings:
        if f.threshold > 0:
            # How far over the threshold? e.g., 0.80 actual vs 0.70 threshold = 0.10 excess
            excess = max(0.0, f.actual_ratio - f.threshold)
            # Scale: each 0.10 excess = ~5 points penalty
            contribution = (excess / 0.10) * 5
        else:
            # Low diversity finding — flat penalty
            contribution = 3.0

        total += contribution * _SEVERITY_MULTIPLIER.get(f.severity, 1.0)

    return min(20.0, round(total, 2))


def _calc_concentration_penalty(bundle: ContextBundle) -> float:
    """
    Concentration penalty (0–20 pts).

    Uses a 2-axis decomposition of source behavior:
    1. Source Dominance (P_dom): over-reliance on a single document.
    2. Entropy Imbalance (P_ent): uneven distribution across multiple sources.
    
    This matches the P_con definition in the methodology paper.
    """
    retrieval_items = bundle.items_by_type(ContextType.RETRIEVAL)

    # 1. Protect "Gold Answer RAG" (Single-chunk lookup)
    if len(retrieval_items) <= 1:
        return 0.0

    # 2. Token-weighted distribution
    source_tokens = defaultdict(int)
    total_tokens = 0
    for item in retrieval_items:
        src = item.source or "unknown"
        source_tokens[src] += item.token_count
        total_tokens += item.token_count

    if total_tokens == 0:
        return 0.0

    num_sources = len(source_tokens)

    # Signal A: Source Dominance (P_dom)
    p_dom = max(source_tokens.values()) / total_tokens

    # Signal B: Entropy Imbalance (P_ent)
    if num_sources <= 1:
        p_ent = 0.0  # Defined as 0 when math would divide-by-zero
    else:
        entropy = 0.0
        for tokens in source_tokens.values():
            p_s = tokens / total_tokens
            if p_s > 0:
                entropy -= p_s * math.log2(p_s)
        p_ent = 1.0 - (entropy / math.log2(num_sources))

    # Combine the signals
    # We weight Dominance slightly higher because it's a stronger failure mode in RAG
    p_con = (0.6 * p_dom) + (0.4 * p_ent)

    return min(20.0, round(p_con * 20.0, 2))


# ── Recommendations ─────────────────────────────────────────────────────


def _generate_recommendations(
    bundle: ContextBundle,
    redundancy_findings: list[RedundancyFinding],
    structure_findings: list[StructureFinding],
    score_breakdown: ScoreBreakdown,
) -> list[Recommendation]:
    """
    Generate actionable recommendations from findings.

    Every recommendation includes:
    - What the issue is
    - How much score improvement to expect
    - How many tokens to save
    - Exactly what to do
    """
    recs: list[Recommendation] = []

    # ── Redundancy recommendations ──────────────────────────────────
    # Group redundant findings (skip expected overlaps)
    real_redundancy = [
        f for f in redundancy_findings
        if f.classification == RedundancyClassification.REDUNDANT_CONTEXT
    ]

    for finding in real_redundancy[:3]:  # Top 3 most impactful
        # Estimate score impact: removing this finding's waste
        if bundle.total_tokens > 0:
            waste_ratio = finding.estimated_waste_tokens / bundle.total_tokens
            estimated_score_gain = waste_ratio * 30  # impacts both redundancy and density
        else:
            estimated_score_gain = 0.0

        recs.append(Recommendation(
            issue=f"Redundant context: {finding.detail}",
            impact_score=round(estimated_score_gain, 1),
            token_savings=finding.estimated_waste_tokens,
            fix=f"Remove the duplicate item ('{finding.item_b_id}') → save {finding.estimated_waste_tokens} tokens",
            severity=FindingSeverity.HIGH if finding.similarity_score > 0.85 else FindingSeverity.MEDIUM,
        ))

    # Boilerplate recommendations
    boilerplate = [
        f for f in redundancy_findings
        if f.classification == RedundancyClassification.BOILERPLATE
    ]
    if boilerplate:
        total_bp_waste = sum(f.estimated_waste_tokens for f in boilerplate)
        recs.append(Recommendation(
            issue=f"Boilerplate repetition detected ({len(boilerplate)} pairs)",
            impact_score=round(total_bp_waste / max(1, bundle.total_tokens) * 15, 1),
            token_savings=total_bp_waste,
            fix="Consolidate repeated instructions into the system prompt",
            severity=FindingSeverity.MEDIUM,
        ))

    # ── Structure recommendations ───────────────────────────────────
    for struct_finding in structure_findings:
        pct = f"{struct_finding.actual_ratio * 100:.0f}%"
        threshold_pct = f"{struct_finding.threshold * 100:.0f}%"

        if struct_finding.issue == "Retrieval dominance":
            fix = f"Reduce retrieval chunks — currently {pct} of context (threshold: {threshold_pct})"
        elif struct_finding.issue == "System prompt bloat":
            fix = f"Trim system prompt — currently {pct} of context (threshold: {threshold_pct})"
        elif struct_finding.issue == "Memory explosion":
            fix = f"Prune old memories — currently {pct} of context (threshold: {threshold_pct})"
        elif struct_finding.issue == "Tool output sprawl":
            fix = f"Summarize tool outputs — currently {pct} of context (threshold: {threshold_pct})"
        else:
            fix = f"Improve context composition — {struct_finding.issue}"

        recs.append(Recommendation(
            issue=struct_finding.issue,
            impact_score=round(score_breakdown.structure_penalty * 0.5, 1),
            token_savings=0,  # structure fixes don't always save tokens directly
            fix=fix,
            severity=struct_finding.severity,
        ))

    # Sort by impact (highest first)
    recs.sort(key=lambda r: r.impact_score, reverse=True)
    return recs
