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
from contextops.core.roast import get_roast


def analyze(
    bundle: ContextBundle,
    model: str = "gpt-4o",
    config: Optional[ContextOpsConfig] = None,
) -> AnalysisResult:
    """
    Run the full ContextOps analysis pipeline.

    Args:
        bundle: Normalized context bundle (from normalizer).
        model: Model name for tiktoken encoding.
        config: Custom thresholds and mode configuration.

    Returns:
        Complete AnalysisResult ready for JSON serialization or CLI rendering.
    """
    config = config or ContextOpsConfig.default()

    # ── Step 1: Token counting ──────────────────────────────────────
    token_breakdown = analyze_tokens(bundle, model=model)

    # ── Step 2: Redundancy detection ────────────────────────────────
    redundancy_findings, final_wasted_tokens = analyze_redundancy(
        bundle,
        strict_semantic=config.strict_semantic,
        config=config,  # passes rs_minimum and future config fields through
    )
    token_breakdown.wasted_tokens = final_wasted_tokens
    if token_breakdown.total_tokens > 0:
        token_breakdown.estimated_reduction_pct = (final_wasted_tokens / token_breakdown.total_tokens) * 100.0

    # ── Step 3: Structure analysis ──────────────────────────────────
    structure_findings = analyze_structure(bundle, config=config)

    # ── Step 3.5: Shadow Density analysis ───────────────────────────
    density_signal = compute_density_signal(bundle)

    # ── Step 4: Compute score ───────────────────────────────────────────
    score_breakdown = _compute_score(
        bundle, token_breakdown, redundancy_findings, structure_findings, density_signal, config
    )

    # Update wasted tokens in token breakdown (already set from analyze_redundancy)
    
    # ── Step 5: Generate recommendations ────────────────────────────
    recommendations = _generate_recommendations(
        bundle, redundancy_findings, structure_findings, score_breakdown, density_signal
    )

    # ── Step 6: Assemble result ─────────────────────────────────────
    result = AnalysisResult(
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
            "version": "0.3.0",
        },
    )

    # ── Step 7: Attach roast (opt-in, non-deterministic) ────────────
    if config.roast_enabled:
        result.roast = get_roast(score=result.score, breakdown=score_breakdown)

    return result

# ── Scoring ─────────────────────────────────────────────────────────────


def _compute_score(
    bundle: ContextBundle,
    token_breakdown: TokenBreakdown,
    redundancy_findings: list[RedundancyFinding],
    structure_findings: list[StructureFinding],
    density_signal: DensitySignal,
    config: ContextOpsConfig,
) -> ScoreBreakdown:
    """
    Compute the 4-axis penalty score.

    Score = 100 - (redundancy + density + structure + concentration)
    Each penalty has a maximum cap to prevent any single axis from dominating.

    Signal contract: each axis reads only from its designated input.
    No cross-axis reading is permitted.
    """
    redundancy = _calc_redundancy_penalty(bundle, redundancy_findings, token_breakdown, config)
    density = _calc_density_penalty(density_signal, config)   # reads DensitySignal only
    structure = _calc_structure_penalty(structure_findings)
    concentration = _calc_concentration_penalty(bundle, config)

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
    config: ContextOpsConfig,
) -> float:
    """
    Redundancy penalty (0–30 pts).

    Findings are the SINGLE SOURCE OF TRUTH for waste.
    The token_breakdown.wasted_tokens value is set from findings by the
    analyze_redundancy() function, so both penalty and UI display are
    always in sync — no split-brain possible.

    Formula:
        (waste_penalty_ratio × 0.6 + cluster_score × 0.4) × 30

    Signal contract: reads only from findings list and token_breakdown.wasted_tokens.
    Must NOT read density_signal or any structural analyzer output.
    """
    wasted = token_breakdown.wasted_tokens  # set from findings, single source of truth
    if wasted == 0:
        return 0.0

    # Exponential penalty curve — same input always same output (deterministic)
    waste_penalty_ratio = 1 - math.exp(-0.001 * wasted)

    # Cluster score: what fraction of items are involved in REDUNDANT_CONTEXT findings?
    # Phase 0 Bug 2 fix: instead of counting all involved item IDs (which inflates score
    # when one item appears in multiple findings), we use a per-item max-waste approach.
    # Each item contributes its single highest waste finding — capping its influence
    # regardless of how many pairs it appears in.
    item_max_waste: dict[str, int] = {}
    for f in findings:
        if f.classification in (
            RedundancyClassification.REDUNDANT_CONTEXT,
            RedundancyClassification.EXACT_DUPLICATE,
            RedundancyClassification.NEAR_DUPLICATE,
        ):
            # Each item in a pair claims the waste from that finding.
            # If the item appears in multiple findings, we keep the max only.
            item_max_waste[f.item_a_id] = max(
                item_max_waste.get(f.item_a_id, 0), f.estimated_waste_tokens
            )
            item_max_waste[f.item_b_id] = max(
                item_max_waste.get(f.item_b_id, 0), f.estimated_waste_tokens
            )

    # cluster_score = fraction of items that have ANY redundant involvement
    cluster_score = len(item_max_waste) / max(1, bundle.item_count)

    penalty = (waste_penalty_ratio * 0.6 + cluster_score * 0.4) * 30.0
    return min(30.0, round(penalty, 2))


def _calc_density_penalty(density_signal: DensitySignal, config: ContextOpsConfig) -> float:
    """
    Structural density penalty (0–30 pts).

    Derived exclusively from DensitySignal — the structural analysis of raw context text.
    DensitySignal measures: format overhead (FO), whitespace waste (WL), entropy compression (EC).

    Phase 3.1 — Log-scale formula (replaces linear):
        penalty = log(1 + total_signal * 2) / log(1 + 2) * 30

    This compresses extreme values (large chunks with formatting noise) while
    preserving full sensitivity at low signal values. Compared to linear:
        signal 0.10 → linear  3.0 pts, log ~2.8 pts  (similar at low end)
        signal 0.50 → linear 15.0 pts, log ~14.6 pts  (similar at mid)
        signal 0.90 → linear 27.0 pts, log ~24.7 pts  (compressed at high end)
        signal 1.00 → linear 30.0 pts, log ~30.0 pts  (same cap)

    Phase 0 Bug 3 fix: divergence amplifier replaced with log-scale formula.
    The old x130 linear amplifier could add up to 30 pts from divergence alone;
    the log formula caps the bonus at ~8 pts for extreme divergence (0.30):
        divergence 0.027 (attack baseline) → ~1.8 bonus pts
        divergence 0.10                    → ~4.5 bonus pts
        divergence 0.30 (extreme)          → ~8.0 bonus pts

    Signal contract: reads ONLY DensitySignal.
    Must NOT read wasted_tokens or any redundancy analyzer output.
    """
    # Phase 3.1: log-scale total density formula
    penalty = math.log1p(density_signal.total_density_signal * 2.0) / math.log1p(2.0) * 30.0

    # Phase 0 Bug 3 fix: log-scale divergence amplifier (replaces linear × 130).
    if density_signal.system_divergence > 0.02:
        excess = density_signal.system_divergence - 0.02
        penalty += math.log1p(excess * 50)

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


def _calc_concentration_penalty(bundle: ContextBundle, config: ContextOpsConfig) -> float:
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
    p_con_adjusted = p_con * config.concentration_weight

    return min(20.0, round(p_con_adjusted * 20.0, 2))


# ── Padding Anomaly Detection (Phase 3.2) ───────────────────────────────


def _is_padding_anomaly(density_signal: DensitySignal, bundle: ContextBundle) -> bool:
    """
    Phase 3.2 — Multi-condition gate for suspicious threshold padding.

    Returns True only when ALL three conditions are met:
      1. System divergence exceeds the adversarial baseline threshold (> 0.02)
      2. Entropy compression is high (≥ 0.5) — indicates repetitive retrieval content
      3. The corpus has fewer than 5 distinct sources — legitimate multi-source
         diversity is NOT an anomaly

    This prevents the common false positives:
      - Multi-domain RAG corpora with natural divergence (fails condition 3)
      - Clean but repetitive FAQ content (fails condition 2 in reverse)
      - Single-document summaries with high format overhead (may fail condition 2)

    When this gate fires, the result is an advisory recommendation (LOW severity),
    NOT a score penalty. Adversarial padding is flagged for human review, not
    automatically counted against the score.
    """
    # Condition 1: divergence above adversarial baseline
    if density_signal.system_divergence <= 0.02:
        return False

    # Condition 2: retrieval content must be highly repetitive
    if density_signal.entropy_compression < 0.5:
        return False

    # Condition 3: not a legitimate multi-source corpus
    sources = {item.source for item in bundle.items if item.source}
    if len(sources) >= 5:
        return False

    return True


# ── Recommendations ─────────────────────────────────────────────────────


def _generate_recommendations(
    bundle: ContextBundle,
    redundancy_findings: list[RedundancyFinding],
    structure_findings: list[StructureFinding],
    score_breakdown: ScoreBreakdown,
    density_signal: DensitySignal | None = None,
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
        if f.classification in (
            RedundancyClassification.REDUNDANT_CONTEXT,
            RedundancyClassification.EXACT_DUPLICATE,
            RedundancyClassification.NEAR_DUPLICATE,
        )
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

    # ── Density Anomaly recommendations (Phase 3.2 — advisory only) ────────
    # The padding anomaly check is now a multi-condition gate. If it fires,
    # we emit a LOW-severity advisory recommendation — NOT a score penalty.
    # High false-positive risk means this should be reviewed, not auto-failed.
    if density_signal and _is_padding_anomaly(density_signal, bundle):
        recs.append(Recommendation(
            issue=f"Suspicious Threshold Padding detected (divergence: {density_signal.system_divergence:.4f})",
            impact_score=0.0,  # Advisory: no score impact claimed
            token_savings=0,
            fix=(
                "Advisory: retrieval context has anomalous density divergence relative to "
                "the system prompt AND is highly repetitive with a small source pool. "
                "Inspect retrieval chunks for malicious padding. "
                f"(divergence: {density_signal.system_divergence:.4f}, "
                f"entropy_compression: {density_signal.entropy_compression:.4f})"
            ),
            severity=FindingSeverity.LOW,  # Advisory — human review, not auto-block
        ))

    # Sort by impact (highest first)
    recs.sort(key=lambda r: r.impact_score, reverse=True)
    return recs
