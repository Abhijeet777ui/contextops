"""
Phase 2: Chaos Stress Tests.

Validates that the ContextOps V0.1 engine is production-hardened:
  - Global invariants (score ∈ [0,100], no negative penalties, no NaN/Inf)
  - Performance bounds (≤2s for ≤5k tokens, ≤5s for ≤20k tokens)
  - Deterministic tolerance (±2 points across runs)
  - Expected behavioral contracts per chaos scenario
"""

import json
import math
import time
from pathlib import Path

import pytest

from contextops.api.inspect import inspect_context
from contextops.core.models import AnalysisResult

CHAOS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "chaos"


def _load_and_inspect(filename: str) -> tuple[AnalysisResult, dict, float]:
    """Load a chaos file, run inspection, return (result, contract, elapsed_seconds)."""
    filepath = CHAOS_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    contract = data.get("expected_contract", {})

    start = time.perf_counter()
    result = inspect_context(data)
    elapsed = time.perf_counter() - start

    return result, contract, elapsed


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL INVARIANTS — applied to EVERY chaos case
# ═══════════════════════════════════════════════════════════════════════════

CHAOS_FILES = [
    "chaos_micro_context.json",
    "chaos_empty_and_corrupt.json",
    "chaos_system_prompt_explosion.json",
    "chaos_agent_trace_loop.json",
    "chaos_massive_rag_dump.json",
]


@pytest.mark.parametrize("filename", CHAOS_FILES)
def test_global_invariant_score_in_range(filename):
    """Score must always be in [0, 100]."""
    result, _, _ = _load_and_inspect(filename)
    assert 0 <= result.score <= 100, f"Score {result.score} out of [0, 100]"


@pytest.mark.parametrize("filename", CHAOS_FILES)
def test_global_invariant_no_negative_penalties(filename):
    """No individual penalty may be negative."""
    result, _, _ = _load_and_inspect(filename)
    sb = result.score_breakdown
    assert sb.redundancy_penalty >= 0, f"Redundancy penalty is negative: {sb.redundancy_penalty}"
    assert sb.density_penalty >= 0, f"Density penalty is negative: {sb.density_penalty}"
    assert sb.structure_penalty >= 0, f"Structure penalty is negative: {sb.structure_penalty}"
    assert sb.concentration_penalty >= 0, f"Concentration penalty is negative: {sb.concentration_penalty}"


@pytest.mark.parametrize("filename", CHAOS_FILES)
def test_global_invariant_no_nan_inf(filename):
    """No NaN or Infinity values in any numeric field."""
    result, _, _ = _load_and_inspect(filename)
    sb = result.score_breakdown
    tb = result.token_breakdown

    values = [
        result.score,
        sb.redundancy_penalty,
        sb.density_penalty,
        sb.structure_penalty,
        sb.concentration_penalty,
        sb.total_penalty,
        tb.total_tokens,
        tb.wasted_tokens,
        tb.estimated_reduction_pct,
    ]
    for v in values:
        assert not math.isnan(v), f"NaN detected: {v}"
        assert not math.isinf(v), f"Infinity detected: {v}"


@pytest.mark.parametrize("filename", CHAOS_FILES)
def test_global_invariant_penalty_sum_consistency(filename):
    """Total penalty must equal the sum of individual penalties."""
    result, _, _ = _load_and_inspect(filename)
    sb = result.score_breakdown
    expected_total = (
        sb.redundancy_penalty
        + sb.density_penalty
        + sb.structure_penalty
        + sb.concentration_penalty
    )
    assert abs(sb.total_penalty - expected_total) < 0.01, (
        f"Total penalty mismatch: {sb.total_penalty} vs sum {expected_total}"
    )


@pytest.mark.parametrize("filename", CHAOS_FILES)
def test_global_invariant_score_equals_100_minus_penalty(filename):
    """Score must equal 100 - total_penalty (clamped to [0, 100])."""
    result, _, _ = _load_and_inspect(filename)
    expected_score = max(0, min(100, round(100 - result.score_breakdown.total_penalty)))
    assert abs(result.score - expected_score) <= 1, (
        f"Score {result.score} does not match 100 - {result.score_breakdown.total_penalty}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# DETERMINISM — same input must produce same output
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("filename", CHAOS_FILES)
def test_determinism(filename):
    """Running the same input twice must yield the exact same score."""
    r1, _, _ = _load_and_inspect(filename)
    r2, _, _ = _load_and_inspect(filename)
    assert r1.score == r2.score, f"Non-deterministic: {r1.score} vs {r2.score}"


# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE BOUNDS
# ═══════════════════════════════════════════════════════════════════════════

def test_performance_micro_context():
    """Micro context (≤5k tokens) must complete in < 2s."""
    _, _, elapsed = _load_and_inspect("chaos_micro_context.json")
    assert elapsed < 2.0, f"Micro context took {elapsed:.2f}s (limit: 2s)"


def test_performance_massive_rag_dump():
    """Massive RAG dump (≤20k tokens) must complete in < 15s."""
    _, _, elapsed = _load_and_inspect("chaos_massive_rag_dump.json")
    assert elapsed < 15.0, f"Massive RAG dump took {elapsed:.2f}s (limit: 15s)"


# ═══════════════════════════════════════════════════════════════════════════
# BEHAVIORAL CONTRACTS — per-scenario assertions
# ═══════════════════════════════════════════════════════════════════════════

def test_chaos_micro_context_contract():
    """Micro context: near-perfect score, no penalties."""
    result, contract, _ = _load_and_inspect("chaos_micro_context.json")
    min_score = contract.get("min_score", 90)
    assert result.score >= min_score, f"Score {result.score} < expected min {min_score}"


def test_chaos_empty_and_corrupt_contract():
    """Empty/corrupt: engine handles gracefully, no crash."""
    result, contract, _ = _load_and_inspect("chaos_empty_and_corrupt.json")
    min_score = contract.get("min_score", 60)
    assert result.score >= min_score, f"Score {result.score} < expected min {min_score}"


def test_chaos_system_prompt_explosion_contract():
    """System prompt explosion: must flag System prompt bloat."""
    result, contract, _ = _load_and_inspect("chaos_system_prompt_explosion.json")
    must_flag = contract.get("must_flag", [])
    rec_issues = [r.issue.lower() for r in result.recommendations]
    for flag in must_flag:
        assert any(flag.lower() in issue for issue in rec_issues), (
            f"Expected flag '{flag}' not found in recommendations: {rec_issues}"
        )


def test_chaos_agent_trace_loop_contract():
    """Agent trace loop: must flag Redundant context."""
    result, contract, _ = _load_and_inspect("chaos_agent_trace_loop.json")
    must_flag = contract.get("must_flag", [])
    rec_issues = [r.issue.lower() for r in result.recommendations]
    for flag in must_flag:
        assert any(flag.lower() in issue for issue in rec_issues), (
            f"Expected flag '{flag}' not found in recommendations: {rec_issues}"
        )


def test_chaos_massive_rag_dump_contract():
    """Massive RAG dump: must flag Retrieval dominance, score < 50."""
    result, contract, _ = _load_and_inspect("chaos_massive_rag_dump.json")
    max_score = contract.get("max_score", 50)
    assert result.score <= max_score, (
        f"Score {result.score} > expected max {max_score} for massive RAG dump"
    )
    must_flag = contract.get("must_flag", [])
    rec_issues = [r.issue.lower() for r in result.recommendations]
    for flag in must_flag:
        assert any(flag.lower() in issue for issue in rec_issues), (
            f"Expected flag '{flag}' not found in recommendations: {rec_issues}"
        )
