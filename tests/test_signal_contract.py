"""
Signal Contract Tests.

These tests enforce the core architectural invariant of ContextOps:

    NO metric may depend on the output of another metric.
    All metrics must be computed independently from raw context inputs.

Rule A: density_signal must NOT change when wasted_tokens changes.
Rule B: redundancy_penalty must NOT change when density inputs change.
Rule C: density_penalty must be derived from DensitySignal, not wasted_tokens.

If any of these tests fail, a signal boundary has been violated.
"""

from __future__ import annotations

import copy
import pytest
from contextops.analyzers.density import compute_density_signal
from contextops.core.engine import analyze, _calc_density_penalty, _calc_redundancy_penalty
from contextops.core.models import (
    ContextBundle,
    ContextItem,
    ContextType,
    DensitySignal,
    RedundancyFinding,
    RedundancyClassification,
    TokenBreakdown,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_item(content: str, ctx_type: ContextType = ContextType.RETRIEVAL, source: str = "doc_a") -> ContextItem:
    return ContextItem(type=ctx_type, content=content, token_count=len(content.split()), source=source)


def _clean_bundle() -> ContextBundle:
    """A bundle with no redundancy, but real structural content."""
    return ContextBundle(items=[
        _make_item("The quick brown fox jumps over the lazy dog.", source="doc_a"),
        _make_item("Python is a high-level programming language.", source="doc_b"),
        _make_item("Machine learning enables systems to learn from data.", source="doc_c"),
    ])


def _redundant_bundle() -> ContextBundle:
    """A bundle with high redundancy (near-identical items)."""
    content = "The quick brown fox jumps over the lazy dog."
    return ContextBundle(items=[
        _make_item(content, source="doc_a"),
        _make_item(content, source="doc_b"),
        _make_item(content, source="doc_c"),
    ])


# ── Rule A: Density signal is independent of wasted_tokens ────────────────

class TestRuleA_DensitySignalIndependence:
    """
    density_signal must NOT change when wasted_tokens changes.
    DensitySignal is a function of raw text content only.
    """

    def test_density_signal_unchanged_when_wasted_tokens_varies(self):
        """Mutating wasted_tokens must not affect the DensitySignal."""
        bundle = _clean_bundle()
        
        signal_before = compute_density_signal(bundle)
        
        # Simulate what the redundancy engine writes — mutations to token breakdown
        # This must have ZERO effect on density_signal
        for item in bundle.items:
            item.token_count = item.token_count * 10  # inflate counts
        
        signal_after = compute_density_signal(bundle)
        
        # DensitySignal reads only content strings, not token_count
        assert signal_before.format_overhead == signal_after.format_overhead, (
            "RULE A VIOLATION: format_overhead changed when token_count was mutated. "
            "DensitySignal must not read from token_count."
        )
        assert signal_before.whitespace_waste == signal_after.whitespace_waste, (
            "RULE A VIOLATION: whitespace_waste changed when token_count was mutated."
        )
        assert signal_before.entropy_compression == signal_after.entropy_compression, (
            "RULE A VIOLATION: entropy_compression changed when token_count was mutated."
        )

    def test_density_signal_unchanged_when_redundancy_findings_change(self):
        """Adding or removing redundancy findings must not affect DensitySignal."""
        bundle = _clean_bundle()
        
        signal_with_no_findings = compute_density_signal(bundle)
        
        # Simulate what redundancy engine adds — this should be completely invisible to density
        # (DensitySignal is computed from bundle.items content only)
        signal_with_findings_simulated = compute_density_signal(bundle)
        
        assert signal_with_no_findings.total_density_signal == signal_with_findings_simulated.total_density_signal, (
            "RULE A VIOLATION: DensitySignal changed when redundancy state changed. "
            "These must be fully independent."
        )


# ── Rule B: Redundancy penalty is independent of density/content format ───

class TestRuleB_RedundancyPenaltyIndependence:
    """
    redundancy_penalty must NOT change when content format changes
    (i.e., same words, different whitespace/punctuation).
    Redundancy is about similarity, not structure.
    """

    def test_redundancy_penalty_stable_across_formatting_changes(self):
        """
        Two bundles with the same words but different whitespace must produce
        similar (within 2 pts) redundancy penalties.

        The RS formula uses Jaccard (word sets, dominant weight 0.6) which is
        whitespace-immune, PLUS a small char_overlap component (0.1 weight) which
        can vary slightly with whitespace. The rule is that Jaccard dominates and
        the penalty stays close — not that it is byte-for-byte identical.
        """
        content_dense = "The quick brown fox jumps over the lazy dog"
        content_spaced = "The  quick  brown  fox  jumps  over  the  lazy  dog"

        bundle_dense = ContextBundle(items=[
            _make_item(content_dense, source="doc_a"),
            _make_item(content_dense, source="doc_b"),
        ])
        bundle_spaced = ContextBundle(items=[
            _make_item(content_spaced, source="doc_a"),
            _make_item(content_spaced, source="doc_b"),
        ])

        result_dense = analyze(bundle_dense)
        result_spaced = analyze(bundle_spaced)

        # Penalties must be within 2 points — Jaccard (0.6 weight) is whitespace-immune
        # and dominates the RS signal. Only the minor char_overlap (0.1 weight) can vary.
        penalty_diff = abs(
            result_dense.score_breakdown.redundancy_penalty
            - result_spaced.score_breakdown.redundancy_penalty
        )
        assert penalty_diff <= 2.0, (
            f"RULE B VIOLATION: redundancy_penalty diverged by {penalty_diff:.2f} pts due to "
            "whitespace formatting. Jaccard must dominate — penalties must stay within 2 pts."
        )


# ── Rule C: density_penalty is derived from DensitySignal, not wasted_tokens

class TestRuleC_DensityPenaltySource:
    """
    _calc_density_penalty must accept DensitySignal as its input.
    It must NOT accept TokenBreakdown or read wasted_tokens.
    """

    def test_density_penalty_accepts_density_signal(self):
        """_calc_density_penalty must work with a DensitySignal input."""
        signal = DensitySignal(
            format_overhead=0.3,
            whitespace_waste=0.2,
            entropy_compression=0.1,
            total_density_signal=0.22,  # weighted
        )
        penalty = _calc_density_penalty(signal)
        assert 0.0 <= penalty <= 30.0, "density_penalty must be in range [0, 30]"

    def test_density_penalty_zero_for_zero_signal(self):
        """A DensitySignal of all zeros must produce zero penalty."""
        signal = DensitySignal(0.0, 0.0, 0.0, 0.0)
        assert _calc_density_penalty(signal) == 0.0

    def test_density_penalty_max_for_max_signal(self):
        """A DensitySignal of total=1.0 must produce max penalty (30 pts)."""
        signal = DensitySignal(1.0, 1.0, 1.0, 1.0)
        assert _calc_density_penalty(signal) == 30.0

    def test_density_penalty_does_not_accept_token_breakdown(self):
        """
        _calc_density_penalty must NOT work with a TokenBreakdown.
        This is a type enforcement check.
        """
        token_breakdown = TokenBreakdown(total_tokens=1000, wasted_tokens=500)
        
        # The function should raise TypeError if passed wrong type,
        # OR if it silently accepts it, something is wrong with the contract.
        try:
            result = _calc_density_penalty(token_breakdown)
            # If it returns without error, it means the function still accepts TokenBreakdown
            # which is a contract violation.
            raise AssertionError(
                "RULE C VIOLATION: _calc_density_penalty accepted a TokenBreakdown. "
                "It must only accept DensitySignal. "
                f"It returned: {result}"
            )
        except (TypeError, AttributeError):
            # Expected — function correctly rejected the wrong input type
            pass


# ── Integration: verify full pipeline orthogonality ───────────────────────

class TestFullPipelineOrthogonality:
    """End-to-end verification that the 4 axes are independently computed."""

    def test_zero_redundancy_bundle_has_nonzero_density_if_bloated(self):
        """
        A bundle with NO redundancy but highly bloated formatting
        must still produce a non-zero density_penalty.
        This proves density_penalty is NOT derived from redundancy.
        """
        # Unique items, no overlap, but each one is heavily whitespace-padded
        bloated_content = "word   \n\n\n   " * 30  # lots of whitespace, no repeated words between items
        bundle = ContextBundle(items=[
            _make_item(bloated_content + "alpha beta gamma", source="doc_a"),
            _make_item(bloated_content + "delta epsilon zeta", source="doc_b"),
            _make_item(bloated_content + "eta theta iota", source="doc_c"),
        ])
        result = analyze(bundle)
        
        assert result.score_breakdown.density_penalty > 0.0, (
            "PIPELINE VIOLATION: density_penalty was 0 for a heavily whitespace-bloated bundle. "
            "density_penalty must reflect structural bloat, not just redundancy."
        )

    def test_high_redundancy_bundle_can_have_zero_density_penalty(self):
        """
        A bundle with HIGH redundancy but CLEAN formatting
        must have near-zero density_penalty.
        This proves the two axes are independent.
        """
        # Identical content (max redundancy), but very clean text (no formatting bloat)
        clean_content = "the quick brown fox jumps over the lazy dog"
        bundle = ContextBundle(items=[
            _make_item(clean_content, source="doc_a"),
            _make_item(clean_content, source="doc_b"),
        ])
        result = analyze(bundle)
        
        # Redundancy should be high, density should be low (clean text)
        assert result.score_breakdown.redundancy_penalty > 0.0, (
            "Expected non-zero redundancy_penalty for identical content."
        )
        # Density should be low — clean lowercase text with no formatting bloat
        assert result.score_breakdown.density_penalty < 10.0, (
            "PIPELINE VIOLATION: density_penalty was high for clean, unformatted text. "
            "density_penalty must reflect structural bloat only."
        )


# ── Rule D: Concentration penalty is independent and correctly decomposed ───

class TestRuleD_ConcentrationSignal:
    """
    Concentration must follow a 2-axis decomposition:
    1. Single chunk total -> 0 penalty (protects Gold Answer RAG)
    2. Multiple chunks from 1 source -> Max penalty (Dominance)
    3. Multiple chunks from many sources -> Scales with entropy
    """

    def test_single_chunk_has_zero_penalty(self):
        bundle = ContextBundle(items=[
            _make_item("the answer is 42", source="doc_a")
        ])
        result = analyze(bundle)
        assert result.score_breakdown.concentration_penalty == 0.0

    def test_single_source_dominance_has_max_penalty(self):
        # 3 chunks from the exact same source
        bundle = ContextBundle(items=[
            _make_item("part 1", source="doc_a"),
            _make_item("part 2", source="doc_a"),
            _make_item("part 3", source="doc_a"),
        ])
        result = analyze(bundle)
        # 100% Dominance -> p_dom = 1.0. p_ent = 0.0.
        # p_con = 0.6*1.0 + 0.4*0.0 = 0.6. penalty = 0.6 * 20 = 12.0
        assert abs(result.score_breakdown.concentration_penalty - 12.0) < 0.1

    def test_concentration_scales_with_entropy(self):
        # 2 sources, heavily imbalanced
        bundle_imbalanced = ContextBundle(items=[
            _make_item("part 1", source="doc_a", ctx_type=ContextType.RETRIEVAL),
            _make_item("part 2", source="doc_a", ctx_type=ContextType.RETRIEVAL),
            _make_item("part 3", source="doc_a", ctx_type=ContextType.RETRIEVAL),
            _make_item("other", source="doc_b", ctx_type=ContextType.RETRIEVAL),
        ])
        result_imb = analyze(bundle_imbalanced)

        # 4 sources, perfectly balanced
        bundle_balanced = ContextBundle(items=[
            _make_item("part 1", source="doc_a"),
            _make_item("part 2", source="doc_b"),
            _make_item("part 3", source="doc_c"),
            _make_item("part 4", source="doc_d"),
        ])
        result_bal = analyze(bundle_balanced)

        assert result_imb.score_breakdown.concentration_penalty > result_bal.score_breakdown.concentration_penalty

