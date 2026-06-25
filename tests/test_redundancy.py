"""Tests for the unified Redundancy Signal (RS) analyzer."""

from contextops.analyzers.redundancy import (
    analyze_redundancy,
    _compute_rs,
    _compute_ngram_overlap,
)
from contextops.core.models import (
    ContextBundle,
    ContextItem,
    ContextType,
    RedundancyClassification,
)


# ── RS unit tests ────────────────────────────────────────────────────────

def test_rs_exact_match_is_near_one() -> None:
    """RS between identical long items should be significantly above 0.5."""
    long_content = "This is a detailed refund policy document stating that all refunds take five business days."
    item_a = ContextItem(type=ContextType.RETRIEVAL, content=long_content, source="a.md", token_count=18)
    item_b = ContextItem(type=ContextType.RETRIEVAL, content=long_content, source="b.md", token_count=18)
    rs, _ = _compute_rs(item_a, item_b)
    assert rs > 0.9, f"Exact match RS for long text should be near 1.0, got {rs}"


def test_rs_completely_different_is_near_zero() -> None:
    """RS between unrelated items should be very low."""
    item_a = ContextItem(type=ContextType.RETRIEVAL, content="Quantum physics governs subatomic particles.", source="a.md", token_count=7)
    item_b = ContextItem(type=ContextType.RETRIEVAL, content="Refunds take 5 business days.", source="b.md", token_count=7)
    rs, _ = _compute_rs(item_a, item_b)
    assert rs < 0.2, f"Unrelated items RS should be near 0, got {rs}"


def test_rs_short_text_is_length_weighted() -> None:
    """RS for short sentences is dampened by the length weight."""
    # A 3-word text should produce a lower RS than an equivalent 15-word text.
    short_a = ContextItem(type=ContextType.RETRIEVAL, content="Refund five days.", source="a.md", token_count=3)
    short_b = ContextItem(type=ContextType.RETRIEVAL, content="Refund five days.", source="b.md", token_count=3)

    long_a = ContextItem(type=ContextType.RETRIEVAL, content="Refunds are typically processed within five business days of approval.", source="a.md", token_count=12)
    long_b = ContextItem(type=ContextType.RETRIEVAL, content="Refunds are typically processed within five business days of approval.", source="b.md", token_count=12)

    rs_short, _ = _compute_rs(short_a, short_b)
    rs_long, _ = _compute_rs(long_a, long_b)
    assert rs_short < rs_long, f"Short RS ({rs_short}) should be < long RS ({rs_long})"


def test_rs_ngram_fallback_on_short_text() -> None:
    """N-gram overlap should gracefully fall back to bigrams for short texts."""
    tokens_a = ["refund", "five", "days"]
    tokens_b = ["refund", "five", "days"]
    overlap = _compute_ngram_overlap(tokens_a, tokens_b)
    assert overlap > 0, "N-gram overlap for identical short tokens should be > 0"


# ── Integration tests ────────────────────────────────────────────────────

def test_exact_match_redundancy() -> None:
    """Identical strings from independent sources must be flagged as EXACT_DUPLICATE.

    Phase 2 change: the hash pre-pass now fires before RS computation, so
    content-identical items are always EXACT_DUPLICATE regardless of source.
    This is the correct behaviour — identical tokens are always wasteful.
    """
    content = "This is a detailed refund policy document stating that all refunds take five business days."
    bundle = ContextBundle(items=[
        ContextItem(type=ContextType.RETRIEVAL, content=content, source="alpha_doc.md", token_count=18),
        ContextItem(type=ContextType.RETRIEVAL, content=content, source="beta_doc.md", token_count=18),
    ])

    findings, wasted = analyze_redundancy(bundle)

    assert len(findings) == 1
    assert findings[0].classification == RedundancyClassification.EXACT_DUPLICATE
    assert findings[0].similarity_score == 1.0
    assert wasted > 0, "Exact match must produce non-zero wasted tokens"


def test_short_text_redundancy_produces_nonzero_penalty() -> None:
    """Short near-duplicate sentences (< 8 words) must produce non-zero waste.

    This is the core fix: the old N-gram engine (minimum window 8) would miss
    5-7 word sentences entirely. The unified RS catches them.
    Uses long enough text to avoid the length_weight floor killing the RS.
    """
    content = "Refunds take five business days to process after approval."
    bundle = ContextBundle(items=[
        ContextItem(type=ContextType.MESSAGE, content=content, source="doc_alpha.md", token_count=11),
        ContextItem(type=ContextType.MESSAGE, content=content, source="doc_beta.md", token_count=11),
    ])

    findings, wasted = analyze_redundancy(bundle)

    assert len(findings) >= 1, "Short duplicate sentences must generate a finding"
    assert wasted > 0, "Short duplicate sentences must produce non-zero wasted tokens"


def test_expected_overlap_adjacent_chunks() -> None:
    """Adjacent chunks from the same source must be EXPECTED_OVERLAP with discounted waste."""
    bundle = ContextBundle(items=[
        ContextItem(
            type=ContextType.RETRIEVAL,
            content="Some text that overlaps heavily with the next chunk.",
            source="doc1.md",
            token_count=20,
            metadata={"index": 1},
        ),
        ContextItem(
            type=ContextType.RETRIEVAL,
            content="overlaps heavily with the next chunk. And more text.",
            source="doc1.md",
            token_count=20,
            metadata={"index": 2},
        ),
    ])

    findings, wasted = analyze_redundancy(bundle)

    assert len(findings) == 1
    assert findings[0].classification == RedundancyClassification.EXPECTED_OVERLAP
    # Adjacent overlap does NOT contribute to wasted_tokens (only REDUNDANT_CONTEXT does)
    assert wasted == 0, "EXPECTED_OVERLAP must not count toward final wasted tokens"


def test_boilerplate_detection() -> None:
    """Repeated instructional boilerplate must be flagged as BOILERPLATE."""
    bundle = ContextBundle(items=[
        ContextItem(
            type=ContextType.SYSTEM,
            content="Please remember to always follow these rules and instructions.",
            source="system_prompt",
            token_count=15,
        ),
        ContextItem(
            type=ContextType.MESSAGE,
            content="Please ensure you always follow the instructions and rules format.",
            source="message_1",
            token_count=15,
        ),
    ])

    findings, wasted = analyze_redundancy(bundle)

    assert len(findings) == 1
    assert findings[0].classification == RedundancyClassification.BOILERPLATE
    # Boilerplate does not count as true wasted tokens
    assert wasted == 0, "BOILERPLATE must not count toward final wasted tokens"


def test_findings_are_single_source_of_truth() -> None:
    """Wasted tokens must exactly equal the sum of penalised finding waste.

    Phase 2 change: EXACT_DUPLICATE and NEAR_DUPLICATE also contribute to wasted
    tokens — they are all true waste sources. The sum must include all three
    penalised classes (EXACT_DUPLICATE, NEAR_DUPLICATE, REDUNDANT_CONTEXT).
    """
    bundle = ContextBundle(items=[
        ContextItem(type=ContextType.RETRIEVAL, content="Policy A: all refunds take 5 days to process fully.", source="doc_a.md", token_count=12),
        ContextItem(type=ContextType.RETRIEVAL, content="Policy A: all refunds take 5 days to process fully.", source="doc_b.md", token_count=12),
        ContextItem(type=ContextType.RETRIEVAL, content="Something completely unrelated about product shipping.", source="doc_c.md", token_count=10),
    ])

    findings, wasted = analyze_redundancy(bundle)

    expected_waste = sum(
        f.estimated_waste_tokens
        for f in findings
        if f.classification in (
            RedundancyClassification.EXACT_DUPLICATE,
            RedundancyClassification.NEAR_DUPLICATE,
            RedundancyClassification.REDUNDANT_CONTEXT,
        )
    )
    assert wasted == expected_waste, (
        f"Wasted tokens ({wasted}) must equal sum of penalised finding waste ({expected_waste})"
    )
