"""Tests for the redundancy analyzer."""

from contextops.analyzers.redundancy import analyze_redundancy
from contextops.core.models import (
    ContextBundle,
    ContextItem,
    ContextType,
    RedundancyClassification,
)


def test_exact_match_redundancy() -> None:
    """Test that identical strings from independent sources are flagged as REDUNDANT_CONTEXT."""
    bundle = ContextBundle(items=[
        ContextItem(
            type=ContextType.RETRIEVAL,
            content="This is the exact same text.",
            source="alpha_doc.md",
            token_count=10,
        ),
        ContextItem(
            type=ContextType.RETRIEVAL,
            content="This is the exact same text.",
            source="beta_doc.md",
            token_count=10,
        ),
    ])

    findings, _ = analyze_redundancy(bundle)

    assert len(findings) == 1
    assert findings[0].classification == RedundancyClassification.REDUNDANT_CONTEXT
    assert findings[0].similarity_score == 1.0
    assert findings[0].estimated_waste_tokens == 10


def test_expected_overlap_adjacent_chunks() -> None:
    """Test that adjacent chunks from the same source are EXPECTED_OVERLAP."""
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

    findings, _ = analyze_redundancy(bundle)

    assert len(findings) == 1
    assert findings[0].classification == RedundancyClassification.EXPECTED_OVERLAP
    assert findings[0].estimated_waste_tokens == 4  # 80% discount (20 * 0.2)


def test_boilerplate_detection() -> None:
    """Test that repeated instructional boilerplate is flagged as BOILERPLATE."""
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

    findings, _ = analyze_redundancy(bundle)

    assert len(findings) == 1
    assert findings[0].classification == RedundancyClassification.BOILERPLATE
